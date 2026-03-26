import json
from builtins import int
import pickle
import numpy as np 
import cv2 
import pdb
from typing import Sequence, Tuple, Dict, Optional, Union, Generator
from multiprocessing import Process
import os
import pathlib
import click
import imageio
import shutil
import fpsample
import gzip
from tqdm import tqdm
from common.cv2_util import get_image_transform_resize_crop, intrinsic_transform_resize
from common.cv_util import back_projection
from common.timestamp_accumulator import get_accumulate_timestamp_idxs
from common.replay_buffer import ReplayBuffer
from common.svo_utils import SVOReader
from common.pose_util import euler_pose_to_mat, mat_to_pose, mat_to_euler_pose, pose_to_mat
from common.interpolation_util import PoseInterpolator, get_interp1d
from human_data.constants import yfxrzu2standard
from scripts_data.entry.vis_helper import parse_to_se3
from human_data.hand_retargeting import Hand_Retargeting
hand_retargeting = Hand_Retargeting("./real/teleop/inspire_hand_0_4_6.yml")


def get_eef_pos_velocity(eef_pos_seq):
    delta = np.linalg.norm(eef_pos_seq[1:] - eef_pos_seq[:-1], axis=-1)
    vel = delta.mean()
    return vel


def fast_mat_inv(mat):
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret


def conversion_single_trajectory(
    mode,
    save_dir, 
    calib_quest2camera,
    speed_downsample_ratio,
    single_arm,
    hand_shrink_coef,
    out_resolutions_resize: Union[None, tuple, Dict[str,tuple]]=None, # (width, height)
    out_resolutions_crop: Union[None, tuple, Dict[str,tuple]]=None, # (width, height)
    out_resolutions_image_final: Union[None, tuple, Dict[str,tuple]]=None, # (width, height)
    network_delay_checking: float=1.0,
    num_points_final: int = 2048, 
    points_max_distance_final: float = 1.25,
    ):

    # ==================== Vision Pro: Load episode & build hand poses ====================

    episode_path = os.path.join(save_dir, "episode.pkl")
    if not os.path.exists(episode_path):
        print(f"[Warning] No episode.pkl found in {save_dir}")
        return None
    try:
        with gzip.open(episode_path, "rb") as f:
            episode_org = pickle.load(f)
    except OSError:
        # fallback for non-gzip pickle
        with open(episode_path, "rb") as f:
            episode_org = pickle.load(f)

    
    print("original intrinsic1:")
    print(episode_org['camera_intrinsics'][0])

    # ---------------------------------------------------------------------
    # 1. timestamps (already aligned to video frames)
    # ---------------------------------------------------------------------
    timestamps = np.asarray(episode_org["frame_timestamps"], dtype=np.float64)
    timestamps = timestamps - timestamps[0]  # make relative time
    T = len(timestamps)

    print(f"[VisionPro] Episode length: {T}")

    # sanity check
    assert len(episode_org["pose_snapshots"]) == T
    assert len(episode_org["camera_extrinsics"]) == T

    # ---------------------------------------------------------------------
    # 2. build camera pose in camera0 frame (SE(3))
    # ---------------------------------------------------------------------

    camera_extrinsics = episode_org["camera_extrinsics"]

    print("camera_extrinsics[0] shape:", camera_extrinsics[0].shape)
    print("camera_extrinsics[0]:\n", camera_extrinsics[0])

    world_T_camera = []

    for t in range(T):

        snap = episode_org["pose_snapshots"][t]

        # -------- device pose --------
        if snap["left"] is not None:
            device_rt = snap["left"]["response"].device
        elif snap["right"] is not None:
            device_rt = snap["right"]["response"].device
        else:
            raise RuntimeError(f"No device pose at frame {t}")

        world_T_device = parse_to_se3(device_rt)

        # -------- camera extrinsics --------
        raw_ext = camera_extrinsics[t]

        # Apple matrix: translation stored in last row
        device_T_camera = raw_ext.T
        device_T_camera = np.linalg.inv(device_T_camera) 

        # world → camera
        world_T_camera_t = world_T_device @ device_T_camera

        world_T_camera.append(world_T_camera_t)

    world_T_camera = np.stack(world_T_camera, axis=0)

    # normalize to camera0 frame
    camera0_T_world = np.linalg.inv(world_T_camera[0])

    camera_pose = camera0_T_world @ world_T_camera





    # ---------------------------------------------------------------------
    # 3. build hand poses (anchor + full skeleton) in camera0 frame
    # ---------------------------------------------------------------------
    hand_poses = {
        "left":  {"anchor": [], "joints": []},
        "right": {"anchor": [], "joints": []},
    }

    for t in range(T):
        snap = episode_org["pose_snapshots"][t]

        for side in ["left", "right"]:
            if snap[side] is None:
                hand_poses[side]["anchor"].append(None)
                hand_poses[side]["joints"].append(None)
                continue

            resp = snap[side]["response"]

            # world_T_anchor
            world_T_anchor = parse_to_se3(resp.hand.anchor_transform)

            # camera0_T_anchor
            cam0_T_anchor = camera0_T_world @ world_T_anchor
            hand_poses[side]["anchor"].append(cam0_T_anchor)

            # joints
            joint_dict = {}
            skel = resp.hand.hand_skeleton

            for field_desc, value in skel.ListFields():
                joint_name = field_desc.name
                anchor_T_joint = parse_to_se3(value)
                joint_dict[joint_name] = cam0_T_anchor @ anchor_T_joint

            hand_poses[side]["joints"].append(joint_dict)

    # At this point we have:
    # - timestamps              (T,)
    # - camera_pose             (T, 4, 4)
    # - hand_poses[side]["anchor"][t]  -> SE(3)
    # - hand_poses[side]["joints"][t][joint_name] -> SE(3)
    # ==============================================================================


    # import pdb; pdb.set_trace()

   

    # ========================== VisionPro Joint Extraction (FIXED COORDINATE SYSTEM) ==========================

    REQUIRED_JOINTS = [
        "wrist",
        "thumb_tip",
        "index_finger_tip",
        "middle_finger_tip",
    ]

    left_joints  = {name: np.zeros((T,6)) for name in REQUIRED_JOINTS}
    right_joints = {name: np.zeros((T,6)) for name in REQUIRED_JOINTS}

    for t in range(T):

        for side, joint_storage in [
            ("left", left_joints),
            ("right", right_joints)
        ]:

            joint_dict = hand_poses[side]["joints"][t]

            if joint_dict is None:

                if t > 0:
                    for j in REQUIRED_JOINTS:
                        joint_storage[j][t] = joint_storage[j][t-1]

                continue

            for j in REQUIRED_JOINTS:

                if j in joint_dict:

                    joint_storage[j][t] = mat_to_pose(joint_dict[j])

                else:

                    if t > 0:
                        joint_storage[j][t] = joint_storage[j][t-1]



    # ========================== Wrist Pose (Robot EEF) ==========================

    left_hand_fix_wrists  = left_joints["wrist"]
    right_hand_fix_wrists = right_joints["wrist"]

    left_hand_wrists  = left_hand_fix_wrists.copy()
    right_hand_wrists = right_hand_fix_wrists.copy()


    # ========================== Compute Panda Gripper Width ==========================

    left_gripper  = np.zeros((T,1))
    right_gripper = np.zeros((T,1))

    #ref = 0.11  # 11 cm

    for t in range(T):

        # RIGHT HAND
        thumb = pose_to_mat(right_joints["thumb_tip"][t])[:3,3]
        index = pose_to_mat(right_joints["index_finger_tip"][t])[:3,3]

        d = np.linalg.norm(thumb - index)
        right_gripper[t,0] = min(d , 0.11)


        # LEFT HAND
        thumb = pose_to_mat(left_joints["thumb_tip"][t])[:3,3]
        index = pose_to_mat(left_joints["index_finger_tip"][t])[:3,3]

        d = np.linalg.norm(thumb - index)
        left_gripper[t,0] = min(d , 0.11)


    # ========================== Episode Construction ==========================

    episode = dict()

    episode['timestamp'] = timestamps
    episode_length = len(episode['timestamp'])

    episode['camera0_pose'] = mat_to_pose(camera_pose)

    episode['left_wrist_pose'] = left_hand_wrists
    episode['right_wrist_pose'] = right_hand_wrists

    episode['left_wrist_fix_pose'] = left_hand_fix_wrists
    episode['right_wrist_fix_pose'] = right_hand_fix_wrists


    # ========================== Robot EEF ==========================

    episode['robot0_eef_pos'] = right_hand_fix_wrists[:, :3]
    episode['robot0_eef_rot_axis_angle'] = right_hand_fix_wrists[:, 3:]

    if single_arm is False:

        episode['robot1_eef_pos'] = left_hand_fix_wrists[:, :3]
        episode['robot1_eef_rot_axis_angle'] = left_hand_fix_wrists[:, 3:]


    # ========================== Panda Gripper ==========================

    episode['gripper0_gripper_pose'] = right_gripper

    if single_arm is False:
        episode['gripper1_gripper_pose'] = left_gripper


    # ========================== Build Action ==========================

    episode['action'] = np.concatenate([
        episode['robot0_eef_pos'],
        episode['robot0_eef_rot_axis_angle'],
        episode['gripper0_gripper_pose'],
    ], axis=-1)

    if single_arm is False:

        episode['action'] = np.concatenate([
            episode['action'],
            episode['robot1_eef_pos'],
            episode['robot1_eef_rot_axis_angle'],
            episode['gripper1_gripper_pose'],
        ], axis=-1)




    # ========================== Transformation to Egocentric View (MP4 backend, mode='o') ==========================

    assert mode == 'o', "MP4 backend only supports mode='o'"

    # ---------------------------------------------------------------------
    # 0. device_id handling (keep compatibility, add fallback)
    # ---------------------------------------------------------------------

    device_id_fp = os.path.join(save_dir, "device_id.txt")
    if os.path.exists(device_id_fp):
        with open(device_id_fp, "r") as f:
            serial_id = f.read().strip()
        if serial_id == "":
            print(f"[Warning] device_id.txt is empty in {save_dir}, fallback to '0'")
            serial_id = "0"
    else:
        print(f"[Info] No device_id.txt found in {save_dir}, fallback to '0'")
        serial_id = "0"

    # ---------------------------------------------------------------------
    # 1. Open MP4 video (replace SVOReader)
    # ---------------------------------------------------------------------

    video_path = os.path.join(save_dir, "main_camera.mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"MP4 file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ---------------------------------------------------------------------
    # 2. Camera intrinsics & camera_info (same logic as original SVO version)
    # ---------------------------------------------------------------------

    camera_info = {}

    camera_info['left_intrinsic'] = intrinsic_transform_resize(
        episode_org['camera_intrinsics'][0].copy(),
        input_res=(width, height),
        output_resize_res=out_resolutions_resize,
        output_crop_res=out_resolutions_crop
    )
    print("left intrinsic:")
    print(camera_info['left_intrinsic'])

    # mode='o' → no stereo, right == left
    camera_info['right_intrinsic'] = camera_info['left_intrinsic']

    camera_info['left_intrinsic_final'] = intrinsic_transform_resize(
        camera_info['left_intrinsic'].copy(),
        input_res=out_resolutions_crop,
        output_resize_res=out_resolutions_image_final,
        output_crop_res=out_resolutions_image_final
    )

    print("left intrinsic final:")
    print(camera_info['left_intrinsic_final'])
    camera_info['right_intrinsic_final'] = camera_info['left_intrinsic_final']

    # stereo not used, keep identity for compatibility
    camera_info['stereo_transform'] = np.eye(4)

    camera_info_key = [
        'stereo_transform',
        'left_intrinsic',
        'right_intrinsic',
        'left_intrinsic_final',
        'right_intrinsic_final'
    ]

    for key in camera_info_key:
        episode["camera0_" + key] = np.array([camera_info[key]] * episode_length)

    # ---------------------------------------------------------------------
    # 3. obs_dict construction (structure preserved)
    # ---------------------------------------------------------------------

    obs_dict = dict()
    episode['camera0_real_timestamp'] = np.zeros((episode_length,), dtype=np.float64)

    transform_img = get_image_transform_resize_crop(
        input_res=(width, height),
        output_resize_res=out_resolutions_resize,
        output_crop_res=out_resolutions_crop,
        bgr_to_rgb=True
    )

    # key naming kept for semantic consistency
    obs_dict['rgb'] = ('image', f'{serial_id}_left', transform_img)

    start_time = episode['timestamp'][0]

    # ---------------------------------------------------------------------
    # 4. frame_cut logic (same semantics as original)
    # ---------------------------------------------------------------------

    frame_cut_fp = os.path.join(save_dir, "frame_cut.txt")
    frame_cut = None
    if os.path.exists(frame_cut_fp):
        with open(frame_cut_fp, "r") as f:
            frame_cut = f.read().strip()
        if frame_cut.isdigit():
            frame_cut = int(
                (episode_org['frame_timestamps'][int(frame_cut)] - start_time) / dt
            )
        else:
            print(f"Frame cut {frame_cut} is not a digit, set to None.")
            frame_cut = None

    if frame_cut is not None:
        episode_length = min(episode_length, frame_cut)

    for episode_key in episode.keys():
        episode[episode_key] = episode[episode_key][:episode_length]

    # ---------------------------------------------------------------------
    # 5. Allocate storage (same pattern as original)
    # ---------------------------------------------------------------------

    episode['camera0_rgb'] = np.zeros(
        (episode_length, out_resolutions_image_final[1], out_resolutions_image_final[0], 3),
        dtype=np.uint8
    )
    print("episode_length:", episode_length)

    camera_rgb_frames = [None] * episode_length

    # ---------------------------------------------------------------------
    # 6. Frame reading loop (MP4 instead of SVO, but same write logic)
    # ---------------------------------------------------------------------

    global_idx = 0

    for t in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        if global_idx >= episode_length:
            break

        # original SVO: data_dict['image'][<key>]
        value = frame[..., :3]  # BGR

        transform = obs_dict['rgb'][2]
        value = transform(value)

        camera_rgb_frames[global_idx] = value.copy()

        # final resize (same as original rgb path)
        value = cv2.resize(
            value,
            out_resolutions_image_final,
            interpolation=cv2.INTER_LINEAR
        )

        episode['camera0_rgb'][global_idx] = value
        episode['camera0_real_timestamp'][global_idx] = episode['timestamp'][global_idx]

        global_idx += 1

    cap.release()

    # ==========================================================================================

    # ========================== Post-process episode length (MP4 version) ==========================

    # In MP4 mode, global_idx is the number of valid frames actually written
    if global_idx < episode_length:
        abandoned_frames = episode_length - global_idx
        for key in episode.keys():
            try:
                episode[key] = episode[key][:global_idx]
            except Exception:
                pass
        episode_length = global_idx
        print(f"[MP4] Warning: only {global_idx} frames written, abandoned {abandoned_frames} frames.")

    # ---------------------------------------------------------------------
    # Final length alignment (keep original semantics)
    # ---------------------------------------------------------------------

    n_length = min(
        episode['timestamp'].shape[0],
        episode['camera0_real_timestamp'].shape[0],
    )

    for key in episode.keys():
        try:
            episode[key] = episode[key][:n_length]
        except Exception:
            pass

    print(f"length: {n_length}")

    # ========================== frame_grasp / frame_release handling ==========================

    frame_grasp_fp = os.path.join(save_dir, "frame_grasp.txt")
    frame_grasp = None
    if os.path.exists(frame_grasp_fp):
        with open(frame_grasp_fp, "r") as f:
            frame_grasp = f.read().strip()
        if frame_grasp.isdigit():
            frame_grasp = int(frame_grasp)
        else:
            print(f"Frame frame_grasp {frame_grasp} is not a digit, set to None.")
            frame_grasp = None

    frame_release_fp = os.path.join(save_dir, "frame_release.txt")
    frame_release = None
    if os.path.exists(frame_release_fp):
        with open(frame_release_fp, "r") as f:
            frame_release = f.read().strip()
        if frame_release.isdigit():
            frame_release = int(frame_release)
        else:
            print(f"Frame frame_release {frame_release} is not a digit, set to None.")
            frame_release = None

    print(f"frame_grasp: {frame_grasp}, frame_release: {frame_release}")

    # Clip grasp / release to valid range
    if frame_grasp is not None:
        frame_grasp = max(0, min(frame_grasp, n_length - 1))
    if frame_release is not None:
        frame_release = max(0, min(frame_release, n_length - 1))

    # ========================== gripper override logic (unchanged semantics) ==========================

    if frame_grasp is not None:
        assert hand_shrink_coef == 1.0, \
            "Can not set hand_shrink_coef != 1.0 when use frame_grasp and frame_release handler."

        if frame_release is None:
            frame_release = n_length - 1

        gripper0_max_pose = np.max(
            episode['gripper0_gripper_pose'][frame_grasp:frame_release + 1],
            axis=0
        )

        if single_arm is False:
            gripper1_max_pose = np.max(
                episode['gripper1_gripper_pose'][frame_grasp:frame_release + 1],
                axis=0
            )

        assert frame_grasp >= 10 + 1, "frame_grasp must be >= 11 for interpolation window"

        episode['gripper0_gripper_pose'][frame_grasp:frame_release + 1] = gripper0_max_pose[None]

        if single_arm is False:
            episode['gripper1_gripper_pose'][frame_grasp:frame_release + 1] = gripper1_max_pose[None]

        # smooth before grasp
        gripper_interp = get_interp1d(
            episode['timestamp'][[frame_grasp - 10 - 1, frame_grasp]],
            episode['gripper0_gripper_pose'][[frame_grasp - 10 - 1, frame_grasp]]
        )
        episode['gripper0_gripper_pose'][frame_grasp - 10:frame_grasp] = \
            gripper_interp(episode['timestamp'][frame_grasp - 10:frame_grasp])

        if single_arm is False:
            gripper_interp = get_interp1d(
                episode['timestamp'][[frame_grasp - 10 - 1, frame_grasp]],
                episode['gripper1_gripper_pose'][[frame_grasp - 10 - 1, frame_grasp]]
            )
            episode['gripper1_gripper_pose'][frame_grasp - 10:frame_grasp] = \
                gripper_interp(episode['timestamp'][frame_grasp - 10:frame_grasp])

        # smooth after release
        if frame_release < n_length - 1:
            frame_release_finish = min(n_length - 1, frame_release + 10)

            gripper_interp = get_interp1d(
                episode['timestamp'][[frame_release, frame_release_finish]],
                episode['gripper0_gripper_pose'][[frame_release, frame_release_finish]]
            )
            episode['gripper0_gripper_pose'][frame_release + 1:frame_release_finish] = \
                gripper_interp(episode['timestamp'][frame_release + 1:frame_release_finish])

            if single_arm is False:
                gripper_interp = get_interp1d(
                    episode['timestamp'][[frame_release, frame_release_finish]],
                    episode['gripper1_gripper_pose'][[frame_release, frame_release_finish]]
                )
                episode['gripper1_gripper_pose'][frame_release + 1:frame_release_finish] = \
                    gripper_interp(episode['timestamp'][frame_release + 1:frame_release_finish])

    # ========================== action construction (unchanged) ==========================

    episode['action'] = np.concatenate(
        [
            episode['robot0_eef_pos'],
            episode['robot0_eef_rot_axis_angle'],
            episode['gripper0_gripper_pose'],
        ],
        axis=-1
    )

    if single_arm is False:
        episode['action'] = np.concatenate(
            [
                episode['action'],
                episode['robot1_eef_pos'],
                episode['robot1_eef_rot_axis_angle'],
                episode['gripper1_gripper_pose'],
            ],
            axis=-1
    )

  ## -------------- VisionPro Gripper Debug Visualization ----------------##

    print("========== VisionPro GRIPPER DEBUG ==========")

    joint_names = [
        "wrist",
        "thumb_tip",
        "index_finger_tip",
        "middle_finger_tip"
    ]

    for idx,name in enumerate(joint_names):
        print(idx,name)

    video_frames = []

    PRINT_FIRST_N_FRAMES = 5
    print("rgb:", len(episode['camera0_rgb']))
    print("right thumb:", len(right_joints["thumb_tip"]))
    print("left thumb:", len(left_joints["thumb_tip"]))
    print("camera_rgb_frames:", len(camera_rgb_frames))

    for t in range(len(camera_rgb_frames)):

        frame = episode['camera0_rgb'][t].copy()

        K = episode["camera0_left_intrinsic_final"][t]

        h_img, w_img = frame.shape[:2]

        for handside,color,joints,gripper in [
            ("left",(0,255,0),left_joints,left_gripper),
            ("right",(255,0,0),right_joints,right_gripper)
        ]:

            joint_uvs = {}

            for joint_name in joint_names:

                pose = joints[joint_name][t]

                # joint in camera0
                T_joint_cam0 = pose_to_mat(pose)

                # transform to camera_t
                cam_t_T_cam0 = np.linalg.inv(camera_pose[t])
                T_joint_camt = cam_t_T_cam0 @ T_joint_cam0

                xyz = T_joint_camt[:3,3]

                if xyz[2] <= 1e-6:
                    print("z invalid:", xyz)
                    joint_uvs[joint_name] = None
                    continue

                u = K[0,0]*(xyz[0]/xyz[2]) + K[0,2]
                v = K[1,1]*(xyz[1]/xyz[2]) + K[1,2]

                uv = np.array([u,v]).astype(int)

                joint_uvs[joint_name] = uv

                # if t < PRINT_FIRST_N_FRAMES:

                #     # print(f"\nFrame {t} | {handside} | {joint_name}")
                #     # print("xyz_cam:",xyz)
                #     # print("uv:",uv)

                #     in_img = (0 <= uv[0] < w_img) and (0 <= uv[1] < h_img)
                #     print("in_image:",in_img)

                if 0 <= uv[0] < w_img and 0 <= uv[1] < h_img:

                    cv2.circle(frame,tuple(uv),6,color,-1)

                    # cv2.putText(
                    #     frame,
                    #     joint_name,
                    #     (uv[0],uv[1]+12),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.4,
                    #     (255,255,255),
                    #     1
                    # )

            # ----- draw finger connections -----

            thumb = joint_uvs.get("thumb_tip")
            index = joint_uvs.get("index_finger_tip")
            middle = joint_uvs.get("middle_finger_tip")

            if thumb is not None and index is not None:
                cv2.line(frame,tuple(thumb),tuple(index),(0,255,255),2)

            if thumb is not None and middle is not None:
                cv2.line(frame,tuple(thumb),tuple(middle),(255,255,0),2)

            # ----- draw gripper width -----

            width_val = gripper[t,0]

            cv2.putText(
                frame,
                f"{handside} gripper: {width_val:.3f}",
                (20,40 if handside=="left" else 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,255),
                2
            )

        video_frames.append(frame)


    vis_dir = os.path.join("/data/zeqingwang/visualizations", save_dir)
    os.makedirs(vis_dir,exist_ok=True)

    video_path = os.path.join(vis_dir,"visionpro_gripper_debug.mp4")

    imageio.mimsave(video_path,video_frames,fps=10,macro_block_size=1)

    print("[OK] VisionPro gripper debug video saved:",video_path)





    ##----------------------------------------##
    
    # # save episode['camera0_rgb_right'] as test.mp4
    # if 'camera0_rgb_right' in episode.keys():
    #     rgb_right = episode['camera0_rgb_right']
    #     if rgb_right.shape[0] > 0:
    #         video_writer = imageio.get_writer("test.mp4", fps=fps)
    #         for i in range(rgb_right.shape[0]):
    #             video_writer.append_data(rgb_right[i])
    #         video_writer.close()
    #     else:
    #         print("No right camera data found, skip saving test.mp4.") 
        
    # for key in episode.keys():
    #     try:
    #         print(f"Key: {key}, shape: {episode[key].shape}")
    #     except:
    #         print(f"Key: {key}, {episode[key]}")
    
    return episode


def conversion_trajectory(input_data_fp_list, calib_quest2camera, speed_downsample_ratio, single_arm,
                          hand_shrink_coef, 
                          mode, 
                          out_resolutions_resize, out_resolutions_crop, resolution_image_final, num_points_final, points_max_distance_final,
                          replay_buffer, 
                          network_delay_checking,
                          process_id):
    pbar = tqdm(enumerate(input_data_fp_list), desc=f"Process {process_id}")
    for i, input_data_fp in pbar:
        save_dir, source, source_idx = input_data_fp
        episode = conversion_single_trajectory(
            mode,
            save_dir, 
            calib_quest2camera,
            speed_downsample_ratio,
            single_arm,
            hand_shrink_coef,
            out_resolutions_resize,
            out_resolutions_crop,
            resolution_image_final,
            network_delay_checking,
            num_points_final, points_max_distance_final,
        )
        if episode is None:
            pbar.write(f"Skip {save_dir} due to network delay or no data.")
            continue
        episode['embodiment'] = np.zeros((len(episode['robot0_eef_pos']), 1))
        episode['source_idx'] = np.ones((len(episode['robot0_eef_pos']), 1)) * source_idx
        if episode is not None:
            replay_buffer.add_episode(episode, compressors='disk')  # with lock mechanism inside replay_buffer instance



@click.command()
@click.option('--input_dir', '-i',  required=True)
@click.option('--output', '-o', required=True)
@click.option('--calib_quest2camera_file', '-cf', required=True)
@click.option('--adapt_config_file', '-acf', default=None, type=str)
@click.option('--single_arm', '-cf', is_flag=True, default=False)
@click.option('--default_speed_downsample_ratio', '-dsdr', default=1.0, type=float)
@click.option('--default_hand_shrink_coef', '-dhsc', default=1.0, type=float)
@click.option('--mode', '-m', required=True, type=click.Choice(['o', 'p', 's', 'a'], case_sensitive=False), default='o',
    help="o: only image, p: with pointcloud, s: with stereo-image, a: with all, including pointcloud and stereo")
@click.option('--resolution_resize', '-ror', default='640x480')
@click.option('--resolution_crop', '-or', default='640x480')
@click.option('--resolution_image_final', '-for', default='224x224')
@click.option('--num_use_source', '-nus', default=None, type=int)
@click.option('--num_points_final', '-npf', type=int, default=2048)
@click.option('--points_max_distance_final', '-pmdf', type=float, default=1.0)
@click.option('--n_encoding_threads', '-ne', default=-1, type=int)
@click.option('--network_delay_checking', '-dl', default=0.5, help="Max network delay for tolerance.")
def main(input_dir, output, calib_quest2camera_file, 
         adapt_config_file, single_arm,
         default_speed_downsample_ratio, default_hand_shrink_coef,
         mode, 
         resolution_resize, resolution_crop, resolution_image_final, num_use_source, num_points_final, points_max_distance_final,
         n_encoding_threads, 
         network_delay_checking
        ):
    out_resolution_resize = tuple(int(x) for x in resolution_resize.split('x'))
    out_resolution_crop = tuple(int(x) for x in resolution_crop.split('x'))
    resolution_image_final = tuple(int(x) for x in resolution_image_final.split('x'))

    if adapt_config_file is not None:
        # load json file
        with open(adapt_config_file, 'r') as f:
            adapt_config = json.load(f)

    input_dir_list = os.listdir(input_dir)
    input_dir_list = [os.path.join(input_dir, x) for x in input_dir_list]

    for input_dir in input_dir_list:
        input_folder = input_dir.split('/')[-1]
        embodiment = input_folder.split('_')[0]
        assert embodiment == "human"
        environment_setting = input_folder.split('_')[1]
        instruction = '_'.join(input_folder.split('_')[2:])
        
        replay_buffer_fp = os.path.join(output, embodiment + "_" + environment_setting + "+" + instruction + "+")
        
        if environment_setting == "me":  # multi-environment
            if num_use_source is not None and num_use_source < 0:
                num_use_source = None
            if num_use_source is not None:
                replay_buffer_fp = replay_buffer_fp + "_src" + str(num_use_source)
        replay_buffer_fp = replay_buffer_fp + ".zarr"
        replay_buffer_fp = pathlib.Path(os.path.expanduser(replay_buffer_fp))

        if replay_buffer_fp.exists():
            shutil.rmtree(replay_buffer_fp)
        
        replay_buffer = ReplayBuffer.create_from_path(replay_buffer_fp, mode='a')

        if environment_setting == "me":  # multi-environment
            input_data_fp_list = []
            source_list = os.listdir(input_dir)
            source_list.sort()
            for sidx, source in enumerate(source_list):
                if num_use_source is not None and sidx == num_use_source:
                    print(f"Use Source: {source_list[:num_use_source]}")
                    break
                tmp_fp_list = os.listdir(os.path.join(input_dir, source))
                tmp_fp_list = [(os.path.join(input_dir, source, fp), source, sidx) for fp in tmp_fp_list]
                input_data_fp_list = input_data_fp_list + tmp_fp_list
                if sidx + 1 == num_use_source:
                    print(f"Use Source: {source_list[:num_use_source]}")
                    print("Use All Sources.")
                    break
        else:
            input_data_fp_list = os.listdir(input_dir)
            input_data_fp_list.sort()
            input_data_fp_list = [(os.path.join(input_dir, fp), "default", 0) for fp in input_data_fp_list] 
        calib_quest2camera = np.load(calib_quest2camera_file)

        speed_downsample_ratio = None
        hand_shrink_coef = None
        if input_folder in adapt_config:
            adapt_param = adapt_config[input_folder]
            if 'speed_downsample_ratio' in adapt_param:
                speed_downsample_ratio = adapt_param['speed_downsample_ratio']
            else:
                speed_downsample_ratio = default_speed_downsample_ratio
            if 'hand_shrink_coef' in adapt_param:
                hand_shrink_coef = adapt_param['hand_shrink_coef']
            else:
                hand_shrink_coef = default_hand_shrink_coef
        else:
            hand_shrink_coef, speed_downsample_ratio = default_hand_shrink_coef, default_speed_downsample_ratio

        print(f"Input Directory: {input_dir}")
        print(f"Output Directory: {replay_buffer_fp}")
        print(f"Hand Shrink Coefficient: {hand_shrink_coef}")
        print(f"Speed Downsample Refer Zarr: {speed_downsample_ratio}")

        if n_encoding_threads > 1:
            input_data_fp_batch_list = []
            for i in range(n_encoding_threads):
                input_data_fp_batch_list.append(input_data_fp_list[i::n_encoding_threads])

            process_list = []
            for i in range(n_encoding_threads):
                p = Process(target=conversion_trajectory, args=(input_data_fp_batch_list[i], calib_quest2camera, speed_downsample_ratio, single_arm, hand_shrink_coef, mode, out_resolution_resize, out_resolution_crop, resolution_image_final, num_points_final, points_max_distance_final, replay_buffer, network_delay_checking, i))
                p.start()
                process_list.append(p)

            for p in process_list:
                p.join()
        else:
            conversion_trajectory(input_data_fp_list, calib_quest2camera, speed_downsample_ratio, single_arm, hand_shrink_coef, mode, out_resolution_resize, out_resolution_crop, resolution_image_final, num_points_final, points_max_distance_final, replay_buffer, network_delay_checking, 0)
        
        print(f"Saving to disk finish: Task {input_dir}")


if __name__ == "__main__":
    # test_conversion()
    main()