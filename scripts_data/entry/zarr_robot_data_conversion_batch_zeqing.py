# ===============================================================
# Robot Teleoperation Data Alignment
# support: image, stereo, pointclouds (from sensor)
# ===============================================================

import pickle
import numpy as np
import cv2
import pyzed.sl as sl
from typing import Sequence, Tuple, Dict, Optional, Union, Generator
import os
import pathlib
import yaml
import click
import fpsample
import shutil
from copy import deepcopy
from multiprocessing import Process
from tqdm import tqdm
from common.cv2_util import get_image_transform_resize_crop
from common.cv_util import egocentric_to_base_obs_transformation
from common.timestamp_accumulator import get_accumulate_timestamp_idxs
from common.cv2_util import get_image_transform_resize_crop, intrinsic_transform_resize
from common.replay_buffer import ReplayBuffer
from common.svo_utils import SVOReader
from common.interpolation_util import PoseInterpolator, get_interp1d
from real.teleop.teleop_utils import pose_to_mat, mat_to_pose

from glob import glob 


def get_eef_pos_velocity(eef_pos_seq):
    delta = np.linalg.norm(eef_pos_seq[1:] - eef_pos_seq[:-1], axis=-1)
    vel = delta.mean()
    return vel

def load_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)  # Use safe_load to prevent code execution
    return data

def project_cam_to_pixel(point_cam, intrinsic):
    point_2d_h = intrinsic @ point_cam
    point_2d = point_2d_h[:2] / point_2d_h[2]
    return point_2d.astype(int)


def conversion_single_trajectory(
        hand_to_eef,
        mode,
        save_dir,
        out_resolutions_resize: Union[None, tuple, Dict[str, tuple]] = None,  # (width, height)
        out_resolutions_crop: Union[None, tuple, Dict[str, tuple]] = None,  # (width, height)
        out_resolutions_image_final: Union[None, tuple, Dict[str, tuple]] = None,  # (width, height)
        num_points_final: int = 2048, 
        points_max_distance_final: float = 1.25,
):
    processed_dir = os.path.join(save_dir, "processed")

    # ==========================================================
    # read pose_data.yaml to load timestamp
    # ==========================================================
    meta_data_file = os.path.join(save_dir, "processed", "metadata.yaml")
    if not os.path.isfile(meta_data_file):
        print(f"{meta_data_file} does not exist. Skipping. episode_dir: {episode_dir}.") 
    else:
        meta_data = load_yaml(meta_data_file)
    camera_names = {val:key for key, val in meta_data["camera_id_to_semantic_name"].items()}
    camera_id = camera_names["scene_right_0"]
    camera_dir = os.path.join(processed_dir, f"images_{camera_id}")
    pose_yaml_path = os.path.join(camera_dir, "pose_data.yaml")

    with open(pose_yaml_path, "r") as f:
        pose_data = yaml.safe_load(f)

    timestamps = []
    for i in sorted(pose_data.keys(), key=lambda x: int(x)):
        timestamps.append(pose_data[i]["timestamp"])

    timestamps = np.array(timestamps, dtype=np.float64)

    timestamps = timestamps / 1e6

    T_ts = len(timestamps)

    # ==========================================================
    # load observations
    # ==========================================================
    obs_path = os.path.join(processed_dir, "observations.npz")
    observations = np.load(obs_path)

    # # ==========================================================
    # # load actions(TODO)
    # # ==========================================================
    # actions_path = os.path.join(processed_dir, "actions.npz")
    # actions_data = np.load(actions_path)
    # actions = actions_data["actions"]   # (83, 20)

    episode = dict()

    episode["timestamp"] = timestamps
    episode["camera0_real_timestamp"] = timestamps.copy()

    # # remove the timestamp where robot0_eef_pos does not change
    # start_time_idx = 0
    # eps = 0.005
    # for i in range(1, len(episode[f'robot0_eef_pos'])):
    #     if np.linalg.norm(episode[f'robot0_eef_pos'][i] - episode[f'robot0_eef_pos'][0]) > eps:
    #         start_time_idx = i
    #         break
    # for key in episode.keys():
    #     episode[key] = episode[key][start_time_idx:]

    # dt = (episode['timestamp'][1:] - episode['timestamp'][:-1]).mean()

    # # ========================== Remove Initialization Stage Data =================================

    # t_init = 0
    # while (episode['action'][t_init] == 0.).all():
    #     t_init += 1
    # for key in episode.keys():
    #     episode[key] = episode[key][t_init:]
    # episode_length = len(episode['timestamp'])

    # ========================== Transformation to Egocentric View =================================
    from scipy.spatial.transform import Rotation as R

    # -------------------------------------------------
    # read pose
    # -------------------------------------------------

    left_xyz_world  = observations["robot__actual__poses__left::panda__xyz"]
    right_xyz_world = observations["robot__actual__poses__right::panda__xyz"]

    left_rot6d_world  = observations["robot__actual__poses__left::panda__rot_6d"]
    right_rot6d_world = observations["robot__actual__poses__right::panda__rot_6d"]

    left_grip  = observations["robot__actual__grippers__left::panda_hand"]
    right_grip = observations["robot__actual__grippers__right::panda_hand"]
    print("Left grip range: ", left_grip.min(), left_grip.max())
    print("Right grip range: ", right_grip.min(), right_grip.max())

    T_obs = left_xyz_world.shape[0]
    assert T_obs == T_ts, "Timestamp length and observation length mismatch"
    T = T_obs


    # -------------------------------------------------
    # rot6d → axis-angle
    # -------------------------------------------------

    def rot6d_to_rotvec(rot6d):
        T = rot6d.shape[0]
        rot_matrix = np.zeros((T, 3, 3))

        for i in range(T):
            a1 = rot6d[i, 0:3]
            a2 = rot6d[i, 3:6]

            b1 = a1 / (np.linalg.norm(a1) + 1e-8)
            b2 = a2 - np.dot(b1, a2) * b1
            b2 = b2 / (np.linalg.norm(b2) + 1e-8)
            b3 = np.cross(b1, b2)

            rot_matrix[i] = np.stack([b1, b2, b3], axis=-1)

        return R.from_matrix(rot_matrix).as_rotvec()


    left_rot_world  = rot6d_to_rotvec(left_rot6d_world)
    right_rot_world = rot6d_to_rotvec(right_rot6d_world)


    # -------------------------------------------------
    # read extrinsics
    # -------------------------------------------------

    extrinsics = np.load(os.path.join(save_dir, "processed", "extrinsics.npz"))[camera_id]
    assert extrinsics.shape[0] == T, "Extrinsics length mismatch"

    # extrinsics shape: (T, 4, 4)
    # which is T_cam_world


    # -------------------------------------------------
    # world → camera 
    # -------------------------------------------------

    left_xyz_cam  = []
    left_rot_cam  = []
    right_xyz_cam = []
    right_rot_cam = []

    for i in range(T):

        T_cam_world = extrinsics[i]
        T_world_cam = np.linalg.inv(T_cam_world)

        # -------- LEFT --------
        T_left_world = np.eye(4)
        T_left_world[:3, :3] = R.from_rotvec(left_rot_world[i]).as_matrix()
        T_left_world[:3, 3]  = left_xyz_world[i]

        T_left_cam = T_world_cam @ T_left_world

        left_xyz_cam.append(T_left_cam[:3, 3])
        left_rot_cam.append(
            R.from_matrix(T_left_cam[:3, :3]).as_rotvec()
        )

        # -------- RIGHT --------
        T_right_world = np.eye(4)
        T_right_world[:3, :3] = R.from_rotvec(right_rot_world[i]).as_matrix()
        T_right_world[:3, 3]  = right_xyz_world[i]

        T_right_cam = T_world_cam @ T_right_world

        right_xyz_cam.append(T_right_cam[:3, 3])
        right_rot_cam.append(
            R.from_matrix(T_right_cam[:3, :3]).as_rotvec()
        )


    left_xyz_cam  = np.array(left_xyz_cam)
    left_rot_cam  = np.array(left_rot_cam)
    right_xyz_cam = np.array(right_xyz_cam)
    right_rot_cam = np.array(right_rot_cam)

    ## Here left should be right and right should be left because of the camera perspective.

    episode["robot0_eef_pos"] = left_xyz_cam
    episode["robot0_eef_rot_axis_angle"] = left_rot_cam
    episode["gripper0_gripper_pose"] = left_grip

    episode["robot1_eef_pos"] = right_xyz_cam
    episode["robot1_eef_rot_axis_angle"] = right_rot_cam
    episode["gripper1_gripper_pose"] = right_grip



    episode["action"] = np.concatenate([
        left_xyz_cam,
        left_rot_cam,
        left_grip,
        right_xyz_cam,
        right_rot_cam,
        right_grip
    ], axis=-1)

    # -------------------------------------------------
    # Add joint positions and velocities
    # -------------------------------------------------

    left_joint_pos = observations["robot__actual__joint_position__left::panda"]
    right_joint_pos = observations["robot__actual__joint_position__right::panda"]

    left_joint_vel = observations["robot__actual__joint_velocity__left::panda"]
    right_joint_vel = observations["robot__actual__joint_velocity__right::panda"]

    assert left_joint_pos.shape[0] == T
    assert right_joint_pos.shape[0] == T
    assert left_joint_vel.shape[0] == T
    assert right_joint_vel.shape[0] == T

    episode["robot1_joint_pos"] = right_joint_pos
    episode["robot0_joint_pos"] = left_joint_pos

    episode["robot1_joint_vel"] = right_joint_vel
    episode["robot0_joint_vel"] = left_joint_vel

        
    # ========================== Camera Action Alignment =================================
    sample_img = observations[camera_id][0]
    height, width = sample_img.shape[:2]

    # Module A: camera info -> episode['camera0_*']
    intrinsics_all = np.load(os.path.join(processed_dir, "intrinsics.npz"))
    extrinsics_all = np.load(os.path.join(processed_dir, "extrinsics.npz"))
    left_intrinsic = intrinsics_all[camera_id]
    extrinsics_cam_to_world = extrinsics_all[camera_id]
    assert extrinsics_cam_to_world.ndim == 3 and extrinsics_cam_to_world.shape[1:] == (4,4)

    frame_count = extrinsics_cam_to_world.shape[0]

    # try to compute resized intrinsics like original pipeline; fallback to copy
    try:
        left_intrinsic_resized = intrinsic_transform_resize(left_intrinsic, input_res=(width, height),
                                                            output_resize_res=out_resolutions_resize,
                                                            output_crop_res=out_resolutions_crop)
        left_intrinsic_final = intrinsic_transform_resize(left_intrinsic, input_res=out_resolutions_crop,
                                                        output_resize_res=out_resolutions_image_final,
                                                        output_crop_res=out_resolutions_image_final)
        print("\n========== Intrinsic Debug ==========")
        print("Original intrinsic:\n", left_intrinsic)
        print("Original resolution:", (width, height))
        print("Resize target:", out_resolutions_resize)
        print("Crop target:", out_resolutions_crop)
        print("Final image resolution:", out_resolutions_image_final)
        print("\nResized intrinsic:\n", left_intrinsic_resized)
        print("\nFinal intrinsic:\n", left_intrinsic_final)
        print("=====================================\n")
    except Exception:
        left_intrinsic_resized = left_intrinsic.copy()
        left_intrinsic_final = left_intrinsic.copy()

    episode["camera0_left_intrinsic"] = np.array([left_intrinsic_resized] * frame_count)
    episode["camera0_left_intrinsic_final"] = np.array([left_intrinsic_final] * frame_count)
    episode["camera0_stereo_transform"] = np.array([np.eye(4)] * frame_count)

    # Module B: build obs_dict for reading from observations
    transform_img = get_image_transform_resize_crop(input_res=(width, height),
                                                    output_resize_res=out_resolutions_resize,
                                                    output_crop_res=out_resolutions_crop,
                                                    bgr_to_rgb=False)
    obs_dict = {}
    obs_dict['rgb'] = ('image', camera_id, transform_img)
    # optional: handle right image / pointcloud if present in observations and mode requires
    if mode in ['s', 'a']:
        # Try find a right-camera id in metadata mapping if available
        # For now skip rgb_right unless present explicitly
        right_id = None
        for k in observations.files:
            if k.endswith("_right") or k.endswith("right"):
                right_id = k
                break
        if right_id is not None:
            obs_dict['rgb_right'] = ('image', right_id, transform_img)
    if mode in ['p', 'a']:
        # if pointcloud stored in observations under a key
        if 'pointcloud' in observations.files:
            transform_pcd = get_image_transform_resize_crop(input_res=(width, height),
                                                            output_resize_res=out_resolutions_resize,
                                                            output_crop_res=out_resolutions_crop,
                                                            is_pcd=True)
            obs_dict['pointcloud'] = ('pointcloud', 'pointcloud', transform_pcd)
        # else: if depth exists and you want to backproject, you'd add code here to backproject depth->pcd

    # Module C: index-based loop to populate episode['camera0_*']
    episode_length = min(frame_count, len(timestamps))
    episode['camera0_real_timestamp'] = np.zeros((episode_length,), dtype=np.float64)

    # pre-check shapes
    assert observations[camera_id].shape[0] >= episode_length, "observations frame count < episode_length"
    assert extrinsics_cam_to_world.shape[0] >= episode_length, "extrinsics frame count < episode_length"

    for i in range(episode_length):
        # for each requested modality, transform and store
        for key in obs_dict.keys():
            src_type, src_name, transform = obs_dict[key]

            if src_type == 'image':
                value = observations[src_name][i]  # raw image (H,W,3/4)
                if value.shape[-1] == 4:
                    value = value[..., :3]
                value = transform(value)
                if out_resolutions_image_final is not None:
                    value = cv2.resize(value, out_resolutions_image_final, interpolation=cv2.INTER_LINEAR)

            elif src_type == 'pointcloud':
                pc_raw = observations[src_name][i]  # expect (H,W,3) xyz
                points_xyz = pc_raw.reshape(-1, 3)
                points_rgb = obs_dict['rgb'][2](observations[camera_id][i][..., :3]).reshape(-1, 3)
                points = np.concatenate([points_xyz, points_rgb / 255.0], axis=-1)
                valid_mask = np.isfinite(points).all(axis=-1) & (np.linalg.norm(points_xyz, axis=-1) <= points_max_distance_final)
                points = points[valid_mask]
                points_xyz = points_xyz[valid_mask]
                if len(points) > num_points_final:
                    points_idx = fpsample.bucket_fps_kdline_sampling(points_xyz, num_points_final, h=7)
                else:
                    if len(points) == 0:
                        # fallback zero points
                        value = np.zeros((num_points_final, 6), dtype=np.float32)
                    else:
                        points_idx = np.array([j % len(points_xyz) for j in range(num_points_final)])
                        value = points[points_idx]

            # write into episode (lazy allocate)
            ep_key = 'camera0_' + key
            if ep_key not in episode:
                episode[ep_key] = np.zeros((episode_length,) + value.shape, dtype=value.dtype)
            episode[ep_key][i] = value

        # store camera real timestamp (use timestamps array from pose_data.yaml)
        episode['camera0_real_timestamp'][i] = timestamps[i]

    # ========================== Remove Idle Frames (Stage 1) =================================
    # Remove leading frames where neither robot0_eef_pos nor robot1_eef_pos has moved from
    # its initial position. All episode arrays are fully populated here, so trimming is uniform.
    start_time_idx = 0
    eps = 0.005
    for i in range(1, len(episode['robot0_eef_pos'])):
        moved0 = np.linalg.norm(episode['robot0_eef_pos'][i] - episode['robot0_eef_pos'][0]) > eps
        moved1 = np.linalg.norm(episode['robot1_eef_pos'][i] - episode['robot1_eef_pos'][0]) > eps
        if moved0 or moved1:
            start_time_idx = i
            break
    for key in list(episode.keys()):
        episode[key] = episode[key][start_time_idx:]
    print(f"Stage 1 idle removal: trimmed {start_time_idx} leading frames (eps={eps})")

    # ========================== Remove Initialization Stage Data (Stage 2) =================================
    # Remove leading frames where the action vector is entirely zero (controller not yet active).
    t_init = 0
    while t_init < len(episode['action']) and (episode['action'][t_init] == 0.).all():
        t_init += 1
    if t_init > 0:
        for key in list(episode.keys()):
            episode[key] = episode[key][t_init:]
    print(f"Stage 2 idle removal: trimmed {t_init} leading zero-action frames")

    # Module D: final trim / sanity
    n_length = min(episode['timestamp'].shape[0], episode['camera0_real_timestamp'].shape[0])
    for k in list(episode.keys()):
        try:
            episode[k] = episode[k][:n_length]
        except Exception:
            pass
    # for key in episode.keys():
    #     try:
    #         print(f"Key: {key}, shape: {episode[key].shape}")
    #     except:
    #         print(f"Key: {key}, {episode[key]}")

    # ========================== DEBUG VISUALIZATION ==========================
    debug = True

    if debug:
        print("Running visualization sanity check...")

        rgb_frames = episode["camera0_rgb"]
        intrinsic = episode["camera0_left_intrinsic_final"][0]

        vis_frames = []

        for i in range(len(rgb_frames)):

            frame = rgb_frames[i].copy()

            # LEFT hand
            pt_left = project_cam_to_pixel(
                episode["robot1_eef_pos"][i],
                intrinsic
            )
            cv2.circle(frame, tuple(pt_left), 5, (0,0,255), -1)
            cv2.putText(frame, "L", tuple(pt_left),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # RIGHT hand
            pt_right = project_cam_to_pixel(
                episode["robot0_eef_pos"][i],
                intrinsic
            )
            cv2.circle(frame, tuple(pt_right), 5, (255,0,0), -1)
            cv2.putText(frame, "R", tuple(pt_right),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

            vis_frames.append(frame)

        save_path = os.path.join(save_dir, "debug_projection.mp4")
        import imageio
        imageio.mimsave(save_path, vis_frames, fps=10)
        print(f"Saved debug projection video to {save_path}")


    return episode


def conversion_trajectory(input_data_fp_list, hand_to_eef, mode, out_resolutions_resize,
                          out_resolutions_crop, 
                          resolution_image_final,
                          num_points_final, points_max_distance_final,
                          replay_buffer,
                          process_id, verbose):
    if verbose is True:
        pbar = tqdm(input_data_fp_list, desc=f"Process {process_id}")
    else:
        pbar = input_data_fp_list
    for input_data_fp in pbar:
        save_dir, source, source_idx = input_data_fp
        episode = conversion_single_trajectory(
            hand_to_eef,
            mode,
            save_dir,
            out_resolutions_resize,
            out_resolutions_crop,
            resolution_image_final,
            num_points_final, points_max_distance_final,
        )
        T = episode['action'].shape[0]

        episode['embodiment'] = np.ones((T, 1), dtype=np.float32)
        episode['source_idx'] = np.ones((T, 1), dtype=np.float32) * source_idx

        replay_buffer.add_episode(episode, compressors='disk')  # with lock mechanism inside replay_buffer instance


@click.command()
@click.option('--input_dir', '-i', required=True)
@click.option('--output', '-o', required=True)
@click.option('--hand_to_eef_file', '-ehf', required=True)
@click.option('--mode', '-m', required=True, type=click.Choice(['o', 'p', 's', 'a'], case_sensitive=False), default='o',
              help="o: only image, p: with pointcloud, s: with stereo-image, a: with all, including pointcloud and stereo")
@click.option('--resolution_resize', '-ror', default='1280x720')
@click.option('--resolution_crop', '-or', default='640x480')
@click.option('--resolution_image_final', '-for', default='224x224')
@click.option('--num_use_source', '-nus', default=None, type=int)
@click.option('--num_points_final', '-npf', type=int, default=2048)
@click.option('--points_max_distance_final', '-pmdf', type=float, default=1.0)
@click.option('--n_encoding_threads', '-ne', default=-1, type=int)
@click.option('--verbose', '-v', is_flag=True, default=True)
def main(input_dir, output, hand_to_eef_file, 
         mode,
         resolution_resize, resolution_crop, resolution_image_final, num_use_source, num_points_final, points_max_distance_final,
         n_encoding_threads,
         verbose):
    hand_to_eef = np.load(hand_to_eef_file)
    out_resolution_resize = tuple(int(x) for x in resolution_resize.split('x'))
    out_resolution_crop = tuple(int(x) for x in resolution_crop.split('x'))
    resolution_image_final = tuple(int(x) for x in resolution_image_final.split('x'))
    
    input_dir_list = os.listdir(input_dir)
    input_dir_list = [os.path.join(input_dir, x) for x in input_dir_list]

    for input_dir in input_dir_list:
        input_folder = input_dir.split('/')[-1]
        embodiment = input_folder.split('_')[0]
        assert embodiment == "robot"
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
        if os.path.exists(replay_buffer_fp):
            shutil.rmtree(replay_buffer_fp)

        replay_buffer = ReplayBuffer.create_from_path(replay_buffer_fp, mode='a')
        if environment_setting == "me":          # multi-environment
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
            # input_data_fp_list = os.listdir(input_dir)
            input_data_fp_list = glob(os.path.join(input_dir, "**", "episode_*"), recursive=True) ###===### ###---##
            input_data_fp_list.sort(key=lambda x:int(x.split("_")[-1]))

            input_data_fp_list = [(os.path.join(input_dir, fp), "default", 0) for fp in input_data_fp_list] 

        if n_encoding_threads > 1:
            input_data_fp_batch_list = []
            for i in range(n_encoding_threads):
                input_data_fp_batch_list.append(input_data_fp_list[i::n_encoding_threads])

            process_list = []
            for i in range(n_encoding_threads):
                p = Process(target=conversion_trajectory, args=(
                input_data_fp_batch_list[i], hand_to_eef, mode, out_resolution_resize, out_resolution_crop, resolution_image_final,
                num_points_final, points_max_distance_final,
                replay_buffer, i, verbose))
                p.start()
                process_list.append(p)

            for p in process_list:
                p.join()
        else:
            conversion_trajectory(input_data_fp_list, hand_to_eef, mode, out_resolution_resize, out_resolution_crop, resolution_image_final,
                                  num_points_final, points_max_distance_final,
                                  replay_buffer, 0, verbose)

        print(f"Saving to disk finish: Task {input_dir}")


if __name__ == "__main__":
    # test_conversion()
    main()

# bash scripts_data/robot_data_conversion_base.sh