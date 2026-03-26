import os
import sys
import numpy as np
import json
import yaml
import pickle
import gzip 
from collections import defaultdict
from glob import glob
from tqdm import tqdm, trange
import pandas as pd
import cv2
import imageio
from scipy.spatial.transform import Rotation
from copy import copy, deepcopy

def project_point_to_image_corrected(point_T, device_extrinsics, camera_extrinsics, K):
    """
    Project a 3D point in world coordinates to camera image coordinates.
    
    Args:
        point_world: 3D position in world frame (3,) array
        device_extrinsics: 4x4 world_T_device (device pose in world)
        camera_extrinsics: 4x4 device_T_camera (camera pose relative to device)
        K: 3x3 camera intrinsics
    
    Returns:
        [u, v] pixel coordinates or None if behind camera
    """
    # The camera_extrinsics has translation in bottom row, so transpose it
    device_T_camera = camera_extrinsics.T
    device_T_camera = np.linalg.inv(device_T_camera) 

    point_world = point_T[:3, 3]
    
    # Compute world_T_camera
    world_T_camera = device_extrinsics @ device_T_camera
    
    # Transform point to camera frame
    camera_T_world = np.linalg.inv(world_T_camera)
    p_cam_h = camera_T_world @ np.hstack([point_world, 1])
    x_cam, y_cam, z_cam = p_cam_h[:3]
    
    # # Check if point is in front of camera (negative Z in Apple Vision Pro convention)
    # if z_cam >= 0:
    #     return None  # Behind camera
    
    # Project to image plane
    u = K[0, 0] * (x_cam / z_cam) + K[0, 2]
    v = K[1, 1] * (y_cam / z_cam) + K[1, 2]
    
    return np.array([u, v])

def parse_to_se3(pose):
    """
    Parse translation and rotation data into SE(3) format (4x4 homogeneous matrix).
    
    Args:
        pose: Object with pose.translation and pose.rotation attributes,
              where each has x, y, z (and w for rotation) attributes
        
    Returns:
        se3_matrix: 4x4 numpy array representing the SE(3) transformation
    """
    # Extract translation values
    t = np.array([pose.translation.x, pose.translation.y, pose.translation.z])
    
    # Extract rotation values (quaternion as x, y, z, w)
    q = np.array([pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w])
    
    # Convert quaternion (x, y, z, w) to rotation matrix
    # scipy uses (x, y, z, w) format
    rot = Rotation.from_quat(q)
    R = rot.as_matrix()
    
    # Create SE(3) matrix (4x4)
    se3 = np.eye(4)
    se3[:3, :3] = R
    se3[:3, 3] = t
    
    return se3

def transform_point_to_camera_frame_xyz(point_T, device_extrinsics, camera_extrinsics):
    """
    Project a 3D point in world coordinates to camera image coordinates.
    
    Args:
        point_world: 3D position in world frame (3,) array
        device_extrinsics: 4x4 world_T_device (device pose in world)
        camera_extrinsics: 4x4 device_T_camera (camera pose relative to device)
        K: 3x3 camera intrinsics
    
    Returns:
        [u, v] pixel coordinates or None if behind camera
    """
    # The camera_extrinsics has translation in bottom row, so transpose it
    device_T_camera = camera_extrinsics.T
    device_T_camera = np.linalg.inv(device_T_camera) 

    point_world = point_T[:3, 3]
    
    # Compute world_T_camera
    world_T_camera = device_extrinsics @ device_T_camera
    
    # Transform point to camera frame
    camera_T_world = np.linalg.inv(world_T_camera)
    p_cam_h = camera_T_world @ np.hstack([point_world, 1])
    return p_cam_h[:3]


def to_camera_plane(p_cam_h, K):
    x_cam, y_cam, z_cam = p_cam_h
    u = K[0, 0] * (x_cam / z_cam) + K[0, 2]
    v = K[1, 1] * (y_cam / z_cam) + K[1, 2]
    
    return np.array([u, v])



    ## --------------  Zeqing ----------------##
    def to_camera_plane(p_cam_h, K):
        x_cam, y_cam, z_cam = p_cam_h
        u = K[0, 0] * (x_cam / z_cam) + K[0, 2]
        v = K[1, 1] * (y_cam / z_cam) + K[1, 2]
        
        return np.array([u, v])
    
    print("start vis")
    initial_idx = 0
    initial_frame = episode['camera0_rgb'][initial_idx].copy()
    intrinsics_t0 = episode['camera0_left_intrinsic_final'][initial_idx]

    for t in range(T):
        intrinsics_t = episode['camera0_left_intrinsic_final'][t]

        for hand, color in [("left", (0, 0, 255)), ("right", (255, 0, 0))]:
            hand_mat = episode_org[f"{hand}_hand_mat"]

            for j in range(J):
                pose_euler = hand_mat[t, j*6:(j+1)*6][None, :]
                T_joint_vr = (euler_pose_to_mat(pose_euler) @ yfxrzu2standard)[0]

                # Camera0
                T_joint_cam0 = vr2camera0 @ T_joint_vr
                xyz_cam0 = T_joint_cam0[:3, 3]
                uv0 = to_camera_plane(xyz_cam0, intrinsics_t0).astype(int)

                # Camera(t)
                T_joint_camt = calib_quest2camera @ fast_mat_inv(episode_org['head_pose_mat'][t]) @ T_joint_vr
                xyz_camt = T_joint_camt[:3, 3]
                uvt = to_camera_plane(xyz_camt, intrinsics_t).astype(int)

                # draw
                cv2.circle(initial_frame, tuple(uv0), 1, color, -1)          
                cv2.circle(initial_frame, tuple(uvt), 1, (255, 255, 255), -1)  
    
    save_dir = "/data/zeqingwang/visualizations"
    os.makedirs(save_dir, exist_ok=True)

    initial_frame_path = os.path.join(save_dir, "camera0_egocentric_overlay.png")

    cv2.imwrite(
        initial_frame_path,
        cv2.cvtColor(initial_frame, cv2.COLOR_RGB2BGR)
    )

    print(f"[OK] Saved egocentric visualization to: {initial_frame_path}")

    ##----------------------------------------##
