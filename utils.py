from typing import Dict, Optional

from avp_stream import VisionProStreamer

import click

import pickle
from pathlib import Path

import tqdm
import copy
import torch
import time
import os
import pinocchio as pin
import numpy as np
import functools

from scipy.spatial.transform import Rotation as R

from utils.geometry_utils import calc_plane_angle, compute_finger_angle


OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)

def calc_delta_mat(start_mat, end_mat):
    """
    Args:
        start_mat: np.ndarray, shape (..., 4, 4)
        end_mat: np.ndarray, shape (..., 4, 4)
    Returns:
        delta_mat: np.ndarray, shape (..., 4, 4), the transformation mat that transforms start_mat to end_mat.
    """
    assert start_mat.shape == end_mat.shape, f"start_mat and end_mat must have the same shape, got {start_mat.shape} and {end_mat.shape}"

    delta_mat = np.linalg.inv(start_mat) @ end_mat
    return delta_mat

def calc_end_mat(start_mat, delta_mat):
    """
    Args:
        start_mat: np.ndarray, shape (..., 4, 4)
        delta_mat: np.ndarray, shape (..., 4, 4)
    Returns:
        end_mat: np.ndarray, shape (..., 4, 4), the transformation mat that transforms start_mat by delta_mat.
    """
    assert start_mat.shape == delta_mat.shape, f"start_mat and delta_mat must have the same shape, got {start_mat.shape} and {delta_mat.shape}"

    end_mat = start_mat @ delta_mat
    return end_mat

def pose_to_pos_euler(pose: np.ndarray) -> np.ndarray:
    """
    Args:
        pose: np.ndarray, shape (..., 7), the pose in the format [x, y, z, qx, qy, qz, qw].
    Returns:
        pos_euler(radians): np.ndarray, shape (..., 6), the position and Euler angles in the format [x, y, z, roll, pitch, yaw].
    """
    assert pose.shape[-1] == 7, f"Expected pose to have 7 elements, got {pose.shape[-1]}"

    pos = pose[..., :3]
    quat = pose[..., 3:]
    R_mat = R.from_quat(quat).as_matrix()
    euler_angles = R.from_matrix(R_mat).as_euler('xyz', degrees=False)

    pos_euler = np.concatenate([pos, euler_angles], axis=-1)  # (..., 6)
    return pos_euler

def pos_euler_to_pose(pos_euler: np.ndarray) -> np.ndarray:
    """
    Args:
        pos_euler: np.ndarray, shape (..., 6), the position and Euler angles in the format [x, y, z, roll, pitch, yaw].
    Returns:
        pose: np.ndarray, shape (..., 7), the pose in the format [x, y, z, qx, qy, qz, qw].
    """
    assert pos_euler.shape[-1] == 6, f"Expected pos_euler to have 6 elements, got {pos_euler.shape[-1]}"

    pos = pos_euler[..., :3]
    euler_angles = pos_euler[..., 3:]  # [roll, pitch, yaw]
    R_mat = R.from_euler('xyz', euler_angles).as_matrix()
    quat = R.from_matrix(R_mat).as_quat()

    pose = np.concatenate([pos, quat], axis=-1)  # (..., 7)
    return pose

def pose_to_mat(pose):
    """
    Args:
        pose: np.ndarray, shape (..., 7), the pose in the format [x, y, z, qx, qy, qz, qw].
    Returns:
        mat: np.ndarray, shape (..., 4, 4), the transformation mat.
    """
    pos = pose[..., :3]
    quat = pose[..., 3:]
    R_mat = R.from_quat(quat).as_matrix()
    mat = np.zeros(pose.shape[:-1] + (4, 4), dtype=np.float32)
    mat[..., :3, :3] = R_mat
    mat[..., :3, 3] = pos
    mat[..., 3, 3] = 1.0
    return mat

def mat_to_pose(mat):
    """
    Args:
        mat: np.ndarray, shape (..., 4, 4), the transformation mat.
    Returns:
        pose: np.ndarray, shape (..., 7), the pose in the format [x, y, z, qx, qy, qz, qw].
    """
    pos = mat[..., :3, 3]
    R_mat = mat[..., :3, :3]
    quat = R.from_matrix(R_mat).as_quat()
    pose = np.concatenate([pos, quat], axis=-1)
    return pose

def sim_pose_to_mat(sim_pose: np.ndarray) -> np.ndarray:
    """
    Convert a simulation pose to a transformation matrix.
    Args:
        sim_pose: np.ndarray, shape (..., 7), the pose in the format [x, y, z, qw, qx, qy, qz].
    Returns:
        mat: np.ndarray, shape (..., 4, 4), the transformation matrix.
    """
    pos = sim_pose[..., :3]
    quat = sim_pose[..., 3:7]  # qw, qx, qy, qz
    # Convert quaternion from (qw, qx, qy, qz) to (qx, qy, qz, qw)
    quat = quat_from_sim(quat)  # (qx, qy, qz, qw)
    R_mat = R.from_quat(quat).as_matrix()
    mat = np.zeros(sim_pose.shape[:-1] + (4, 4), dtype=np.float32)
    mat[..., :3, :3] = R_mat
    mat[..., :3, 3] = pos
    mat[..., 3, 3] = 1.0
    return mat

def mat_to_sim_pose(mat: np.ndarray) -> np.ndarray:
    """
    Convert a transformation matrix to a simulation pose.
    Args:
        mat: np.ndarray, shape (..., 4, 4), the transformation matrix.
    Returns:
        sim_pose: np.ndarray, shape (..., 7), the pose in the format [x, y, z, qw, qx, qy, qz].
    """
    pos = mat[..., :3, 3]
    R_mat = mat[..., :3, :3]
    quat = R.from_matrix(R_mat).as_quat()
    # Convert quaternion from (qx, qy, qz, qw) to (qw, qx, qy, qz)
    quat = quat_to_sim(quat)  # (qw, qx, qy, qz)
    sim_pose = np.concatenate([pos, quat], axis=-1)
    return sim_pose

def pose_to_sim_pose(pose: np.ndarray) -> np.ndarray:
    """
    Args:
        pose: np.ndarray, shape (..., 7), the pose in the format [x, y, z, qx, qy, qz, qw].
    Returns:
        sim_pose: np.ndarray, shape (..., 7), the pose in the format [x, y, z, qw, qx, qy, qz].
    """

    return mat_to_sim_pose(pose_to_mat(pose))

def sim_pose_to_pose(sim_pose: np.ndarray) -> np.ndarray:
    """
    Args:
        sim_pose: np.ndarray, shape (..., 7), the pose in the format [x, y, z, qw, qx, qy, qz].
    Returns:
        pose: np.ndarray, shape (..., 7), the pose in the format [x, y, z, qx, qy, qz, qw].
    """

    return mat_to_pose(sim_pose_to_mat(sim_pose))

def pos_euler_to_mat(pos_euler: np.ndarray) -> np.ndarray:
    """
    Convert position and Euler angles to a transformation matrix.
    Args:
        pos_euler: np.ndarray, shape (..., 6,), first 3 elements are position [x, y, z], last 3 elements are Euler angles [roll, pitch, yaw] in radians.
    Returns:
        pose: np.ndarray, shape (..., 4, 4), the transformation matrix.
    """
    assert pos_euler.shape[-1] == 6, f"Expected pos_euler to have 6 elements, got {pos_euler.shape[-1]}"

    pos = pos_euler[..., :3]
    euler_angles = pos_euler[..., 3:]  # [roll, pitch, yaw]
    R_mat = R.from_euler('xyz', euler_angles).as_matrix()
    mat = np.zeros(pos_euler.shape[:-1] + (4, 4), dtype=np.float32)
    mat[..., :3, :3] = R_mat
    mat[..., :3, 3] = pos
    mat[..., 3, 3] = 1.0
    return mat

def j25_to_j21(j25_pos: np.ndarray) -> np.ndarray:
    """
    Args:
        j25_pos: shape (25, 3), the positions of the 25 hand joints.
        resources/hand_skeleton_25.png
        resources/vision_os_hand_skeleton.png
    Returns:
        j21_pos: shape (21, 3), the positions of the 21 hand joints.
        resources/hand_skeleton_21.png
    """
    j21_pos = np.zeros((21, 3), dtype=np.float32)
    j21_pos[0:5, :] = j25_pos[0:5, :]  # thumb
    j21_pos[5:9, :] = j25_pos[6:10, :]  # index finger
    j21_pos[9:13, :] = j25_pos[11:15, :]  # middle finger
    j21_pos[13:17, :] = j25_pos[16:20, :]  # ring finger
    j21_pos[17:21, :] = j25_pos[21:25, :]  # little finger

    return j21_pos

def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """
    Compute the 3D coordinate frame (orientation only) from detected 3d key points
    :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
    :return: the coordinate frame of wrist in MANO convention
    """
    assert keypoint_3d_array.shape == (21, 3)
    points = keypoint_3d_array[[0, 5, 9], :]

    # Compute vector from palm to the first joint of middle finger
    x_vector = points[0] - points[2]

    # Normal fitting with SVD
    points = points - np.mean(points, axis=0, keepdims=True)
    u, s, v = np.linalg.svd(points)

    normal = v[2, :]

    # Gram–Schmidt Orthonormalize
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)

    # We assume that the vector from pinky to index is similar the z axis in MANO convention
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    frame = np.stack([x, normal, z], axis=1)
    return frame

def solve_avp_3d(keypoint_3d_array: np.ndarray, left_right: str) -> np.ndarray:
    operator2mano = OPERATOR2MANO_RIGHT if left_right == "right" else OPERATOR2MANO_LEFT

    keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :] # shape (21, 3)
    mediapipe_wrist_rot = estimate_frame_from_hand_points(keypoint_3d_array)
    joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ operator2mano # shape (21, 3)
    # joint_pos = keypoint_3d_array @ mediapipe_wrist_rot # shape (21, 3)

    return joint_pos

def np2tensor(data: Dict[str, np.ndarray], device) -> Dict[str, torch.Tensor]:
    for key in data.keys():
        data[key] = torch.tensor(data[key], dtype = torch.float32, device = device)
    return data

def post_process_raw_data(raw_data: np.ndarray):
    """
    Apply post-processing to the raw data from AVP.
    Args:
        raw_data: np.ndarray, the raw data from AVP, shape (53, 7), where each row is [x, y, z, qx, qy, qz, qw].
    Returns:
        raw_data: np.ndarray, the post-processed raw data, shape (53, 7).
    """
    # vision os y forward, but sapien is -x forward, so swap the x and y axes
    # raw_data[:, :3] = raw_data[:, :3] @ np.array([
    #     [0, 1, 0],
    #     [1, 0, 0],
    #     [0, 0, 1],
    # ], dtype=np.float32)

    # convert left wrist pose
    left_wrist_pose = raw_data[2, 3:] # (4, )
    # convert from right-hand-coordinate to left-hand-coordinate
    left_wrist_pose = np.array([-left_wrist_pose[0], -left_wrist_pose[1], left_wrist_pose[2], left_wrist_pose[3]], dtype=np.float32) # (4, )
    left_wrist_pose = R.from_quat(left_wrist_pose).as_matrix() # (3, 3)
    # rotate left wrist pose 180 degrees around x-axis to match sapien's coordinate system
    left_wrist_pose = left_wrist_pose @ R.from_euler("x", 180, degrees=True).as_matrix() # (3, 3)
    raw_data[2, 3:] = R.from_matrix(left_wrist_pose).as_quat() # (4, )

    return raw_data

def quat_to_sim(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion from (qx, qy, qz, qw) to (qw, qx, qy, qz).
    Args:
        quat: np.ndarray, shape (..., 4) or (..., 7), quaternion in (qx, qy, qz, qw) format.
    Returns:
        np.ndarray, shape (..., 4) or (..., 7), quaternion in (qw, qx, qy, qz) format.
    """
    if quat.shape[-1] == 7:
        # If the input is in (x, y, z, qx, qy, qz, qw) format, we need to convert it to (x, y, z, qw, qx, qy, qz)
        return np.stack([
            quat[..., 0],  # x
            quat[..., 1],  # y
            quat[..., 2],  # z
            quat[..., 6],  # qw
            quat[..., 3],  # qx
            quat[..., 4],  # qy
            quat[..., 5],  # qz
        ], axis=-1, dtype=np.float32)
    elif quat.shape[-1] == 4:
        return np.stack([
            quat[..., 3],  # qw
            quat[..., 0],  # qx
            quat[..., 1],  # qy
            quat[..., 2],  # qz
        ], axis=-1, dtype=np.float32) # (qx, qy, qz, qw) to (qw, qx, qy, qz)
    raise ValueError(f"Unsupported quaternion shape: {quat.shape}. Expected last dimension to be 4 or 7.")

def quat_from_sim(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion from (qw, qx, qy, qz) to (qx, qy, qz, qw).
    Args:
        quat: np.ndarray, shape (..., 4) or (..., 7), quaternion in (qw, qx, qy, qz) format.
    Returns:
        np.ndarray, shape (..., 4) or (..., 7), quaternion in (qx, qy, qz, qw) format.
    """
    if quat.shape[-1] == 7:
        # If the input is in (x, y, z, qw, qx, qy, qz) format, we need to convert it to (x, y, z, qx, qy, qz, qw)
        return np.stack([
            quat[..., 0],  # x
            quat[..., 1],  # y
            quat[..., 2],  # z
            quat[..., 4],  # qx
            quat[..., 5],  # qy
            quat[..., 6],  # qz
            quat[..., 3],  # qw
        ], axis=-1, dtype=np.float32)
    elif quat.shape[-1] == 4:
        return np.stack([
            quat[..., 1],  # qx
            quat[..., 2],  # qy
            quat[..., 3],  # qz
            quat[..., 0],  # qw
        ], axis=-1, dtype=np.float32) # (qw, qx, qy, qz) to (qx, qy, qz, qw)
    raise ValueError(f"Unsupported quaternion shape: {quat.shape}. Expected last dimension to be 4 or 7.")

def compensente_default_quat_for_pos_to_pose(pos: np.ndarray):
    """
    Args:
        pos: np.ndarray, shape (..., 3), the position in the format [x, y, z].
    """
    quat_shape = pos.shape[:-1] + (4,) if pos.ndim > 1 else (4,)
    quat = np.zeros(quat_shape, dtype=np.float32)
    quat[..., 3] = 1.0  # Set qw to 1.0

    pose = np.concatenate([pos, quat], axis=-1)  # (..., 7)

    return pose  # (..., 7), where each row is [x, y, z, qx, qy, qz, qw] with qw = 1.0

def pos_euler_to_pinse3(pos_euler: np.ndarray):
    """
    Args:
        pos_euler: (6, ) array, first 3 elements are position, last 3 elements are euler angles in radians
    """

    return pin.SE3(R.from_euler('xyz', pos_euler[3:]).as_matrix(), pos_euler[:3])

def pose_to_pinse3(pose: np.ndarray):
    """
    Args:
        pose: (7, ) array, first 3 elements are position, last 4 elements are quaternion [qx, qy, qz, qw]
    """
    assert pose.shape == (7,), f"Expected pose to be of shape (7,), got {pose.shape}"

    return pin.SE3(R.from_quat(pose[3:]).as_matrix(), pose[:3])

def mat_to_pinse3(mat: np.ndarray):
    """
    Args:
        mat: (4, 4) array, the transformation matrix.
    Returns:
        pin.SE3: the transformation in pinocchio SE3 format.
    """
    assert mat.shape == (4, 4), f"Expected mat to be of shape (4, 4), got {mat.shape}"

    return pin.SE3(mat[:3, :3], mat[:3, 3])

def np_to_torch(np_array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np_array).to(device="cuda:0", dtype=torch.float32)

def rotate_pose_90_z(mat: np.ndarray) -> np.ndarray:
    """
    Args:
        mat: np.ndarray, shape (..., 4, 4), the transformation matrix.
    """
    mat = mat.copy()  # Avoid modifying the original matrix

    angle = -90
    R_z = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,             1]
    ])

    if mat.ndim == 2:
        mat[:3, 3] = R_z @ mat[:3, 3]  # Rotate the translation part
    elif mat.ndim == 3:
        for i in range(mat.shape[0]):
            mat[i, :3, 3] = R_z @ mat[i, :3, 3]
    else:
        raise ValueError(f"Unsupported matrix shape: {mat.shape}. Expected 2D or 3D array.")

    return mat

def world_rot(mat: np.ndarray, axis: str, angle: float) -> np.ndarray:
    """
    left multi means rotate in world coordinate system

    Rotate a pose around the specified axis in the world coordinate system.
    Args:
        mat: np.ndarray, shape (..., 4, 4), the transformation matrix.
        axis: str, 'x', 'y', or 'z'.
        angle: float, angle in degrees.
    Returns:
        np.ndarray, shape (..., 4, 4), the rotated transformation matrix.
    """
    mat = mat.copy()  # Avoid modifying the original matrix

    if axis == 'x':
        R = np.array([
            [1, 0, 0],
            [0, np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
            [0, np.sin(np.radians(angle)), np.cos(np.radians(angle))]
        ])
    elif axis == 'y':
        R = np.array([
            [np.cos(np.radians(angle)), 0, np.sin(np.radians(angle))],
            [0, 1, 0],
            [-np.sin(np.radians(angle)), 0, np.cos(np.radians(angle))]
        ])
    elif axis == 'z':
        R = np.array([
            [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
            [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError(f"Unsupported axis: {axis}. Expected 'x', 'y', or 'z'.")

    if mat.ndim == 2:
        mat[:3, :3] = R @ mat[:3, :3]  # Rotate the rotation part
    elif mat.ndim == 3:
        for i in range(mat.shape[0]):
            mat[i, :3, :3] = R @ mat[i, :3, :3]
    else:
        raise ValueError(f"Unsupported matrix shape: {mat.shape}. Expected 2D or 3D array.")

    return mat

def local_rot(mat: np.ndarray, axis: str, angle: float) -> np.ndarray:
    """
    right multi means rotate in local coordinate system

    Rotate a pose around the specified axis in the local coordinate system.
    Args:
        mat: np.ndarray, shape (..., 4, 4), the transformation matrix.
        axis: str, 'x', 'y', or 'z'.
        angle: float, angle in degrees.
    Returns:
        np.ndarray, shape (..., 4, 4), the rotated transformation matrix.
    """
    mat = mat.copy()  # Avoid modifying the original matrix

    if axis == 'x':
        R = np.array([
            [1, 0, 0],
            [0, np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
            [0, np.sin(np.radians(angle)), np.cos(np.radians(angle))]
        ])
    elif axis == 'y':
        R = np.array([
            [np.cos(np.radians(angle)), 0, np.sin(np.radians(angle))],
            [0, 1, 0],
            [-np.sin(np.radians(angle)), 0, np.cos(np.radians(angle))]
        ])
    elif axis == 'z':
        R = np.array([
            [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
            [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError(f"Unsupported axis: {axis}. Expected 'x', 'y', or 'z'.")

    if mat.ndim == 2:
        mat[:3, :3] = mat[:3, :3] @ R  # Rotate the rotation part
    elif mat.ndim == 3:
        for i in range(mat.shape[0]):
            mat[i, :3, :3] = mat[i, :3, :3] @ R
    else:
        raise ValueError(f"Unsupported matrix shape: {mat.shape}. Expected 2D or 3D array.")

    return mat

def transform_avp_raw_data(avp_raw_data: dict[str, np.ndarray]):
    """
    transform VisionProTeleop's coordination system to the simulator unified frame.

    Args:
        avp_raw_data: dict[str, np.ndarray], the raw data from AVP, containing:
            - 'head': (1, 4, 4) transformation mat of the head in world frame
            - 'right_wrist': (1, 4, 4) transformation mat of the right wrist in world frame
            - 'left_wrist': (1, 4, 4) transformation mat of the left wrist in world frame
            - 'right_fingers': (25, 4, 4) transformation matrices of the right fingers relative to the right wrist
            - 'left_fingers': (25, 4, 4) transformation matrices of the left fingers relative to the left wrist
    Returns:
        dict[str, np.ndarray]: transformed poses in world frame:
            - 'head_mat_w': (1, 4, 4) pose of the
                head in world frame
            - 'right_wrist_mat_w': (1, 4, 4) pose of the right wrist in world frame
            - 'left_wrist_mat_w': (1, 4, 4) pose of the left wrist in world frame
            - 'right_fingers_mat_w': (25, 4, 4) poses of the right fingers in world frame
            - 'left_fingers_mat_w': (25, 4, 4) poses of the left fingers in world frame

        # dict[str, np.ndarray]: transformed poses in world frame:
        #     - 'head_pose_w': (7,) pose of the head in world frame
        #     - 'right_wrist_pose_w': (7,) pose of the right wrist in world frame
        #     - 'left_wrist_pose_w': (7,) pose of the left wrist in world frame
        #     - 'right_fingers_pose_w': (25, 7) poses of the right fingers in world frame
        #     - 'left_fingers_pose_w': (25, 7) poses of the left fingers in world frame
    """
    # in world frame
    head_mat_w = avp_raw_data["head"] # (1, 4, 4)
    right_wrist_mat_w = avp_raw_data["right_wrist"] # (1, 4, 4)
    left_wrist_mat_w = avp_raw_data["left_wrist"] # (1, 4, 4)

    head_mat_w[..., :3, :3] = np.eye(3)

    # _rw means relative to wrist
    right_fingers_mat_rw = avp_raw_data["right_fingers"] # (25, 4, 4)
    left_fingers_mat_rw = avp_raw_data["left_fingers"] # (25, 4, 4)

    # in world frame
    right_fingers_mat_w = np.concatenate(
        [right_wrist_mat_w @ right_finger for right_finger in right_fingers_mat_rw],
        axis=0
    ) # (25, 4, 4)
    left_fingers_mat_w = np.concatenate(
        [left_wrist_mat_w @ left_finger for left_finger in left_fingers_mat_rw],
        axis=0
    ) # (25, 4, 4)

    # Rotate 90 degrees around z-axis to match simulator's coordinate system
    world_rot_func = functools.partial(world_rot, axis="z", angle=90)
    head_mat_w = world_rot_func(head_mat_w) # (1, 4, 4)
    # right_wrist_mat_w = world_rot_func(right_wrist_mat_w) # (1, 4, 4)
    # left_wrist_mat_w = world_rot_func(left_wrist_mat_w) # (1, 4, 4)
    # right_fingers_mat_w = world_rot_func(right_fingers_mat_w)  # (25, 4, 4)
    # left_fingers_mat_w = world_rot_func(left_fingers_mat_w) # (25, 4, 4)

    # rotate quaternions to match simulator's coordinate system
    right_wrist_mat_w = local_rot(right_wrist_mat_w, "y", -90)  # (1, 4, 4)
    right_wrist_mat_w = local_rot(right_wrist_mat_w, "z", -90)  # (1, 4, 4)

    left_wrist_mat_w = local_rot(left_wrist_mat_w, "y", 90)  # (1, 4, 4)
    left_wrist_mat_w = local_rot(left_wrist_mat_w, "z", 90)  # (1, 4, 4)

    return {
        "head_mat_w": head_mat_w,  # (1, 4, 4)
        "right_wrist_mat_w": right_wrist_mat_w,  # (1, 4, 4)
        "left_wrist_mat_w": left_wrist_mat_w,  # (1, 4, 4)
        "right_fingers_mat_w": right_fingers_mat_w,  # (25, 4, 4)
        "left_fingers_mat_w": left_fingers_mat_w,  # (25, 4, 4)
        # "left_fingers_mat_rw": left_fingers_mat_rw,  # (25, 4, 4), relative to left wrist
        # "right_fingers_mat_rw": right_fingers_mat_rw,  # (25, 4, 4), relative to right wrist
    }

    # # convert to pose
    # head_pose_w = mat_to_pose(head_mat_w[0])  # (7,)
    # right_wrist_pose_w = mat_to_pose(right_wrist_mat_w[0])  # (7,)
    # left_wrist_pose_w = mat_to_pose(left_wrist_mat_w[0])  # (7,)

    # right_fingers_pose_w = mat_to_pose(right_fingers_mat_w)  # (25, 7)
    # left_fingers_pose_w = mat_to_pose(left_fingers_mat_w)  # (25, 7)

    # return {
    #     "head_pose_w": head_pose_w,  # (7,)
    #     "right_wrist_pose_w": right_wrist_pose_w,  # (7,)
    #     "left_wrist_pose_w": left_wrist_pose_w,  # (7,)
    #     "right_fingers_pose_w": right_fingers_pose_w,  # (25, 7)
    #     "left_fingers_pose_w": left_fingers_pose_w,  # (25, 7)
    # }


def retarget_frame_6dof(hand_joint_pos: np.ndarray, left_right: str):
    """
    Args:
        hand_joint_pos: np.ndarray, shape (21, 3), the positions of the 21 joints in the hand.
        left_right: str, 'left' or 'right', indicating the hand side.
    Returns:
        np.ndarray, shape (6,), the retargeted joint positions for the hand.
    """
    assert hand_joint_pos.shape == (21, 3), f"Expected hand_joint_pos to have shape (21, 3), got {hand_joint_pos.shape}"

    thumb_1_joint_pos = calc_plane_angle(hand_joint_pos[4], hand_joint_pos[0], hand_joint_pos[5], hand_joint_pos[17])  # angle in degrees
    thumb_1_joint_pos = 180 - thumb_1_joint_pos

    # thumb_2_joint_pos = angle_between_segments(hand_joint_pos[0], hand_joint_pos[1], hand_joint_pos[4])  # angle in degrees
    # index_1_joint_pos = angle_between_segments(hand_joint_pos[0], hand_joint_pos[5], hand_joint_pos[8])  # angle in degrees
    # middle_1_joint_pos = angle_between_segments(hand_joint_pos[0], hand_joint_pos[9], hand_joint_pos[12])  # angle in degrees
    # ring_1_joint_pos = angle_between_segments(hand_joint_pos[0], hand_joint_pos[13], hand_joint_pos[16])  # angle in degrees
    # pinky_1_joint_pos = angle_between_segments(hand_joint_pos[0], hand_joint_pos[17], hand_joint_pos[20])  # angle in degrees

    thumb_2_joint_pos = compute_finger_angle(hand_joint_pos[0], hand_joint_pos[5], hand_joint_pos[2], hand_joint_pos[4])
    index_1_joint_pos = compute_finger_angle(hand_joint_pos[0], hand_joint_pos[9], hand_joint_pos[5], hand_joint_pos[8])
    middle_1_joint_pos = compute_finger_angle(hand_joint_pos[0], hand_joint_pos[13], hand_joint_pos[9], hand_joint_pos[12])
    ring_1_joint_pos = compute_finger_angle(hand_joint_pos[0], hand_joint_pos[17], hand_joint_pos[13], hand_joint_pos[16])
    pinky_1_joint_pos = compute_finger_angle(hand_joint_pos[0], hand_joint_pos[13], hand_joint_pos[17], hand_joint_pos[20])

    if left_right == "left":
        pinky_1_joint_pos = -pinky_1_joint_pos
    elif left_right == "right":
        index_1_joint_pos = -index_1_joint_pos
        middle_1_joint_pos = -middle_1_joint_pos
        ring_1_joint_pos = -ring_1_joint_pos
        thumb_2_joint_pos = -thumb_2_joint_pos

    return np.array([
        thumb_1_joint_pos,
        thumb_2_joint_pos,
        index_1_joint_pos,
        middle_1_joint_pos,
        ring_1_joint_pos,
        pinky_1_joint_pos
    ], dtype=np.float32) # (6, )
