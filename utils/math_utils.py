import numpy as np
from scipy.spatial.transform import Rotation as R

__all__ = ['compute_delta', 'update_pose', 'axis_angle_to_xyzw', "VIVETRACKER_quat", 'posrot_convert', 'pose_to_matrix', 'matrix_to_pose']

def pose_to_matrix(pose):
    """
    将六维位姿转换为 4x4 齐次变换矩阵。
    pose: [x, y, z, rx, ry, rz]，其中旋转部分为旋转向量（弧度）。
    """
    pos = np.array(pose[:3])
    rotvec = np.array(pose[3:])
    R_mat = R.from_rotvec(rotvec).as_matrix()
    mat = np.eye(4)
    mat[:3, :3] = R_mat
    mat[:3, 3] = pos
    return mat


def matrix_to_pose(mat):
    """
    将 4x4 齐次变换矩阵转换为六维位姿 [x,y,z,rx,ry,rz]。
    其中旋转部分以旋转向量表示。
    """
    pos = mat[:3, 3]
    R_mat = mat[:3, :3]
    rotvec = R.from_matrix(R_mat).as_rotvec()
    return np.concatenate([pos, rotvec])

def VIVETRACKER_quat(quat):
    rot = R.from_quat(quat)

    rotate_trans = R.from_euler('x', 90, degrees=True)
    rot = rotate_trans * rot
    return rot.as_quat()

def compute_delta(start_pos, start_quat, pos, quat):
    """
    计算 delta_pos 和 delta_quat

    :param start_pos: 初始位置 (numpy array, shape=(3,))
    :param start_quat: 初始四元数 (numpy array, shape=(4,), scalar-last convention [x, y, z, w])
    :param pos: 目标位置 (numpy array, shape=(3,))
    :param quat: 目标四元数 (numpy array, shape=(4,), scalar-last convention [x, y, z, w])
    :return: delta_pos (numpy array, shape=(3,)), delta_quat (numpy array, shape=(4,))
    """
    # 计算位置变化
    delta_pos = pos - start_pos

    # 计算四元数变化
    start_rotation = R.from_quat(start_quat)
    target_rotation = R.from_quat(quat)

    # 计算四元数相对变化
    delta_rotation = target_rotation * start_rotation.inv()
    delta_quat = delta_rotation.as_quat()

    return delta_pos, delta_quat

def update_pose(start_pos, start_quat, delta_pos, delta_quat):
    """
    计算新的位姿（位置和四元数）。

    :param start_pos: 初始位置 (x, y, z)，列表或NumPy数组
    :param start_quat: 初始四元数 (x, y, z, w)，列表或NumPy数组
    :param delta_pos: 位置增量 (dx, dy, dz)，列表或NumPy数组
    :param delta_quat: 旋转增量四元数 (x, y, z, w)，列表或NumPy数组
    :return: (cur_pos, cur_quat) 新的位置和四元数
    """
    # 确保输入为 NumPy 数组
    start_pos = np.array(start_pos)
    start_quat = np.array(start_quat)
    delta_pos = np.array(delta_pos)
    delta_quat = np.array(delta_quat)

    # 计算新的四元数 (旋转组合)
    start_rot = R.from_quat(start_quat)
    delta_rot = R.from_quat(delta_quat)
    cur_rot = start_rot * delta_rot  # 旋转合成
    cur_quat = cur_rot.as_quat()  # 转换回四元数 (x, y, z, w)

    # 计算新的位置 (增量应用于初始坐标系)
    rotated_delta_pos = start_rot.apply(delta_pos)  # 旋转后的增量位置
    cur_pos = start_pos + rotated_delta_pos  # 计算新位置

    return cur_pos, cur_quat

import numpy as np

def axis_angle_to_xyzw(axis_angle):
    """
    将轴角（rx, ry, rz）转换为四元数 (x, y, z, w)

    :param axis_angle: np.array([rx, ry, rz])，轴角表示
    :return: np.array([x, y, z, w])，四元数
    """
    theta = np.linalg.norm(axis_angle)  # 计算旋转角度
    if theta < 1e-6:  # 处理接近零角度的情况，直接返回单位四元数
        return np.array([1.0, 0.0, 0.0, 0.0])

    axis = axis_angle / theta  # 归一化旋转轴
    half_theta = theta / 2.0
    w = np.cos(half_theta)
    xyz = axis * np.sin(half_theta)

    return np.array([xyz[0], xyz[1], xyz[2], w])

def xyzw_to_axis_angle(quat):
    """
    将四元数 (x, y, z, w) 转换为轴角 (rx, ry, rz)

    :param quat: np.array([x, y, z, w])，四元数表示
    :return: np.array([rx, ry, rz])，轴角
    """
    quat = np.array(quat)
    xyz = quat[:3]
    w = quat[3]

    theta = 2 * np.arccos(w)
    if np.abs(theta) < 1e-6:
        return np.array([0.0, 0.0, 0.0])

    axis = xyz / np.sin(theta / 2.0)

    return axis * theta

def posrot_convert(posrot, convert_mat, inv=False):
    # 提取位姿中的位置(x, y, z)和旋转部分(rx, ry, rz)
    position = np.array([posrot[0], posrot[1], posrot[2], 1])  # 位置需要转换成齐次坐标
    rotation = posrot[3:]  # 旋转部分（rx, ry, rz）

    # 根据需要选择是正变换还是逆变换
    if inv:
        convert_mat = np.linalg.inv(convert_mat)  # 求逆变换矩阵

    # 对位置进行变换
    transformed_position = np.dot(convert_mat, position)  # 位置变换

    # 将旋转向量转换为旋转矩阵
    rotation_matrix = R.from_rotvec(rotation).as_matrix()

    # 提取转换矩阵的旋转部分（前3x3的子矩阵）
    convert_rotation_matrix = convert_mat[:3, :3]

    # 对旋转矩阵进行变换
    transformed_rotation_matrix = np.dot(convert_rotation_matrix, rotation_matrix)

    # 将变换后的旋转矩阵转换回旋转向量
    transformed_rotation = R.from_matrix(transformed_rotation_matrix).as_rotvec()

    # 输出变换后的位姿，包括变换后的位置和旋转
    transformed_pose = (transformed_position[0], transformed_position[1], transformed_position[2]) + tuple(transformed_rotation)

    return transformed_pose

def pose_euler_convert(pose: np.ndarray, convert_mat: np.ndarray, inv=False):
    """
    Args:
        pose: [x, y, z, rx, ry, rz], np.ndarray, in radians
    Returns:
        converted_pose: [x, y, z, rx, ry, rz], np.ndarray, in radians
    """
    if inv:
        convert_mat = np.linalg.inv(convert_mat)

    position = np.array([pose[0], pose[1], pose[2], 1])
    euler_angles = np.array(pose[3:])  # 旋转部分为欧拉角（弧度）

    # 对位置进行变换
    transformed_position = np.dot(convert_mat, position)  # 位置变换

    # 将欧拉角转换为旋转矩阵
    rotation_matrix = R.from_euler('xyz', euler_angles).as_matrix()

    # 提取转换矩阵的旋转部分（前3x3的子矩阵）
    convert_rotation_matrix = convert_mat[:3, :3]

    # 对旋转矩阵进行变换
    transformed_rotation_matrix = np.dot(convert_rotation_matrix, rotation_matrix)

    # 将变换后的旋转矩阵转换回欧拉角
    transformed_euler_angles = R.from_matrix(transformed_rotation_matrix).as_euler('xyz')

    # 输出变换后的位姿，包括变换后的位置和欧拉角
    transformed_pose = (transformed_position[0], transformed_position[1], transformed_position[2]) + tuple(transformed_euler_angles)

    return np.array(transformed_pose)

def pose_convert(pose: np.ndarray, convert_mat: np.ndarray, inv=False):
    """
    Args:
        pose: [x, y, z, qx, qy, qz, qw], np.ndarray
    Returns:
        converted_pose: [x, y, z, qx, qy, qz, qw], np.ndarray
    """
    if inv:
        convert_mat = np.linalg.inv(convert_mat)

    # 提取位置和四元数
    position = np.array([pose[0], pose[1], pose[2], 1])  # 位置需要转换成齐次坐标
    quaternion = np.array(pose[3:])

    # 对位置进行变换
    transformed_position = np.dot(convert_mat, position)  # 位置变换

    # 将四元数转换为旋转矩阵
    rotation_matrix = R.from_quat(quaternion).as_matrix()

    # 提取转换矩阵的旋转部分（前3x3的子矩阵）
    convert_rotation_matrix = convert_mat[:3, :3]

    # 对旋转矩阵进行变换
    transformed_rotation_matrix = np.dot(convert_rotation_matrix, rotation_matrix)

    # 将变换后的旋转矩阵转换回四元数
    transformed_quaternion = R.from_matrix(transformed_rotation_matrix).as_quat()

    # 输出变换后的位姿，包括变换后的位置和四元数
    transformed_pose = (transformed_position[0], transformed_position[1], transformed_position[2]) + tuple(transformed_quaternion)

    return np.array(transformed_pose)

# def posrot_convert(posrot, convert_mat, inv=False):
#     # 提取位姿中的位置(x, y, z)和旋转部分(rx, ry, rz)
#     position = np.array([posrot[0], posrot[1], posrot[2], 1])  # 位置需要转换成齐次坐标
#     rotation = posrot[3:]  # 旋转部分（rx, ry, rz）

#     # 根据需要选择是正变换还是逆变换
#     if inv:
#         convert_mat = np.linalg.inv(convert_mat)  # 求逆变换矩阵

#     # 仅对位置进行变换
#     transformed_position = np.dot(convert_mat, position)  # 位置变换

#     # 输出变换后的位姿，旋转部分保持不变
#     transformed_pose = (transformed_position[0], transformed_position[1], transformed_position[2]) + tuple(rotation)

#     return transformed_pose
