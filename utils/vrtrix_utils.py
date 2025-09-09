import numpy as np
from scipy.spatial.transform import Rotation as R


def extract_finger_angles_from_quaternions_relative(joints_quat, joint_indices, wrist_quat, hand='right',
                                                   proximal_weight=0.7, intermediate_weight=0.3):
    """
    Extract finger bending angles from joint quaternions using relative rotations.

    Args:
        joints_quat: 16x4 array of quaternions [w,x,y,z]
        joint_indices: list of joint indices for a finger [proximal, intermediate, distal]
        wrist_quat: wrist quaternion [w,x,y,z]
        hand: 'right' or 'left' to handle coordinate system differences
        proximal_weight: weight for proximal joint angle
        intermediate_weight: weight for intermediate joint angle

    Returns:
        angle: combined finger bending angle in degrees
    """
    angles = []

    # Convert wrist quaternion to rotation
    wrist_rot = R.from_quat([wrist_quat[1], wrist_quat[2], wrist_quat[3], wrist_quat[0]])  # scipy uses [x,y,z,w]

    # Previous rotation starts with wrist
    prev_rot = wrist_rot

    for i, idx in enumerate(joint_indices[:2]):  # Only use proximal and intermediate
        # Current joint rotation (absolute)
        quat = joints_quat[idx]
        curr_rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])

        # Calculate relative rotation: R_rel = R_prev^T * R_curr
        rel_rot = prev_rot.inv() * curr_rot

        # Get relative rotation matrix
        rel_matrix = rel_rot.as_matrix()

        # For finger flexion, we want rotation around Z-axis in the local joint frame
        # Extract flexion angle from the relative rotation matrix
        flexion_angle = np.degrees(np.arctan2(-rel_matrix[1, 0], rel_matrix[0, 0]))

        # Handle hand-specific coordinate conventions
        if hand == 'left':
            flexion_angle = -flexion_angle
        elif hand == 'right':
            flexion_angle = -flexion_angle

        # Normalize to positive angles for flexion
        if flexion_angle < 0:
            flexion_angle = 0

        # Weight the angles based on joint type
        if i == 0:  # Proximal
            angles.append(flexion_angle * proximal_weight)
        else:  # Intermediate
            angles.append(flexion_angle * intermediate_weight)

        # Update previous rotation for next iteration
        prev_rot = curr_rot

    # Combine angles
    return sum(angles)

def extract_thumb_abduction(joints_quat, joint_indices, wrist_quat, hand='right'):
    """
    Extract thumb abduction/adduction angle from joint quaternions.

    Args:
        joints_quat: 16x4 array of quaternions [w,x,y,z]
        joint_indices: list of joint indices for thumb [proximal, intermediate, distal]
        wrist_quat: wrist quaternion [w,x,y,z]
        hand: 'right' or 'left' to handle coordinate system differences

    Returns:
        thumb_abduction: abduction/adduction angle in degrees (movement away from palm)
    """
    # Convert wrist quaternion to rotation
    wrist_rot = R.from_quat([wrist_quat[1], wrist_quat[2], wrist_quat[3], wrist_quat[0]])  # scipy uses [x,y,z,w]

    # Get proximal joint rotation (CMC joint - responsible for abduction/adduction)
    proximal_quat = joints_quat[joint_indices[0]]
    proximal_rot = R.from_quat([proximal_quat[1], proximal_quat[2], proximal_quat[3], proximal_quat[0]])

    # Calculate relative rotation: R_rel = R_wrist^T * R_proximal
    rel_rot = wrist_rot.inv() * proximal_rot
    euler_angles = rel_rot.as_euler('xyz', degrees=True)
    abduction_angle = euler_angles[2]  # Z rotation
    return abduction_angle


def extract_thumb_flexion(joints_quat, joint_indices, wrist_quat, hand='right'):
    """
    Extract thumb flexion angles from joint quaternions.

    Args:
        joints_quat: 16x4 array of quaternions [w,x,y,z]
        joint_indices: list of joint indices for thumb [proximal, intermediate, distal]
        wrist_quat: wrist quaternion [w,x,y,z]
        hand: 'right' or 'left' to handle coordinate system differences

    Returns:
        thumb flexion angle in degrees (bending movement)
    """
    # Convert wrist quaternion to rotation
    wrist_rot = R.from_quat([wrist_quat[1], wrist_quat[2], wrist_quat[3], wrist_quat[0]])  # scipy uses [x,y,z,w]

    # Get proximal joint rotation
    proximal_quat = joints_quat[joint_indices[0]]
    proximal_rot = R.from_quat([proximal_quat[1], proximal_quat[2], proximal_quat[3], proximal_quat[0]])

    # Get intermediate joint rotation
    intermediate_quat = joints_quat[joint_indices[1]]
    intermediate_rot = R.from_quat([intermediate_quat[1], intermediate_quat[2], intermediate_quat[3], intermediate_quat[0]])

    # Get distal joint rotation
    distal_quat = joints_quat[joint_indices[2]]
    distal_rot = R.from_quat([distal_quat[1], distal_quat[2], distal_quat[3], distal_quat[0]])


    # Proximal to intermediate flexion
    rel_rot_proximal_to_intermediate = proximal_rot.inv() * intermediate_rot
    rel_matrix_proximal_to_intermediate = rel_rot_proximal_to_intermediate.as_matrix()
    flexion_proximal = np.degrees(np.arctan2(-rel_matrix_proximal_to_intermediate[1, 0],
                                              rel_matrix_proximal_to_intermediate[0, 0]))

    # Intermediate to distal flexion
    rel_rot_intermediate_to_distal = intermediate_rot.inv() * distal_rot
    rel_matrix_intermediate_to_distal = rel_rot_intermediate_to_distal.as_matrix()
    flexion_intermediate = np.degrees(np.arctan2(-rel_matrix_intermediate_to_distal[1, 0],
                                                 rel_matrix_intermediate_to_distal[0, 0]))

    # Handle hand-specific coordinate conventions
    if hand == 'left':
        flexion_proximal = -flexion_proximal
        flexion_intermediate = -flexion_intermediate
    elif hand == 'right':
        flexion_proximal = -flexion_proximal
        flexion_intermediate = -flexion_intermediate

    if flexion_proximal < 0:
        flexion_proximal = 0
    if flexion_intermediate < 0:
        flexion_intermediate = 0
    thumb_flexion = 0.3 * flexion_proximal + 0.7 * flexion_intermediate


    return thumb_flexion


def vrtrix_to_realman_6dof(joints_quat, hand='right', proximal_weight=0.7, intermediate_weight=0.3):
    """
    Map VRTrix 16 joint quaternions to Realman 6DOF hand control using relative rotations.

    Args:
        joints_quat: 16x4 array of quaternions [w,x,y,z]
        hand: 'right' or 'left'
        proximal_weight: weight for proximal joint angle
        intermediate_weight: weight for intermediate joint angle

    Returns:
        hand_angles: 6-element array of angles in degrees
                    [thumb_1, thumb_2, index_1, middle_1, ring_1, pinky_1]
    """
    # Get wrist quaternion
    wrist_quat = joints_quat[0]

    # Define joint indices for each finger
    finger_joints = {
        'thumb': [1, 2, 3],    # Proximal, Intermediate, Distal
        'index': [4, 5, 6],
        'middle': [7, 8, 9],
        'ring': [10, 11, 12],
        'pinky': [13, 14, 15]
    }

    # 特别处理大拇指

    thumb_1= extract_thumb_abduction(
        joints_quat, finger_joints['thumb'], wrist_quat, hand)


    thumb_2= extract_thumb_flexion(
        joints_quat, finger_joints['thumb'], wrist_quat, hand)

    # 其他四指继续使用原来的方法
    index_angle = extract_finger_angles_from_quaternions_relative(
        joints_quat, finger_joints['index'], wrist_quat, hand, proximal_weight, intermediate_weight)
    middle_angle = extract_finger_angles_from_quaternions_relative(
        joints_quat, finger_joints['middle'], wrist_quat, hand, proximal_weight, intermediate_weight)
    ring_angle = extract_finger_angles_from_quaternions_relative(
        joints_quat, finger_joints['ring'], wrist_quat, hand, proximal_weight, intermediate_weight)
    pinky_angle = extract_finger_angles_from_quaternions_relative(
        joints_quat, finger_joints['pinky'], wrist_quat, hand, proximal_weight, intermediate_weight)

    # 组装6DOF数组

    hand_angles = np.array([
        thumb_1,
        thumb_2,
        index_angle,
        middle_angle,
        ring_angle,
        pinky_angle
    ], dtype=np.float64)

    # 应用缩放因子
    # 调整大拇指的缩放因子以获得更好的映射效果
    scaling_factors = np.array([
        1.3,    # thumb_1
        1.8,    # thumb_2
        1.0,    # index
        1.0,    # middle
        1.0,    # ring
        1.0     # pinky
    ])
    hand_angles = hand_angles * scaling_factors

    return hand_angles