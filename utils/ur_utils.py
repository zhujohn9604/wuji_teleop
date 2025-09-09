import numpy as np


def move_to_pose(rtde_c, rtde_r, start_pose_w: np.ndarray, error: float, max_attempts: int, velocity: float, acceleration: float, fps: int, lookahead_time: float, gain: float):
    attempt = 0
    while attempt < max_attempts:
        current_pose = np.array(rtde_r.getActualTCPPose())
        pos_error = np.linalg.norm(current_pose[:3] - start_pose_w[:3])
        rot_error = np.linalg.norm(current_pose[3:] - start_pose_w[3:])
        total_error = pos_error + rot_error
        if total_error < error:
            print("Reached initial pose.")
            break
        rtde_c.servoL(
            start_pose_w,
            velocity, acceleration,
            1.0 / fps,
            lookahead_time,
            gain
        )
        attempt += 1

