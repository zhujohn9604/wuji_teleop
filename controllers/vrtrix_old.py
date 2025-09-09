from dataclasses import field, dataclass
from typing import Optional, Any, Dict
import enum
import multiprocessing as mp
import time
import traceback
import os
import copy
from collections import deque

from pink.tasks import FrameTask
import numpy as np
from scipy.spatial.transform import Rotation as R

from .base import BaseController, BaseControllerConfig

from utils.retargeting_utils import (
    transform_avp_raw_data,
    pose_to_mat,
    calc_delta_mat,
    calc_end_mat,
    mat_to_pose,
    retarget_frame_6dof,
    j25_to_j21,
)
from utils.pink_ik_utils import PinkIKController


@dataclass
class VRTrixControllerConfig(BaseControllerConfig):
    ip: str = ""

    robot_r_mat_w: np.ndarray = None
    robot_l_mat_w: np.ndarray = None

    assume_avp_pos_offset: np.ndarray = None
    assume_avp_mat_w: np.ndarray = None

    def validate(self):
        super().validate()

        assert isinstance(self.robot_l_mat_w, np.ndarray) and self.robot_l_mat_w.shape == (4, 4), \
            f"robot_l_mat_w should be a 4x4 numpy array, got {type(self.robot_l_mat_w)} with shape {self.robot_l_mat_w.shape}"
        assert isinstance(self.robot_r_mat_w, np.ndarray) and self.robot_r_mat_w.shape == (4, 4), \
            f"robot_r_mat_w should be a 4x4 numpy array, got {type(self.robot_r_mat_w)} with shape {self.robot_r_mat_w.shape}"

        assert isinstance(self.assume_avp_pos_offset, np.ndarray) and self.assume_avp_pos_offset.shape == (3,), \
            f"assume_avp_pos_offset should be a 3-element numpy array, got {type(self.assume_avp_pos_offset)} with shape {self.assume_avp_pos_offset.shape}"

        self.assume_avp_mat_w = pose_to_mat(
            np.concatenate([
                self.assume_avp_pos_offset,
                np.array([0.0, 0.0, 0.0, 1.0])
            ])
        )  # (4, 4)

        self.feedback_sample = {
            "right_q": np.zeros((7,), dtype=np.float64),
            "left_q": np.zeros((7,), dtype=np.float64),
            "right_hand_cmd": np.zeros((6, ), dtype=np.float64),
            "left_hand_cmd": np.zeros((6, ), dtype=np.float64),
            "receive_timestamp": np.zeros((1,), dtype=np.float64),
            "timestamp": np.zeros((1,), dtype=np.float64),
        }

def init_pink_ik_controller(dt: float):
    urdf_path = "/home/wuji/code/dex-real-deployment/urdf/rm75b/rm_75_b_description.urdf"
    mesh_path = "/home/wuji/code/dex-real-deployment/urdf/rm75b"

    left_pink_ik_controller = PinkIKController(
        urdf_path=urdf_path,
        mesh_path=mesh_path,
        variable_input_tasks=[
            FrameTask(
                "Link7",
                position_cost=1.0,  # [cost] / [m]
                orientation_cost=1.0,  # [cost] / [rad]
                lm_damping=10,  # dampening for solver for step jumps
                gain=0.1,
            ),
        ],
        fixed_input_tasks=[],
        dt=dt,
    )

    right_pink_ik_controller = PinkIKController(
        urdf_path=urdf_path,
        mesh_path=mesh_path,
        variable_input_tasks=[
            FrameTask(
                "Link7",
                position_cost=1.0,  # [cost] / [m]
                orientation_cost=1.0,  # [cost] / [rad]
                lm_damping=10,  # dampening for solver for step jumps
                gain=0.1,
            ),
        ],
        fixed_input_tasks=[],
        dt=dt,
    )

    return right_pink_ik_controller, left_pink_ik_controller


class VRTrixController(BaseController):
    config: VRTrixControllerConfig

    def __init__(self, config: VRTrixControllerConfig):
        super().__init__(config)

        self._is_ready = self.mp_manager.Value("b", True)

    ################## cls methods ##################
    def check_ready(self):
        return self._is_ready.value and not self._stop_event.is_set()

    ################## abstract methods ##################
    def _initialize(self):
        self.avp_streamer = VisionProStreamer(self.config.ip, False)

        self.right_pink_ik_controller, self.left_pink_ik_controller = init_pink_ik_controller(self.dt)

    def _update(self):
        assume_avp_mat_w = self.config.assume_avp_mat_w  # (4, 4)
        robot_r_mat_w = self.config.robot_r_mat_w  # (4, 4)
        robot_l_mat_w = self.config.robot_l_mat_w  # (4, 4)
        right_pink_ik_controller = self.right_pink_ik_controller
        left_pink_ik_controller = self.left_pink_ik_controller


        avp_raw_data = copy.deepcopy(self.avp_streamer.latest) # keys: ['head', 'right_wrist', 'left_wrist', 'right_fingers', 'left_fingers'], np.ndarray
        timestamp = time.time()



        avp_transformed_data = transform_avp_raw_data(avp_raw_data)

        # _fw means fake world frame(world frame in the avp world)
        head_mat_fw = avp_transformed_data["head_mat_w"].squeeze(0)  # (4, 4)
        right_wrist_mat_fw = avp_transformed_data["right_wrist_mat_w"].squeeze(0)  # (4, 4)
        left_wrist_mat_fw = avp_transformed_data["left_wrist_mat_w"].squeeze(0)  # (4, 4)
        right_fingers_mat_fw = avp_transformed_data["right_fingers_mat_w"]  # (25, 4, 4)
        left_fingers_mat_fw = avp_transformed_data["left_fingers_mat_w"]  # (25, 4, 4)

        right_fingers_mat_right_wrist = calc_delta_mat(
            right_wrist_mat_fw.reshape(1, 4, 4).repeat(25, axis=0), # (25, 4, 4)
            right_fingers_mat_fw, # (25, 4, 4)
        ) # (25, 4, 4)
        left_fingers_mat_left_wrist = calc_delta_mat(
            left_wrist_mat_fw.reshape(1, 4, 4).repeat(25, axis=0), # (25, 4, 4)
            left_fingers_mat_fw, # (25, 4, 4)
        ) # (25, 4, 4)

        # in avp world, +y is forward, +x is right, +z is up

        # # _fw means fake world frame(world frame in the avp world)
        # head_pose_fw = avp_transformed_data["head_pose_w"]  # (7,)
        # right_wrist_pose_fw = avp_transformed_data["right_wrist_pose_w"]  # (7,)
        # left_wrist_pose_fw = avp_transformed_data["left_wrist_pose_w"]  # (7,)
        # right_fingers_pose_fw = avp_transformed_data["right_fingers_pose_w"]  # (25, 7)
        # left_fingers_pose_fw = avp_transformed_data["left_fingers_pose_w"]  # (25, 7)

        right_wrist_mat_avp = calc_delta_mat(
            head_mat_fw,
            right_wrist_mat_fw,
        ) # (4, 4)
        left_wrist_mat_avp = calc_delta_mat(
            head_mat_fw,
            left_wrist_mat_fw,
        ) # (4, 4)

        right_wrist_mat_w = calc_end_mat(
            assume_avp_mat_w,
            right_wrist_mat_avp,
        ) # (4, 4)
        left_wrist_mat_w = calc_end_mat(
            assume_avp_mat_w,
            left_wrist_mat_avp,
        ) # (4, 4)

        right_fingers_mat_w = calc_end_mat(
            right_wrist_mat_w.reshape(1, 4, 4).repeat(25, axis=0), # (25, 4, 4)
            right_fingers_mat_right_wrist, # (25, 4, 4)
        ) # (25, 4, 4)
        left_fingers_mat_w = calc_end_mat(
            left_wrist_mat_w.reshape(1, 4, 4).repeat(25, axis=0), # (25, 4, 4)
            left_fingers_mat_left_wrist, # (25, 4, 4)
        ) # (25, 4, 4)

        right_fingers_pose_w = mat_to_pose(right_fingers_mat_w)  # (25, 7)
        left_fingers_pose_w = mat_to_pose(left_fingers_mat_w)  # (25, 7)

        right_wrist_mat_right_base_link = calc_delta_mat(
            robot_r_mat_w,
            right_wrist_mat_w,
        ) # (4, 4)
        left_wrist_mat_left_base_link = calc_delta_mat(
            robot_l_mat_w,
            left_wrist_mat_w,
        ) # (4, 4)

        right_pink_ik_controller.set_target(right_wrist_mat_right_base_link)
        left_pink_ik_controller.set_target(left_wrist_mat_left_base_link)

        right_q = right_pink_ik_controller.compute() # (7, )
        left_q = left_pink_ik_controller.compute() # (7, )

        right_hand_qpos = retarget_frame_6dof(j25_to_j21(right_fingers_pose_w[:, :3])) # (6, ), in degrees
        left_hand_qpos = retarget_frame_6dof(j25_to_j21(left_fingers_pose_w[:, :3])) # (6, ), in degrees

        angle_min, angle_max = 0, 180

        right_hand_qpos = right_hand_qpos.clip(angle_min, angle_max) # clip to [angle_min, angle_max] degrees # (6, )
        left_hand_qpos = left_hand_qpos.clip(angle_min, angle_max) # clip to [angle_min, angle_max] degrees # (6, )

        right_hand_cmd = right_hand_qpos / angle_max * 65535 # scale to [0, 65535] # (6, )
        left_hand_cmd = left_hand_qpos / angle_max * 65535 # scale to [0, 65535] # (6, )

        # 0: thumb_1
        # 1: thumb_2
        # 2: index_1
        # 3: middle_1
        # 4: ring_1
        # 5: pinky_1

        # ->

        # 0: thumb_2
        # 1: index_1
        # 2: middle_1
        # 3: ring_1
        # 4: pinky_1
        # 5: thumb_1

        right_hand_cmd = np.concatenate([
            right_hand_cmd[1:],
            right_hand_cmd[:1]
        ])  # (6, )

        left_hand_cmd = np.concatenate([
            left_hand_cmd[1:],
            left_hand_cmd[:1]
        ])  # (6, )

        state_dict = {
            "right_q": np.array(right_q, dtype=np.float64),
            "left_q": np.array(left_q, dtype=np.float64),
            "right_hand_cmd": np.array(right_hand_cmd, dtype=np.float64),
            "left_hand_cmd": np.array(left_hand_cmd, dtype=np.float64),
            "receive_timestamp": np.array([timestamp], dtype=np.float64),
            "timestamp": np.array([timestamp], dtype=np.float64)
        }
        self.feedback_queue.put(state_dict)

    def _process_commands(self):
        pass

    def _close(self):
        pass

    def reset(self):
        pass
