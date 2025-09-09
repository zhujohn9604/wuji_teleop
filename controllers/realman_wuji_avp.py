from dataclasses import field, dataclass
from typing import Optional, Any, Dict
import enum
import multiprocessing as mp
import time
import traceback
import warnings
import os
import redis
import json
import copy


from collections import deque

from pink.tasks import FrameTask
import numpy as np
from scipy.spatial.transform import Rotation as R

from avp_stream import VisionProStreamer

from .base import BaseController, BaseControllerConfig

from utils.retargeting_utils import (
    transform_avp_raw_data,
    pose_to_mat,
    calc_delta_mat,
    calc_end_mat,
    retarget_frame_6dof,
    j25_to_j21,
)
from utils.wuji_utils import retarget_wuji_hand
from utils.mediapipe_utils import convert_vision_pro_to_mediapipe_format

from utils.pink_ik_utils import PinkIKController
from utils.optimizer_utils import LPFilter


@dataclass
class RealmanAvpControllerConfig(BaseControllerConfig):
    ip: str = ""

    redis_ip: Optional[str] = None
    redis_port: int = 6379

    use_redis_server: bool = None

    low_pass_alpha: float = 1.0

    robot_r_mat_w: np.ndarray = None
    robot_l_mat_w: np.ndarray = None

    pink_ik_damp: float = 0.5
    pink_ik_gain: float = 1.0

    retarget_config_path_r: str = ""
    retarget_config_path_l: str = ""

    default_right_q: np.ndarray = field(default_factory=lambda: np.zeros((7,), dtype=np.float64))
    default_left_q: np.ndarray = field(default_factory=lambda: np.zeros((7,), dtype=np.float64))

    enable_right: bool = True
    enable_left: bool = True

    assume_avp_pos_offset: np.ndarray = None
    assume_avp_mat_w: np.ndarray = None

    hand_cmd_len: int = 6

    def validate(self):
        super().validate()

        self.use_redis_server = self.redis_ip is not None

        assert 0.0 <= self.low_pass_alpha <= 1.0, \
            f"low_pass_alpha should be in [0.0, 1.0], got {self.low_pass_alpha}"

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
            "right_mat_w": np.zeros((4, 4, ), dtype=np.float64),
            "left_mat_w": np.zeros((4, 4, ), dtype=np.float64),
            "right_hand_cmd": np.zeros((self.hand_cmd_len, ), dtype=np.float64),
            "left_hand_cmd": np.zeros((self.hand_cmd_len, ), dtype=np.float64),
            "receive_timestamp": np.zeros((1,), dtype=np.float64),
            "timestamp": np.zeros((1,), dtype=np.float64),
        }

def init_pink_ik_controller(dt: float, damping: float, gain: float):
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
                # lm_damping=10,  # dampening for solver for step jumps
                # gain=0.1,

                lm_damping=damping,  # dampening for solver for step jumps
                gain=gain,
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
                # lm_damping=10,  # dampening for solver for step jumps
                # gain=0.1,

                lm_damping=damping,  # dampening for solver for step jumps
                gain=gain,
            ),
        ],
        fixed_input_tasks=[],
        dt=dt,
    )

    return right_pink_ik_controller, left_pink_ik_controller


# 在文件开头添加全局变量（给手指单独维护缓冲区）
_finger_buffers = {
    "right_fingers": None,
    "left_fingers": None
}
_buffer_size = 10

class RealmanAvpController(BaseController):
    config: RealmanAvpControllerConfig

    def __init__(self, config: RealmanAvpControllerConfig):
        super().__init__(config)

        self._is_ready = self.mp_manager.Value("b", True)
        self.data_list = []
        self.init_dex_retargeter(config)

    def init_dex_retargeter(self, config=None):
        from dex_retargeting.retargeting_config import RetargetingConfig

        if config.retarget_config_path_r is not None:
            config_r = RetargetingConfig.load_from_file(config.retarget_config_path_r)
            self.retarget_right = config_r.build()
            self.retargeting_type_right = self.retarget_right.optimizer.retargeting_type
            self.indices_right = self.retarget_right.optimizer.target_link_human_indices
        if config.retarget_config_path_l is not None:
            config_l = RetargetingConfig.load_from_file(config.retarget_config_path_l)
            self.retarget_left = config_l.build()
            self.retargeting_type_left = self.retarget_left.optimizer.retargeting_type
            self.indices_left = self.retarget_left.optimizer.target_link_human_indices

    ################## cls methods ##################
    @staticmethod
    def calc_pos_dist(a_mat: np.ndarray, b_mat: np.ndarray) -> float:
        """
        Args:
            a_mat: np.ndarray, shape (4, 4)
            b_mat: np.ndarray, shape (4, 4)

        Returns:
            float: distance between the two matrices, calculated as the Frobenius norm of their difference
        """
        a_pos = a_mat[:3, 3]  # Extract position from the matrix
        b_pos = b_mat[:3, 3]  # Extract position from the matrix

        return np.linalg.norm(a_pos - b_pos)

    def check_ready(self):
        return self._is_ready.value and not self._stop_event.is_set()

    def load_fake_data(self, fake_data_dir: str):
        self.right_pose_ws = np.load(f"{fake_data_dir}/right_pose_ws.npy") # (N, 7)
        self.left_pose_ws = np.load(f"{fake_data_dir}/left_pose_ws.npy") # (N, 7)
        self.right_qs = np.load(f"{fake_data_dir}/right_qs.npy") # (N, 7)
        self.left_qs = np.load(f"{fake_data_dir}/left_qs.npy") # (N, 7)
        self.right_fingers_pose_ws = np.load(f"{fake_data_dir}/right_fingers_pose_ws.npy") # (N, 25, 7)
        self.left_fingers_pose_ws = np.load(f"{fake_data_dir}/left_fingers_pose_ws.npy") # (N, 25, 7)

        self.head_pose_fws = np.load(f"{fake_data_dir}/head_pose_fws.npy") # (N, 7)

        self.index = 0

        self.fake_data = True

    def get_fake_data_feedback(self):
        if self.index >= len(self.right_qs):
            self.index = 0

        timestamp = time.time()

        right_pose_w = self.right_pose_ws[self.index]  # (7,)
        left_pose_w = self.left_pose_ws[self.index]  # (7,)
        right_q = self.right_qs[self.index]  # (7,)
        left_q = self.left_qs[self.index]  # (7,)
        right_fingers_pose_w = self.right_fingers_pose_ws[self.index]  # (25, 7)
        left_fingers_pose_w = self.left_fingers_pose_ws[self.index]  # (25, 7)

        right_hand_qpos = retarget_frame_6dof(j25_to_j21(right_fingers_pose_w[:, :3]), left_right="right") # (6, ), in degrees
        left_hand_qpos = retarget_frame_6dof(j25_to_j21(left_fingers_pose_w[:, :3]), left_right="left") # (6, ), in degrees

        angle_min, angle_max = 0, 60

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
            "right_mat_w": np.zeros((4, 4, ), dtype=np.float64), # TODO:
            "left_mat_w": np.zeros((4, 4, ), dtype=np.float64),
            "right_hand_cmd": np.array(right_hand_cmd, dtype=np.float64),
            "left_hand_cmd": np.array(left_hand_cmd, dtype=np.float64),
            "receive_timestamp": np.array([timestamp], dtype=np.float64),
            "timestamp": np.array([timestamp], dtype=np.float64)
        }

        self.index += 1

        return state_dict

    ################## abstract methods ##################
    def get_feedback(self):
        if hasattr(self, "fake_data") and self.fake_data:
            return self.get_fake_data_feedback()

        return self.feedback_queue.get()

    def _initialize(self):
        if self.config.use_redis_server:
            self.redis_client = redis.Redis(host=self.config.redis_ip, port=self.config.redis_port, db=0)
        else:
            self.avp_streamer = VisionProStreamer(self.config.ip, False)

        self.right_pink_ik_controller, self.left_pink_ik_controller = init_pink_ik_controller(self.dt, self.config.pink_ik_damp, self.config.pink_ik_gain)

        # self.lp_filter = LPFilter(self.config.low_pass_alpha)

    def process_right_hand_data(self, right_fingers_mat):
        """Process VisionPro RIGHT-hand data and send to robot."""

        joint_commands = retarget_wuji_hand(right_fingers_mat)  # 不传self.rh_filter

        human_mediapipe_pose = convert_vision_pro_to_mediapipe_format(
            right_fingers_mat, hand_type="Right"
        )


        origin_indices = self.indices_right[0, :]
        task_indices = self.indices_right[1, :]
        ref_value = human_mediapipe_pose[task_indices, :] - human_mediapipe_pose[origin_indices, :]

        robot_qpos = self.retarget_right.retarget(ref_value)
        hand_positions = robot_qpos.reshape(5, 4)

        hand_positions[0, 2] = joint_commands[0, 2]
        hand_positions[0, 3] = joint_commands[0, 3]


        hand_positions[1, 0] = joint_commands[1, 0]
        hand_positions[1, 2] = joint_commands[1, 2]

        hand_positions[2, 0] = joint_commands[2, 0]
        hand_positions[2, 2] = joint_commands[2, 2]

        hand_positions[3, 0] = joint_commands[3, 0]
        hand_positions[3, 2] = joint_commands[3, 2]

        hand_positions[4, 0] = joint_commands[4, 0]
        hand_positions[4, 2] = joint_commands[4, 2]

        return hand_positions.flatten()


    def process_left_hand_data(self, left_fingers_mat):
        """Process VisionPro LEFT-hand data and send to robot."""

        joint_commands = retarget_wuji_hand(left_fingers_mat)  # 不传self.rh_filter

        human_mediapipe_pose = convert_vision_pro_to_mediapipe_format(
            left_fingers_mat, hand_type="Left"
        )

        origin_indices = self.indices_left[0, :]
        task_indices = self.indices_left[1, :]
        ref_value = human_mediapipe_pose[task_indices, :] - human_mediapipe_pose[origin_indices, :]

        robot_qpos = self.retarget_left.retarget(ref_value)
        hand_positions = robot_qpos.reshape(5, 4)
        hand_positions[0, 2] = joint_commands[0, 2]
        hand_positions[0, 3] = joint_commands[0, 3]


        hand_positions[1, 0] = joint_commands[1, 0]
        hand_positions[1, 1] = joint_commands[1, 1]
        hand_positions[1, 2] = joint_commands[1, 2]

        hand_positions[2, 0] = joint_commands[2, 0]
        hand_positions[2, 1] = joint_commands[2, 1]
        hand_positions[2, 2] = joint_commands[2, 2]

        hand_positions[3, 0] = joint_commands[3, 0]
        hand_positions[3, 1] = joint_commands[3, 1]
        hand_positions[3, 2] = joint_commands[3, 2]

        hand_positions[4, 0] = joint_commands[4, 0]
        hand_positions[4, 1] = joint_commands[4, 1]
        hand_positions[4, 2] = joint_commands[4, 2]

        hand_positions[1, 1] = -1 * hand_positions[1, 1]
        hand_positions[2, 1] = -1 * hand_positions[2, 1]
        hand_positions[3, 1] = -1 * hand_positions[3, 1]
        hand_positions[4, 1] = -1 * hand_positions[4, 1]

        return hand_positions.flatten()


    def smooth_finger_data(self, avp_raw_data):
        """对VisionPro手指数据进行多帧平滑处理"""
        global _finger_buffers

        smoothed_data = {}

        for key in avp_raw_data:
            if key in ["right_fingers", "left_fingers"]:
                finger_data = avp_raw_data[key]

                # 初始化缓冲区
                if _finger_buffers[key] is None:
                    _finger_buffers[key] = [finger_data.copy() for _ in range(_buffer_size)]
                    smoothed_data[key] = finger_data
                    continue

                # 添加新帧
                _finger_buffers[key].append(finger_data.copy())

                # 保持缓冲区大小
                if len(_finger_buffers[key]) > _buffer_size:
                    _finger_buffers[key].pop(0)

                # 计算加权平均（最近的帧权重更大）
                weights = np.linspace(0.1, 1.0, len(_finger_buffers[key]))
                weights = weights / np.sum(weights)

                smoothed = np.zeros_like(finger_data)
                for frame, weight in zip(_finger_buffers[key], weights):
                    smoothed += weight * frame

                smoothed_data[key] = smoothed
            else:
                smoothed_data[key] = avp_raw_data[key].copy()

        return smoothed_data

    def _update(self):
        assume_avp_mat_w = self.config.assume_avp_mat_w  # (4, 4)
        robot_r_mat_w = self.config.robot_r_mat_w  # (4, 4)
        robot_l_mat_w = self.config.robot_l_mat_w  # (4, 4)
        right_pink_ik_controller = self.right_pink_ik_controller
        left_pink_ik_controller = self.left_pink_ik_controller


        if self.config.use_redis_server:
            def load_redis_np_array(key: str):
                return np.array(json.loads(self.redis_client.get(key)))
            avp_raw_data = {
                "head": load_redis_np_array("head"),
                "right_wrist": load_redis_np_array("right_wrist"),
                "left_wrist": load_redis_np_array("left_wrist"),
                "right_fingers": load_redis_np_array("right_fingers"),
                "left_fingers": load_redis_np_array("left_fingers"),
            }
        else:
            avp_raw_data = copy.deepcopy(self.avp_streamer.latest) # keys: ['head', 'right_wrist', 'left_wrist', 'right_fingers', 'left_fingers'], np.ndarray

        right_wrist_head_dist = self.calc_pos_dist(avp_raw_data["head"][0], avp_raw_data["right_wrist"][0]) # float
        left_wrist_head_dist = self.calc_pos_dist(avp_raw_data["head"][0], avp_raw_data["left_wrist"][0]) # float

        right_wrist_out_of_view = right_wrist_head_dist < 0.05
        left_wrist_out_of_view = left_wrist_head_dist < 0.05


        timestamp = time.time()
        avp_raw_data = self.smooth_finger_data(avp_raw_data)
        avp_transformed_data = transform_avp_raw_data(avp_raw_data)


        """
        import pickle
        self.data_list.append(avp_raw_data)
        with open('/home/wuji/dummy_data/avp_raw_data_flow.pkl', 'wb') as fp:
            pickle.dump(self.data_list, fp)
        print('save data!')
        """
        # _fw means fake world frame(world frame in the avp world)
        head_mat_fw = avp_transformed_data["head_mat_w"].squeeze(0)  # (4, 4)
        right_wrist_mat_fw = avp_transformed_data["right_wrist_mat_w"].squeeze(0)  # (4, 4)
        left_wrist_mat_fw = avp_transformed_data["left_wrist_mat_w"].squeeze(0)  # (4, 4)
        right_fingers_mat_fw = avp_raw_data["right_fingers"]  # (25, 4, 4)
        left_fingers_mat_fw = avp_raw_data["left_fingers"]  # (25, 4, 4)



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


        #right_fingers_pose_w = mat_to_pose(right_fingers_mat_w)  # (25, 7)
        #left_fingers_pose_w = mat_to_pose(left_fingers_mat_w)  # (25, 7)

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

        if self.config.enable_right:
            if right_wrist_out_of_view:
                print(f"Right wrist out of view, dist to head: {right_wrist_head_dist:.2f} m")
                right_q = self.config.default_right_q.copy()  # (7, )
            else:
                right_q = right_pink_ik_controller.compute() # (7, )
        else:
            right_q = self.config.default_right_q.copy() # (7, )

        if self.config.enable_left:
            if left_wrist_out_of_view:
                print(f"Left wrist out of view, dist to head: {left_wrist_head_dist:.2f} m")
                left_q = self.config.default_left_q.copy()  # (7, )
            else:
                left_q = left_pink_ik_controller.compute() # (7, )
        else:
            left_q = self.config.default_left_q.copy() # (7, )


        right_hand_cmd = self.process_right_hand_data(right_fingers_mat_fw) # (20， )
        left_hand_cmd = self.process_left_hand_data(left_fingers_mat_fw) # （20， ）

        state_dict = {
            "right_q": np.array(right_q, dtype=np.float64),
            "left_q": np.array(left_q, dtype=np.float64),
            "right_mat_w": np.array(right_wrist_mat_fw, dtype=np.float64),
            "left_mat_w": np.array(left_wrist_mat_fw, dtype=np.float64),
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
