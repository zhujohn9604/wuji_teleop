from dataclasses import field, dataclass
from typing import Optional, Any, Dict
import enum
import multiprocessing as mp
import time
import traceback
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.shared_memory import Empty
from .base import BaseController, BaseControllerConfig

from utils.spacemouse import pyspacemouse



@dataclass
class SpaceMouseControllerConfig(BaseControllerConfig):
    name: str = ""
    enable_left: bool = True
    enable_right: bool = True
    dev_num: int = None

    pos_threshold: float = 0.0
    rot_threshold: float = 0.0
    pos_coef: float = 0.3
    rot_coef: float = 1.0

    command_sample = None

    feedback_sample: Dict[str, Any] = field(
        default_factory=lambda: {
            'left_open_button': np.zeros((1,), dtype=bool),
            'left_close_button': np.zeros((1,), dtype=bool),
            'left_pos': np.zeros((3,), dtype=np.float64),
            'left_rot': np.zeros((3,), dtype=np.float64),
            'right_open_button': np.zeros((1,), dtype=bool),
            'right_close_button': np.zeros((1,), dtype=bool),
            'right_pos': np.zeros((3,), dtype=np.float64),
            'right_rot': np.zeros((3,), dtype=np.float64),
            'spacemouse_timestamp': np.zeros((1,), dtype=np.float64),
            'is_still': np.zeros((1,), dtype=bool),
        }
    )

    def validate(self):
        super().validate()
        assert isinstance(self.name, str) and len(self.name) > 0, f"Invalid name: {self.name}"
        assert isinstance(self.enable_left, bool), "enable_left must be boolean"
        assert isinstance(self.enable_right, bool), "enable_right must be boolean"
        assert self.pos_threshold >= 0, "Position threshold must be non-negative"
        assert self.rot_threshold >= 0, "Rotation threshold must be non-negative"
        assert self.pos_coef > 0, "Position coefficient must be positive"
        assert self.rot_coef > 0, "Rotation coefficient must be positive"

        assert isinstance(self.dev_num, int), f"dev_num={self.dev_num} is not int"

        self.dev_num = int(self.enable_left) + int(self.enable_right)

def clip_threshold(x: np.array, threshold):
    gt_cond = x > threshold
    lt_cond = x < -threshold
    in_cond = np.logical_and(x > -threshold, x < threshold)
    x = np.where(gt_cond, x - threshold, x)
    x = np.where(lt_cond, x + threshold, x)
    x = np.where(in_cond, 0, x)

    return x

def deal_space_state(space_state, spacemouse_pos_threshold, spacemouse_rot_threshold, spacemouse_pos_coef, spacemouse_rot_coef, left_right: str):
    assert left_right in ["left", "right"], f"left_right={left_right} is not in ['left', 'right']"

    open_button = bool(space_state.buttons[-1])
    close_button = bool(space_state.buttons[0])

    if left_right == "left":
        pos = np.array([space_state.y, -space_state.x, space_state.z])
        rot = np.array([space_state.roll, space_state.pitch, -space_state.yaw])
    elif left_right == "right":
        pos = np.array([-space_state.y, space_state.x, space_state.z])
        rot = np.array([-space_state.roll, -space_state.pitch, -space_state.yaw])

    pos = clip_threshold(pos, spacemouse_pos_threshold)
    rot = clip_threshold(rot, spacemouse_rot_threshold)

    pos *= spacemouse_pos_coef
    rot *= spacemouse_rot_coef

    rot = R.from_euler("xyz", rot)

    return open_button, close_button, pos, rot

class SpaceMouseController(BaseController):
    config: SpaceMouseControllerConfig

    def __init__(self, config: SpaceMouseControllerConfig):
        super().__init__(config)

    ################## cls methods ##################

    ################## abstract methods ##################
    def _process_commands(self):
        pass

    def _initialize(self):
        assert pyspacemouse.open(DeviceNumber=self.config.dev_num)

    def _update(self):
        timestamp = time.time()
        state_dict = {
            'spacemouse_timestamp': np.array([timestamp], dtype=np.float64)
        }

        space_states = pyspacemouse.read_all()
        assert len(space_states) == self.config.dev_num, f"len(space_states)={len(space_states)} != {self.config.dev_num}"

        fetch_idx = 0

        is_still = True

        if self.config.enable_left:
            open_button, close_button, pos, rot = deal_space_state(space_states[fetch_idx], self.config.pos_threshold, self.config.rot_threshold, self.config.pos_coef, self.config.rot_coef, "left")

            state_dict["left_open_button"] = np.array([open_button], dtype=bool)
            state_dict["left_close_button"] = np.array([close_button], dtype=bool)
            state_dict["left_pos"] = pos.astype(np.float64)
            state_dict["left_rot"] = rot.as_euler('xyz').astype(np.float64)

            pos_norm = np.linalg.norm(pos)
            rot_norm = np.linalg.norm(rot.as_euler('xyz'))

            if pos_norm > 0.01 or rot_norm > 0.01:
                is_still = False
            if open_button or close_button:
                is_still = False

            fetch_idx += 1

        if self.config.enable_right:
            open_button, close_button, pos, rot = deal_space_state(space_states[fetch_idx], self.config.pos_threshold, self.config.rot_threshold, self.config.pos_coef, self.config.rot_coef, "right")

            state_dict["right_open_button"] = np.array([open_button], dtype=bool)
            state_dict["right_close_button"] = np.array([close_button], dtype=bool)
            state_dict["right_pos"] = pos.astype(np.float64)
            state_dict["right_rot"] = rot.as_euler('xyz').astype(np.float64)

            pos_norm = np.linalg.norm(pos)
            rot_norm = np.linalg.norm(rot.as_euler('xyz'))

            if pos_norm > 0.01 or rot_norm > 0.01:
                is_still = False
            if open_button or close_button:
                is_still = False

            fetch_idx += 1

        state_dict["is_still"] = np.array([is_still], dtype=bool)

        self.feedback_queue.put(state_dict)

    def _close(self):
        pass

    def reset(self):
        # SpaceMouse doesn't need a reset operation
        pass
