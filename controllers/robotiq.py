from dataclasses import field, dataclass
from typing import Optional, Any, Dict
import enum
import multiprocessing as mp
import time
import traceback

import numpy as np

from utils.precise_sleep_utils import precise_wait
from utils.pos_rotvec_interp_utils import PoseTrajectoryInterpolator
from utils.shared_memory import Empty
from utils.robotiq_driver_utils import RobotiqDriver

from .base import BaseController, BaseControllerConfig

class RobotiqControllerCommand(enum.Enum):
    SCHEDULE_WAYPOINT = 0

@dataclass
class RobotiqControllerConfig(BaseControllerConfig):
    ip: str = ""
    port: int = 63352
    open_width: float = 0.082  # unit: m
    closed_width: float = 0.0  # unit: m
    calibrate_on_activation: bool = False
    min_position: int = 3
    max_position: int = 220
    receive_latency: float = 0.0
    reset_delay: float = 1.0

    command_sample: Dict[str, Any] = field(
        default_factory=lambda: {
            'cmd': RobotiqControllerCommand.SCHEDULE_WAYPOINT.value,
            'target_width': 0.0,
            'target_time': 0.0
        }
    )
    feedback_sample: Dict[str, Any] = field(
        default_factory=lambda: {
            'gripper_width': np.zeros((1,), dtype=np.float64),
            'gripper_receive_timestamp': np.zeros((1,), dtype=np.float64),
            'gripper_timestamp': np.zeros((1,), dtype=np.float64),
        }
    )

    def validate(self):
        super().validate()
        assert isinstance(self.ip, str) and len(self.ip) > 0, f"Invalid Robotiq IP: {self.ip}"
        assert 0 <= self.port <= 65535, f"Invalid port: {self.port}"
        assert self.open_width > self.closed_width, "Open width must be greater than closed width"
        assert 0 <= self.min_position <= 255, "min_position should be between 0 and 255"
        assert 0 <= self.max_position <= 255, "max_position should be between 0 and 255"
        assert self.min_position < self.max_position, "min_position should be less than max_position"

class RobotiqController(BaseController):
    config: RobotiqControllerConfig

    def __init__(self, config: RobotiqControllerConfig):
        super().__init__(config)
        self.rbtq: Optional[RobotiqDriver] = None

    ################## cls methods ##################
    def move_to_width(self, width: float, target_time: float):
        """Schedule a waypoint for the gripper to move to a specific width at a specific time."""
        self.command_queue.put({
            'cmd': RobotiqControllerCommand.SCHEDULE_WAYPOINT.value,
            'target_width': width,
            'target_time': target_time, # time.monotonic
        })

    def clip_width(self, width: float) -> float:
        """Clip the width to be within the range of closed and open width."""
        return max(min(width, self.config.open_width), self.config.closed_width)

    def width_to_pos(self, width: float) -> int:
        width = self.clip_width(width)

        return int((width - self.config.closed_width) / (self.config.open_width - self.config.closed_width) *
                (self.open_pos - self.closed_pos) + self.closed_pos)

    def pos_to_width(self, pos: int) -> float:
        """Convert gripper position (0-255) to width (m)."""
        return ((pos - self.closed_pos) / (self.open_pos - self.closed_pos) *
                (self.config.open_width - self.config.closed_width) + self.config.closed_width)

    ################## abstract methods ##################
    def _process_commands(self):
        # send command to gripper
        t_now = time.monotonic()
        # target_width = self.width_interp(t_now)[0]
        # self._curr_width = target_width
        target_pos = self.width_to_pos(self._curr_width)

        self.rbtq.move_and_wait_for_recv(position=target_pos, speed=255, force=255)

        # if t_now < self.last_waypoint_time:
        #     # if curr_time is not scheduled, we do not execute the action
        #     # i.e. we only do interpolation not extrpolation
        #     self.rbtq.move_and_wait_for_recv(position=target_pos, speed=255, force=255)

        try:
            commands = self.command_queue.get_all()
            n_cmd = len(commands['cmd'])
        except Empty:
            n_cmd = 0
        except Exception as e:
            traceback.print_exc()
            raise e

        # execute commands
        for i in range(n_cmd):
            cmd = commands['cmd'][i]
            self._curr_width = self.clip_width(commands['target_width'][i])
            target_time = commands['target_time'][i]

            # if cmd == RobotiqControllerCommand.SCHEDULE_WAYPOINT.value:
            #     curr_time = t_now + self.dt
            #     self.width_interp = self.width_interp.schedule_waypoint(
            #         pose=[target_width, 0, 0, 0, 0, 0],
            #         time=target_time,
            #         curr_time=curr_time,
            #         last_waypoint_time=self.last_waypoint_time
            #     )
            #     self.last_waypoint_time = target_time
            # else:
            #     raise ValueError(f"Unknown command: {cmd}")

    def _initialize(self):
        self.rbtq = RobotiqDriver(
            hostname=self.config.ip,
            port=self.config.port,
            calibrate_on_activation=self.config.calibrate_on_activation
        )
        self.rbtq.__enter__()

        # Initialize positions
        if self.config.calibrate_on_activation:
            self.closed_pos = self.rbtq.get_closed_position()
            self.open_pos = self.rbtq.get_open_position()
        else:
            self.closed_pos = self.config.max_position
            self.open_pos = self.config.min_position

        # Move to open position initially
        self.rbtq.move_and_wait_for_pos(self.open_pos, 255, 255)

        # Get initial state
        curr_pos = self.rbtq.get_current_position()
        curr_width = self.pos_to_width(curr_pos)
        curr_t = time.monotonic()

        self.last_waypoint_time = curr_t
        self.width_interp = PoseTrajectoryInterpolator(
            times=[curr_t],
            poses=[[curr_width, 0, 0, 0, 0, 0]]
        )

        self._curr_width = curr_width

    def _update(self):
        # curr_pos = self.rbtq.get_current_position() # very big latency, deprecated
        gripper_receive_timestamp = time.time()
        state_dict = {
            'gripper_width': np.array([self._curr_width,], dtype=np.float64),
            'gripper_receive_timestamp': np.array([gripper_receive_timestamp,], dtype=np.float64),
            'gripper_timestamp': np.array([gripper_receive_timestamp - self.config.receive_latency,], dtype=np.float64),
        }
        self.feedback_queue.put(state_dict)

    def _close(self):
        if self.rbtq is not None:
            self.rbtq.disconnect()
            self.rbtq = None

    def reset(self):
        self.move_to_width(self.config.open_width, time.monotonic() + self.config.reset_delay)
