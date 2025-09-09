from dataclasses import field, dataclass
from typing import Optional, Any, Dict
import threading
import time
import enum
import numpy as np
import traceback

from utils.shared_memory import Empty
from utils.wsg_utils import WSGBinaryDriver

from .base import BaseController, BaseControllerConfig


class WSGControllerCommand(enum.Enum):
    RESET = 0
    MOVE = 1

@dataclass
class WSGControllerConfig(BaseControllerConfig):
    ip: str = None
    port: int = 1000

    # Maximum gripper opening in m
    max_width: float = 0.11

    allowed_max_width: float = 0.082

    # Default speed in mm/s when not specified
    default_speed: float = 1000.0

    command_sample: Dict[str, Any] = field(
        default_factory=lambda: {
            'cmd': 0,
            'target_width': 0.0,
        }
    )
    feedback_sample: Dict[str, Any] = field(
        default_factory=lambda: {
            'width': np.zeros((1,), dtype=np.float64),
            'wsg_receive_timestamp': np.zeros((1,), dtype=np.float64),
            'wsg_timestamp': np.zeros((1,), dtype=np.float64),
        }
    )

    def validate(self):
        super().validate()
        assert self.ip is not None, f"Invalid IP address: {self.ip}"


class WSGController(BaseController):
    config: WSGControllerConfig

    def __init__(self, config: WSGControllerConfig):
        super().__init__(config)

    ################## cls methods ##################

    def move_to_width(self, width: float):
        """Schedule a waypoint for the gripper to move to a specific width at a specific time."""
        width = max(min(width, self.config.max_width), 0.0)
        # Clip the width to the allowed maximum width
        width = min(width, self.config.allowed_max_width)

        self.command_queue.put({
            'cmd': WSGControllerCommand.MOVE.value,
            'target_width': width,
        })

    def reset_to_home(self):
        """Schedule a command to reset the gripper to its home position."""
        self.command_queue.put({
            'cmd': WSGControllerCommand.RESET.value,
            'target_width': 0.0,
        })

    ################## abstract methods ##################
    def _initialize(self):
        # Initialize the WSG binary driver with the configuration
        self.wsg = WSGBinaryDriver(hostname=self.config.ip, port=self.config.port)
        self.wsg.start()
        self.wsg.ack_fault()
        self.wsg.homing()

        self._cur_width = None
        self._tgt_width = None

    def _process_commands(self):
        if self._cur_width is not None and self._tgt_width is not None:
            vel = -self.config.default_speed if self._tgt_width < self._cur_width else self.config.default_speed

            self.wsg.script_position_pd(position=self._tgt_width * 1000, velocity=vel)

        try:
            commands = self.command_queue.get_all()
            n_cmd = len(commands['cmd'])
        except Empty:
            n_cmd = 0
        except Exception as e:
            traceback.print_exc()
            raise e

        for i in range(n_cmd):
            cmd = commands['cmd'][i]
            target_width = commands['target_width'][i]

            if cmd == WSGControllerCommand.MOVE.value:
                # clip to [0, 0.11]
                target_width = max(min(target_width, self.config.max_width), 0.0)
                self._tgt_width = target_width
            elif cmd == WSGControllerCommand.RESET.value:
                self.wsg.ack_fault()
                self.wsg.homing()
                self._tgt_width = None
                self._cur_width = None

    def _update(self):
        # Get current position from the gripper
        msg = self.wsg.script_query()
        timestamp = time.time()
        cur_width = float(msg["position"]) / 1000.0  # Convert to m

        cur_width = max(min(cur_width, self.config.max_width), 0.0)
        # Clip the width to the allowed maximum width
        cur_width = min(cur_width, self.config.allowed_max_width)

        self._cur_width = cur_width

        # Create the feedback state dictionary
        state_dict = {
            'width': np.array([cur_width], dtype=np.float64),
            'wsg_receive_timestamp': np.array([timestamp], dtype=np.float64),
            'wsg_timestamp': np.array([timestamp], dtype=np.float64),
        }

        # Send feedback to the queue
        self.feedback_queue.put(state_dict)

    def _close(self):
        if self.wsg is not None:
            # Close the connection to the gripper
            del self.wsg
            self.wsg = None

    def reset(self):
        self.reset_to_home()
