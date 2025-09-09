from dataclasses import field, dataclass
from typing import Optional, Any, Dict
import enum
import multiprocessing as mp
import time
import traceback
import os
import copy
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.tracker_utils import ViveTrackerUpdater

from .base import BaseController, BaseControllerConfig


@dataclass
class TrackerControllerConfig(BaseControllerConfig):
    receive_latency: float = 0.0

    tracker_names: list[str] = field(default_factory=list)
    tracker_serials: list[str] = field(default_factory=list)

    not_ready_return_to_init_dev_wait_time: float = 10.0
    not_ready_sleep_time: float = 0.1

    tracker_check_hist: float = 0.5
    idle_time: float = 30.0
    idle_pos_eps: float = 0.005

    num_devs: int = None

    feedback_sample: Dict[str, Any] = None

    def validate(self):
        super().validate()

        assert len(self.tracker_names) == len(self.tracker_serials), \
            f"Number of tracker names {len(self.tracker_names)} does not match number of serials {len(self.tracker_serials)}"

        assert len(set(self.tracker_names)) == len(self.tracker_names), \
            f"Tracker names must be unique, found duplicates: {self.tracker_names}"
        assert len(set(self.tracker_serials)) == len(self.tracker_serials), \
            f"Tracker serials must be unique, found duplicates: {self.tracker_serials}"

        self.num_devs = len(self.tracker_names)

        self.feedback_sample = {}
        for tracker_name, tracker_serial in zip(self.tracker_names, self.tracker_serials):
            self.feedback_sample[f"{tracker_name}_pose"] = np.zeros((7,), dtype=np.float64)
            self.feedback_sample[f"{tracker_name}_receive_timestamp"] = np.zeros((1,), dtype=np.float64)
            self.feedback_sample[f"{tracker_name}_timestamp"] = np.zeros((1,), dtype=np.float64)


def get_all_tracker_serials():
    """
    Get a list of all available tracker serial numbers connected to the system.

    Returns:
        list[str]: A list of tracker serial numbers
    """
    try:
        # Create a temporary tracker updater to access connected devices
        tracker = ViveTrackerUpdater()

        # Get the tracking devices and their serial numbers
        tracker_devices = tracker.tracking_devices_keys
        tracker_serials = [tracker.vive_tracker_module.devices[device].get_serial()
                           for device in tracker_devices]

        # Clean up the tracker instance
        del tracker

        print(f"Found {len(tracker_serials)} trackers with serials: {tracker_serials}")
        return tracker_serials

    except Exception as e:
        print(f"Error getting tracker serials: {e}")
        return []

class TrackerController(BaseController):
    config: TrackerControllerConfig

    def __init__(self, config: TrackerControllerConfig):
        super().__init__(config)

        self.vive_tracker_updater: Optional[ViveTrackerUpdater] = None

        self._is_idle = self.mp_manager.Value("b", False)
        self._is_ready = self.mp_manager.Value("b", False)

    ################## cls methods ##################
    def _check_is_idle(self):
        if self.cur_is_idle:
            self.cur_idle_steps += 1
        else:
            self.cur_idle_steps = 0

        self._is_idle.value = (self.cur_idle_steps >= self.idle_steps)

    def _check_ready(self):
        if len(self.history_poss) == 0:
            ready = False
        else:
            ready = True
            for tracker_idx in range(self.config.num_devs):

                pos = self.poss[tracker_idx]
                quat = self.quats[tracker_idx]
                hist_pos = self.history_poss[0][tracker_idx]
                hist_quat = self.history_quats[0][tracker_idx]

                # True is ready, False is not ready
                pos_at_zero_check = np.sum(np.array(pos) ** 2) != 0
                pos_change_check = np.sum(np.abs(np.array(hist_pos) - np.array(pos))) != 0

                ready = ready and pos_at_zero_check and pos_change_check

        self._is_ready.value = ready

    def check_ready(self):
        return self._is_ready.value and not self._stop_event.is_set()

    def is_idle(self):
        return self._is_idle.value

    ################## abstract methods ##################
    def _initialize(self):
        self.tracker = ViveTrackerUpdater()

        while len(self.tracker.tracking_devices_keys) != self.config.num_devs:
            print(f"not enough trackers found, {len(self.tracker.tracking_devices_keys)}/{self.config.num_devs}, retrying...")
            del self.tracker
            self.tracker = ViveTrackerUpdater()

        self.tracker_devices: list[str] = self.tracker.tracking_devices_keys
        print(f"Tracker devices: {self.tracker_devices}")
        self.tracker_serials: list[str] = [self.tracker.vive_tracker_module.devices[device].get_serial() for device in self.tracker_devices]

        self.tracker_names: list[str] = [self.config.tracker_names[self.config.tracker_serials.index(serial)] for serial in self.tracker_serials]

        for config_tracker_serial in self.config.tracker_serials:
            assert config_tracker_serial in self.tracker_serials, \
                f"Tracker serial {config_tracker_serial} not found in connected devices: {self.tracker_serials}"

        self.cur_is_idle = False
        self.cur_idle_steps = 0

        # stop collecting data if tracker is idle for self.idle_steps_threshold steps
        self.idle_steps = self.config.idle_time * self.fps
        # move 0.01m in 0.5s will be considered as not idle
        self.idle_pos_eps = self.config.idle_pos_eps
        self.tracker_check_steps = int(self.config.tracker_check_hist * self.fps)

        self.history_poss = deque(maxlen=self.tracker_check_steps)
        self.history_quats = deque(maxlen=self.tracker_check_steps)

        self.sec = None
        self.poss = [[0.0, 0.0, 0.0]] * self.config.num_devs
        self.quats = [[0.0, 0.0, 0.0, 1.0]] * self.config.num_devs

    def _update(self):
        self.tracker.update()
        self.sec = time.time()

        self.history_poss.append(copy.deepcopy(self.poss))
        self.history_quats.append(copy.deepcopy(self.quats))

        this_is_idle = True

        for tracker_idx in range(self.config.num_devs):
            pos = copy.deepcopy(self.tracker.tracking_result[self.tracker_devices[tracker_idx]]["pos"])
            quat = copy.deepcopy(self.tracker.tracking_result[self.tracker_devices[tracker_idx]]["quat"])

            # old one
            hist_pos = self.history_poss[0][tracker_idx]
            hist_quat = self.history_quats[0][tracker_idx]

            hist_dist = np.linalg.norm(np.array(hist_pos) - np.array(pos))
            this_is_idle &= (hist_dist < self.idle_pos_eps)

            self.poss[tracker_idx] = pos
            self.quats[tracker_idx] = quat

        self.cur_is_idle = this_is_idle
        self._check_is_idle()
        self._check_ready()

        state_dict = {}
        for tracker_idx in range(self.config.num_devs):
            pos = self.poss[tracker_idx]
            quat = self.quats[tracker_idx]
            sec = self.sec

            pose = np.array([pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]], dtype=np.float64)

            state_dict[f"{self.tracker_names[tracker_idx]}_pose"] = pose
            state_dict[f"{self.tracker_names[tracker_idx]}_receive_timestamp"] = np.array([sec], dtype=np.float64)
            state_dict[f"{self.tracker_names[tracker_idx]}_timestamp"] = np.array([sec], dtype=np.float64)

        self.feedback_queue.put(state_dict)

    def stop(self):
        super().stop()

    def _process_commands(self):
        pass

    def _close(self):
        del self.tracker
        self.tracker = None

    def reset(self):
        pass
