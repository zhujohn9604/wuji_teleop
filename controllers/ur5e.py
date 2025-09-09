from dataclasses import field, dataclass
from typing import Optional, Any, Dict

import enum
import multiprocessing as mp
import time
import traceback
import copy

import numpy as np
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from utils.precise_sleep_utils import precise_wait
from utils.pos_rotvec_interp_utils import PoseTrajectoryInterpolator
from utils.math_utils import posrot_convert
from utils.shared_memory import Empty
from utils.print_utils import print_cyan

from .base import BaseController, BaseControllerConfig



class UR5EControllerCommand(enum.Enum):
    SCHEDULE_WAYPOINT = 0


@dataclass
class UR5EControllerConfig(BaseControllerConfig):
    ip: str = ""
    lookahead_time: float = 0.1
    gain: int = 300
    max_pos_speed: float = 0.15
    max_rot_speed: float = 0.6
    launch_timeout: int = 3
    tcp_offset_pose: Optional[np.ndarray] = None
    payload_mass: Optional[float] = None
    payload_cog: Optional[np.ndarray] = None  # center of payload
    joints_init: Optional[np.ndarray] = None
    joints_init_speed: float = 1.05
    receive_latency: float = 0.0
    calib_matrix: np.ndarray = np.eye(4)
    reset_delay: float = 3.0

    bbox: Optional[np.ndarray] = None
    reset_pose: Optional[np.ndarray] = None
    reset_pose_rand_range: Optional[np.ndarray] = None

    command_sample: Dict[str, Any] = field(
        default_factory=lambda: {
            'cmd': UR5EControllerCommand.SCHEDULE_WAYPOINT.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0,
            'is_intervened': False,
            "empty_interp_cache": False,
        }
    )

    feedback_sample: Dict[str, Any] = field(
        default_factory=lambda :{
            'ActualTCPPose': np.zeros((6,), dtype=np.float64),
            'ActualTCPSpeed': np.zeros((6,), dtype=np.float64),
            'ActualQ': np.zeros((6,), dtype=np.float64),
            'ActualQd': np.zeros((6,), dtype=np.float64),
            'TargetTCPPose': np.zeros((6,), dtype=np.float64),
            'TargetTCPSpeed': np.zeros((6,), dtype=np.float64),
            'TargetQ': np.zeros((6,), dtype=np.float64),
            'TargetQd': np.zeros((6,), dtype=np.float64),
            'robot_receive_timestamp': np.zeros((1,), dtype=np.float64),
            'robot_timestamp': np.zeros((1,), dtype=np.float64),
        }
    )

    def validate(self):
        super().validate()
        assert isinstance(self.ip, str) and len(self.ip) > 0, f"Invalid UR IP: {self.ip}"
        assert 0.03 <= self.lookahead_time <= 0.2, f"Invalid lookahead time: {self.lookahead_time}"
        assert 100 <= self.gain <= 2000, f"Invalid gain: {self.gain}"
        assert 0 < self.max_pos_speed, f"Invalid max position speed: {self.max_pos_speed}"
        assert 0 < self.max_rot_speed, f"Invalid max rotation speed: {self.max_rot_speed}"
        assert 0 < self.launch_timeout, f"Invalid launch timeout: {self.launch_timeout}"
        if self.tcp_offset_pose is not None:
            assert self.tcp_offset_pose.shape == (6,), f"Invalid TCP offset pose shape: {self.tcp_offset_pose.shape}"
        if self.payload_mass is not None:
            assert 0 <= self.payload_mass <= 5, f"Invalid payload mass: {self.payload_mass}"
        if self.payload_cog is not None:
            assert self.payload_cog.shape == (3,), f"Invalid payload COG shape: {self.payload_cog.shape}"
            assert self.payload_mass is not None, "Payload mass must be specified if payload COG is provided"
        if self.joints_init is not None:
            assert self.joints_init.shape == (6,), f"Invalid joints init shape: {self.joints_init.shape}"
        assert isinstance(self.calib_matrix, np.ndarray) and self.calib_matrix.shape == (4, 4), \
            f"Invalid calibration matrix: {type(self.calib_matrix)}, {self.calib_matrix}"

        if self.bbox is not None:
            assert self.bbox.shape == (2, 6), f"Invalid bbox shape: {self.bbox.shape}"
            assert np.all(self.bbox[0] <= self.bbox[1]), f"Invalid bbox: {self.bbox}"
        if self.reset_pose is not None:
            assert self.reset_pose.shape == (6,), f"Invalid reset pose shape: {self.reset_pose.shape}"
        if self.reset_pose_rand_range is not None:
            assert self.reset_pose_rand_range.shape == (2, 6), f"Invalid reset pos rand range shape: {self.reset_pose_rand_range.shape}"
            assert np.all(self.reset_pose_rand_range[0] <= self.reset_pose_rand_range[1]), f"Invalid reset pos rand range: {self.reset_pose_rand_range}"

class UR5EController(BaseController):
    config: UR5EControllerConfig

    def __init__(self, config: UR5EControllerConfig):
        super().__init__(config)

    ################## cls methods ##################
    def servoL(self, target_posrot):
        # print(f"target rot: {target_posrot[-1]}")
        target_posrot_w = posrot_convert(target_posrot, self.config.calib_matrix, inv=True)

        assert self.rtde_c.servoL(target_posrot_w, 0.5, 0.5, 1 / self.config.fps, self.config.lookahead_time, self.config.gain)

    def _apply_bbox_clip(self, pose: np.ndarray):
        assert isinstance(pose, np.ndarray) and pose.shape == (6,), f"Invalid pose shape: {pose.shape}"
        if self.config.bbox is not None:
            pose = np.clip(pose, self.config.bbox[0], self.config.bbox[1])
        return pose

    def _generate_reset_pose(self):
        assert self.config.reset_pose is not None, "Reset pose is not specified"
        if self.config.reset_pose_rand_range is None:
            return self.config.reset_pose
        reset_pose = self.config.reset_pose + np.random.uniform(
            low=self.config.reset_pose_rand_range[0],
            high=self.config.reset_pose_rand_range[1],
            size=(6,)
        )
        return reset_pose

    def _schedule_waypoint(self, pose, target_time, is_intervened=False, empty_interp_cache=False):
        pose = np.array(pose)
        assert pose.shape == (6,)

        pose = self._apply_bbox_clip(pose)

        message = {
            'cmd': UR5EControllerCommand.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time,
            'is_intervened': is_intervened,
            'empty_interp_cache': empty_interp_cache,
        }
        self.command_queue.put(message)

    ################## abstract methods ##################
    def _process_commands(self):
        # send command to robot
        t_now = time.monotonic()
        if self._is_intervened or t_now < self.last_waypoint_time:
            # if curr_time is not scheduled, we do not execute the action
            # i.e. we only do interpolation not extrpolation
            self.servoL(self.pose_interp(t_now))

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
            target_pose = commands['target_pose'][i]
            target_time = commands['target_time'][i]
            self._is_intervened = commands['is_intervened'][i]
            empty_interp_cache = commands['empty_interp_cache'][i]

            if cmd == UR5EControllerCommand.SCHEDULE_WAYPOINT.value:
                curr_time = t_now + self.dt
                if empty_interp_cache:
                    self.pose_interp = PoseTrajectoryInterpolator(
                        times=[curr_time],
                        poses=[self._last_unifyed_pose]
                    )
                self.pose_interp = self.pose_interp.schedule_waypoint(
                    pose=target_pose,
                    time=target_time,
                    max_pos_speed=self.config.max_pos_speed,
                    max_rot_speed=self.config.max_rot_speed,
                    curr_time=curr_time,
                    last_waypoint_time=self.last_waypoint_time
                )
                self.last_waypoint_time = target_time
            else:
                raise ValueError(f"Unknown command: {cmd}")

    def _initialize(self):
        self.rtde_c = RTDEControlInterface(self.config.ip)
        self.rtde_r = RTDEReceiveInterface(self.config.ip)

        # set parameters
        if self.config.tcp_offset_pose is not None:
            self.rtde_c.setTcp(self.config.tcp_offset_pose)
        if self.config.payload_mass is not None:
            if self.config.payload_cog is not None:
                assert self.rtde_c.setPayload(self.config.payload_mass, self.config.payload_cog)
            else:
                assert self.rtde_c.setPayload(self.config.payload_mass)

        # init pose
        if self.config.joints_init is not None:
            assert self.rtde_c.moveJ(self.config.joints_init, self.config.joints_init_speed, 1.4)

        # main loop
        curr_pose_w = self.rtde_r.getActualTCPPose()   # in arm's base frame
        curr_pose = posrot_convert(curr_pose_w, self.config.calib_matrix)  # in unified frame
        # use monotonic time to make sure the control loop never go backward
        curr_t = time.monotonic()

        self.last_waypoint_time = curr_t
        # calculate interpolation in base frame
        self.pose_interp = PoseTrajectoryInterpolator(
            times=[curr_t],
            poses=[curr_pose]
        )

        self._is_intervened = False
        self._last_unifyed_pose = curr_pose # (6, )

    def _update(self):
        state_dict = {}
        state_dict["ActualTCPPose"] = np.array(self.rtde_r.getActualTCPPose())
        state_dict["ActualTCPSpeed"] = np.array(self.rtde_r.getActualTCPSpeed())
        state_dict["ActualQ"] = np.array(self.rtde_r.getActualQ())
        state_dict["ActualQd"] = np.array(self.rtde_r.getActualQd())
        state_dict["TargetTCPPose"] = np.array(self.rtde_r.getTargetTCPPose())
        state_dict["TargetTCPSpeed"] = np.array(self.rtde_r.getTargetTCPSpeed())
        state_dict["TargetQ"] = np.array(self.rtde_r.getTargetQ())
        state_dict["TargetQd"] = np.array(self.rtde_r.getTargetQd())

        robot_receive_timestamp = time.time()
        state_dict['robot_receive_timestamp'] = np.array([robot_receive_timestamp,], dtype=np.float64)
        state_dict['robot_timestamp'] = np.array([robot_receive_timestamp - self.config.receive_latency,], dtype=np.float64)

        # convert
        state_dict["ActualTCPPose"] = posrot_convert(state_dict["ActualTCPPose"], self.config.calib_matrix)
        state_dict["ActualTCPSpeed"] = posrot_convert(state_dict["ActualTCPSpeed"], self.config.calib_matrix)
        state_dict["TargetTCPPose"] = posrot_convert(state_dict["TargetTCPPose"], self.config.calib_matrix)
        state_dict["TargetTCPSpeed"] = posrot_convert(state_dict["TargetTCPSpeed"], self.config.calib_matrix)

        self._last_unifyed_pose = copy.deepcopy(state_dict["ActualTCPPose"]) # (6, )

        self.feedback_queue.put(state_dict)

    def _close(self):
        self.rtde_c.servoStop()
        self.rtde_c.stopScript()
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()

    def reset(self):
        if self.config.reset_pose is None:
            print_cyan("No reset pose specified, skipping reset")
            return
        reset_pose = self._generate_reset_pose()
        self._schedule_waypoint(reset_pose, time.monotonic() + self.config.reset_delay, is_intervened=False, empty_interp_cache=False)
