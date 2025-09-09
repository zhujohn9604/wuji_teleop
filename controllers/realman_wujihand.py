from dataclasses import field, dataclass
from typing import Optional, Any, Dict, Tuple

import enum
import multiprocessing as mp
import time
import traceback
import copy
import math

import numpy as np

from utils.precise_sleep_utils import precise_wait
from utils.pos_euler_interp_utils import PosEulerTrajectoryInterpolator
from utils.anything_interp_utils import AnythingTrajectoryInterpolator, zip_x, zip_xs, unzip_x
from utils.math_utils import pose_euler_convert
from utils.shared_memory import Empty
from utils.print_utils import print_cyan

from scipy.spatial.transform import Rotation as R

from .base import BaseController, BaseControllerConfig

from Robotic_Arm.rm_robot_interface import *
from controllers.wuji_hand_pdo import WujiHand

class RealmanControllerCommand(enum.Enum):
    SCHEDULE_WAYPOINT = 0
    SCHEDULE_JOINT = 1
    SCHEDULE_HAND_POS = 2
    SCHEDULE_HAND_ANGLE = 3

@dataclass
class RealmanWujiControllerConfig(BaseControllerConfig):
    """
    all pose means in pos + euler format, i.e. [x, y, z, rx, ry, rz] (radians)

    attention: quat in realman is [qw, qx, qy, qz]
    """

    # arm related
    ip: str = ""
    port: int = 8080

    callback_port: int = 8088

    # max_line_speed: float = 0.25
    # max_line_acc: float = 1.6
    # max_angular_speed: float = 0.6
    # max_angular_acc: float = 4.0

    # max_line_speed: float = 1.0
    # max_line_acc: float = 3.0
    # max_angular_speed: float = 2.0
    # max_angular_acc: float = 8.0

    max_line_speed: float = 10.0
    max_line_acc: float = 100.0
    max_angular_speed: float = 20.0
    max_angular_acc: float = 200.0

    joint_drive_max_speed: float = 180 # degrees/s
    joint_drive_max_acc: float = 600 # degrees/s^2

    # collision_stage: int = 6

    tcp_offset_pose: np.ndarray = field(default_factory=lambda: np.zeros((6,), dtype=np.float64))  # (x, y, z, rx, ry, rz), euler
    payload_mass: float = 1.0

    pose_init: Optional[np.ndarray] = None  # (6, ), in unified frame, i.e. in the same frame as the robot's base frame
    joints_init: Optional[np.ndarray] = None # (dof, ), in radians

    hand_angle_init: np.ndarray = field(default_factory=lambda: np.zeros((6,), dtype=np.float32))  # (hand_dof, ), initial hand joint angles, in [-32768, 32767]
    # 0: thumb_2, 	2.26° ~ 36.76°
    # 1: index_1, 100.22°~178.37°
    # 2: middle_1, 97.81° ~ 176.06°
    # 3: ring_1, 101.38°~176.54°
    # 4: pinky_1, 98.84°~174.86°
    # 5: thumb_1, 	0° ~ 90°

    # hand_speed: int = 500
    # hand_force: int = 100

    hand_speed: int = 1000
    hand_force: int = 50

    receive_latency: float = 0.0
    calib_matrix: np.ndarray = np.eye(4)
    reset_delay: float = 3.0

    dof: int = 7

    connection_level: int = 3

    # Thread mode (0: single, 1: dual, 2: triple). Defaults to 2.
    thread_mode: int = 2

    # hand related
    hand_type: str = "rohand" # "rohand", "inspire", "wuji"

    hand_pos_range: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "rohand": (0, 65535),  # rohand has 16-bit position range
        "inspire": (0, 1000)   # inspire has 10-bit position range
    })
    hand_angle_range: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "rohand": (-32768, 32767),  # rohand has angle range in degrees
        "inspire": (0, 2000),   # inspire has angle range in degrees
    })
    hand_dof: Dict[str, int] = field(default_factory=lambda: {
        "rohand": 6,  # rohand has 6 DOF
        "inspire": 6,  # inspire has 6 DOF
        "wuji": 20,
    })

    cmd_c_len = 30

    command_sample: Dict[str, Any] = None
    feedback_sample: Dict[str, Any] = None

    def validate(self):
        super().validate()
        assert isinstance(self.ip, str) and len(self.ip) > 0, f"Invalid UR IP: {self.ip}"

        assert int(self.pose_init is not None) + int(self.joints_init is not None) <= 1, \
            f"Only one of pose_init or joints_init can be specified, got pose_init: {self.pose_init}, joints_init: {self.joints_init}"

        #self.hand_pos_range = self.hand_pos_range[self.hand_type]
        #self.hand_angle_range = self.hand_angle_range[self.hand_type]
        self.hand_dof = self.hand_dof[self.hand_type]

        self.command_sample = {
            'cmd': RealmanControllerCommand.SCHEDULE_WAYPOINT.value,
            'cmd_c': np.zeros((self.cmd_c_len,), dtype=np.float64), # pos + euler
            'target_time': 0.0,
            'is_intervened': False,
            "empty_interp_cache": False,
        }

        self.feedback_sample = {
            'cur_q': np.zeros((self.dof,), dtype=np.float64),  # current joint positions
            'cur_hand_angle': np.zeros((self.hand_dof,), dtype=np.float64),  # current hand positions
            'cur_pose_b': np.zeros((6,), dtype=np.float64), # pos + euler
            'receive_timestamp': np.zeros((1,), dtype=np.float64),
            'timestamp': np.zeros((1,), dtype=np.float64),
        }


class RealmanWujiController(BaseController):
    config: RealmanWujiControllerConfig

    def __init__(self, config: RealmanWujiControllerConfig):
        super().__init__(config)


    ################## cls methods ##################
    def set_hand_follow_pos(self, target_hand_pos: np.ndarray):
        """
        Args:
            target_hand_pos: np.ndarray, shape (hand_dof, ), target hand angles
                ro_hand has hand_dof=6 joints
        """

        self.wuji_hand.set_joint_positions_pdo(target_hand_pos.reshape(5, 4))
        time.sleep(0.07)


    def set_hand_follow_angle(self, target_hand_pos: np.ndarray):
        """
        Args:
            target_hand_pos: np.ndarray, shape (hand_dof, ), target hand angles
                ro_hand has hand_dof=6 joints
        """

        target_hand_pos = target_hand_pos.astype(np.int64)

        self.wuji_hand.set_joint_positions_pdo(target_hand_pos.tolist())
        #self.wuji_hand.set_joint_positions_async(target_hand_pos.tolist())

    def rm_movep_follow(self, target_pose_b: np.ndarray):
        """
        Args:
            target_pose_b: np.ndarray, shape (6, ), [x, y, z, rx, ry, rz], in radians
                in unified frame
        """
        target_pose_w = pose_euler_convert(target_pose_b, self.config.calib_matrix, inv=True)

        # realman use rad for euler rot
        assert self.robot.rm_movep_follow(target_pose_w.tolist()) == 0, f"Failed to move to target pose: {target_pose_w}"

    def rm_movej_follow(self, target_joints: np.ndarray):
        """
        Args:
            target_joints: np.ndarray, shape (dof,), target joint positions
        """
        assert target_joints.shape == (self.config.dof,), f"Invalid joints shape: {target_joints.shape}, expected ({self.config.dof},)"

        target_joints = target_joints * 180.0 / math.pi  # convert to degrees

        assert self.robot.rm_movej_follow(target_joints) == 0, f"Failed to move to target joints: {target_joints}"

    def rm_movej_canfd(self, target_joints: np.ndarray):
        """
        Args:
            target_joints: np.ndarray, shape (dof,), target joint positions
        """
        assert target_joints.shape == (self.config.dof,), f"Invalid joints shape: {target_joints.shape}, expected ({self.config.dof},)"

        target_joints = target_joints * 180.0 / math.pi  # convert to degrees

        assert self.robot.rm_movej_canfd(target_joints, follow=True) == 0, f"Failed to move to target joints: {target_joints}"
        # assert self.robot.rm_movej_canfd(target_joints, follow=True, expand=1, radio=200) == 0, f"Failed to move to target joints: {target_joints}"

    def schedule_waypoint(self, pose_b: np.ndarray, target_time, is_intervened=False, empty_interp_cache=False):
        """
        Args:
            pose: np.ndarray, shape (6,), [x, y, z, rx, ry, rz]
             in unified frame
        """
        pose_b = np.array(pose_b)
        assert pose_b.shape == (6, ), f"Invalid pose shape: {pose_b.shape}, expected (6,)"

        cmd_c = np.zeros((self.config.cmd_c_len,), dtype=np.float64)
        cmd_c[:6] = pose_b[:6]  # copy position and euler angles

        message = {
            'cmd': RealmanControllerCommand.SCHEDULE_WAYPOINT.value,
            'cmd_c': cmd_c,
            'target_time': target_time,
            'is_intervened': is_intervened,
            'empty_interp_cache': empty_interp_cache,
        }
        self.command_queue.put(message)

    def schedule_joint(self, target_joints: np.ndarray, target_time, is_intervened=False, empty_interp_cache=False):
        """
        Args:
            target_joints: np.ndarray, shape (dof,), target joint positions, in radians
            target_time: float, target time to reach the joint positions
            is_intervened: bool, whether the command is an intervention
            empty_interp_cache: bool, whether to empty the interpolation cache
        """
        target_joints = np.array(target_joints)
        assert target_joints.shape == (self.config.dof,), f"Invalid joints shape: {target_joints.shape}, expected ({self.config.dof},)"

        cmd_c = np.zeros((self.config.cmd_c_len,), dtype=np.float64)
        cmd_c[:self.config.dof] = target_joints[:self.config.dof]  # copy joint positions

        message = {
            'cmd': RealmanControllerCommand.SCHEDULE_JOINT.value,
            'cmd_c': cmd_c,
            'target_time': target_time,
            'is_intervened': is_intervened,
            'empty_interp_cache': empty_interp_cache,
        }
        self.command_queue.put(message)

    def schedule_hand_pos(self, target_hand_pos: np.ndarray, target_time, is_intervened=False, empty_interp_cache=False):
        """
        Args:
            target_hand_pos: np.ndarray, shape (hand_dof,), target hand joint positions
            target_time: float, target time to reach the hand joint positions
            is_intervened: bool, whether the command is an intervention
            empty_interp_cache: bool, whether to empty the interpolation cache
        """
        target_hand_pos = np.array(target_hand_pos)
        assert target_hand_pos.shape == (self.config.hand_dof,), f"Invalid hand joints shape: {target_hand_pos.shape}, expected ({self.config.hand_dof},)"

        cmd_c = np.zeros((self.config.cmd_c_len,), dtype=np.float64)
        cmd_c[:self.config.hand_dof] = target_hand_pos[:self.config.hand_dof]

        message = {
            'cmd': RealmanControllerCommand.SCHEDULE_HAND_POS.value,
            'cmd_c': cmd_c,
            'target_time': target_time,
            'is_intervened': is_intervened,
            'empty_interp_cache': empty_interp_cache,
        }
        self.command_queue.put(message)

    def schedule_hand_angle(self, target_hand_angle: np.ndarray, target_time, is_intervened=False, empty_interp_cache=False):
        """
        Args:
            target_hand_angle: np.ndarray, shape (hand_dof,), target hand joint angles, in [-32768, 32767] for rohand
            target_time: float, target time to reach the hand joint angles
            is_intervened: bool, whether the command is an intervention
            empty_interp_cache: bool, whether to empty the interpolation cache
        """
        target_hand_angle = np.array(target_hand_angle)
        assert target_hand_angle.shape == (self.config.hand_dof,), f"Invalid hand angles shape: {target_hand_angle.shape}, expected ({self.config.hand_dof},)"

        cmd_c = np.zeros((self.config.cmd_c_len,), dtype=np.float64)
        cmd_c[:self.config.hand_dof] = target_hand_angle[:self.config.hand_dof]

        message = {
            'cmd': RealmanControllerCommand.SCHEDULE_HAND_ANGLE.value,
            'cmd_c': cmd_c,
            'target_time': target_time,
            'is_intervened': is_intervened,
            'empty_interp_cache': empty_interp_cache,
        }
        self.command_queue.put(message)

    def arm_state_callback(self, data):
        """
        Callback function for arm state.

        Args:
            data: The data containing the arm state information.
        """
        self.cur_q[:] = np.array([data.joint_status.joint_position[i] for i in range(self.config.dof)]) # (dof, )
        # convert from angles to radians
        self.cur_q = self.cur_q * math.pi / 180.0  # (dof, )

        cur_pos = np.array([data.waypoint.position.x, data.waypoint.position.y, data.waypoint.position.z])  # (3, )
        cur_euler = np.array([data.waypoint.euler.rx, data.waypoint.euler.ry, data.waypoint.euler.rz])  # (3, )
        self.cur_pose_w[:] = np.concatenate((cur_pos, cur_euler), axis=0)  # (6, ), in unified frame

        #self.cur_hand_angle[:] = np.array([data.handState.hand_angle[i] for i in range(self.config.hand_dof)])  # (hand_dof, )
        #self.cur_hand_pos[:] = np.array([data.handState.hand_pos[i] for i in range(self.config.hand_dof)])  # (hand_dof, )

    def get_cur_pose_b(self) -> np.ndarray:
        return pose_euler_convert(copy.deepcopy(self.cur_pose_w), self.config.calib_matrix)  # (6, ), in unified frame

    ################## abstract methods ##################
    def _initialize(self):
        self.thread_mode = rm_thread_mode_e(self.config.thread_mode)
        self.robot = RoboticArm(self.thread_mode)
        self.wuji_hand = WujiHand(tpdo_id=1, interval=1000)
        self.begin_time_ = time.monotonic_ns()
        self.handle = self.robot.rm_create_robot_arm(self.config.ip, self.config.port, self.config.connection_level)

        realtime_push_config = rm_realtime_push_config_t(
            cycle=1,
            enable=True,
            port=self.config.callback_port,
            # force_coordinate=0,
            ip=self.config.ip,
            custom_config=rm_udp_custom_config_t(
                hand_state=1,
                arm_current_status=1,
            ),
        )
        self.robot.rm_set_realtime_push(realtime_push_config)

        self.robot.rm_set_arm_max_line_speed(self.config.max_line_speed)
        self.robot.rm_set_arm_max_line_acc(self.config.max_line_acc)
        self.robot.rm_set_arm_max_angular_speed(self.config.max_angular_speed)
        self.robot.rm_set_arm_max_angular_acc(self.config.max_angular_acc)

        # for i in range(1, self.config.dof + 1):
        #     assert self.robot.rm_set_joint_clear_err(i) == 0, f"Failed to clear joint {i} error"
        #     assert self.robot.rm_set_joint_en_state(i, 1) == 0, f"Failed to enable joint {i}"
        #     # assert self.robot.rm_set_joint_drive_max_acc(i, self.config.joint_drive_max_acc) == 0, f"Failed to set joint {i} max acceleration to {self.config.joint_drive_max_acc} degrees/s^2"
        #     # assert self.robot.rm_set_joint_drive_max_speed(i, self.config.joint_drive_max_speed) == 0, f"Failed to set joint {i} max speed to {self.config.joint_drive_max_speed} degrees/s"

        self.robot.rm_set_hand_speed(self.config.hand_speed)
        self.robot.rm_set_hand_force(self.config.hand_force)

        # self.robot.rm_set_manual_tool_frame(
        #     rm_frame_t(
        #         frame_name="test",
        #         pose=[0, 0, 0, self.config.tcp_offset_pose[3], self.config.tcp_offset_pose[4], self.config.tcp_offset_pose[5]],
        #         payload=self.config.payload_mass,
        #         x=0, y=0, z=0
        #     )
        # )

        self.cur_q = np.zeros((self.config.dof,), dtype=np.float32)  # (dof, )
        self.cur_pose_w = np.zeros((6,), dtype=np.float32)  # (x, y, z, rx, ry, rz), in unified frame
        self.cur_hand_pos = np.zeros((self.config.hand_dof,), dtype=np.float32)  # (hand_dof, )
        self.cur_hand_angle = np.zeros((self.config.hand_dof,), dtype=np.float32)  # (hand_dof, )

        #assert self.handle.id != -1, f"Failed to create robot arm handle with IP {self.config.ip} and port {self.config.port}"

        self.arm_state = rm_realtime_arm_state_callback_ptr(self.arm_state_callback)
        self.robot.rm_realtime_arm_state_call_back(self.arm_state)

        # init joints
        if self.config.joints_init is not None:
            print(f"joints_init: {self.config.joints_init}")
            joints_init_degrees = self.config.joints_init * 180.0 / math.pi  # convert to degrees
            assert self.robot.rm_movej(joints_init_degrees, 20, 0, rm_trajectory_connect_config_e.RM_TRAJECTORY_DISCONNECT_E, RM_MOVE_MULTI_BLOCK) == 0, f"Failed to move to initial joints: {self.config.joints_init}"

        # init pose
        if self.config.pose_init is not None:
            print(f"pose_init: {self.config.pose_init}")
            assert self.robot.rm_movej_p(self.config.pose_init, 40, 0, rm_trajectory_connect_config_e.RM_TRAJECTORY_DISCONNECT_E, RM_MOVE_MULTI_BLOCK) == 0, f"Failed to move to initial pose: {self.config.pose_init}"

        # init hand joint
        if self.config.hand_angle_init is not None:
            print(f"hand_angle_init: {self.config.hand_angle_init}")
            #assert self.robot.rm_set_hand_follow_angle(self.config.hand_angle_init.astype(np.int64), 1) == 0, f"Failed to set initial hand joints: {self.config.hand_angle_init}"
            pass

        while np.linalg.norm(self.cur_q) == 0.0:
            print(f"cur_q: {self.cur_q}, waiting for arm state to be updated...")
            time.sleep(0.5)

        # while np.linalg.norm(self.cur_hand_angle) == 0.0:
        #     print(f"cur_hand_angle: {self.cur_hand_angle}, waiting for hand state to be updated...")
        #     time.sleep(0.5)

        # while np.linalg.norm(self.cur_hand_pos) == 0.0:
        #     print(f"cur_hand_pos: {self.cur_hand_pos}, waiting for hand state to be updated...")
        #     time.sleep(0.5)

        self._is_intervened = False

        self.to_save_qs = []

    def _process_commands(self):
        t_now = time.monotonic()

        ###
        # arm control
        ###
        if hasattr(self, 'target_joints'):
            self.rm_movej_follow(self.target_joints)
        elif hasattr(self, 'joint_interp'):
            target_joints = unzip_x(self.joint_interp(t_now), self.config.dof)
            self.rm_movej_follow(target_joints)

            # self.rm_movej_canfd(target_joints)

        if hasattr(self, 'target_pose_b'):
            self.rm_movep_follow(self.target_pose_b)
        elif hasattr(self, 'pos_euler_interp'):
            target_pose_b = self.pos_euler_interp(t_now)
            self.rm_movep_follow(target_pose_b)

        ###
        # hand control
        ###
        if hasattr(self, 'target_hand_angle'):
            self.set_hand_follow_angle(self.target_hand_angle)
        elif hasattr(self, 'hand_angle_interp'):
            target_hand_angle = unzip_x(self.hand_angle_interp(t_now), self.config.hand_dof)

            self.rm_set_hand_follow_angle(target_hand_angle)

        if hasattr(self, 'target_hand_pos'):
            self.set_hand_follow_pos(self.target_hand_pos)
        elif hasattr(self, 'hand_pos_interp'):
            target_hand_pos = unzip_x(self.hand_pos_interp(t_now), self.config.hand_dof)

            self.rm_set_hand_follow_pos(target_hand_pos)

        ###
        # recv cmds
        ###
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
            cmd_c = commands['cmd_c'][i]
            target_time = commands['target_time'][i]
            self._is_intervened = commands['is_intervened'][i]
            empty_interp_cache = commands['empty_interp_cache'][i]

            if cmd == RealmanControllerCommand.SCHEDULE_WAYPOINT.value:
                target_pose_b = cmd_c[:6]  # (6,), [x, y, z, rx, ry, rz], in unified frame

                #1: add interpolation
                curr_time = t_now + self.dt
                if empty_interp_cache or not hasattr(self, 'pos_euler_interp'):
                    self.pos_euler_interp = PosEulerTrajectoryInterpolator(
                        times=[curr_time],
                        poses=[self.get_cur_pose_b()]
                    )
                    if not hasattr(self, 'last_pose_time'):
                        self.last_pose_time = curr_time
                self.pos_euler_interp = self.pos_euler_interp.schedule_waypoint(
                    pose=target_pose_b,
                    time=target_time,
                    curr_time=curr_time,
                    last_waypoint_time=self.last_pose_time
                )
                self.last_pose_time = target_time

                #2: control directly
                # self.target_pose_b = target_pose_b
            elif cmd == RealmanControllerCommand.SCHEDULE_JOINT.value:
                target_joints = cmd_c[:self.config.dof]  # (dof,)

                #1: add interpolation
                curr_time = t_now + self.dt
                if empty_interp_cache or not hasattr(self, 'joint_interp'):
                    self.joint_interp = AnythingTrajectoryInterpolator(
                        times=[curr_time],
                        xs=zip_xs([self.cur_q], AnythingTrajectoryInterpolator.x_max_len)
                    )
                    if not hasattr(self, 'last_joint_time'):
                        self.last_joint_time = curr_time
                self.joint_interp = self.joint_interp.schedule_waypoint(
                    x=zip_x(target_joints, AnythingTrajectoryInterpolator.x_max_len),
                    time=target_time,
                    curr_time=curr_time,
                    last_waypoint_time=self.last_joint_time
                )
                self.last_joint_time = target_time

                #2: control directly
                # self.target_joints = target_joints

            elif cmd == RealmanControllerCommand.SCHEDULE_HAND_POS.value: # deprecated
                target_hand_pos = cmd_c[:self.config.hand_dof]  # (hand_dof,)

                self.target_hand_pos = target_hand_pos
            elif cmd == RealmanControllerCommand.SCHEDULE_HAND_ANGLE.value:
                target_hand_angle = cmd_c[:self.config.hand_dof]  # (hand_dof,)

                hand_angle_option = 2
                # TODO: use interpolation will result in fingers still

                #1: add interpolation
                if hand_angle_option == 1:
                    curr_time = t_now + self.dt
                    if empty_interp_cache or not hasattr(self, 'hand_angle_interp'):
                        self.hand_angle_interp = AnythingTrajectoryInterpolator(
                            times=[curr_time],
                            xs=zip_xs([self.cur_hand_angle], AnythingTrajectoryInterpolator.x_max_len)
                        )
                        if not hasattr(self, 'last_hand_angle_time'):
                            self.last_hand_angle_time = curr_time
                    self.hand_angle_interp = self.hand_angle_interp.schedule_waypoint(
                        x=zip_x(target_hand_angle, AnythingTrajectoryInterpolator.x_max_len),
                        time=target_time,
                        curr_time=curr_time,
                        last_waypoint_time=self.last_hand_angle_time
                    )
                    self.last_hand_angle_time = target_time

                #2: control directly
                elif hand_angle_option == 2:
                    self.target_hand_angle = target_hand_angle
            else:
                raise ValueError(f"Unknown command: {cmd}")

    def _update(self):
        state_dict = {}

        receive_timestamp = time.time()
        state_dict['receive_timestamp'] = np.array([receive_timestamp,], dtype=np.float64)
        state_dict['timestamp'] = np.array([receive_timestamp - self.config.receive_latency,], dtype=np.float64)

        state_dict["cur_pose_b"] = self.get_cur_pose_b()  # (6, ), in unified frame
        state_dict["cur_q"] = copy.deepcopy(self.cur_q).astype(np.float64)  # (dof, ), in radians
        # state_dict["cur_hand_pos"] = copy.deepcopy(self.cur_hand_pos).astype(np.int32)  # (hand_dof, )
        state_dict["cur_hand_angle"] = copy.deepcopy(self.cur_hand_angle).astype(np.float64)  # (hand_dof, )

        self.feedback_queue.put(state_dict)

    def _close(self):
        pass
    #    handle = self.robot.rm_delete_robot_arm()
    #    if handle == 0:
    #        print(f"Successfully disconnected from the robot arm with IP {self.config.ip} and port {self.config.port}, handle: {handle}")
    #    else:
    #        print(f"Failed to disconnect from the robot arm with IP {self.config.ip} and port {self.config.port}, handle: {handle}")

    def reset(self):
        if hasattr(self, 'joint_interp'):
            del self.joint_interp

        if self.config.joints_init is not None:
            print(f"Resetting to initial joints: {self.config.joints_init}")
            self.schedule_joint(self.config.joints_init, time.monotonic() + self.config.reset_delay, is_intervened=False, empty_interp_cache=False)

        if hasattr(self, 'joint_interp'):
            del self.joint_interp

