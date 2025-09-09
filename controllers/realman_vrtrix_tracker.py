from dataclasses import field, dataclass
from typing import Optional, Any, Dict
import multiprocessing as mp
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy
import socket
import struct
import traceback
import threading
import json
import pinocchio as pin

from pink.tasks import FrameTask

from .base import BaseController, BaseControllerConfig
from utils.vrtrix_utils import vrtrix_to_realman_6dof
from utils.pink_ik_utils import PinkIKController
from utils.retargeting_utils import (
    pose_to_mat,
    mat_to_pose,
    calc_delta_mat,
    calc_end_mat,
)

# VRTrix command constants
CMD_STOP = 0
CMD_TriggerHaptics = 1
CMD_ToggleHaptics = 2
CMD_HardwareCalibration = 3
CMD_TPoseCalibration = 4
CMD_SetAdvancedMode = 5
CMD_SetHardwareVersion = 6
CMD_SetRadioLimit = 7
CMD_ChannelHopping = 8
CMD_AlgorithmTuning = 9
CMD_SetRotationFormat = 10
CMD_OKPoseCalibration = 11


@dataclass
class RealmanVRTrixTrackerControllerConfig(BaseControllerConfig):
    # VRTrix glove settings
    vrtrix_ip: str = "127.0.0.1"
    vrtrix_port: int = 11002

    # Vive Tracker settings
    tracker_ip: str = "192.168.2.68"
    tracker_port: int = 12345

    # Robot base positions in world frame
    robot_r_mat_w: np.ndarray = None
    robot_l_mat_w: np.ndarray = None

    # Pink IK parameters
    pink_ik_damp: float = 0.5
    pink_ik_gain: float = 1.0

    # Default robot joint positions
    default_right_q: np.ndarray = field(default_factory=lambda: np.zeros((7,), dtype=np.float64))
    default_left_q: np.ndarray = field(default_factory=lambda: np.zeros((7,), dtype=np.float64))

    # Hand control parameters
    angle_min: float = 0.0  # degrees
    angle_max: float = 90.0  # degrees

    # Enable flags
    enable_right: bool = True
    enable_left: bool = True

    # Mapping parameters
    proximal_weight: float = 0.7  # Weight for proximal joint
    intermediate_weight: float = 0.3  # Weight for intermediate joint

    # Debug mode
    debug_mode: bool = False
    debug_interval: int = 30  # Print debug info every N frames

    # VRTrix connection settings
    auto_calibrate_on_start: bool = True
    recv_buffer_size: int = 273  # Fixed packet size
    connection_timeout: float = 5.0
    max_packets_per_update: int = 3
    use_threading: bool = True

    # Tracker settings
    tracker_timeout: float = 0.1  # Maximum age of tracker data to use
    tracker_buffer_size: int = 4096

    # Tracker mounting calibration
    # These matrices transform from tracker frame to expected hand frame
    tracker_to_hand_right: np.ndarray = None
    tracker_to_hand_left: np.ndarray = None

    def validate(self):
        super().validate()

        assert isinstance(self.robot_l_mat_w, np.ndarray) and self.robot_l_mat_w.shape == (4, 4), \
            f"robot_l_mat_w should be a 4x4 numpy array"
        assert isinstance(self.robot_r_mat_w, np.ndarray) and self.robot_r_mat_w.shape == (4, 4), \
            f"robot_r_mat_w should be a 4x4 numpy array"

        # Set default tracker to hand transformations if not provided
        if self.tracker_to_hand_right is None:
            # Default transformation for right hand
            # This assumes tracker is mounted on back of hand with:
            # - Tracker Z pointing up from hand back
            # - Tracker X pointing towards fingers
            # - Tracker Y pointing left (when looking at back of right hand)
            # Transform to hand frame where:
            # - X: forward (towards fingers)
            # - Y: left
            # - Z: up
            self.tracker_to_hand_right = np.array([
                [1, 0, 0, 0],   # Tracker X -> Hand X (forward)
                [0, 0, -1, 0],  # Tracker Z -> Hand -Y (right)
                [0, 1, 0, 0],   # Tracker Y -> Hand Z (up)
                [0, 0, 0, 1]
            ])

        if self.tracker_to_hand_left is None:
            # Default transformation for left hand
            # Mirror of right hand
            self.tracker_to_hand_left = np.array([
                [1, 0, 0, 0],   # Tracker X -> Hand X (forward)
                [0, 0, 1, 0],   # Tracker Z -> Hand Y (left)
                [0, 1, 0, 0],   # Tracker Y -> Hand Z (up)
                [0, 0, 0, 1]
            ])

        self.feedback_sample = {
            "right_q": np.zeros((7,), dtype=np.float64),
            "left_q": np.zeros((7,), dtype=np.float64),
            "right_mat_w": np.eye(4, dtype=np.float64),
            "left_mat_w": np.eye(4, dtype=np.float64),
            "right_hand_cmd": np.zeros((6,), dtype=np.float64),
            "left_hand_cmd": np.zeros((6,), dtype=np.float64),
            "receive_timestamp": np.zeros((1,), dtype=np.float64),
            "timestamp": np.zeros((1,), dtype=np.float64),
        }


def init_pink_ik_controller(dt: float, damping: float, gain: float):
    """Initialize Pink IK controllers for both arms"""
    urdf_path = "/home/wuji/code/dex-real-deployment/urdf/rm75b/rm_75_b_description.urdf"
    mesh_path = "/home/wuji/code/dex-real-deployment/urdf/rm75b"

    left_pink_ik_controller = PinkIKController(
        urdf_path=urdf_path,
        mesh_path=mesh_path,
        variable_input_tasks=[
            FrameTask(
                "Link7",
                position_cost=1.0,
                orientation_cost=1.0,
                lm_damping=damping,
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
                position_cost=1.0,
                orientation_cost=1.0,
                lm_damping=damping,
                gain=gain,
            ),
        ],
        fixed_input_tasks=[],
        dt=dt,
    )

    return right_pink_ik_controller, left_pink_ik_controller


class RealmanVRTrixTrackerController(BaseController):
    config: RealmanVRTrixTrackerControllerConfig

    def __init__(self, config: RealmanVRTrixTrackerControllerConfig):
        super().__init__(config)
        self._is_ready = self.mp_manager.Value("b", False)
        self.frame_count = 0

    def check_ready(self):
        return self._is_ready.value and not self._stop_event.is_set()

    def _initialize(self):
        """Initialize VRTrix and Tracker connections"""
        try:
            # Initialize VRTrix connection
            self._init_vrtrix()

            # Initialize Tracker connection
            self._init_tracker()

            # Initialize Pink IK controllers
            self.right_pink_ik_controller, self.left_pink_ik_controller = init_pink_ik_controller(
                self.dt, self.config.pink_ik_damp, self.config.pink_ik_gain
            )

            # Initialize tracker reference poses
            self.tracker_initial_right_mat = None
            self.tracker_initial_left_mat = None
            self.robot_initial_right_mat = None
            self.robot_initial_left_mat = None
            self.initial_poses_saved = False

            # Set initial joint positions
            self.right_pink_ik_controller.q = self.config.default_right_q.copy()
            self.left_pink_ik_controller.q = self.config.default_left_q.copy()

            self._is_ready.value = True
            print("VRTrix-Tracker teleop controller ready!")

        except Exception as e:
            print(f"Failed to initialize controller: {e}")
            traceback.print_exc()
            raise

    def _init_vrtrix(self):
        """Initialize VRTrix connection"""
        # Create TCP socket for VRTrix
        self.vrtrix_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.vrtrix_sock.settimeout(self.config.connection_timeout)
        self.vrtrix_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.vrtrix_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)

        # Connect to VRTrix server
        vrtrix_address = (self.config.vrtrix_ip, self.config.vrtrix_port)
        print(f"Connecting to VRTrix server at {vrtrix_address}")
        self.vrtrix_sock.connect(vrtrix_address)

        # Set to quaternion mode
        self._send_vrtrix_command(CMD_SetRotationFormat, 'B', 1)

        # Auto calibrate if requested
        if self.config.auto_calibrate_on_start:
            print("Performing T-Pose calibration...")
            if self.config.enable_right:
                self._send_vrtrix_command(CMD_TPoseCalibration, 'R', 0)
            if self.config.enable_left:
                self._send_vrtrix_command(CMD_TPoseCalibration, 'L', 0)
            time.sleep(2.0)

        # Initialize VRTrix data storage
        self.latest_right_vrtrix_data = None
        self.latest_left_vrtrix_data = None

        # Start VRTrix receiver thread
        if self.config.use_threading:
            self.vrtrix_data_lock = threading.Lock()
            self.vrtrix_thread = threading.Thread(target=self._vrtrix_receiver_loop, daemon=True)
            self.vrtrix_thread.start()

    def _init_tracker(self):
        """Initialize Tracker connection"""
        # Create TCP socket for Tracker
        self.tracker_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tracker_address = (self.config.tracker_ip, self.config.tracker_port)
        print(f"Connecting to Tracker server at {tracker_address}")
        self.tracker_sock.connect(tracker_address)

        # Initialize tracker data storage
        self.latest_tracker_data = None
        self.tracker_buffer = ""

        # Start tracker receiver thread
        self.tracker_data_lock = threading.Lock()
        self.tracker_thread = threading.Thread(target=self._tracker_receiver_loop, daemon=True)
        self.tracker_thread.start()

    def _send_vrtrix_command(self, cmd: int, hand: str, param: int):
        """Send command to VRTrix gloves"""
        message = struct.pack('<hshhffff', cmd, hand.encode('utf-8'), param, 0, 0, 0, 0, 0)
        self.vrtrix_sock.sendall(message)

    def _vrtrix_receiver_loop(self):
        """Receiver thread for VRTrix data"""
        self.vrtrix_sock.setblocking(False)
        buffer = bytearray()

        while not self._stop_event.is_set():
            try:
                data = self.vrtrix_sock.recv(4096)
                if not data:
                    time.sleep(0.001)
                    continue

                buffer.extend(data)

                while len(buffer) >= 273:
                    packet = bytes(buffer[:273])
                    buffer = buffer[273:]

                    parsed_data = self._parse_vrtrix_data(packet)
                    if parsed_data:
                        with self.vrtrix_data_lock:
                            if parsed_data['hand'] == 'R' and self.config.enable_right:
                                self.latest_right_vrtrix_data = parsed_data
                            elif parsed_data['hand'] == 'L' and self.config.enable_left:
                                self.latest_left_vrtrix_data = parsed_data

            except socket.error:
                time.sleep(0.001)
            except Exception as e:
                if self.config.verbose:
                    print(f"Error in VRTrix receiver loop: {e}")
                time.sleep(0.001)

    def _tracker_receiver_loop(self):
        """Receiver thread for Tracker data"""
        while not self._stop_event.is_set():
            try:
                data = self.tracker_sock.recv(self.config.tracker_buffer_size).decode('utf-8')
                if not data:
                    print("Tracker server disconnected")
                    break

                # Process received data
                self.tracker_buffer += data
                lines = self.tracker_buffer.split('\n')
                self.tracker_buffer = lines[-1]  # Keep incomplete line

                for line in lines[:-1]:
                    if line.strip():
                        try:
                            tracker_data = json.loads(line)
                            with self.tracker_data_lock:
                                self.latest_tracker_data = tracker_data
                        except json.JSONDecodeError as e:
                            if self.config.verbose:
                                print(f"JSON parse error: {e}")

            except Exception as e:
                if self.config.verbose:
                    print(f"Error in tracker receiver loop: {e}")
                time.sleep(0.001)

    def _parse_vrtrix_data(self, data: bytes) -> Optional[Dict[str, Any]]:
        """Parse VRTrix glove data packet"""
        try:
            if len(data) != 273 or data[0:6] != b'VRTRIX':
                return None

            hand_char = chr(data[6])
            if hand_char not in ['L', 'R']:
                return None

            # Parse quaternions
            quat_data = np.frombuffer(data, dtype=np.float32, count=64, offset=9).reshape(16, 4)
            joints_quat_scipy = np.column_stack((quat_data[:, 1:4], quat_data[:, 0]))
            joints_quat = np.column_stack((joints_quat_scipy[:, 3], joints_quat_scipy[:, 0:3]))

            # Parse metadata
            metadata_offset = 265
            radio_strength, = struct.unpack('h', data[metadata_offset:metadata_offset+2])
            battery, = struct.unpack('f', data[metadata_offset+2:metadata_offset+6])
            cal_score, = struct.unpack('h', data[metadata_offset+6:metadata_offset+8])

            return {
                'hand': hand_char,
                'joints_quat': joints_quat.astype(np.float64),
                'radio_strength': radio_strength,
                'battery': battery,
                'cal_score': cal_score,
                'timestamp': time.time()
            }

        except Exception as e:
            if self.config.verbose:
                print(f"Error parsing VRTrix data: {e}")
            return None

    def _tracker_pose_to_mat(self, tracker_data: list) -> np.ndarray:
        """Convert tracker data [x, y, z, roll, pitch, yaw] to 4x4 matrix"""
        if len(tracker_data) != 6:
            return None

        # Extract position and orientation
        pos = np.array(tracker_data[:3])  # [x, y, z]
        euler_deg = np.array(tracker_data[3:])  # [roll, pitch, yaw] in degrees

        # Convert euler angles to rotation matrix
        r = R.from_euler('xyz', euler_deg, degrees=True)
        rot_mat = r.as_matrix()

        # Construct 4x4 transformation matrix
        mat = np.eye(4)
        mat[:3, :3] = rot_mat
        mat[:3, 3] = pos

        return mat

    def _save_initial_poses(self, tracker_data: dict):
        """Save initial poses for tracker and robot"""
        if self.initial_poses_saved:
            return

        # Get tracker data
        tracker_1 = tracker_data.get('tracker_1', [])  # Right hand
        tracker_2 = tracker_data.get('tracker_2', [])  # Left hand

        if self.config.enable_right and len(tracker_1) == 6:
            # Get tracker pose and apply coordinate transformation
            tracker_mat = self._tracker_pose_to_mat(tracker_1)
            self.tracker_initial_right_mat = self.config.tracker_to_hand_right @ tracker_mat

            # Get current robot end-effector pose using forward kinematics
            model = self.right_pink_ik_controller.robot_wrapper.model
            data = self.right_pink_ik_controller.robot_wrapper.data
            q = self.right_pink_ik_controller.q

            # Perform forward kinematics
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)

            # Get the frame ID for Link7
            link7_id = model.getFrameId("Link7")
            # Get the pose
            ee_pose = data.oMf[link7_id]
            self.robot_initial_right_mat = np.eye(4)
            self.robot_initial_right_mat[:3, :3] = ee_pose.rotation
            self.robot_initial_right_mat[:3, 3] = ee_pose.translation
            print("Saved initial right poses")

        if self.config.enable_left and len(tracker_2) == 6:
            # Get tracker pose and apply coordinate transformation
            tracker_mat = self._tracker_pose_to_mat(tracker_2)
            self.tracker_initial_left_mat = self.config.tracker_to_hand_left @ tracker_mat

            # Get current robot end-effector pose using forward kinematics
            model = self.left_pink_ik_controller.robot_wrapper.model
            data = self.left_pink_ik_controller.robot_wrapper.data
            q = self.left_pink_ik_controller.q

            # Perform forward kinematics
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)

            # Get the frame ID for Link7
            link7_id = model.getFrameId("Link7")
            # Get the pose
            ee_pose = data.oMf[link7_id]
            self.robot_initial_left_mat = np.eye(4)
            self.robot_initial_left_mat[:3, :3] = ee_pose.rotation
            self.robot_initial_left_mat[:3, 3] = ee_pose.translation
            print("Saved initial left poses")

        # Check if all required poses are saved
        if ((not self.config.enable_right or self.tracker_initial_right_mat is not None) and
            (not self.config.enable_left or self.tracker_initial_left_mat is not None)):
            self.initial_poses_saved = True
            print("All initial poses saved!")

    def _update(self):
        """Update loop - get VRTrix and Tracker data and compute robot commands"""
        try:
            timestamp = time.time()
            self.frame_count += 1

            # Get latest tracker data
            with self.tracker_data_lock:
                tracker_data = copy.deepcopy(self.latest_tracker_data) if self.latest_tracker_data else None

            # Save initial poses if not done yet
            if tracker_data and not self.initial_poses_saved:
                self._save_initial_poses(tracker_data)

            # Get VRTrix data
            with self.vrtrix_data_lock:
                right_vrtrix_data = self.latest_right_vrtrix_data.copy() if self.latest_right_vrtrix_data else None
                left_vrtrix_data = self.latest_left_vrtrix_data.copy() if self.latest_left_vrtrix_data else None

            # Initialize outputs
            right_q = self.config.default_right_q.copy()
            left_q = self.config.default_left_q.copy()
            right_hand_cmd = np.zeros(6, dtype=np.float64)
            left_hand_cmd = np.zeros(6, dtype=np.float64)

            # Process right arm
            if self.config.enable_right and self.initial_poses_saved:
                # Get tracker pose
                if tracker_data and tracker_data.get('timestamp', 0) > 0:
                    tracker_age = timestamp - tracker_data['timestamp']
                    if tracker_age < self.config.tracker_timeout:
                        tracker_1 = tracker_data.get('tracker_1', [])
                        if len(tracker_1) == 6:
                            # Compute current tracker pose and apply coordinate transformation
                            tracker_raw_mat = self._tracker_pose_to_mat(tracker_1)
                            tracker_current_mat = self.config.tracker_to_hand_right @ tracker_raw_mat

                            # Compute relative transformation
                            tracker_delta_mat = calc_delta_mat(
                                self.tracker_initial_right_mat,
                                tracker_current_mat
                            )

                            # Apply relative transformation to robot
                            robot_target_mat = calc_end_mat(
                                self.robot_initial_right_mat,
                                tracker_delta_mat
                            )

                            # Convert to robot base frame
                            # robot_target_mat is in base frame, so no conversion needed
                            robot_target_mat_base = robot_target_mat

                            # Set IK target and compute joint angles
                            self.right_pink_ik_controller.set_target(robot_target_mat_base)
                            right_q = self.right_pink_ik_controller.compute()

                # Process hand commands from VRTrix
                if right_vrtrix_data and (timestamp - right_vrtrix_data['timestamp']) < 0.1:
                    right_joints_quat = right_vrtrix_data['joints_quat']
                    right_hand_angles = vrtrix_to_realman_6dof(
                        right_joints_quat, 'right',
                        self.config.proximal_weight,
                        self.config.intermediate_weight
                    )
                    right_hand_angles = np.clip(right_hand_angles, self.config.angle_min, self.config.angle_max)
                    right_hand_cmd = right_hand_angles / self.config.angle_max * 65535
                    right_hand_cmd = np.concatenate([right_hand_cmd[1:], right_hand_cmd[:1]])

            # Process left arm
            if self.config.enable_left and self.initial_poses_saved:
                # Get tracker pose
                if tracker_data and tracker_data.get('timestamp', 0) > 0:
                    tracker_age = timestamp - tracker_data['timestamp']
                    if tracker_age < self.config.tracker_timeout:
                        tracker_2 = tracker_data.get('tracker_2', [])
                        if len(tracker_2) == 6:
                            # Compute current tracker pose and apply coordinate transformation
                            tracker_raw_mat = self._tracker_pose_to_mat(tracker_2)
                            tracker_current_mat = self.config.tracker_to_hand_left @ tracker_raw_mat

                            # Compute relative transformation
                            tracker_delta_mat = calc_delta_mat(
                                self.tracker_initial_left_mat,
                                tracker_current_mat
                            )

                            # Apply relative transformation to robot
                            robot_target_mat = calc_end_mat(
                                self.robot_initial_left_mat,
                                tracker_delta_mat
                            )

                            # Convert to robot base frame
                            # robot_target_mat is in base frame, so no conversion needed
                            robot_target_mat_base = robot_target_mat

                            # Set IK target and compute joint angles
                            self.left_pink_ik_controller.set_target(robot_target_mat_base)
                            left_q = self.left_pink_ik_controller.compute()

                # Process hand commands from VRTrix
                if left_vrtrix_data and (timestamp - left_vrtrix_data['timestamp']) < 0.1:
                    left_joints_quat = left_vrtrix_data['joints_quat']
                    left_hand_angles = vrtrix_to_realman_6dof(
                        left_joints_quat, 'left',
                        self.config.proximal_weight,
                        self.config.intermediate_weight
                    )
                    left_hand_angles = np.clip(left_hand_angles, self.config.angle_min, self.config.angle_max)
                    left_hand_cmd = left_hand_angles / self.config.angle_max * 65535
                    left_hand_cmd = np.concatenate([left_hand_cmd[1:], left_hand_cmd[:1]])

            # Debug output
            if self.config.debug_mode and self.frame_count % self.config.debug_interval == 0:
                print(f"\n=== Frame {self.frame_count} ===")
                if tracker_data:
                    print(f"Tracker data age: {timestamp - tracker_data.get('timestamp', 0):.3f}s")
                    if self.config.enable_right:
                        tracker_1 = tracker_data.get('tracker_1', [])
                        if len(tracker_1) == 6:
                            print(f"Right tracker: pos=({tracker_1[0]:.3f}, {tracker_1[1]:.3f}, {tracker_1[2]:.3f})")
                            print(f"              rot=({tracker_1[3]:.1f}°, {tracker_1[4]:.1f}°, {tracker_1[5]:.1f}°)")
                    if self.config.enable_left:
                        tracker_2 = tracker_data.get('tracker_2', [])
                        if len(tracker_2) == 6:
                            print(f"Left tracker:  pos=({tracker_2[0]:.3f}, {tracker_2[1]:.3f}, {tracker_2[2]:.3f})")
                            print(f"              rot=({tracker_2[3]:.1f}°, {tracker_2[4]:.1f}°, {tracker_2[5]:.1f}°)")

            # Prepare state dict
            state_dict = {
                "right_q": right_q,
                "left_q": left_q,
                "right_mat_w": np.eye(4, dtype=np.float64),  # Could compute actual pose if needed
                "left_mat_w": np.eye(4, dtype=np.float64),
                "right_hand_cmd": right_hand_cmd,
                "left_hand_cmd": left_hand_cmd,
                "receive_timestamp": np.array([timestamp], dtype=np.float64),
                "timestamp": np.array([timestamp], dtype=np.float64)
            }

            self.feedback_queue.put(state_dict)

        except Exception as e:
            print(f"Error in update loop: {e}")
            traceback.print_exc()

    def _process_commands(self):
        """Process any pending commands"""
        pass

    def _close(self):
        """Clean up resources"""
        try:
            self._stop_event.set()

            # Close VRTrix connection
            if hasattr(self, 'vrtrix_thread'):
                self.vrtrix_thread.join(timeout=1.0)
            if hasattr(self, 'vrtrix_sock'):
                self.vrtrix_sock.close()

            # Close Tracker connection
            if hasattr(self, 'tracker_thread'):
                self.tracker_thread.join(timeout=1.0)
            if hasattr(self, 'tracker_sock'):
                self.tracker_sock.close()

        except Exception as e:
            print(f"Error closing controller: {e}")

    def reset(self):
        """Reset controller state"""
        # Reset initial poses
        self.initial_poses_saved = False
        self.tracker_initial_right_mat = None
        self.tracker_initial_left_mat = None
        self.robot_initial_right_mat = None
        self.robot_initial_left_mat = None

        # Reset joint positions
        if hasattr(self, 'right_pink_ik_controller'):
            self.right_pink_ik_controller.q = self.config.default_right_q.copy()
        if hasattr(self, 'left_pink_ik_controller'):
            self.left_pink_ik_controller.q = self.config.default_left_q.copy()

        # Clear data
        with self.vrtrix_data_lock:
            self.latest_right_vrtrix_data = None
            self.latest_left_vrtrix_data = None
        with self.tracker_data_lock:
            self.latest_tracker_data = None