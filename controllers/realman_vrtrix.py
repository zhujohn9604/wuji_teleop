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

from pink.tasks import FrameTask

from .base import BaseController, BaseControllerConfig
from utils.vrtrix_utils import vrtrix_to_realman_6dof
from utils.retargeting_utils import pose_to_mat, mat_to_pose, calc_delta_mat, calc_end_mat
from utils.pink_ik_utils import PinkIKController

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
class RealmanVRTrixTeleopControllerConfig(BaseControllerConfig):
    # VRTrix glove settings
    vrtrix_ip: str = "127.0.0.1"
    vrtrix_port: int = 11002

    # Tracker settings
    enable_tracker: bool = False
    tracker_ip: str = "192.168.2.68"
    tracker_port: int = 12345
    tracker_position_scale: float = 1.0  # Scale factor for position mapping
    tracker_smoothing_factor: float = 0.8  # Exponential smoothing factor (0-1)

    # Robot arm configuration
    robot_r_mat_w: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    robot_l_mat_w: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))

    # Pink IK parameters
    pink_ik_damp: float = 0.5
    pink_ik_gain: float = 1.0

    # Default robot joint positions
    default_right_q: np.ndarray = field(default_factory=lambda: np.zeros((7,), dtype=np.float64))
    default_left_q: np.ndarray = field(default_factory=lambda: np.zeros((7,), dtype=np.float64))

    # Default end effector position (relative to robot base)
    default_ee_position: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.0, 0.3], dtype=np.float64))

    # Default end effector rotation (Euler angles in degrees, XYZ order)
    default_ee_rotation_xyz: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64))

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

    # Rotation scaling factor
    rotation_scale: float = 1.0

    def validate(self):
        super().validate()

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


class RealmanVRTrixTeleopController(BaseController):
    config: RealmanVRTrixTeleopControllerConfig

    def __init__(self, config: RealmanVRTrixTeleopControllerConfig):
        super().__init__(config)
        self._is_ready = self.mp_manager.Value("b", False)
        self.frame_count = 0

    def check_ready(self):
        return self._is_ready.value and not self._stop_event.is_set()

    def _initialize(self):
        """Initialize VRTrix connection and Pink IK controllers"""
        try:
            # Initialize Pink IK controllers
            self.right_pink_ik_controller, self.left_pink_ik_controller = init_pink_ik_controller(
                self.dt, self.config.pink_ik_damp, self.config.pink_ik_gain
            )

            # Create TCP socket for VRTrix
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.config.connection_timeout)

            # Set TCP_NODELAY to reduce latency
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            # Increase receive buffer size
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)

            # Connect to VRTrix server
            server_address = (self.config.vrtrix_ip, self.config.vrtrix_port)
            print(f"Connecting to VRTrix server at {server_address}")
            self.sock.connect(server_address)

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

            # Initialize data storage
            self.latest_right_data = None
            self.latest_left_data = None
            self.last_update_time = time.time()

            # Initialize reference quaternions (will be set on first valid data)
            self.right_wrist_ref_quat = None
            self.left_wrist_ref_quat = None

            # Initialize default end effector poses
            self._init_default_ee_poses()

            # Define coordinate transformation matrix from VRTrix to Robot
            # VRTrix coordinate system: X-right, Y-forward, Z-up
            # Robot coordinate system: X-right, Y-up, Z-forward
            # So we need to swap Y and Z axes
            self.vrtrix_to_robot_transform = np.array([
                [1, 0, 0],  # X remains X
                [0, 0, 1],  # Z becomes Y
                [0, 1, 0]   # Y becomes Z
            ], dtype=np.float64)

            # Define coordinate transformation matrix from Tracker to Robot
            # Tracker: +Y = Robot: +Z
            # Tracker: +X = Robot: -Y
            # Tracker: +Z = Robot: +X
            self.tracker_to_robot_transform = np.array([
                [0, 0, -1],   # tracker_z -> robot_x
                [-1, 0, 0],  # -tracker_x -> robot_y
                [0, 1, 0]    # tracker_y -> robot_z
            ], dtype=np.float64)

            # Initialize tracker if enabled
            if self.config.enable_tracker:
                self._init_tracker()

            # If using threading mode, start receiver thread
            if self.config.use_threading:
                self.data_lock = threading.Lock()
                self.receiver_thread = threading.Thread(target=self._vrtrix_receiver_loop, daemon=True)
                self.receiver_thread.start()

            self._is_ready.value = True
            print("VRTrix teleop controller ready!")

        except Exception as e:
            print(f"Failed to initialize VRTrix connection: {e}")
            traceback.print_exc()
            raise

    def _init_tracker(self):
        """Initialize tracker connection and data structures"""
        try:
            # Create TCP socket for tracker
            self.tracker_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tracker_sock.settimeout(self.config.connection_timeout)

            # Connect to tracker server
            tracker_address = (self.config.tracker_ip, self.config.tracker_port)
            print(f"Connecting to tracker server at {tracker_address}")
            self.tracker_sock.connect(tracker_address)
            print("Tracker connection established!")

            # Initialize tracker data storage
            self.tracker_data_lock = threading.Lock()
            self.latest_tracker_data = {
                'tracker_1': None,  # Right hand
                'tracker_2': None,  # Left hand
                'timestamp': 0
            }

            # Initialize reference positions (will be set on first valid data)
            self.right_tracker_ref_pos = None
            self.left_tracker_ref_pos = None

            # Initialize smoothed positions
            self.right_tracker_smoothed_pos = None
            self.left_tracker_smoothed_pos = None

            # Start tracker receiver thread
            self.tracker_receiver_thread = threading.Thread(target=self._tracker_receiver_loop, daemon=True)
            self.tracker_receiver_thread.start()

        except Exception as e:
            print(f"Failed to initialize tracker connection: {e}")
            print("Tracker will be disabled.")
            self.config.enable_tracker = False

    def _tracker_receiver_loop(self):
        """Receiver thread for tracker data"""
        buffer = ""

        while not self._stop_event.is_set():
            try:
                # Receive data
                data = self.tracker_sock.recv(4096).decode('utf-8')

                if not data:
                    print("Tracker connection lost, attempting to reconnect...")
                    self._reconnect_tracker()
                    continue

                # Process received data
                buffer += data
                lines = buffer.split('\n')
                buffer = lines[-1]  # Keep incomplete line

                for line in lines[:-1]:
                    if line.strip():
                        try:
                            # Parse JSON data
                            tracker_data = json.loads(line)

                            with self.tracker_data_lock:
                                self.latest_tracker_data = tracker_data

                                # Set reference positions on first valid data
                                if tracker_data.get('tracker_1') and self.right_tracker_ref_pos is None:
                                    self.right_tracker_ref_pos = np.array(tracker_data['tracker_1'][:3])
                                    self.right_tracker_smoothed_pos = self.right_tracker_ref_pos.copy()
                                    print(f"Right tracker reference position set: {self.right_tracker_ref_pos}")

                                if tracker_data.get('tracker_2') and self.left_tracker_ref_pos is None:
                                    self.left_tracker_ref_pos = np.array(tracker_data['tracker_2'][:3])
                                    self.left_tracker_smoothed_pos = self.left_tracker_ref_pos.copy()
                                    print(f"Left tracker reference position set: {self.left_tracker_ref_pos}")

                        except json.JSONDecodeError as e:
                            if self.config.verbose:
                                print(f"Tracker JSON decode error: {e}")
                        except Exception as e:
                            if self.config.verbose:
                                print(f"Error processing tracker data: {e}")

            except socket.timeout:
                continue
            except Exception as e:
                if self.config.verbose:
                    print(f"Error in tracker receiver loop: {e}")
                time.sleep(0.1)

    def _reconnect_tracker(self):
        """Attempt to reconnect to tracker server"""
        try:
            self.tracker_sock.close()
            self.tracker_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tracker_sock.settimeout(self.config.connection_timeout)
            tracker_address = (self.config.tracker_ip, self.config.tracker_port)
            self.tracker_sock.connect(tracker_address)
            print("Tracker reconnection successful!")
        except Exception as e:
            if self.config.verbose:
                print(f"Tracker reconnection failed: {e}")
            time.sleep(5)

    def _transform_tracker_position_to_robot(self, tracker_pos: np.ndarray) -> np.ndarray:
        """Transform position from tracker coordinate system to robot coordinate system"""
        # Apply the coordinate transformation
        robot_pos = self.tracker_to_robot_transform @ tracker_pos
        return robot_pos

    def _apply_position_smoothing(self, current_pos: np.ndarray, smoothed_pos: np.ndarray) -> np.ndarray:
        """Apply exponential smoothing to position"""
        alpha = self.config.tracker_smoothing_factor
        return alpha * smoothed_pos + (1 - alpha) * current_pos

    def _init_default_ee_poses(self):
        """Initialize default end effector poses with position and rotation"""
        # Create rotation from Euler angles (XYZ order)
        euler_angles_rad = np.deg2rad(self.config.default_ee_rotation_xyz)
        default_rotation = R.from_euler('xyz', euler_angles_rad)

        # Right arm default pose
        self.right_ee_default_mat = np.eye(4, dtype=np.float64)
        right_pos = self.config.default_ee_position.copy()
        right_pos[0] = -right_pos[0]
        right_pos[1] = -right_pos[1]+0.1
        self.right_ee_default_mat[:3, 3] = right_pos
        self.right_ee_default_mat[:3, :3] = default_rotation.as_matrix()

        # Left arm default pose (mirror x position)
        self.left_ee_default_mat = np.eye(4, dtype=np.float64)
        left_pos = self.config.default_ee_position.copy()
        left_pos[0] = -left_pos[0]  # Mirror x position
        self.left_ee_default_mat[:3, 3] = left_pos
        self.left_ee_default_mat[:3, :3] = default_rotation.as_matrix()

        # Print initialization info
        print(f"Initialized default EE poses:")
        print(f"  Position: {self.config.default_ee_position}")
        print(f"  Rotation (XYZ Euler angles in degrees): {self.config.default_ee_rotation_xyz}")
        print(f"  Rotation matrix:\n{default_rotation.as_matrix()}")

    def _send_vrtrix_command(self, cmd: int, hand: str, param: int):
        """Send command to VRTrix gloves"""
        message = struct.pack('<hshhffff', cmd, hand.encode('utf-8'), param, 0, 0, 0, 0, 0)
        self.sock.sendall(message)

    def _vrtrix_receiver_loop(self):
        """Receiver thread for VRTrix data"""
        self.sock.setblocking(False)
        buffer = bytearray()

        while not self._stop_event.is_set():
            try:
                # Try to read data
                data = self.sock.recv(4096)
                if not data:
                    time.sleep(0.001)
                    continue

                buffer.extend(data)

                # Process complete packets
                while len(buffer) >= 273:
                    packet = bytes(buffer[:273])
                    buffer = buffer[273:]

                    parsed_data = self._parse_vrtrix_data(packet)
                    if parsed_data:
                        with self.data_lock:
                            if parsed_data['hand'] == 'R' and self.config.enable_right:
                                self.latest_right_data = parsed_data
                                # Set reference quaternion on first valid data
                                if self.right_wrist_ref_quat is None:
                                    self.right_wrist_ref_quat = parsed_data['joints_quat'][0].copy()
                                    print(f"Right wrist reference quaternion set: {self.right_wrist_ref_quat}")
                            elif parsed_data['hand'] == 'L' and self.config.enable_left:
                                self.latest_left_data = parsed_data
                                # Set reference quaternion on first valid data
                                if self.left_wrist_ref_quat is None:
                                    self.left_wrist_ref_quat = parsed_data['joints_quat'][0].copy()
                                    print(f"Left wrist reference quaternion set: {self.left_wrist_ref_quat}")

            except socket.error:
                time.sleep(0.001)
            except Exception as e:
                if self.config.verbose:
                    print(f"Error in VRTrix receiver loop: {e}")
                time.sleep(0.001)

    def _parse_vrtrix_data(self, data: bytes) -> Optional[Dict[str, Any]]:
        """Parse VRTrix glove data packet"""
        try:
            # Quick check
            if len(data) != 273 or data[0:6] != b'VRTRIX':
                return None

            hand_char = chr(data[6])
            if hand_char not in ['L', 'R']:
                return None

            # Parse quaternions (16 quaternions, 4 floats each)
            quat_data = np.frombuffer(data, dtype=np.float32, count=64, offset=9).reshape(16, 4)

            # Convert to scipy format [x, y, z, w] and then to our format [w, x, y, z]
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

    def _transform_vrtrix_rotation_to_robot(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """Transform rotation matrix from VRTrix coordinate system to robot coordinate system"""
        # Apply the coordinate transformation to the rotation matrix
        # R_robot = T * R_vrtrix * T^T
        transformed_rotation = self.vrtrix_to_robot_transform @ rotation_matrix @ self.vrtrix_to_robot_transform.T
        return transformed_rotation

    def _compute_relative_rotation(self, current_quat: np.ndarray, ref_quat: np.ndarray) -> np.ndarray:
        """Compute relative rotation from reference quaternion to current quaternion"""
        # Convert to scipy Rotation objects (expects [x, y, z, w] format)
        ref_rot = R.from_quat(np.concatenate([ref_quat[1:4], ref_quat[0:1]]))
        curr_rot = R.from_quat(np.concatenate([current_quat[1:4], current_quat[0:1]]))

        # Compute relative rotation: delta = curr * ref^(-1)
        relative_rot = curr_rot * ref_rot.inv()

        # Get rotation matrix
        relative_rot_matrix = relative_rot.as_matrix()

        # Transform from VRTrix coordinate system to robot coordinate system
        transformed_rot_matrix = self._transform_vrtrix_rotation_to_robot(relative_rot_matrix)

        # Apply rotation scaling if needed
        if self.config.rotation_scale != 1.0:
            # Convert to axis-angle, scale, and convert back
            scaled_rot = R.from_matrix(transformed_rot_matrix)
            axis_angle = scaled_rot.as_rotvec()
            scaled_axis_angle = axis_angle * self.config.rotation_scale
            scaled_rot = R.from_rotvec(scaled_axis_angle)
            transformed_rot_matrix = scaled_rot.as_matrix()

        return transformed_rot_matrix

    def _update(self):
        """Update loop - get VRTrix data and compute robot commands"""
        try:
            timestamp = time.time()
            self.frame_count += 1

            # Get VRTrix data
            if self.config.use_threading:
                # Threading mode - read from latest data
                with self.data_lock:
                    right_data = self.latest_right_data.copy() if self.latest_right_data else None
                    left_data = self.latest_left_data.copy() if self.latest_left_data else None
            else:
                # Non-threading mode - read limited packets
                self.sock.setblocking(False)
                packets_processed = 0

                while packets_processed < self.config.max_packets_per_update:
                    try:
                        data = self.sock.recv(273)
                        if not data or len(data) != 273:
                            break

                        parsed_data = self._parse_vrtrix_data(data)
                        if parsed_data:
                            if parsed_data['hand'] == 'R' and self.config.enable_right:
                                self.latest_right_data = parsed_data
                                # Set reference quaternion on first valid data
                                if self.right_wrist_ref_quat is None:
                                    self.right_wrist_ref_quat = parsed_data['joints_quat'][0].copy()
                                    print(f"Right wrist reference quaternion set: {self.right_wrist_ref_quat}")
                            elif parsed_data['hand'] == 'L' and self.config.enable_left:
                                self.latest_left_data = parsed_data
                                # Set reference quaternion on first valid data
                                if self.left_wrist_ref_quat is None:
                                    self.left_wrist_ref_quat = parsed_data['joints_quat'][0].copy()
                                    print(f"Left wrist reference quaternion set: {self.left_wrist_ref_quat}")

                        packets_processed += 1

                    except socket.error:
                        break

                right_data = self.latest_right_data
                left_data = self.latest_left_data

            # Get tracker data if enabled
            tracker_data = None
            if self.config.enable_tracker:
                with self.tracker_data_lock:
                    tracker_data = self.latest_tracker_data.copy() if self.latest_tracker_data else None

            # Initialize outputs
            right_q = self.config.default_right_q.copy()
            left_q = self.config.default_left_q.copy()
            right_hand_cmd = np.zeros(6, dtype=np.float64)
            left_hand_cmd = np.zeros(6, dtype=np.float64)
            right_mat_w = self.right_ee_default_mat.copy()
            left_mat_w = self.left_ee_default_mat.copy()

            # Process right hand
            if (self.config.enable_right and right_data and
                (timestamp - right_data['timestamp']) < 0.1 and
                self.right_wrist_ref_quat is not None):

                right_joints_quat = right_data['joints_quat']  # 16x4

                # Get wrist quaternion (first joint)
                right_wrist_quat = right_joints_quat[0]

                # Compute relative rotation (with coordinate transformation)
                relative_rotation = self._compute_relative_rotation(right_wrist_quat, self.right_wrist_ref_quat)

                # Apply relative rotation to default end effector pose
                right_ee_mat = self.right_ee_default_mat.copy()
                right_ee_mat[:3, :3] = relative_rotation @ self.right_ee_default_mat[:3, :3]

                # Apply tracker position if enabled
                if (self.config.enable_tracker and tracker_data and
                    tracker_data.get('tracker_1') and
                    self.right_tracker_ref_pos is not None):

                    # Get current tracker position (first 3 elements)
                    current_tracker_pos = np.array(tracker_data['tracker_1'][:3])

                    # Compute relative position in tracker space
                    relative_tracker_pos = current_tracker_pos - self.right_tracker_ref_pos

                    # Transform to robot coordinate system
                    relative_robot_pos = self._transform_tracker_position_to_robot(relative_tracker_pos)

                    # Apply scaling
                    scaled_robot_pos = relative_robot_pos * self.config.tracker_position_scale

                    # Apply smoothing
                    if self.right_tracker_smoothed_pos is not None:
                        target_pos = self.right_ee_default_mat[:3, 3] + scaled_robot_pos  # 使用修改后的位置
                        self.right_tracker_smoothed_pos = self._apply_position_smoothing(
                            target_pos, self.right_tracker_smoothed_pos)
                        right_ee_mat[:3, 3] = self.right_tracker_smoothed_pos
                    else:
                        right_ee_mat[:3, 3] = self.right_ee_default_mat[:3, 3] + scaled_robot_pos  # 使用修改后的位置

                # Transform to robot base frame
                right_ee_mat_base = calc_delta_mat(self.config.robot_r_mat_w, right_ee_mat)

                # Solve IK
                self.right_pink_ik_controller.set_target(right_ee_mat_base)
                right_q = self.right_pink_ik_controller.compute()

                # Update world matrix for feedback
                right_mat_w = right_ee_mat

                # Process hand angles
                right_hand_angles = vrtrix_to_realman_6dof(
                    right_joints_quat, 'right', self.config.proximal_weight, self.config.intermediate_weight)

                # Debug output
                if self.config.debug_mode and self.frame_count % self.config.debug_interval == 0:
                    print("\n=== Right Hand Debug Info ===")
                    print(f"Frame: {self.frame_count}")
                    print(f"Battery: {right_data['battery']:.1f}%, Signal: {right_data['radio_strength']}dB")
                    print(f"Wrist quaternion: {right_wrist_quat}")
                    print(f"Relative rotation (after coordinate transform):\n{relative_rotation}")
                    if self.config.enable_tracker and tracker_data and tracker_data.get('tracker_1'):
                        print(f"Tracker position: {tracker_data['tracker_1'][:3]}")
                        print(f"EE position: {right_ee_mat[:3, 3]}")
                    print("Finger angles (degrees):")
                    print(f"  Thumb:  abd={right_hand_angles[0]:.1f}°, flex={right_hand_angles[1]:.1f}°")
                    print(f"  Index:  {right_hand_angles[2]:.1f}°")
                    print(f"  Middle: {right_hand_angles[3]:.1f}°")
                    print(f"  Ring:   {right_hand_angles[4]:.1f}°")
                    print(f"  Pinky:  {right_hand_angles[5]:.1f}°")

                # Clip to valid range
                right_hand_angles = np.clip(right_hand_angles, self.config.angle_min, self.config.angle_max)

                # Convert to command values (0-65535)
                right_hand_cmd = right_hand_angles / self.config.angle_max * 65535

                # Reorder as per Realman format
                right_hand_cmd = np.concatenate([
                    right_hand_cmd[1:],  # thumb_2, index_1, middle_1, ring_1, pinky_1
                    right_hand_cmd[:1]   # thumb_1
                ])

            # Process left hand
            if (self.config.enable_left and left_data and
                (timestamp - left_data['timestamp']) < 0.1 and
                self.left_wrist_ref_quat is not None):

                left_joints_quat = left_data['joints_quat']  # 16x4

                # Get wrist quaternion (first joint)
                left_wrist_quat = left_joints_quat[0]

                # Compute relative rotation (with coordinate transformation)
                relative_rotation = self._compute_relative_rotation(left_wrist_quat, self.left_wrist_ref_quat)

                # Apply relative rotation to default end effector pose
                left_ee_mat = self.left_ee_default_mat.copy()
                left_ee_mat[:3, :3] = relative_rotation @ self.left_ee_default_mat[:3, :3]

                # Apply tracker position if enabled
                if (self.config.enable_tracker and tracker_data and
                    tracker_data.get('tracker_2') and
                    self.left_tracker_ref_pos is not None):

                    # Get current tracker position (first 3 elements)
                    current_tracker_pos = np.array(tracker_data['tracker_2'][:3])

                    # Compute relative position in tracker space
                    relative_tracker_pos = current_tracker_pos - self.left_tracker_ref_pos

                    # Transform to robot coordinate system
                    relative_robot_pos = self._transform_tracker_position_to_robot(relative_tracker_pos)

                    # Apply scaling
                    scaled_robot_pos = relative_robot_pos * self.config.tracker_position_scale

                    # Apply smoothing
                    if self.left_tracker_smoothed_pos is not None:
                        target_pos = self.left_ee_default_mat[:3, 3] + scaled_robot_pos  # 使用修改后的位置
                        self.left_tracker_smoothed_pos = self._apply_position_smoothing(
                            target_pos, self.left_tracker_smoothed_pos)
                        left_ee_mat[:3, 3] = self.left_tracker_smoothed_pos
                    else:
                        left_ee_mat[:3, 3] = self.left_ee_default_mat[:3, 3] + scaled_robot_pos  # 使用修改后的位置

                # Transform to robot base frame
                left_ee_mat_base = calc_delta_mat(self.config.robot_l_mat_w, left_ee_mat)

                # Solve IK
                self.left_pink_ik_controller.set_target(left_ee_mat_base)
                left_q = self.left_pink_ik_controller.compute()

                # Update world matrix for feedback
                left_mat_w = left_ee_mat

                # Process hand angles
                left_hand_angles = vrtrix_to_realman_6dof(
                    left_joints_quat, 'left', self.config.proximal_weight, self.config.intermediate_weight)

                # Debug output
                if self.config.debug_mode and self.frame_count % self.config.debug_interval == 0:
                    print("\n=== Left Hand Debug Info ===")
                    print(f"Battery: {left_data['battery']:.1f}%, Signal: {left_data['radio_strength']}dB")
                    print(f"Wrist quaternion: {left_wrist_quat}")
                    print(f"Relative rotation (after coordinate transform):\n{relative_rotation}")
                    if self.config.enable_tracker and tracker_data and tracker_data.get('tracker_2'):
                        print(f"Tracker position: {tracker_data['tracker_2'][:3]}")
                        print(f"EE position: {left_ee_mat[:3, 3]}")
                    print("Finger angles (degrees):")
                    print(f"  Thumb:  abd={left_hand_angles[0]:.1f}°, flex={left_hand_angles[1]:.1f}°")
                    print(f"  Index:  {left_hand_angles[2]:.1f}°")
                    print(f"  Middle: {left_hand_angles[3]:.1f}°")
                    print(f"  Ring:   {left_hand_angles[4]:.1f}°")
                    print(f"  Pinky:  {left_hand_angles[5]:.1f}°")
                    print("="*40)

                # Clip to valid range
                left_hand_angles = np.clip(left_hand_angles, self.config.angle_min, self.config.angle_max)

                # Convert to command values (0-65535)
                left_hand_cmd = left_hand_angles / self.config.angle_max * 65535

                # Reorder as per Realman format
                left_hand_cmd = np.concatenate([
                    left_hand_cmd[1:],  # thumb_2, index_1, middle_1, ring_1, pinky_1
                    left_hand_cmd[:1]   # thumb_1
                ])

            # Prepare state dict
            state_dict = {
                "right_q": right_q,
                "left_q": left_q,
                "right_mat_w": right_mat_w,
                "left_mat_w": left_mat_w,
                "right_hand_cmd": right_hand_cmd,
                "left_hand_cmd": left_hand_cmd,
                "receive_timestamp": np.array([timestamp], dtype=np.float64),
                "timestamp": np.array([timestamp], dtype=np.float64)
            }

            self.feedback_queue.put(state_dict)

        except Exception as e:
            print(f"Error in VRTrix teleop update: {e}")
            import traceback
            traceback.print_exc()

    def _process_commands(self):
        """Process any pending commands"""
        pass

    def _close(self):
        """Clean up resources"""
        try:
            self._stop_event.set()

            if self.config.use_threading and hasattr(self, 'receiver_thread'):
                self.receiver_thread.join(timeout=1.0)

            if self.config.enable_tracker and hasattr(self, 'tracker_receiver_thread'):
                self.tracker_receiver_thread.join(timeout=1.0)
                if hasattr(self, 'tracker_sock'):
                    self.tracker_sock.close()

            if hasattr(self, 'sock'):
                self.sock.close()
        except Exception as e:
            print(f"Error closing VRTrix teleop controller: {e}")

    def reset(self):
        """Reset controller state"""
        if self.config.use_threading:
            with self.data_lock:
                self.latest_right_data = None
                self.latest_left_data = None
                # Reset reference quaternions
                self.right_wrist_ref_quat = None
                self.left_wrist_ref_quat = None
        else:
            self.latest_right_data = None
            self.latest_left_data = None
            # Reset reference quaternions
            self.right_wrist_ref_quat = None
            self.left_wrist_ref_quat = None

        # Reset tracker references if enabled
        if self.config.enable_tracker:
            with self.tracker_data_lock:
                self.right_tracker_ref_pos = None
                self.left_tracker_ref_pos = None
                self.right_tracker_smoothed_pos = None
                self.left_tracker_smoothed_pos = None

    def get_feedback(self):
        """Get latest feedback data"""
        return self.feedback_queue.get()

    def trigger_haptics(self, hand: str, duration_ms: int = 500):
        """Trigger haptic feedback on specified hand"""
        if self._is_ready.value:
            self._send_vrtrix_command(CMD_TriggerHaptics, hand, duration_ms)