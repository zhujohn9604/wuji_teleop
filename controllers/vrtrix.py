from dataclasses import field, dataclass
from typing import Optional, Any, Dict, Tuple
import multiprocessing as mp
import time
import socket
import struct
import traceback
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import deque
import threading

import click

from .base import BaseController, BaseControllerConfig

# Command constants
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
class VRTrixControllerConfig(BaseControllerConfig):
    # Server address for gloves
    ip: str = ""
    port: int = 11002

    enable_right: bool = True
    enable_left: bool = True

    # Calibration settings
    auto_calibrate_on_start: bool = True

    # Connection settings
    recv_buffer_size: int = 273  # Fixed packet size per protocol
    connection_timeout: float = 5.0

    # Performance settings
    max_packets_per_update: int = 3  # 减少每次更新处理的包数
    use_threading: bool = True  # 使用独立线程接收数据

    def validate(self):
        super().validate()

        # Define feedback sample structure
        self.feedback_sample = {
            "right_joints_quat": np.zeros((16, 4), dtype=np.float64),  # All 16 joints x 4 (qw,qx,qy,qz)
            "left_joints_quat": np.zeros((16, 4), dtype=np.float64),
            "right_joint_poses": np.zeros((16, 4, 4), dtype=np.float64),  # All 16 joints x 4x4 pose matrix
            "left_joint_poses": np.zeros((16, 4, 4), dtype=np.float64),
            "right_fingertip_poses": np.zeros((5, 4, 4), dtype=np.float64),  # 5 fingertips x 4x4 pose matrix
            "left_fingertip_poses": np.zeros((5, 4, 4), dtype=np.float64),
            "receive_timestamp": np.zeros((1,), dtype=np.float64),
            "timestamp": np.zeros((1,), dtype=np.float64),
        }


class VRTrixController(BaseController):
    """
    Optimized controller for VRTrix data gloves.

    Joint mapping (0-15):
    0: Wrist
    1: Thumb_Proximal
    2: Thumb_Intermediate
    3: Thumb_Distal (tip)
    4: Index_Proximal
    5: Index_Intermediate
    6: Index_Distal (tip)
    7: Middle_Proximal
    8: Middle_Intermediate
    9: Middle_Distal (tip)
    10: Ring_Proximal
    11: Ring_Intermediate
    12: Ring_Distal (tip)
    13: Pinky_Proximal
    14: Pinky_Intermediate
    15: Pinky_Distal (tip)
    """
    config: VRTrixControllerConfig

    def __init__(self, config: VRTrixControllerConfig):
        super().__init__(config)
        self._is_ready = self.mp_manager.Value("b", False)

        # 使用线程安全的数据结构
        if config.use_threading:
            self.data_lock = threading.Lock()
            self.receiver_thread = None

        # 预计算常量矩阵，避免重复计算
        self._setup_kinematics_constants()

    def _setup_kinematics_constants(self):
        """预计算运动学常量，避免重复计算"""
        # Define finger segment lengths in meters
        self.segment_lengths = {
            'thumb': {
                'proximal': 0.043,    # Thumb metacarpal
                'intermediate': 0.025, # Thumb proximal phalanx
                'distal': 0.028       # Thumb distal phalanx
            },
            'index': {
                'proximal': 0.030,    # Index metacarpal
                'intermediate': 0.027, # Index middle phalanx
                'distal': 0.020       # Index distal phalanx
            },
            'middle': {
                'proximal': 0.033,    # Middle metacarpal
                'intermediate': 0.027, # Middle middle phalanx
                'distal': 0.020       # Middle distal phalanx
            },
            'ring': {
                'proximal': 0.030,    # Ring metacarpal
                'intermediate': 0.027, # Ring middle phalanx
                'distal': 0.020       # Ring distal phalanx
            },
            'pinky': {
                'proximal': 0.024,    # Pinky metacarpal
                'intermediate': 0.019, # Pinky middle phalanx
                'distal': 0.012       # Pinky distal phalanx
            }
        }

        # Define finger base positions relative to wrist center (in meters)
        # These represent where each finger attaches to the palm
        self.finger_base_positions = {
            'left': {
                'thumb': np.array([-0.008, -0.02, 0.047]),   # Forward, down, and left
                'index': np.array([-0.091, 0, 0.031]),       # Forward and slightly left
                'middle': np.array([-0.093, 0, 0.006]),     # Forward and center
                'ring': np.array([-0.091, 0, -0.006]),       # Forward and slightly right
                'pinky': np.array([-0.087, 0, -0.026])      # Forward and more right
            },
            'right': {  # Mirror the Z coordinates for right hand
                'thumb': np.array([-0.008, -0.02, -0.047]),  # Forward, down, and right
                'index': np.array([-0.091, 0, -0.031]),      # Forward and slightly right
                'middle': np.array([-0.093, 0, -0.006]),     # Forward and center
                'ring': np.array([-0.091, 0, 0.006]),        # Forward and slightly left
                'pinky': np.array([-0.087, 0, 0.026])       # Forward and more left
            }
        }

        # 预计算基础变换矩阵
        self.base_transforms = {'left': {}, 'right': {}}
        for hand in ['left', 'right']:
            for finger, pos in self.finger_base_positions[hand].items():
                T = np.eye(4, dtype=np.float64)
                T[:3, 3] = pos
                self.base_transforms[hand][finger] = T

        # 预计算指尖索引映射
        self.fingertip_indices = [3, 6, 9, 12, 15]
        self.finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

        # 预分配工作矩阵，避免重复分配
        self._work_matrix = np.eye(4, dtype=np.float64)
        self._work_rotation = np.zeros((3, 3), dtype=np.float64)

    def check_ready(self):
        return self._is_ready.value and not self._stop_event.is_set()

    def _initialize(self):
        """Initialize TCP connection to VRTrix gloves server"""
        try:
            # Create TCP socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.config.connection_timeout)

            # 设置TCP_NODELAY减少延迟
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

            # 增加接收缓冲区大小
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)

            # Connect to server
            server_address = (self.config.ip, self.config.port)
            print(f"Connecting to VRTrix server at {server_address}")
            self.sock.connect(server_address)

            # Set to quaternion mode
            self._send_command(CMD_SetRotationFormat, 'B', 1)

            # Auto calibrate if requested
            if self.config.auto_calibrate_on_start:
                print("Performing T-Pose calibration...")
                if self.config.enable_right:
                    self._send_command(CMD_TPoseCalibration, 'R', 0)
                if self.config.enable_left:
                    self._send_command(CMD_TPoseCalibration, 'L', 0)
                time.sleep(2.0)

            # Initialize data storage
            self.latest_right_data = None
            self.latest_left_data = None
            self.last_update_time = time.time()

            # 如果使用线程模式，启动接收线程
            if self.config.use_threading:
                self.receiver_thread = threading.Thread(target=self._receiver_loop, daemon=True)
                self.receiver_thread.start()

            self._is_ready.value = True

        except Exception as e:
            print(f"Failed to initialize VRTrix controller: {e}")
            traceback.print_exc()
            raise

    def _send_command(self, cmd: int, hand: str, param: int):
        """Send command to gloves"""
        message = struct.pack('<hshhffff', cmd, hand.encode('utf-8'), param, 0, 0, 0, 0, 0)
        self.sock.sendall(message)

    def _receiver_loop(self):
        """独立线程用于接收数据，减少主线程负担"""
        self.sock.setblocking(False)
        buffer = bytearray()

        while not self._stop_event.is_set():
            try:
                # 尝试读取数据
                data = self.sock.recv(4096)  # 一次读取更多数据
                if not data:
                    time.sleep(0.001)
                    continue

                buffer.extend(data)

                # 处理完整的数据包
                while len(buffer) >= 273:
                    packet = bytes(buffer[:273])
                    buffer = buffer[273:]

                    parsed_data = self._parse_glove_data(packet)
                    if parsed_data:
                        with self.data_lock:
                            if parsed_data['hand'] == 'R' and self.config.enable_right:
                                self.latest_right_data = parsed_data
                            elif parsed_data['hand'] == 'L' and self.config.enable_left:
                                self.latest_left_data = parsed_data

            except socket.error:
                time.sleep(0.001)
            except Exception as e:
                if self.config.verbose:
                    print(f"Error in receiver loop: {e}")
                time.sleep(0.001)

    def compute_all_joint_poses_optimized(self, all_joints_quat, hand='right'):
        """
        优化的关节姿态计算，减少矩阵分配和计算

        返回:
            joint_poses: 16x4x4 array of pose matrices
            fingertip_poses: 5x4x4 array of fingertip pose matrices
        """
        joint_poses = np.zeros((16, 4, 4), dtype=np.float64)
        fingertip_poses = np.zeros((5, 4, 4), dtype=np.float64)

        # Wrist is identity
        joint_poses[0] = np.eye(4)

        # Get wrist rotation - 直接使用scipy的高效实现
        wrist_rot = R.from_quat(all_joints_quat[0]).as_matrix()

        # 批量转换所有四元数为旋转矩阵（更高效）
        all_rotations = R.from_quat(all_joints_quat).as_matrix()

        # Process each finger
        finger_data = [
            ('thumb', [1, 2, 3], 0),
            ('index', [4, 5, 6], 1),
            ('middle', [7, 8, 9], 2),
            ('ring', [10, 11, 12], 3),
            ('pinky', [13, 14, 15], 4)
        ]

        for finger_name, joint_indices, finger_idx in finger_data:
            segments = self.segment_lengths[finger_name]
            link_lengths = [segments['proximal'], segments['intermediate'], segments['distal']]

            # 使用预计算的基础变换
            T_current = self.base_transforms[hand][finger_name].copy()

            # 计算相对旋转
            R_prev = wrist_rot

            for j, (joint_idx, length) in enumerate(zip(joint_indices, link_lengths)):
                R_curr = all_rotations[joint_idx]
                R_rel = R_prev.T @ R_curr

                # 构建变换矩阵
                T_joint = np.eye(4)
                T_joint[:3, :3] = R_rel

                if j > 0:
                    T_joint[:3, 3] = [-link_lengths[j-1], 0, 0]

                T_current = T_current @ T_joint
                joint_poses[joint_idx] = T_current.copy()

                # 如果是指尖关节，计算指尖位置
                if j == 2:  # Distal joint
                    T_tip = np.eye(4)
                    T_tip[:3, 3] = [-length, 0, 0]
                    fingertip_poses[finger_idx] = T_current @ T_tip

                R_prev = R_curr

        return joint_poses, fingertip_poses

    def _parse_glove_data(self, data: bytes) -> Optional[Dict[str, Any]]:
        """优化的数据解析，减少内存分配"""
        try:
            # 快速检查
            if len(data) != 273 or data[0:6] != b'VRTRIX':
                return None

            hand_char = chr(data[6])
            if hand_char not in ['L', 'R']:
                return None

            # 使用numpy的fromstring进行批量解析（更快）
            # 跳过头部(9字节)，读取16个四元数(每个16字节)
            quat_data = np.frombuffer(data, dtype=np.float32, count=64, offset=9).reshape(16, 4)

            # 转换为scipy格式 [x, y, z, w]
            joints_quat = np.column_stack((quat_data[:, 1:4], quat_data[:, 0]))

            # 解析元数据
            metadata_offset = 265
            radio_strength, = struct.unpack('h', data[metadata_offset:metadata_offset+2])
            battery, = struct.unpack('f', data[metadata_offset+2:metadata_offset+6])
            cal_score, = struct.unpack('h', data[metadata_offset+6:metadata_offset+8])

            return {
                'hand': hand_char,
                'all_joints_quat': joints_quat.astype(np.float64),  # 保持精度
                'radio_strength': radio_strength,
                'battery': battery,
                'cal_score': cal_score,
                'timestamp': time.time()
            }

        except Exception as e:
            if self.config.verbose:
                print(f"Error parsing glove data: {e}")
            return None

    def _update(self):
        """优化的更新函数"""
        try:
            timestamp = time.time()

            if self.config.use_threading:
                # 使用线程模式，直接从最新数据读取
                with self.data_lock:
                    right_data = self.latest_right_data.copy() if self.latest_right_data else None
                    left_data = self.latest_left_data.copy() if self.latest_left_data else None
            else:
                # 非线程模式，读取有限数量的包
                self.sock.setblocking(False)
                packets_processed = 0

                while packets_processed < self.config.max_packets_per_update:
                    try:
                        data = self.sock.recv(273)
                        if not data or len(data) != 273:
                            break

                        parsed_data = self._parse_glove_data(data)
                        if parsed_data:
                            if parsed_data['hand'] == 'R' and self.config.enable_right:
                                self.latest_right_data = parsed_data
                            elif parsed_data['hand'] == 'L' and self.config.enable_left:
                                self.latest_left_data = parsed_data

                        packets_processed += 1

                    except socket.error:
                        break

                right_data = self.latest_right_data
                left_data = self.latest_left_data

            # 准备反馈数据
            state_dict = {
                "receive_timestamp": np.array([timestamp], dtype=np.float64),
                "timestamp": np.array([timestamp], dtype=np.float64)
            }

            # 处理右手数据
            if right_data and (timestamp - right_data['timestamp']) < 0.1:
                right_joints_quat = right_data['all_joints_quat']
                joint_poses, fingertip_poses = self.compute_all_joint_poses_optimized(right_joints_quat, 'right')

                # 转换回[w,x,y,z]格式
                right_joints_quat_wxyz = np.column_stack((
                    right_joints_quat[:, 3],
                    right_joints_quat[:, 0:3]
                ))

                state_dict.update({
                    "right_joints_quat": right_joints_quat_wxyz,
                    "right_joint_poses": joint_poses,
                    "right_fingertip_poses": fingertip_poses
                })
            else:
                state_dict.update({
                    "right_joints_quat": np.zeros((16, 4), dtype=np.float64),
                    "right_joint_poses": np.zeros((16, 4, 4), dtype=np.float64),
                    "right_fingertip_poses": np.zeros((5, 4, 4), dtype=np.float64)
                })

            # 处理左手数据
            if left_data and (timestamp - left_data['timestamp']) < 0.1:
                left_joints_quat = left_data['all_joints_quat']
                joint_poses, fingertip_poses = self.compute_all_joint_poses_optimized(left_joints_quat, 'left')

                # 转换回[w,x,y,z]格式
                left_joints_quat_wxyz = np.column_stack((
                    left_joints_quat[:, 3],
                    left_joints_quat[:, 0:3]
                ))

                state_dict.update({
                    "left_joints_quat": left_joints_quat_wxyz,
                    "left_joint_poses": joint_poses,
                    "left_fingertip_poses": fingertip_poses
                })
            else:
                state_dict.update({
                    "left_joints_quat": np.zeros((16, 4), dtype=np.float64),
                    "left_joint_poses": np.zeros((16, 4, 4), dtype=np.float64),
                    "left_fingertip_poses": np.zeros((5, 4, 4), dtype=np.float64)
                })

            self.feedback_queue.put(state_dict)

        except Exception as e:
            if self.config.verbose:
                print(f"Error in VRTrix update: {e}")
                traceback.print_exc()

    def _process_commands(self):
        """Process any pending commands"""
        pass

    def _close(self):
        """Clean up resources"""
        try:
            self._stop_event.set()

            if self.config.use_threading and self.receiver_thread:
                self.receiver_thread.join(timeout=1.0)

            if hasattr(self, 'sock'):
                self.sock.close()
        except Exception as e:
            print(f"Error closing VRTrix controller: {e}")

    def reset(self):
        """Reset controller state"""
        if self.config.use_threading:
            with self.data_lock:
                self.latest_right_data = None
                self.latest_left_data = None
        else:
            self.latest_right_data = None
            self.latest_left_data = None

    def trigger_haptics(self, hand: str, duration_ms: int = 500):
        """Trigger haptic feedback on specified hand"""
        if self._is_ready.value:
            self._send_command(CMD_TriggerHaptics, hand, duration_ms)


#  test
####################################################


def format_quaternion(q):
    """Format quaternion for display"""
    return f"({q[0]:6.3f}, {q[1]:6.3f}, {q[2]:6.3f}, {q[3]:6.3f})"


def format_pose_matrix(pose):
    """Format 4x4 pose matrix for display"""
    lines = []
    for i in range(4):
        line = "  [" + ", ".join(f"{pose[i,j]:7.3f}" for j in range(4)) + "]"
        lines.append(line)
    return "\n".join(lines)


class HandVisualizer:
    """3D hand visualization handler that maintains persistent figure"""

    def __init__(self, show_all_joints=False):
        """Initialize the visualization figure and axes"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        self.plt = plt
        self.show_all_joints = show_all_joints
        self.fig = plt.figure(figsize=(14, 7))

        # Create subplots for left and right hands
        self.ax_left = self.fig.add_subplot(121, projection='3d')
        self.ax_right = self.fig.add_subplot(122, projection='3d')

        # Configure axes with proper viewing angles
        # Left hand viewed from right side
        self.ax_left.view_init(elev=20, azim=-60)
        self.ax_left.set_title('Left Hand')

        # Right hand viewed from left side
        self.ax_right.view_init(elev=20, azim=60)
        self.ax_right.set_title('Right Hand')

        # Configure both axes
        for ax in [self.ax_left, self.ax_right]:
            ax.set_xlabel('X (Forward)')
            ax.set_ylabel('Y (Up)')
            ax.set_zlabel('Z (Lateral)')
            ax.set_xlim([-0.15, 0.15])
            ax.set_ylim([-0.15, 0.15])
            ax.set_zlim([-0.15, 0.15])

        # Show the figure
        plt.ion()  # Interactive mode
        plt.show()

    def update(self, data):
        """Update the visualization with new data"""
        # Clear previous plot elements
        for ax in [self.ax_right, self.ax_left]:
            ax.clear()

            # Reconfigure axis after clearing
            ax.set_xlabel('X (Forward)')
            ax.set_ylabel('Y (Up)')
            ax.set_zlabel('Z (Lateral)')
            ax.set_xlim([-0.15, 0.15])
            ax.set_ylim([-0.15, 0.15])
            ax.set_zlim([-0.15, 0.15])

        # Update hands
        self._update_hand(self.ax_left, data['left_joint_poses'], data['left_fingertip_poses'],
                         data['left_joints_quat'], 'Left Hand', 'left')
        self._update_hand(self.ax_right, data['right_joint_poses'], data['right_fingertip_poses'],
                         data['right_joints_quat'], 'Right Hand', 'right')

        # Redraw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _update_hand(self, ax, joint_poses, fingertip_poses, joints_quat, title, hand_type):
        """Update a single hand subplot with full kinematic chain visualization"""
        ax.set_title(title)

        # Set view angle based on hand type
        if hand_type == 'left':
            ax.view_init(elev=20, azim=-60)
        else:
            ax.view_init(elev=20, azim=60)

        # Plot wrist at origin
        ax.scatter([0], [0], [0], c='red', s=100, label='Wrist', marker='o')

        # Add coordinate frame at wrist
        axis_length = 0.03
        ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', arrow_length_ratio=0.3, linewidth=2)    # X axis
        ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', arrow_length_ratio=0.3, linewidth=2)  # Y axis
        ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.3, linewidth=2)   # Z axis

        # Define joint groups and colors
        finger_info = [
            ('Thumb', [1, 2, 3], 'orange'),
            ('Index', [4, 5, 6], 'yellow'),
            ('Middle', [7, 8, 9], 'green'),
            ('Ring', [10, 11, 12], 'cyan'),
            ('Pinky', [13, 14, 15], 'purple')
        ]

        # Plot joints and connections
        for finger_name, joint_indices, color in finger_info:
            # Get positions of all joints in this finger
            positions = []

            # Add wrist position as starting point
            positions.append([0, 0, 0])

            # Add positions of all joints
            for joint_idx in joint_indices:
                pose = joint_poses[joint_idx]
                positions.append(pose[:3, 3])

            # Add fingertip position
            finger_idx = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky'].index(finger_name)
            fingertip_pose = fingertip_poses[finger_idx]
            positions.append(fingertip_pose[:3, 3])

            positions = np.array(positions)

            # Plot connections between joints
            for i in range(len(positions) - 1):
                ax.plot([positions[i, 0], positions[i+1, 0]],
                       [positions[i, 1], positions[i+1, 1]],
                       [positions[i, 2], positions[i+1, 2]],
                       c=color, alpha=0.7, linewidth=2)

            # Plot joints if requested
            if self.show_all_joints:
                # Plot intermediate joints
                for i, joint_idx in enumerate(joint_indices):
                    joint_type = ['Proximal', 'Intermediate', 'Distal'][i]
                    ax.scatter(*positions[i+1], c=color, s=40, alpha=0.6, marker='o')

                    # Add small coordinate frame at each joint
                    if i < 2:  # Don't show frame for distal joint to reduce clutter
                        R_joint = joint_poses[joint_idx][:3, :3]
                        pos = positions[i+1]
                        joint_axis_length = 0.015

                        ax.quiver(pos[0], pos[1], pos[2],
                                 -R_joint[0,0]*joint_axis_length,
                                 -R_joint[1,0]*joint_axis_length,
                                 -R_joint[2,0]*joint_axis_length,
                                 color='red', alpha=0.3, arrow_length_ratio=0.3)
                        ax.quiver(pos[0], pos[1], pos[2],
                                 R_joint[0,1]*joint_axis_length,
                                 R_joint[1,1]*joint_axis_length,
                                 R_joint[2,1]*joint_axis_length,
                                 color='green', alpha=0.3, arrow_length_ratio=0.3)

            # Always plot fingertip
            tip_pos = positions[-1]
            ax.scatter(*tip_pos, c=color, s=80, label=finger_name, marker='*')

            # Add coordinate frame at fingertip
            R_tip = fingertip_pose[:3, :3]
            tip_axis_length = 0.02

            # X axis (finger extension direction - negative in VRTrix)
            ax.quiver(tip_pos[0], tip_pos[1], tip_pos[2],
                     -R_tip[0,0]*tip_axis_length,
                     -R_tip[1,0]*tip_axis_length,
                     -R_tip[2,0]*tip_axis_length,
                     color='red', alpha=0.5, arrow_length_ratio=0.3)
            # Y axis (dorsal direction)
            ax.quiver(tip_pos[0], tip_pos[1], tip_pos[2],
                     R_tip[0,1]*tip_axis_length,
                     R_tip[1,1]*tip_axis_length,
                     R_tip[2,1]*tip_axis_length,
                     color='green', alpha=0.5, arrow_length_ratio=0.3)
            # Z axis (lateral direction)
            ax.quiver(tip_pos[0], tip_pos[1], tip_pos[2],
                     R_tip[0,2]*tip_axis_length,
                     R_tip[1,2]*tip_axis_length,
                     R_tip[2,2]*tip_axis_length,
                     color='blue', alpha=0.5, arrow_length_ratio=0.3)

        ax.legend(loc='upper right', fontsize='small')
        ax.set_box_aspect([1,1,1])  # Equal aspect ratio

    def close(self):
        """Close the visualization window"""
        self.plt.close(self.fig)


def display_glove_data(data, controller=None, show_poses=False, show_all_joints=False):
    """Display glove data in a formatted way"""
    print("\n" + "="*80)
    print(f"Timestamp: {data['timestamp'][0]:.3f}")

    # Show battery and signal info if available
    if controller and hasattr(controller, 'latest_right_data') and controller.latest_right_data:
        print(f"Right Hand - Battery: {controller.latest_right_data['battery']:.1f}%, "
              f"Signal: {controller.latest_right_data['radio_strength']}dB")
    if controller and hasattr(controller, 'latest_left_data') and controller.latest_left_data:
        print(f"Left Hand  - Battery: {controller.latest_left_data['battery']:.1f}%, "
              f"Signal: {controller.latest_left_data['radio_strength']}dB")

    print("-"*80)

    # Joint names for all 16 joints
    all_joint_names = [
        'Wrist',
        'Thumb_Proximal', 'Thumb_Intermediate', 'Thumb_Distal',
        'Index_Proximal', 'Index_Intermediate', 'Index_Distal',
        'Middle_Proximal', 'Middle_Intermediate', 'Middle_Distal',
        'Ring_Proximal', 'Ring_Intermediate', 'Ring_Distal',
        'Pinky_Proximal', 'Pinky_Intermediate', 'Pinky_Distal'
    ]

    # Display right hand data
    print("RIGHT HAND:")
    if show_all_joints:
        for i, joint_name in enumerate(all_joint_names):
            quat = data['right_joints_quat'][i]
            print(f"  {joint_name:20s}: {format_quaternion(quat)}")
    else:
        # Show only key joints
        key_indices = [0, 3, 6, 9, 12, 15]  # Wrist and fingertips
        key_names = ['Wrist', 'Thumb_Tip', 'Index_Tip', 'Middle_Tip', 'Ring_Tip', 'Pinky_Tip']
        for idx, name in zip(key_indices, key_names):
            quat = data['right_joints_quat'][idx]
            print(f"  {name:12s}: {format_quaternion(quat)}")

    if show_poses:
        print("\nRIGHT HAND FINGERTIP POSES (relative to wrist):")
        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        for i, name in enumerate(finger_names):
            pose = data['right_fingertip_poses'][i]
            translation = pose[:3, 3]
            print(f"\n{name}: Position = ({translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}) m")
            if show_all_joints:  # Show full matrix only in verbose mode
                print(format_pose_matrix(pose))

    print("\nLEFT HAND:")
    if show_all_joints:
        for i, joint_name in enumerate(all_joint_names):
            quat = data['left_joints_quat'][i]
            print(f"  {joint_name:20s}: {format_quaternion(quat)}")
    else:
        # Show only key joints
        key_indices = [0, 3, 6, 9, 12, 15]  # Wrist and fingertips
        key_names = ['Wrist', 'Thumb_Tip', 'Index_Tip', 'Middle_Tip', 'Ring_Tip', 'Pinky_Tip']
        for idx, name in zip(key_indices, key_names):
            quat = data['left_joints_quat'][idx]
            print(f"  {name:12s}: {format_quaternion(quat)}")

    if show_poses:
        print("\nLEFT HAND FINGERTIP POSES (relative to wrist):")
        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        for i, name in enumerate(finger_names):
            pose = data['left_fingertip_poses'][i]
            translation = pose[:3, 3]
            print(f"\n{name}: Position = ({translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}) m")
            if show_all_joints:  # Show full matrix only in verbose mode
                print(format_pose_matrix(pose))


@click.command()
@click.option('--ip', default='127.0.0.1', help='IP address of VRTrix server')
@click.option('--port', default=11002, help='Port of VRTrix server')
@click.option('--fps', default=60, help='Target FPS for data reading')
@click.option('--enable_left', default=True, help='Enable left hand')
@click.option('--enable_right', default=True, help='Enable right hand')
@click.option('--calibrate', default=True, help='Auto-calibrate on start')
@click.option('--verbose', default=False, help='Verbose output')
@click.option('--duration', default=0, help='Test duration in seconds (0 for infinite)')
@click.option('--show_poses', default=True, help='Show fingertip positions')
@click.option('--show_all_joints', default=True, help='Show all 16 joints (verbose mode)')
@click.option('--visualize', default=True, help='Enable 3D visualization')
@click.option('--visualize_all_joints', default=False, help='Visualize all joints in 3D')


def main(ip, port, fps, enable_left, enable_right, calibrate, verbose, duration, show_poses, show_all_joints, visualize, visualize_all_joints):
    """Test VRTrix data glove controller with full kinematics"""

    print(f"Connecting to VRTrix server at {ip}:{port}")
    print(f"Target FPS: {fps}")
    print(f"Left hand: {'Enabled' if enable_left else 'Disabled'}")
    print(f"Right hand: {'Enabled' if enable_right else 'Disabled'}")

    # Configure controller
    config = VRTrixControllerConfig(
        name="vrtrix_test",
        fps=fps,
        put_desired_frequency=fps,
        ip=ip,
        port=port,
        enable_left=enable_left,
        enable_right=enable_right,
        auto_calibrate_on_start=calibrate,
        verbose=verbose
    )

    # Create and start controller
    controller = VRTrixController(config)

    # Initialize visualizer if enabled
    visualizer = None
    if visualize:
        try:
            visualizer = HandVisualizer(show_all_joints=visualize_all_joints)
        except ImportError:
            print("Warning: matplotlib not available, disabling visualization")
            visualize = False

    try:
        # Start the controller process
        controller.start()

        # Wait for initialization
        print("Waiting for controller initialization...")
        for i in range(50):  # 5 second timeout
            if controller.check_ready():
                print("Controller ready!")
                break
            time.sleep(0.1)
        else:
            print("Controller failed to initialize!")
            return

        # If calibrating, give user instructions
        if calibrate:
            print("\nCalibration performed. Make sure hands are in T-pose position.")
            time.sleep(2.0)

        print("\nStarting data collection. Press Ctrl+C to stop.")
        print("="*80)

        # Main loop
        start_time = time.time()
        dt = 1.0 / fps
        frame_count = 0
        vis_update_interval = max(1, int(fps / 30))  # Update visualization at ~30 FPS

        while True:
            loop_start = time.time()

            # Check duration
            if duration > 0 and (loop_start - start_time) > duration:
                print(f"\nTest duration of {duration} seconds reached.")
                break

            # Get feedback data
            feedback = controller.get_feedback()

            if feedback is not None:
                frame_count += 1

                # Display data
                if verbose or frame_count % fps == 0:  # Show every second if not verbose
                    display_glove_data(feedback, controller, show_poses, show_all_joints)

                    # Calculate and display FPS
                    elapsed = time.time() - start_time
                    actual_fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"\nActual FPS: {actual_fps:.1f}")

                # Update visualization if enabled
                if visualizer and frame_count % vis_update_interval == 0:
                    visualizer.update(feedback)

            # Sleep to maintain target FPS
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nStopping controller...")

    finally:
        # Clean shutdown
        controller.stop()
        controller.join(timeout=2.0)

        # Display statistics
        total_time = time.time() - start_time
        print(f"\nTest completed:")
        print(f"  Total time: {total_time:.1f} seconds")
        print(f"  Total frames: {frame_count}")
        print(f"  Average FPS: {frame_count/total_time:.1f}")

        # Close visualizer if active
        if visualizer:
            visualizer.close()


if __name__ == '__main__':
    main()