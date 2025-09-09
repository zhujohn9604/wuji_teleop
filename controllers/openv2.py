import time
import serial
import re
import os
import threading
import numpy as np
import multiprocessing
from dataclasses import dataclass
from typing import Optional, Any, List

from utils.crc_utils import validate_crc
from utils.print_utils import print_blue, print_green, print_yellow, print_red
from .base import BaseController, BaseControllerConfig


openv2_sleep_time = 1 / 200
openv2_baudrate = 115200
left_data_to_send = bytes([0x01, 0x03, 0x03, 0x80, 0x00, 0x04, 0x45, 0xA5])
right_data_to_send = bytes([0x02, 0x03, 0x03, 0x80, 0x00, 0x04, 0x45, 0x96])


@dataclass
class OpenV2ControllerConfig(BaseControllerConfig):
    receive_latency: float = 0.0

    left_dev_id: Optional[int] = None
    right_dev_id: Optional[int] = None

    left_min_angle: float = None
    left_max_angle: float = None

    right_min_angle: float = None
    right_max_angle: float = None

    def validate(self):
        super().validate()
        if self.left_dev_id is not None:
            assert isinstance(self.left_dev_id, int), f"Invalid left device ID: {self.left_dev_id}"
            assert self.left_min_angle is not None, "left_min_angle is None"
            assert self.left_max_angle is not None, "left_max_angle is None"
            assert 0 <= self.left_min_angle <= 360, f"Invalid left_min_angle: {self.left_min_angle}"
            assert 0 <= self.left_max_angle <= 360, f"Invalid left_max_angle: {self.left_max_angle}"
        if self.right_dev_id is not None:
            assert isinstance(self.right_dev_id, int), f"Invalid right device ID: {self.right_dev_id}"
            assert self.right_min_angle is not None, "right_min_angle is None"
            assert self.right_max_angle is not None, "right_max_angle is None"
            assert 0 <= self.right_min_angle <= 360, f"Invalid right_min_angle: {self.right_min_angle}"
            assert 0 <= self.right_max_angle <= 360, f"Invalid right_max_angle: {self.right_max_angle}"

        feedback_sample = {}
        if self.left_dev_id is not None:
            feedback_sample["left_openv2"] = {
                'width': np.zeros((1,), dtype=np.float64),
                'timestamp': np.zeros((1,), dtype=np.float64),
            }
        if self.right_dev_id is not None:
            feedback_sample["right_openv2"] = {
                'width': np.zeros((1,), dtype=np.float64),
                'timestamp': np.zeros((1,), dtype=np.float64),
            }

        self.feedback_sample = feedback_sample


def calculate_angle(data):
    """
    解析并计算编码器角度：
    - 从数据中提取第3到第6个字节（索引2到5）
    - 转换为十进制整数
    - 计算角度值

    Args:
        data (bytes): 接收到的完整字节数据

    Returns:
        float: 计算出的角度值（度）
    """
    if len(data) != 13:
        print("数据长度不符合预期")
        return None

    raw_angle_bytes = data[3:7]

    # 将字节数组转换为整数（大端字节序）
    raw_angle_value = int.from_bytes(raw_angle_bytes, byteorder='big')

    # 计算弧度 (rad) = 值 / 2^21
    radian_value = raw_angle_value / (2 ** 21)

    # 转换为角度 (°) = rad * 360
    angle_in_degrees = radian_value * 360

    return angle_in_degrees


def test_is_openv2_left(dev: str):
    ser = serial.Serial(dev, openv2_baudrate, timeout=0)
    # send 01 03 03 80 00 04 45 A5 to serial
    ser.write(left_data_to_send)
    time.sleep(openv2_sleep_time)
    response = ser.readline()
    print(f"{dev}, response: {response}")
    # start with b'\x01\x03'
    is_button = response.startswith(b'\x01\x03')
    ser.close()

    return is_button


def test_is_openv2_right(dev: str):
    ser = serial.Serial(dev, openv2_baudrate, timeout=0)
    # send 02 03 03 80 00 04 45 96 to serial
    ser.write(right_data_to_send)
    time.sleep(openv2_sleep_time)
    response = ser.readline()
    print(f"{dev}, response: {response}")
    # start with b'\x02\x03'
    is_button = response.startswith(b'\x02\x03')
    ser.close()

    return is_button


def get_all_openv2_dev_id(exclude_dev_ids: List[int] = []):
    """
    Get all OpenV2 device IDs connected to the system

    Args:
        exclude_dev_ids: List of device IDs to exclude

    Returns:
        Tuple[Optional[int], Optional[int]]: (left_dev_id, right_dev_id)
    """
    # check all devs are /dev/ttyUSB*
    devs = os.listdir('/dev')
    devs = [dev for dev in devs if dev.startswith('ttyUSB')]
    dev_ids = [int(dev.split('ttyUSB')[-1]) for dev in devs]
    dev_ids = [dev_id for dev_id in dev_ids if dev_id not in exclude_dev_ids]

    left_dev_id, right_dev_id = None, None
    for dev_id in dev_ids:
        if test_is_openv2_left(f'/dev/ttyUSB{dev_id}'):
            left_dev_id = dev_id
        elif test_is_openv2_right(f'/dev/ttyUSB{dev_id}'):
            right_dev_id = dev_id

    print(f"Found OpenV2 devices - Left: {left_dev_id}, Right: {right_dev_id}")
    return left_dev_id, right_dev_id


class OpenV2Controller(BaseController):
    """
    Controller class for OpenV2 encoders
    """
    config: OpenV2ControllerConfig

    def __init__(self, config: OpenV2ControllerConfig):
        super().__init__(config)

    ################## cls methods ##################
    def _update_openv2(self, connection: serial.Serial, side):
        """Update a single encoder reading"""
        # Send the appropriate request based on side
        if side == "left":
            connection.write(left_data_to_send)
        else:
            connection.write(right_data_to_send)

        time.sleep(openv2_sleep_time)

        response = connection.readline()
        timestamp = time.time()

        if len(response) == 13 and validate_crc(response):
            angle = calculate_angle(response)
            width = self._calc_width(angle, self.config.left_min_angle if side == "left" else self.config.right_min_angle,
                                    self.config.left_max_angle if side == "left" else self.config.right_max_angle)
            if side == "left":
                self.left_width = width
                self.left_timestamp = timestamp
            else:
                self.right_width = width
                self.right_timestamp = timestamp
        else:
            print_red(f"Invalid response from {side} encoder: {response}")

    def _calc_width(self, angle: float, min_angle: float, max_angle: float) -> float:
        """
        Calculate the width based on the angle and the min/max angles
        the angle is circular
        for example, min_angle = 200, max_angle = 50, the angle is in [200, 360] + [0, 50]

        return in [0, 1]
        """
        if min_angle < max_angle:
            angle_range = max_angle - min_angle
        else:
            angle_range = 360 - min_angle + max_angle

        if angle < min_angle:
            angle += 360

        width = (angle - min_angle) / angle_range

        return width

    ################## abstract methods ##################
    def _process_commands(self):
        pass  # No commands to process for encoders

    def _initialize(self):
        """Initialize the OpenV2 encoders"""
        if self.config.left_dev_id is not None:
            left_tty = f"/dev/ttyUSB{self.config.left_dev_id}"
            self.left_connection = serial.Serial(left_tty, openv2_baudrate, timeout=0)
            print(f"Left OpenV2 initialized on {left_tty}")
            self.left_width = 0
            self.left_timestamp = 0

        if self.config.right_dev_id is not None:
            right_tty = f"/dev/ttyUSB{self.config.right_dev_id}"
            self.right_connection = serial.Serial(right_tty, openv2_baudrate, timeout=0)
            print(f"Right OpenV2 initialized on {right_tty}")
            self.right_width = 0
            self.right_timestamp = 0

    def _update(self):
        """Update encoder readings"""
        state_dict = {}

        # Update left encoder
        if self.config.left_dev_id is not None:
            self._update_openv2(self.left_connection, "left")

            state_dict["left_openv2"] = {
                'width': np.array([self.left_width], dtype=np.float64),
                'timestamp': np.array([self.left_timestamp], dtype=np.float64),
            }

        # Update right encoder
        if self.config.right_dev_id is not None:
            self._update_openv2(self.right_connection, "right")

            state_dict["right_openv2"] = {
                'width': np.array([self.right_width], dtype=np.float64),
                'timestamp': np.array([self.right_timestamp], dtype=np.float64),
            }

        self.feedback_queue.put(state_dict)

    def _close(self):
        """Close connections to encoders"""
        if self.left_connection is not None:
            self.left_connection.close()
            print("Left OpenV2 connection closed")

        if self.right_connection is not None:
            self.right_connection.close()
            print("Right OpenV2 connection closed")

    def reset(self):
        self.left_width = 0
        self.right_width = 0
        self.left_timestamp = 0
        self.right_timestamp = 0
