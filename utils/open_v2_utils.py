from __future__ import annotations

import time
import serial
import os
import threading

from utils.crc_utils import validate_crc


__all__ = ['get_all_openv2_dev_id', 'OpenV2']

openv2_sleep_time = 1 / 200
openv2_baudrate = 115200
left_data_to_send = bytes([0x01, 0x03, 0x03, 0x80, 0x00, 0x04, 0x45, 0xA5])
right_data_to_send = bytes([0x02, 0x03, 0x03, 0x80, 0x00, 0x04, 0x45, 0x96])


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
    print(f"write to {dev}, msg: {left_data_to_send}")
    ser.write(left_data_to_send)
    time.sleep(openv2_sleep_time)
    response = ser.readline()
    print(f"{dev}, response: {response}")
    # start with b'\x01\x01'
    is_button = response.startswith(b'\x01\x03')
    ser.close()

    return is_button

def test_is_openv2_right(dev: str):
    ser = serial.Serial(dev, openv2_baudrate, timeout=0)
    # send 02 03 03 80 00 04 45 96 to serial
    print(f"write to {dev}, msg: {right_data_to_send}")
    ser.write(right_data_to_send)
    time.sleep(openv2_sleep_time)
    response = ser.readline()
    print(f"{dev}, response: {response}")
    # start with b'\x01\x01'
    is_button = response.startswith(b'\x02\x03')
    ser.close()

    return is_button

def get_all_openv2_dev_id(exclude_dev_ids: list[int]):
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

    return left_dev_id, right_dev_id

class OpenV2():
    def __init__(self, cfg, dev_id, logger, dev_name: str):
        """
        Args:
            cfg: Config object
            dev_id: Open device ID
        """
        self.cfg = cfg
        self.dev_id = dev_id
        self.logger = logger
        self.dev_name = dev_name

        # 这里的串口 ID
        self.tty_id = f"/dev/ttyUSB{dev_id}"

        self.angle_lock = threading.Lock()
        self.angle = None
        self.sec = None

        self._is_connected = False

        self.stop_update_flag = threading.Event()

        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.daemon = True
        self.update_thread.start()

    def _update_loop(self):
        self.serial_connection = serial.Serial(self.tty_id, openv2_baudrate, timeout=0)
        self._is_connected = True

        while not self.stop_update_flag.is_set():
            try:
                self._update_once()
            except Exception as e:
                self.logger.error(f"Open {self.tty_id} update error: {e}")
                self._is_connected = False
                break
            time.sleep(0.0001)  # Prevent busy-waiting

        print(f"Openv2 {self.tty_id} update loop exited")

    def _update_once(self):
        if self.dev_name == 'open_left':
            self.serial_connection.write(left_data_to_send)
        elif self.dev_name == 'open_right':
            self.serial_connection.write(right_data_to_send)

        time.sleep(openv2_sleep_time)

        response = self.serial_connection.readline()
        if len(response) == 13:
            if validate_crc(response):
                angle = calculate_angle(response)
                if angle is not None and \
                        0 <= angle <= 360:
                    with self.angle_lock:
                        if self.angle is not None:
                            angle_diff = min(
                                abs(angle - self.angle),
                                abs(angle - self.angle - 360),
                                abs(angle - self.angle + 360)
                            )
                            if angle_diff < 80:
                                self.sec = time.time()
                                self.angle = angle
                        else:
                            self.sec = time.time()
                            self.angle = angle
            else:
                self.logger.warning(f"recv {response}, Open {self.tty_id} CRC16 validation failed")

    def release(self):
        self.stop_update_flag.set()
        self.update_thread.join()

        self.serial_connection.close()

        print(f"Open {self.tty_id} released")

