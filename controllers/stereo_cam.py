from dataclasses import field, dataclass
from typing import Optional, Any, Dict, Callable
import multiprocessing as mp
import traceback

import cv2
import time
import subprocess
import re
import copy

import numpy as np

from .cam import CamController, CamControllerConfig

from utils.uvc_cam_utils import kill_process_by_device

@dataclass
class StereoCamControllerConfig(CamControllerConfig):
    cam_dev_id: int = None
    cv2_prop_auto_exposure: int = 3
    cv2_rgb_cam_prop_exposure: int = 250
    prevent_busy_waiting_time: int = 0.0001
    width: int = 1280
    height: int = 480

    transformed_width: int = 1280
    transformed_height: int = 480

    def validate(self):
        super().validate()
        assert isinstance(self.cam_dev_id, int) and self.cam_dev_id >= 0, f"Invalid camera device ID: {self.cam_dev_id}"


def get_all_stereo_dev_id(kill_process=True):
    # 获取 v4l2-ctl --list-devices 命令的输出
    process = subprocess.run(
        ['v4l2-ctl', '--list-devices'],
        capture_output=True,
        text=True
    )
    output = process.stdout + process.stderr  # 合并标准输出和错误输出
    devices = output.split('\n\n')  # 每个设备块用双换行符分隔

    left_dev_id, right_dev_id = None, None
    for device in devices:
        if "USB Camera1" in device:
            # 匹配设备路径，例如 /dev/video1, /dev/video8 等
            paths = re.findall(r'(/dev/video\d+)', device)
            if paths:
                right_dev_id = int(paths[0].split('video')[-1])
        elif "USB Camera" in device:
            paths = re.findall(r'(/dev/video\d+)', device)
            if paths:
                left_dev_id = int(paths[0].split('video')[-1])

    if kill_process:
        for dev_id in [left_dev_id, right_dev_id]:
            if dev_id is not None:
                kill_process_by_device(dev_id)
        time.sleep(1.0)

    return left_dev_id, right_dev_id

def reverse_stereo_img(img):
    rot_img = cv2.rotate(img, cv2.ROTATE_180)  # (480, 1280, 3)

    left_img = rot_img[0:480, 640:1280]  # (480, 640, 3)
    right_img = rot_img[0:480, 0:640]  # (480, 640, 3)
    final_img = np.concatenate((left_img, right_img), axis=1)

    return final_img

class StereoCamController(CamController):
    config: StereoCamControllerConfig

    def __init__(self, config: StereoCamControllerConfig):
        super().__init__(config)

    ################## cls methods ##################

    ################## abstract methods ##################
    def _process_commands(self):
        super()._process_commands()

    def _initialize(self):
        self.cap: cv2.VideoCapture = None
        while self.cap is None or \
                not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.config.cam_dev_id, cv2.CAP_V4L)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            for _ in range(3):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
                time.sleep(0.1)
            for _ in range(3):
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, self.config.cv2_prop_auto_exposure)
                time.sleep(0.1)
            for _ in range(3):
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.config.cv2_rgb_cam_prop_exposure)
                time.sleep(0.1)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

            time.sleep(0.1)  # Wait for camera to initialize

        for _ in range(10):
            ret, img = self.cap.read()
            assert img is not None and ret, f"Failed to read image from camera {self.config.cam_dev_id}"

        super()._initialize()

    def _update(self):
        ret, img = self.cap.read()
        img_receive_timestamp = time.time()
        assert img is not None and ret, f"Failed to read image from camera {self.config.cam_dev_id}"

        self.last_img = cv2.cvtColor(reverse_stereo_img(img), cv2.COLOR_BGR2RGB)
        self.last_timestamp = img_receive_timestamp

        super()._update()

    def _close(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

        super()._close()

    def reset(self):
        super().reset()
