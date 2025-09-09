import copy
from dataclasses import field, dataclass
from typing import Callable, Optional, Any, Dict
import multiprocessing as mp
import traceback

import sys
import os
import time
from ctypes import *

import cv2
import numpy as np

import re
import subprocess

from .cam import CamController, CamControllerConfig

from utils.img_utils import ImageViewer


@dataclass
class USBCamControllerConfig(CamControllerConfig):
    dev_id: int = None

    crop_func: Callable = None

    cap_prop_auto_exposure: Optional[int] = None
    cap_prop_exposure: Optional[int] = None

    def validate(self):
        super().validate()
        assert self.dev_id is not None, "MonoControllerConfig: dev_id must be set"


def get_all_osmo_dev_id():
    """
    Args:
        usb_name (str): The name of the USB camera to search for
            e.g., "Qualcomm, Inc. OsmoAction5pro_SN:04702F4E".
    """
    osmo_usb_name_prefix = "OsmoAction5pro_SN"

    process = subprocess.run(
        ['v4l2-ctl', '--list-devices'],
        capture_output=True,
        text=True
    )
    output = process.stdout + process.stderr
    devices = output.split('\n\n')

    serial_numbers = []
    dev_ids = []

    for device in devices:
        if osmo_usb_name_prefix in device:
            print(f"device: {device}")
            # paths = re.findall(r'(/dev/video\d+)', device)
            # if paths:
            #     right_dev_id = int(paths[0].split('video')[-1])

            match = re.search(r'OsmoAction5pro_SN:(\S+):', device)
            if match:
                serial_number = match.group(1)
                serial_numbers.append(serial_number)

            paths = re.findall(r'(/dev/video\d+)', device)
            if paths:
                dev_id = int(paths[0].split('video')[-1])
                dev_ids.append(dev_id)

    assert len(serial_numbers) == len(dev_ids), f"Mismatch in number of serial numbers and device IDs found: {len(serial_numbers)} vs {len(dev_ids)}"

    return serial_numbers, dev_ids

def get_osmo_dev_id_by_serial(serial: str):
    """
    Args:
        serial (str): The serial number of the USB camera to search for
            e.g., "04702F4E".
    """
    osmo_usb_name_prefix = "OsmoAction5pro_SN"

    process = subprocess.run(
        ['v4l2-ctl', '--list-devices'],
        capture_output=True,
        text=True
    )
    output = process.stdout + process.stderr
    devices = output.split('\n\n')

    for device in devices:
        if osmo_usb_name_prefix in device and serial in device:
            print(f"device: {device}")
            paths = re.findall(r'(/dev/video\d+)', device)
            if paths:
                right_dev_id = int(paths[0].split('video')[-1])
                return right_dev_id

    raise ValueError(f"Device with serial {serial} not found.")

def get_all_usb_cam_dev_id(usb_name: str):
    """
    Args:
        usb_name (str): The name of the USB camera to search for
    """
    process = subprocess.run(
        ['v4l2-ctl', '--list-devices'],
        capture_output=True,
        text=True
    )
    output = process.stdout + process.stderr
    devices = output.split('\n\n')

    dev_ids = []

    for device in devices:
        if usb_name in device:
            print(f"device: {device}")
            paths = re.findall(r'(/dev/video\d+)', device)
            if paths:
                right_dev_id = int(paths[0].split('video')[-1])
                dev_ids.append(right_dev_id)

    return dev_ids

class USBCamController(CamController):
    config: USBCamControllerConfig

    def __init__(self, config: USBCamControllerConfig):
        super().__init__(config)

        self._is_ready = self.mp_manager.Value('b', True)

    ################## cls methods ##################
    def check_ready(self):
        return self._is_ready.value and not self._stop_event.is_set()

    ################## abstract methods ##################
    def _initialize(self):
        self.cap: cv2.VideoCapture = None
        while self.cap is None or \
                not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.config.dev_id, cv2.CAP_V4L)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            for _ in range(3):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
                time.sleep(0.1)

            if self.config.cap_prop_auto_exposure is not None:
                for _ in range(3):
                    self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, self.config.cap_prop_auto_exposure)
                    time.sleep(0.1)

            if self.config.cap_prop_exposure is not None:
                for _ in range(3):
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, self.config.cap_prop_exposure)
                    time.sleep(0.1)

            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

            time.sleep(0.1)  # Wait for camera to initialize

        color_image = None
        while color_image is None:
            ret, color_image = self.cap.read()

        if self.config.enable_ui:
            self.image_viewer = ImageViewer(window_name=self.config.name)
            if self.config.enable_transformed_frame:
                self.transformed_image_viewer = ImageViewer(
                    window_name=f"{self.config.name} (transformed)",
                )

    def _process_commands(self):
        pass

    def _update(self):
        ret, frame = self.cap.read() # (H, W, C)
        # bgr to rgb
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = time.time()

        # resize
        transformed_frame = cv2.resize(
            frame,
            (self.config.transformed_width, self.config.transformed_height),
            interpolation=cv2.INTER_LINEAR
        )

        self.feedback_queue.put({
            'img': frame.copy(),
            'timestamp': np.array([timestamp,], dtype=np.float64),
            'receive_timestamp': np.array([timestamp - self.config.receive_latency,], dtype=np.float64),
        })

        self.transformed_feedback_queue.put({
            'img': transformed_frame.copy(),
            'timestamp': np.array([timestamp,], dtype=np.float64),
            'receive_timestamp': np.array([timestamp - self.config.receive_latency,], dtype=np.float64),
        })

        if self.config.enable_ui:
            self.image_viewer.update(frame)
            if self.config.enable_transformed_frame:
                self.transformed_image_viewer.update(transformed_frame)

    def _close(self):
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def reset(self):
        pass
