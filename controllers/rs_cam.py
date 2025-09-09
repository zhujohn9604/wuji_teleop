from dataclasses import field, dataclass
import copy
import time
import cv2
import numpy as np
import pyrealsense2 as rs
from typing import Optional, Any, Dict
import multiprocessing as mp

from utils.img_utils import ImageViewer
from utils.print_utils import print_blue

from .cam import CamController, CamControllerConfig

@dataclass
class RSControllerConfig(CamControllerConfig):
    """
    Hard code to L515 setting currently
    """
    serial_no: str = ""  # RealSense serial number
    fps: int = 30

    raw_rgb_width: int = 1920
    raw_rgb_height: int = 1080

    width: int = 853
    height: int = 480

    gray_width: int = 1024
    gray_height: int = 768

    # umi dataset config transform
    transformed_width: int = 683
    transformed_height: int = 384

    def validate(self):
        super().validate()
        assert isinstance(self.serial_no, str) and len(self.serial_no) > 0, "Invalid RealSense serial number"

        self.feedback_sample = {
            'img': np.zeros((self.height, self.width, 3), dtype=np.uint8),
            'gray16': np.zeros((self.gray_height, self.gray_width), dtype=np.uint16),
            'img_receive_timestamp': np.zeros((1,), dtype=np.float64),
            'img_timestamp': np.zeros((1,), dtype=np.float64),
        }
        self.transformed_feedback_sample = {
            'img': np.zeros((self.transformed_height, self.transformed_width, 3), dtype=np.uint8),
            'img_receive_timestamp': np.zeros((1,), dtype=np.float64),
            'img_timestamp': np.zeros((1,), dtype=np.float64),
        }

class RSController(CamController):
    config: RSControllerConfig

    def __init__(self, config: RSControllerConfig):
        super().__init__(config)
        self.pipeline = None
        self.profile = None

        self.camera_info = mp.Manager().dict()

    ################## cls methods ##################
    def copy_camera_info(self):
        return copy.deepcopy(dict(self.camera_info))

    ################## abstract methods ##################
    def _process_commands(self):
        super()._process_commands()

    def _initialize(self):
        self.pipeline = rs.pipeline()
        config = rs.config()

        context = rs.context()
        devices = context.query_devices()
        if devices.size() == 0:
            raise RuntimeError("No RealSense devices connected")
        else:
            print_blue(f"Found {len(devices)} RealSense devices")

        config.enable_device(self.config.serial_no)

        # Enable RGB & Depth stream
        config.enable_stream(
            rs.stream.depth,
            self.config.gray_width,
            self.config.gray_height,
            rs.format.z16,
            self.config.fps
        )
        config.enable_stream(
            rs.stream.color,
            self.config.raw_rgb_width,
            self.config.raw_rgb_height,
            rs.format.rgb8,
            self.config.fps
        )

        # Start pipeline
        self.profile = self.pipeline.start(config)

        # get intrinsics
        device = self.profile.get_device()
        depth_sensor = device.first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        depth_stream = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        intrinsics = depth_stream.get_intrinsics()

        self.camera_info["width"] = intrinsics.width
        self.camera_info["height"] = intrinsics.height
        self.camera_info["fx"] = intrinsics.fx
        self.camera_info["fy"] = intrinsics.fy
        self.camera_info["cx"] = intrinsics.ppx
        self.camera_info["cy"] = intrinsics.ppy
        self.camera_info["depth_scale"] = depth_scale

        # device = self.profile.get_device()
        # device.hardware_reset()

        # Warmup camera
        for _ in range(3):
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError("Failed to get RGB frame during initialization")

        if self.config.enable_ui:
            self.img_viewer = ImageViewer(window_name=self.config.name)
            self.gray_img_viewer = ImageViewer(window_name=self.config.name + '_gray')

        if self.config.enable_transformed_ui and self.config.img_transform_func is not None:
            self.transformed_img_viewer = ImageViewer(window_name=self.config.name + '_transformed')
            if not self.config.enable_ui:
                self.gray_img_viewer = ImageViewer(window_name=self.config.name + '_gray')

    def _update(self):
        # Wait for frames
        frames = self.pipeline.wait_for_frames()

        img_receive_timestamp = time.time()

        # Get RGB frame
        color_frame = frames.get_color_frame()
        assert color_frame is not None, "Failed to get RGB frame"
        color_frame = np.asanyarray(color_frame.get_data()) # (H, W, 3)
        # resize (raw rgb to target rgb)
        assert color_frame.shape == (self.config.raw_rgb_height, self.config.raw_rgb_width, 3), \
            f"Color frame is not of shape ({self.config.raw_rgb_height}, {self.config.raw_rgb_width}, 3), got {color_frame.shape}"
        # (H, W, 3), (W, H)
        color_frame = cv2.resize(color_frame, (self.config.width, self.config.height), interpolation=cv2.INTER_LINEAR)
        assert color_frame.shape == (self.config.height, self.config.width, 3), \
            f"Color frame is not of shape ({self.config.height}, {self.config.width}, 3), got {color_frame.shape}"
        assert color_frame.dtype == np.uint8, f"Color frame is not of type uint8, got {color_frame.dtype}"

        # Get depth frame
        depth_frame = frames.get_depth_frame()
        assert depth_frame is not None, "Failed to get depth frame"
        depth_frame = np.asanyarray(depth_frame.get_data())
        assert depth_frame.dtype == np.uint16, f"Depth frame is not of type uint16, got {depth_frame.dtype}"

        self.feedback_queue.put({
            'img': color_frame.copy(),
            'gray16': depth_frame.copy(),
            'img_timestamp': np.array([img_receive_timestamp,], dtype=np.float64),
            'img_receive_timestamp': np.array([img_receive_timestamp - self.config.receive_latency,], dtype=np.float64),
        })

        if self.config.img_transform_func is not None:
            transformed_img = self.config.img_transform_func(color_frame).copy()
            self.transformed_feedback_queue.put({
                'img': transformed_img.copy(),
                'img_timestamp': np.array([img_receive_timestamp,], dtype=np.float64),
                'img_receive_timestamp': np.array([img_receive_timestamp - self.config.receive_latency,], dtype=np.float64),
            })

        if self.config.enable_ui:
            self.img_viewer.update(color_frame)
            self.gray_img_viewer.update(depth_frame)

        if self.config.enable_transformed_ui and self.config.img_transform_func is not None:
            self.transformed_img_viewer.update(transformed_img)
            if not self.config.enable_ui:
                self.gray_img_viewer.update(depth_frame)

    def _close(self):
        if hasattr(self, 'pipeline') and self.pipeline is not None:
            self.pipeline.stop()

        super()._close()

    def reset(self):
        super().reset()
