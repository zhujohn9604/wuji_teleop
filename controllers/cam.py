from dataclasses import field, dataclass
from typing import Callable, Optional, Any, Dict
import multiprocessing as mp
import threading

import sys
import cv2
import numpy as np

from utils.img_utils import ImageViewer

from utils.shared_memory import (
    Empty, SharedMemoryQueue, SharedMemoryRingBuffer, SharedMemoryManager
)

from controllers.base import BaseController, BaseControllerConfig


@dataclass
class CamControllerConfig(BaseControllerConfig):
    receive_latency: float = 0.0
    img_transform_func: Optional[Callable] = None

    width: int = 1
    height: int = 1

    transformed_width: int = 1
    transformed_height: int = 1

    enable_transformed_frame: bool = False

    transformed_feedback_sample: Optional[Dict[str, np.ndarray]] = None

    enable_ui: bool = False
    enable_transformed_ui: bool = False

    def validate(self):
        super().validate()

        self.enable_transformed_frame = self.transformed_width != 1

        self.feedback_sample = {
            'img': np.zeros((self.height, self.width, 3), dtype=np.uint8),
            'receive_timestamp': np.zeros((1,), dtype=np.float64),
            'timestamp': np.zeros((1,), dtype=np.float64),
        }
        self.transformed_feedback_sample = {
            'img': np.zeros((self.transformed_height, self.transformed_width, 3), dtype=np.uint8),
            'receive_timestamp': np.zeros((1,), dtype=np.float64),
            'timestamp': np.zeros((1,), dtype=np.float64),
        }

class CamController(BaseController):
    config: CamControllerConfig

    def __init__(self, config: CamControllerConfig):
        super().__init__(config)

        if config.img_transform_func is not None:
            shm_manager = SharedMemoryManager()
            shm_manager = shm_manager.__enter__()
            self.transformed_feedback_queue = SharedMemoryRingBuffer.create_from_examples(
                shm_manager=shm_manager,
                examples=self.config.transformed_feedback_sample,
                get_max_k=config.get_max_k,
                get_time_budget=config.get_time_budget,
                put_desired_frequency=config.put_desired_frequency,
            )

        self.last_img: np.ndarray = None
        self.last_timestamp: float = None

    ################## abstract methods ##################
    def get_transformed_feedback(self):
        return self.transformed_feedback_queue.get()

    def _process_commands(self):
        pass

    def _initialize(self):
        if self.config.enable_ui:
            print(f"{self.config.name} initializing UI...")
            self.img_viewer = ImageViewer(window_name=self.config.name)
            print(f"{self.config.name} UI initialized.")

        if self.config.enable_transformed_ui and self.config.img_transform_func is not None:
            print(f"{self.config.name} initializing transformed UI...")
            self.transformed_img_viewer = ImageViewer(window_name=self.config.name + '_transformed')
            print(f"{self.config.name} transformed UI initialized.")

    def _update(self):
        assert self.last_img is not None and self.last_timestamp is not None, f"last_img or last_timestamp is None, last_img: {type(self.last_img)}, last_timestamp: {type(self.last_timestamp)}"
        self.feedback_queue.put({
            'img': self.last_img.copy(),
            'img_timestamp': np.array([self.last_timestamp,], dtype=np.float64),
            'img_receive_timestamp': np.array([self.last_timestamp - self.config.receive_latency,], dtype=np.float64),
        })

        if self.config.img_transform_func is not None:
            transformed_img = self.config.img_transform_func(self.last_img).copy()
            self.transformed_feedback_queue.put({
                'img': transformed_img.copy(),
                'img_timestamp': np.array([self.last_timestamp,], dtype=np.float64),
                'img_receive_timestamp': np.array([self.last_timestamp - self.config.receive_latency,], dtype=np.float64),
            })

        if self.config.enable_ui:
            rgb_img = self.last_img
            self.img_viewer.update(rgb_img)

        if self.config.enable_transformed_ui and self.config.img_transform_func is not None:
            self.transformed_img_viewer.update(transformed_img)

    def _close(self):
        pass

    def reset(self):
        pass
