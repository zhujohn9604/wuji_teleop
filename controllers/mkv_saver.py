from dataclasses import dataclass
from typing import Optional, Dict, List, Any
import multiprocessing as mp
import traceback
import os
import time
import numpy as np
from copy import deepcopy
import cv2

from .base import BaseControllerConfig, BaseController
from utils.moviepy_utils import MoviePyVideoStreamerWithTimestamp
from utils.print_utils import print_green, print_cyan, print_red
from utils.hdf5_utils import HDF5Appender
from utils.shared_memory import Empty


@dataclass
class MKVSaverControllerConfig(BaseControllerConfig):
    """
    Configuration for MKV video saver controller.
    """
    mkv_path: str = ""

    codec: str = "libx264" # libx264, ffv1
    preset: str = "medium"
    crf: int = 23
    ffv1_level: Optional[int] = None

    frame_width: int = None
    frame_height: int = None

    add_timestamp: bool = True

    hdf5_dataset_name: str = "float_array"
    hdf5_timestamp_dataset_name: str = "timestamp_array"

    hdf5_compression: str = "gzip"
    hdf5_compression_opts: int = 9
    hdf5_timestamp_file: Optional[str] = None

    def validate(self):
        super().validate()
        assert isinstance(self.mkv_path, str) and len(self.mkv_path) > 0, f"Invalid video path: {self.mkv_path}"

        self.hdf5_timestamp_file = f"{self.mkv_path}.timestamp.hdf5"

        assert self.codec in ["libx264", "ffv1"], f"Invalid codec: {self.codec}"
        assert self.preset in ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"], \
            f"Invalid preset: {self.preset}"
        assert isinstance(self.crf, int) and 0 <= self.crf <= 51, f"Invalid CRF value: {self.crf}"
        assert self.ffv1_level is None or (isinstance(self.ffv1_level, int) and 0 <= self.ffv1_level <= 3), \
            f"Invalid FFV1 level: {self.ffv1_level}"
        assert isinstance(self.frame_width, int) and self.frame_width > 0, f"Invalid frame width: {self.frame_width}"
        assert isinstance(self.frame_height, int) and self.frame_height > 0, f"Invalid frame height: {self.frame_height}"
        assert isinstance(self.add_timestamp, bool), f"Invalid add_timestamp value: {self.add_timestamp}"

        self.command_sample = {
            "img": np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8),
            'timestamp': np.zeros((1,), dtype=np.float64),
        }


class MKVSaverController(BaseController):
    """
    Controller for saving video data to MKV files.

    Uses the BaseController framework with multiprocessing to efficiently save
    video frames with timestamps.
    """
    config: MKVSaverControllerConfig

    def __init__(self, config: MKVSaverControllerConfig):
        super().__init__(config)

    def _initialize(self):
        assert self.config.mkv_path != "topass", f"Invalid video path: {self.config.mkv_path}"
        self.video_saver = MoviePyVideoStreamerWithTimestamp(
            output_file=self.config.mkv_path,
            fps=self.config.fps,
            frame_size=(self.config.frame_width, self.config.frame_height),
            codec=self.config.codec,
            preset=self.config.preset,
            crf=self.config.crf,
            add_timestamp=self.config.add_timestamp,
            timestamp_file=self.config.hdf5_timestamp_file,
            timestamp_compression=self.config.hdf5_compression,
            timestamp_compression_opts=self.config.hdf5_compression_opts,
            ffv1_level=self.config.ffv1_level,
        )

    def _process_commands(self):
        try:
            commands = self.command_queue.get_all()
            n_cmd = len(commands['img'])
        except Empty:
            n_cmd = 0
        except Exception as e:
            traceback.print_exc()
            raise e

        for i in range(n_cmd):
            img = commands['img'][i]
            timestamp = commands['timestamp'][i]

            self.video_saver.add_frame(img, timestamp)

    def _update(self):
        pass

    def _close(self):
        self.video_saver.release()

    def reset(self):
        pass

