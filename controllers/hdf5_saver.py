from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import multiprocessing as mp
import os
import traceback
import time
import numpy as np
from copy import deepcopy

from .base import BaseControllerConfig, BaseController
from utils.hdf5_utils import HDF5Appender
from utils.print_utils import print_green, print_cyan, print_red
from utils.shared_memory import Empty


@dataclass
class HDF5SaverControllerConfig(BaseControllerConfig):
    """
    Configuration for HDF5 data saver controller.
    """
    hdf5_path: str = ""
    dataset_name: str = "float_array"
    timestamp_dataset_name: str = "timestamp_array"

    hdf5_compression: str = "gzip"
    hdf5_compression_opts: int = 9

    sample_shape: Optional[Tuple[int]] = None
    sample_dtype: np.dtype = None

    def validate(self):
        super().validate()
        assert isinstance(self.hdf5_path, str) and len(self.hdf5_path) > 0, f"Invalid hdf5_path: {self.hdf5_path}"

        assert self.sample_shape is not None, f"sample_shape must be specified, but got {self.sample_shape}"
        assert isinstance(self.sample_shape, tuple), f"sample_shape must be a tuple, but got {type(self.sample_shape)}"
        assert len(self.sample_shape) > 0, f"sample_shape must be a non-empty tuple, but got {self.sample_shape}"
        assert all(isinstance(dim, int) and dim > 0 for dim in self.sample_shape), \
            f"sample_shape must contain positive integers, but got {self.sample_shape}"
        assert isinstance(self.dataset_name, str) and len(self.dataset_name) > 0, \
            f"Invalid dataset_name: {self.dataset_name}"
        assert isinstance(self.timestamp_dataset_name, str) and len(self.timestamp_dataset_name) > 0, \
            f"Invalid timestamp_dataset_name: {self.timestamp_dataset_name}"
        assert self.sample_dtype is not None, f"sample_dtype must be specified, but got {self.sample_dtype}"

        self.command_sample = {
            'sample': np.zeros(self.sample_shape, dtype=self.sample_dtype),
            'timestamp': np.zeros((1,), dtype=np.float64),
        }


class HDF5SaverController(BaseController):
    """
    Controller for saving data to HDF5 files.

    Uses the BaseController framework with multiprocessing to efficiently save
    arbitrary data with timestamps.
    """
    config: HDF5SaverControllerConfig

    def __init__(self, config: HDF5SaverControllerConfig):
        super().__init__(config)

    def _initialize(self):
        assert self.config.hdf5_path != "topass", \
            f"HDF5 path is not set. Please set it in the config file."
        self.hdf5_appender = HDF5Appender(
            filename=self.config.hdf5_path,
            dataset_name=self.config.dataset_name,
            timestamp_dataset_name=self.config.timestamp_dataset_name,
            compression=self.config.hdf5_compression,
            compression_opts=self.config.hdf5_compression_opts,
        )

    def _process_commands(self):
        try:
            commands = self.command_queue.get_all()
            n_cmd = len(commands['sample'])
        except Empty:
            n_cmd = 0
        except Exception as e:
            traceback.print_exc()
            raise e

        for i in range(n_cmd):
            sample = commands['sample'][i]
            timestamp = commands['timestamp'][i]

            self.hdf5_appender.append(sample, timestamp)

    def _update(self):
        pass

    def _close(self):
        self.hdf5_appender.close()

    def reset(self):
        pass
