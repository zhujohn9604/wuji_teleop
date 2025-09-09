from .hdf5_saver import HDF5SaverController, HDF5SaverControllerConfig
from .mkv_saver import MKVSaverController, MKVSaverControllerConfig

import copy
import os

from typing import Union, Dict, Tuple


def get_saver_controllers(
    mkv_saver_configs: list[dict],
    hdf5_saver_configs: list[dict],
    data_dir: str,
) -> Tuple[Dict[str, MKVSaverController], Dict[str, HDF5SaverController]]:
    """
    Args:
        mkv_saver_configs (list[dict]): List of MKV saver configurations.
        hdf5_saver_configs (list[dict]): List of HDF5 saver configurations.
        data_dir (str): Directory where the data is stored.
    Returns:
        tuple: A tuple containing two dictionaries:
            - mkv_saver_controllers: Dictionary of MKV saver controllers keyed by their names.
            - hdf5_saver_controllers: Dictionary of HDF5 saver controllers keyed by their names.
    """

    mkv_saver_controllers, hdf5_saver_controllers = {}, {}

    for mkv_saver_config in mkv_saver_configs:
        mkv_saver_config = copy.deepcopy(mkv_saver_config)
        mkv_name = mkv_saver_config.pop("mkv_name")
        mkv_saver_config["mkv_path"] = os.path.join(data_dir, mkv_name)
        mkv_saver_controller = MKVSaverController(MKVSaverControllerConfig(**mkv_saver_config))
        mkv_saver_controller.start()
        mkv_saver_controller._ready_event.wait()
        mkv_saver_controllers[mkv_saver_controller.config.name] = mkv_saver_controller

    for hdf5_saver_config in hdf5_saver_configs:
        hdf5_saver_config = copy.deepcopy(hdf5_saver_config)
        hdf5_name = hdf5_saver_config.pop("hdf5_name")
        hdf5_saver_config["hdf5_path"] = os.path.join(data_dir, hdf5_name)
        hdf5_saver_controller = HDF5SaverController(HDF5SaverControllerConfig(**hdf5_saver_config))
        hdf5_saver_controller.start()
        hdf5_saver_controller._ready_event.wait()
        hdf5_saver_controllers[hdf5_saver_controller.config.name] = hdf5_saver_controller

    return mkv_saver_controllers, hdf5_saver_controllers
