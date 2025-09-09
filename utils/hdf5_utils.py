import h5py
import os
import cv2
import numpy as np

__all__ = ['HDF5Appender', 'play_video_hdf5', 'Hdf5Reader']

def play_video_hdf5(hdf5_file, fps):
    frame_interval = 1.0 / fps

    with h5py.File(hdf5_file, 'r') as f:
        if 'float_array' not in f or 'timestamp_array' not in f:
            raise ValueError(f"HDF5 文件中缺少 'float_array' 或 'timestamp_array' 数据集, have {list(f.keys())}")

        frames = f['float_array']
        timestamps = f['timestamp_array']

        num_frames = len(frames)
        print(f"Playing {num_frames} frames at {fps} FPS...")
        for i in range(num_frames):
            frame = frames[i]
            timestamp = timestamps[i]

            cv2.imshow('Video Playback', frame.astype(np.uint8))

            if cv2.waitKey(int(frame_interval * 1000)) & 0xFF == ord('q'):
                print("Playback interrupted by user.")
                break

            print(f"Frame {i + 1}/{num_frames}, Timestamp: {timestamp:.3f}")

    cv2.destroyAllWindows()

class Hdf5Reader:
    def __init__(self, hdf5_file_path: str):
        self.hdf5_file_path = hdf5_file_path
        self.f = h5py.File(hdf5_file_path, 'r')
        self.float_array_data = self.f["float_array"]
        self.timestamp_data = self.f["timestamp_array"]

    def __len__(self):
        return len(self.timestamp_data)

    def __getitem__(self, idx):
        return self.float_array_data[idx]

class HDF5Appender:
    def __init__(
        self,
        filename,
        dataset_name='float_array',
        timestamp_dataset_name='timestamp_array',
        compression='gzip',
        compression_opts=5,
        meta_info=None
    ):
        self.filename = filename
        self.dataset_name = dataset_name
        self.timestamp_dataset_name = timestamp_dataset_name
        self.compression = compression
        self.compression_opts = compression_opts
        self.meta_info = meta_info

        self.file = None
        self.dataset = None
        self.timestamp_dataset = None
        self.data_shape = None

        self.open()

    def open(self):
        self.file = h5py.File(self.filename, 'a')

        if self.dataset_name in self.file:
            self.dataset = self.file[self.dataset_name]
            self.data_shape = self.dataset.shape[1:]
        else:
            self.dataset = None

        if self.timestamp_dataset_name in self.file:
            self.timestamp_dataset = self.file[self.timestamp_dataset_name]
        else:
            self.timestamp_dataset = None

        if self.meta_info is not None and isinstance(self.meta_info, dict):
            for key, value in self.meta_info.items():
                if key not in self.file.attrs:
                    self.file.attrs[key] = value

    def append(self, value, timestamp):
        if self.file is None:
            raise ValueError("HDF5 file is not opened. Call 'open' first.")
        if not isinstance(value, np.ndarray):
            raise ValueError("Value must be a numpy array.")

        if self.dataset is None:
            self.data_shape = value.shape
            self.dataset = self.file.create_dataset(
                self.dataset_name,
                shape=(0, *self.data_shape),
                maxshape=(None, *self.data_shape),
                dtype=value.dtype,
                chunks=(1, *self.data_shape),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )
        elif value.shape != self.data_shape:
            raise ValueError(
                f"All arrays must have the same shape ({self.data_shape}). Got {value.shape}"
            )

        if self.timestamp_dataset is None:
            self.timestamp_dataset = self.file.create_dataset(
                self.timestamp_dataset_name,
                shape=(0,),
                maxshape=(None,),
                dtype='f8'
            )

        current_size = self.dataset.shape[0]
        new_size = current_size + 1
        self.dataset.resize(new_size, axis=0)
        self.dataset[current_size, ...] = value
        self.dataset.flush()

        current_ts_size = self.timestamp_dataset.shape[0]
        new_ts_size = current_ts_size + 1
        self.timestamp_dataset.resize(new_ts_size, axis=0)
        self.timestamp_dataset[new_ts_size - 1] = timestamp
        self.timestamp_dataset.flush()

    def append_multi(self, values, timestamps):
        if self.file is None:
            raise ValueError("HDF5 file is not opened. Call 'open' first.")
        if not isinstance(values, np.ndarray):
            raise ValueError("Values must be a numpy array.")
        if len(values) != len(timestamps):
            raise ValueError("Values and timestamps must have the same length.")

        if self.dataset is None:
            self.data_shape = values.shape[1:]
            self.dataset = self.file.create_dataset(
                self.dataset_name,
                shape=(0, *self.data_shape),
                maxshape=(None, *self.data_shape),
                dtype=values.dtype,
                chunks=(1, *self.data_shape),
                compression=self.compression,
                compression_opts=self.compression_opts,
            )

        elif values.shape[1:] != self.data_shape:
            raise ValueError(
                f"All arrays must have the same shape ({self.data_shape}). Got {values.shape[1:]}"
            )

        if self.timestamp_dataset is None:
            self.timestamp_dataset = self.file.create_dataset(
                self.timestamp_dataset_name,
                shape=(0,),
                maxshape=(None,),
                dtype='f8'
            )

        current_size = self.dataset.shape[0]
        new_size = current_size + len(values)
        self.dataset.resize(new_size, axis=0)
        self.dataset[current_size:new_size, ...] = values
        self.dataset.flush()

        current_ts_size = self.timestamp_dataset.shape[0]
        new_ts_size = current_ts_size + len(timestamps)
        self.timestamp_dataset.resize(new_ts_size, axis=0)
        self.timestamp_dataset[current_ts_size:new_ts_size] = timestamps
        self.timestamp_dataset.flush()

    def close(self):
        self.file.flush()
        self.file.close()
        self.file = None
        self.dataset = None
        self.timestamp_dataset = None

def test_write_hdf5():
    import time

    filename = 'tensor_data.h5'
    meta_info = {'author': 'test_user', 'description': 'Sample HDF5 data'}
    appender = HDF5Appender(filename, meta_info=meta_info)

    try:
        arr1 = np.random.rand(3, 4).astype(np.float32)
        ts1 = time.time()

        arr2 = np.random.rand(3, 4).astype(np.float32)
        ts2 = time.time()

        appender.append(arr1, ts1)
        appender.append(arr2, ts2)
    finally:
        appender.close()

    with h5py.File(filename, 'r') as f:
        data = f['float_array'][:]
        timestamps = f['timestamp_array'][:]
        print("Data shape:", data.shape)
        print("Data dtype:", data.dtype)
        print("Timestamps shape:", timestamps.shape)
        print("Timestamps dtype:", timestamps.dtype)
        print("Timestamps:", timestamps)
        print("Meta info:", dict(f.attrs))

def test_play_hdf5():
    filename = '/umi_data/ep_153/left_Stereo.hdf5'
    play_video_hdf5(filename, fps=30)

if __name__ == "__main__":
    test_write_hdf5()
    # test_play_hdf5()
