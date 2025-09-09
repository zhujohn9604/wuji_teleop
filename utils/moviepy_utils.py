import sys
sys.path.append(".")

import numpy as np
import time
import cv2
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from utils.hdf5_utils import HDF5Appender

__all__ = ["MoviePyVideoStreamerWithTimestamp"]


class MoviePyVideoStreamerWithTimestamp:
    def __init__(self, output_file, fps, frame_size, codec, preset, crf, add_timestamp=False, timestamp_file=None, timestamp_compression=None, timestamp_compression_opts=None, ffv1_level=None):
        self.output_file = output_file
        self.timestamp_file = f"{output_file}.timestamps.hdf5"
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        self.preset = preset
        self.crf = crf
        self.ffv1_level = ffv1_level

        self.add_timestamp = add_timestamp
        self.timestamp_file = timestamp_file
        self.timestamp_compression = timestamp_compression
        self.timestamp_compression_opts = timestamp_compression_opts

        ffmpeg_params = []

        if self.crf is not None:
            ffmpeg_params.extend(["-crf", f"{crf}"])

        if self.codec.lower() == "ffv1" and self.ffv1_level is not None:
            ffmpeg_params.extend(["-level", f"{self.ffv1_level}"])
            ffmpeg_params.extend(["-coder", "1"])
            ffmpeg_params.extend(["-context", "1"])
            ffmpeg_params.extend(["-g", "15"])
            ffmpeg_params.extend(["-slices", "16"])

        # 创建 moviepy FFMPEG VideoWriter
        # ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
        self.writer = FFMPEG_VideoWriter(
            filename=output_file,
            size=frame_size,
            fps=fps,
            codec=codec,
            preset=preset,
            ffmpeg_params=ffmpeg_params,
        )

        if self.add_timestamp:
            self.timestamp_appender = HDF5Appender(
                self.timestamp_file,
                compression=self.timestamp_compression,
                compression_opts=self.timestamp_compression_opts
            )

    def add_frame(self, frame, timestamp):
        """
        发送一帧视频到 moviepy，并记录时间戳
        """
        # 如果是单通道灰度，转换成 BGR24
        if len(frame.shape) == 2:  # H x W
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # 检查尺寸
        height, width = frame.shape[:2]
        if (width, height) != self.frame_size:
            raise ValueError(
                f"Frame size does not match the initialized frame size, "
                f"expected {self.frame_size}, but got {(width, height)}"
            )

        # 写入视频
        self.writer.write_frame(frame)

        if self.add_timestamp:
            self.timestamp_appender.append(np.array([timestamp]), timestamp)

    def release(self):
        """
        关闭视频写入器，关闭 HDF5
        """
        print(f"Releasing moviepy writer")
        self.writer.close()  # 关闭 moviepy 视频写入器
        print(f"Moviepy writer closed")

        if self.add_timestamp:
            self.timestamp_appender.close()
            print(f"Timestamp HDF5 file closed")


# 示例用法
if __name__ == "__main__":
    import time
    import random

    output_file = "output_video_moviepy.mp4"
    fps = 30
    frame_size = (480, 1280)

    streamer = MoviePyVideoStreamerWithTimestamp(output_file, fps, frame_size)

    try:
        start_time = time.time()
        for i in range(100):
            # 随机生成帧
            if i % 2 == 0:
                frame = np.random.randint(0, 256, (frame_size[1], frame_size[0]), dtype=np.uint8)
            else:
                frame = np.random.randint(0, 256, (frame_size[1], frame_size[0], 3), dtype=np.uint8)

            # 时间戳
            start = time.time()
            current_time = 110000 + time.time() - start_time + random.uniform(0.01, 0.1)
            streamer.add_frame(frame, current_time)
            end = time.time()
            print(f"Step {i}: {current_time:.2f}, takes {end - start:.2f}s")
    except KeyboardInterrupt:
        print("Interrupted!")
    finally:
        streamer.release()
        print(f"Video saved to {output_file}")
