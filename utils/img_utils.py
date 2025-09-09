import multiprocessing as mp
import cv2
import os
import shutil
import numpy as np

from utils.print_utils import print_blue

class ImageViewer:
    def __init__(self, window_name="Image Viewer", window_mode=cv2.WINDOW_NORMAL):
        """
        初始化图像可视化窗口。

        :param window_name: 窗口名称
        :param window_mode: 窗口模式（默认 cv2.WINDOW_NORMAL，可改为 cv2.WINDOW_AUTOSIZE 等）
        """
        self.window_name = window_name
        self.step = 0

        self.tmp_dir = f"./tmp/tmpimgs/{self.window_name}"
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

        return

        # print_blue(f"visual ui window_name: {window_name}")
        # cv2.startWindowThread()
        # 创建窗口
        print(f"visual ui window_name: {window_name}")
        cv2.namedWindow(self.window_name, window_mode)
        print(f"visual ui window_name: {window_name} created")

    def update(self, image: np.ndarray):
        """
        更新并显示图像。

        :param image: 要显示的图像，格式为 OpenCV 默认 BGR 或灰度
        """
        if image.ndim == 2:
            if image.dtype == np.uint16:
                # clip gray 16 to gray 8
                image = image / 5
                gray8 = np.clip(image, 0, 255).astype(np.uint8)
                # gray8 -> bgr
                image = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
            elif image.dtype == np.uint8:
                # gray8 -> bgr
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3:
            # rgb -> bgr
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("Unsupported image format: {}".format(image.shape))

        # save to ./tmp/{window_name}_{step}.jpg
        cv2.imwrite(f"{self.tmp_dir}/{self.step}.jpg", image)

        self.step += 1

        # # resize to the image size
        # cv2.resizeWindow(self.window_name, image.shape[1], image.shape[0])

        # cv2.imshow(self.window_name, image)
        # # 这里 waitKey(1) 可以让窗口及时刷新
        # # 返回值是按键的 ASCII 码，若需要实现按键退出可自行扩展
        # cv2.waitKey(1)

    def __del__(self):
        """
        析构时关闭窗口。
        """
        cv2.destroyWindow(self.window_name)
