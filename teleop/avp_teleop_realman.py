import sys
sys.path.append('.')
import click
import time

import numpy as np

import os

from scripts.sim.rm75_utils import get_robot_init_pose

from controllers.realman_avp import (
    RealmanAvpControllerConfig,
    RealmanAvpController,
)
from controllers import get_saver_controllers

from scripts.common_utils import init_realman_controller, init_osmo

import threading

from pynput import keyboard


class KeyboardListener:
    def __init__(self):
        self.key_status: dict = {}
        self.key_status_locks: dict = {}

        self.key_status["collecting"] = False
        self.key_status_locks["collecting"] = threading.Lock()

        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        print(f"key pressed: {key}")
        if key == keyboard.Key.space:
            with self.key_status_locks["collecting"]:
                print(f"Toggling collecting status, current status: {self.key_status['collecting']}")
                self.key_status["collecting"] = not self.key_status["collecting"]
                print(f"New collecting status: {self.key_status['collecting']}")

    def get(self, key: str):
        with self.key_status_locks[key]:
            return self.key_status[key]


def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))

@click.command()
@click.option('--right_ip', default='192.168.2.18', help='IP address of the robotic arm')
@click.option('--right_port', default=8080, help='Port number of the robotic arm')
@click.option('--left_ip', default='192.168.2.20', help='IP address of the left robotic arm')
@click.option('--left_port', default=8080, help='Port number of the left robotic arm')
@click.option('--avp_ip', default=None, help='IP address of the AVP (optional)')
@click.option('--redis_ip', default=None, help='IP address of the AVP (optional)')
@click.option('--fps', default=30, help='Frames per second for the teleoperation')
@click.option('--data_dir', default=None, help='Directory to save data (optional)')
@click.option('--enable_osmo', is_flag=True, help='Enable Osmo camera for data collection')
@click.option('--fake_data', is_flag=True, help='Use fake data for testing')
@click.option('--enable_left', is_flag=True, help='Enable left arm control')
@click.option('--enable_right', is_flag=True, help='Enable right arm control')
def main(right_ip: str, right_port: int, left_ip: str, left_port: int, avp_ip: str, redis_ip: str, fps: int, data_dir: str, enable_osmo: bool, fake_data: bool, enable_left: bool, enable_right: bool):
    save_data = data_dir is not None
    print(f"Save data: {save_data}, Data directory: {data_dir}")

    dt = 1.0 / fps

    robot_r_mat_w, robot_l_mat_w, assume_avp_pos_offset, assume_avp_mat_w = get_robot_init_pose()

    right_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    left_q = np.array([-0.9013055, -0.71556759, -0.91614085, -1.31098664, -1.65623021, 0.78862703, 6.2336359], dtype=np.float64)

    realman_avp_controller_config = RealmanAvpControllerConfig(
        name="realman_avp",
        fps=fps,
        put_desired_frequency=fps,
        ip=avp_ip,
        # redis_ip=redis_ip,
        robot_r_mat_w=robot_r_mat_w,
        robot_l_mat_w=robot_l_mat_w,
        assume_avp_pos_offset=assume_avp_pos_offset,
        default_right_q=right_q,
        default_left_q=left_q,
        enable_left=enable_left,
        enable_right=enable_right,
    )
    realman_avp_controller = RealmanAvpController(realman_avp_controller_config)
    if fake_data:
        realman_avp_controller.load_fake_data("/home/wuji/code/dex-real-deployment/tmp/rm75_ik")
    else:
        realman_avp_controller.start()
        realman_avp_controller._ready_event.wait()

    # time.sleep(2.0)  # wait for the controller to be ready

    # realman_avp_feedback_state_dict = realman_avp_controller.get_feedback()

    # right_q = realman_avp_feedback_state_dict["right_q"]
    # left_q = realman_avp_feedback_state_dict["left_q"]

    right_realman_controller, left_realman_controller = init_realman_controller(
        right_ip=right_ip,
        right_port=right_port,
        left_ip=left_ip,
        left_port=left_port,
        dof=7,
        controlller_fps=500,
        right_joints_init=right_q,
        left_joints_init=left_q,
        right_hand_angle_init=None,
        left_hand_angle_init=None,
        enable_right=enable_right,
        enable_left=enable_left,
    )

    # 创建可视化器
    # from scripts.teleop.visualizer import RobotArmVisualizer
    # visualizer = RobotArmVisualizer()

    if save_data:
        keyboard_listener = KeyboardListener()
        collect_step = -1
        last_collecting_status = False

        if enable_osmo:
            osmo_frame_transformed_width=640
            osmo_frame_transformed_height=360
            osmo_name, osmo_controller = init_osmo(
                osmo_serial="04702F4E",
                osmo_frame_width=1280,
                osmo_frame_height=720,
                osmo_frame_transformed_width=osmo_frame_transformed_width,
                osmo_frame_transformed_height=osmo_frame_transformed_height,
                fps=fps,
            )

        mkv_saver_controllers, hdf5_saver_controllers = None, None
        def get_saver_controllers_call(timestamp_data_dir: str):
            if enable_osmo:
                mkv_saver_configs = [
                    {
                        "name": osmo_name,
                        "fps": fps,
                        "put_desired_frequency": fps,
                        "mkv_name": f"{osmo_name}.mkv",
                        "codec": "ffv1",
                        "preset": "medium",
                        "crf": 23,
                        "ffv1_level": 3,
                        "frame_width": osmo_frame_transformed_width,
                        "frame_height": osmo_frame_transformed_height,
                    }
                ]
            else:
                mkv_saver_configs = []

            hdf5_saver_controllers = [
                {
                    "name": "right_q",
                    "fps": fps,
                    "put_desired_frequency": fps,
                    "hdf5_name": "right_q.hdf5",
                    "sample_shape": (7,),
                    "sample_dtype": np.float64,
                },
                {
                    "name": "right_hand_angle",
                    "fps": fps,
                    "put_desired_frequency": fps,
                    "hdf5_name": "right_hand_angle.hdf5",
                    "sample_shape": (6,),
                    "sample_dtype": np.int32,
                },
            ]
            hdf5_saver_controllers += [
                {
                    "name": "left_q",
                    "fps": fps,
                    "put_desired_frequency": fps,
                    "hdf5_name": "left_q.hdf5",
                    "sample_shape": (7,),
                    "sample_dtype": np.float64,
                },
                {
                    "name": "left_hand_angle",
                    "fps": fps,
                    "put_desired_frequency": fps,
                    "hdf5_name": "left_hand_angle.hdf5",
                    "sample_shape": (6,),
                    "sample_dtype": np.int32,
                },
            ]

            mkv_saver_controllers, hdf5_saver_controllers = get_saver_controllers(
                mkv_saver_configs=mkv_saver_configs,
                hdf5_saver_configs=hdf5_saver_controllers,
                data_dir=timestamp_data_dir,
            )

            return mkv_saver_controllers, hdf5_saver_controllers

    start_time = time.monotonic()
    global_step = 0

    while True:
        if save_data:
            cur_collecting_status = keyboard_listener.get("collecting")
            if not last_collecting_status and cur_collecting_status:
                collect_step = 0

                cur_timestamp_str = get_timestamp()
                timestamp_data_dir = os.path.join(data_dir, cur_timestamp_str)
                os.makedirs(timestamp_data_dir, exist_ok=True)

                mkv_saver_controllers, hdf5_saver_controllers = get_saver_controllers_call(timestamp_data_dir)

            if last_collecting_status and not cur_collecting_status:
                collect_step = -1
                for controller in list(mkv_saver_controllers.values()) + list(hdf5_saver_controllers.values()):
                    controller.stop()
                    controller.join()

            last_collecting_status = cur_collecting_status



        end_time = start_time + (global_step + 1) * dt
        global_step += 1

        realman_avp_feedback_state_dict = realman_avp_controller.get_feedback()

        right_q = realman_avp_feedback_state_dict["right_q"] # (7, )
        left_q = realman_avp_feedback_state_dict["left_q"] # (7, )
        right_hand_cmd = realman_avp_feedback_state_dict["right_hand_cmd"]
        left_hand_cmd = realman_avp_feedback_state_dict["left_hand_cmd"]

        # visualizer.update_plot(left_q)

        if enable_right:
            right_realman_controller.schedule_joint(right_q, end_time)
            right_realman_controller.schedule_hand_pos(right_hand_cmd, end_time)

        if enable_left:
            left_realman_controller.schedule_joint(left_q, end_time)
            left_realman_controller.schedule_hand_pos(left_hand_cmd, end_time)

        if save_data:
            if cur_collecting_status:
                if enable_right:
                    right_realman_feedback_state_dict = right_realman_controller.get_feedback()
                    right_realman_cur_q = right_realman_feedback_state_dict["cur_q"] # (7, )
                    right_realman_cur_q_timestamp = right_realman_feedback_state_dict["timestamp"]
                    right_hand_angle = right_realman_feedback_state_dict["cur_hand_angle"] # (6, )
                    hdf5_saver_controllers["right_q"].send_command({
                        "sample": right_realman_cur_q,
                        "timestamp": right_realman_cur_q_timestamp,
                    })
                    hdf5_saver_controllers["right_hand_angle"].send_command({
                        "sample": right_hand_angle,
                        "timestamp": right_realman_cur_q_timestamp,
                    })

                if enable_left:
                    left_realman_feedback_state_dict = left_realman_controller.get_feedback()
                    left_realman_cur_q = left_realman_feedback_state_dict["cur_q"] # (7, )
                    left_realman_cur_q_timestamp = left_realman_feedback_state_dict["timestamp"]
                    left_hand_angle = left_realman_feedback_state_dict["cur_hand_angle"] # (6, )
                    hdf5_saver_controllers["left_q"].send_command({
                        "sample": left_realman_cur_q,
                        "timestamp": left_realman_cur_q_timestamp,
                    })
                    hdf5_saver_controllers["left_hand_angle"].send_command({
                        "sample": left_hand_angle,
                        "timestamp": left_realman_cur_q_timestamp,
                    })

                if enable_osmo:
                    # osmo_feedback_state_dict = osmo_controller.get_feedback()
                    osmo_feedback_state_dict = osmo_controller.get_transformed_feedback()
                    mkv_saver_controllers[osmo_name].send_command({
                        "img": osmo_feedback_state_dict["img"], # (height, width, 3), aligh with mkv saver
                        "timestamp": osmo_feedback_state_dict["timestamp"],
                    })

                collect_step += 1
                print(f"Collecting step: {collect_step}, global step: {global_step}, time elapsed: {end_time - start_time:.2f}s")

        time.sleep(max(0, end_time - time.monotonic()))

if __name__ == '__main__':
    import sys
    sys.argv = [
        sys.argv[0],  # 保留脚本名称
        '--right_ip', '192.168.2.18',
        '--right_port', '8080',
        '--left_ip', '192.168.2.20',
        '--left_port', '8080',
        '--avp_ip', '192.168.2.13',
        '--fps', '30',
        '--data_dir', '/nfs_data1/teleop_data/test',
        '--enable_left',
        #'--enable_right',
        '--fake_data',
        # '--enable_osmo',  # 需要时取消注释
    ]
    main()
