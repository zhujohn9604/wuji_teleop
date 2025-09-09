import sys
sys.path.append('.')
import click
import time
import numpy as np
import os
import threading
from pynput import keyboard

from controllers.realman_vrtrix import (
    RealmanVRTrixTeleopControllerConfig,
    RealmanVRTrixTeleopController,
)

from controllers import get_saver_controllers
from scripts.common_utils import init_realman_controller, init_osmo
from scripts.sim.rm75_utils import get_robot_init_pose  # Import robot pose utilities


class KeyboardListener:
    def __init__(self):
        self.key_status: dict = {}
        self.key_status_locks: dict = {}

        self.key_status["collecting"] = False
        self.key_status_locks["collecting"] = threading.Lock()

        self.key_status["reset"] = False
        self.key_status_locks["reset"] = threading.Lock()

        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        print(f"key pressed: {key}")
        if key == keyboard.Key.space:
            with self.key_status_locks["collecting"]:
                print(f"Toggling collecting status, current status: {self.key_status['collecting']}")
                self.key_status["collecting"] = not self.key_status["collecting"]
                print(f"New collecting status: {self.key_status['collecting']}")
        elif key == keyboard.KeyCode.from_char('r') or key == keyboard.KeyCode.from_char('R'):
            with self.key_status_locks["reset"]:
                self.key_status["reset"] = True
                print("Reset reference quaternion and tracker position requested")

    def get(self, key: str):
        with self.key_status_locks[key]:
            if key == "reset":
                # For reset, return and clear the flag
                val = self.key_status[key]
                self.key_status[key] = False
                return val
            return self.key_status[key]


def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))


@click.command()
@click.option('--right_ip', default='192.168.2.18', help='IP address of the right robotic arm')
@click.option('--right_port', default=8080, help='Port number of the right robotic arm')
@click.option('--left_ip', default='192.168.2.20', help='IP address of the left robotic arm')
@click.option('--left_port', default=8080, help='Port number of the left robotic arm')
@click.option('--vrtrix_ip', default='127.0.0.1', help='IP address of the VRTrix server')
@click.option('--vrtrix_port', default=11002, help='Port number of the VRTrix server')
@click.option('--fps', default=60, help='Frames per second for the teleoperation')
@click.option('--data_dir', default=None, help='Directory to save data (optional)')
@click.option('--enable_osmo', is_flag=True, help='Enable Osmo camera for data collection')
@click.option('--enable_left', is_flag=True, help='Enable left arm control')
@click.option('--enable_right', is_flag=True, help='Enable right arm control')
@click.option('--debug', is_flag=True, help='Enable debug output for angle mapping')
@click.option('--rotation_scale', default=1.0, help='Scaling factor for rotation (0.1-2.0)')
@click.option('--ee_x', default=-0.5, help='End effector default X position')
@click.option('--ee_y', default=0.4, help='End effector default Y position')
@click.option('--ee_z', default=0.1, help='End effector default Z position')
@click.option('--ee_rx', default=0.0, help='End effector default rotation around X axis (degrees)')
@click.option('--ee_ry', default=90.0, help='End effector default rotation around Y axis (degrees)')
@click.option('--ee_rz', default=0.0, help='End effector default rotation around Z axis (degrees)')
@click.option('--enable_tracker', is_flag=True, help='Enable tracker for position control')
@click.option('--tracker_ip', default='192.168.2.38', help='IP address of the tracker server')
@click.option('--tracker_port', default=12345, help='Port number of the tracker server')
@click.option('--tracker_position_scale', default=1.0, help='Scaling factor for tracker position mapping (0.1-5.0)')
@click.option('--tracker_smoothing', default=0, help='Smoothing factor for tracker position (0.0-1.0, higher = more smoothing)')
def main(right_ip: str, right_port: int, left_ip: str, left_port: int,
         vrtrix_ip: str, vrtrix_port: int, fps: int, data_dir: str,
         enable_osmo: bool, enable_left: bool, enable_right: bool, debug: bool,
         rotation_scale: float, ee_x: float, ee_y: float, ee_z: float,
         ee_rx: float, ee_ry: float, ee_rz: float,
         enable_tracker: bool, tracker_ip: str, tracker_port: int,
         tracker_position_scale: float, tracker_smoothing: float):

    save_data = data_dir is not None
    print(f"Save data: {save_data}, Data directory: {data_dir}")
    print(f"VRTrix server: {vrtrix_ip}:{vrtrix_port}")
    print(f"Debug mode: {debug}")
    print(f"Rotation scale: {rotation_scale}")
    print(f"Default EE position: [{ee_x:.3f}, {ee_y:.3f}, {ee_z:.3f}]")
    print(f"Default EE rotation (XYZ Euler angles in degrees): [{ee_rx:.1f}, {ee_ry:.1f}, {ee_rz:.1f}]")

    if enable_tracker:
        print(f"\nTracker enabled:")
        print(f"  Server: {tracker_ip}:{tracker_port}")
        print(f"  Position scale: {tracker_position_scale}")
        print(f"  Smoothing factor: {tracker_smoothing}")
        print(f"  Tracker coordinate mapping:")
        print(f"    tracker +Y -> robot +Z")
        print(f"    tracker +X -> robot -Y")
        print(f"    tracker +Z -> robot +X")

    dt = 1.0 / fps

    # Get robot base transformation matrices
    robot_r_mat_w, robot_l_mat_w, _, _ = get_robot_init_pose()

    # Set default joint positions
    right_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    left_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    # left_q = np.array([-0.9013055, -0.71556759, -0.91614085, -1.31098664, -1.65623021, 0.78862703, 6.2336359], dtype=np.float64)
    # left_q = np.array([-0.9013055, -0.71556759, -0.91614085, -1.31098664, -1.65623021, 0.78862703, 6.2336359], dtype=np.float64)

    # Set default end effector position and rotation
    default_ee_position = np.array([ee_x, ee_y, ee_z], dtype=np.float64)
    default_ee_rotation_xyz = np.array([ee_rx, ee_ry, ee_rz], dtype=np.float64)

    # Create VRTrix teleop controller config
    vrtrix_teleop_config = RealmanVRTrixTeleopControllerConfig(
        name="realman_vrtrix_teleop",
        fps=fps,
        put_desired_frequency=fps,
        vrtrix_ip=vrtrix_ip,
        vrtrix_port=vrtrix_port,
        robot_r_mat_w=robot_r_mat_w,
        robot_l_mat_w=robot_l_mat_w,
        default_right_q=right_q,
        default_left_q=left_q,
        default_ee_position=default_ee_position,
        default_ee_rotation_xyz=default_ee_rotation_xyz,
        enable_left=enable_left,
        enable_right=enable_right,
        debug_mode=debug,
        rotation_scale=rotation_scale,
        # Tracker settings
        enable_tracker=enable_tracker,
        tracker_ip=tracker_ip,
        tracker_port=tracker_port,
        tracker_position_scale=tracker_position_scale,
        tracker_smoothing_factor=tracker_smoothing,
    )

    # Create and start VRTrix teleop controller
    vrtrix_teleop_controller = RealmanVRTrixTeleopController(vrtrix_teleop_config)
    vrtrix_teleop_controller.start()

    # Wait for controller to be ready
    print("Waiting for VRTrix teleop controller to be ready...")
    for i in range(100):  # 10 second timeout
        if vrtrix_teleop_controller.check_ready():
            print("VRTrix teleop controller ready!")
            break
        time.sleep(0.1)
    else:
        print("VRTrix teleop controller failed to initialize!")
        return

    # Initialize Realman controllers
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

    # Setup data collection if enabled
    if save_data:
        keyboard_listener = KeyboardListener()
        collect_step = -1
        last_collecting_status = False

        if enable_osmo:
            osmo_frame_transformed_width = 640
            osmo_frame_transformed_height = 360
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

            hdf5_saver_configs = []

            if enable_right:
                hdf5_saver_configs.extend([
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
                ])

            if enable_left:
                hdf5_saver_configs.extend([
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
                ])

            mkv_saver_controllers, hdf5_saver_controllers = get_saver_controllers(
                mkv_saver_configs=mkv_saver_configs,
                hdf5_saver_configs=hdf5_saver_configs,
                data_dir=timestamp_data_dir,
            )

            return mkv_saver_controllers, hdf5_saver_controllers
    else:
        keyboard_listener = KeyboardListener()  # Still need for reset functionality

    print("\n" + "="*80)
    print("Starting teleoperation with VRTrix rotation control")
    if enable_tracker:
        print("Tracker position control is ENABLED")
    print("Controls:")
    print("  - Move your wrist to control robot end effector rotation")
    if enable_tracker:
        print("  - Move your hand (with tracker) to control robot end effector position")
    print("  - Move your fingers to control robot hand")
    print("  - Press 'R' to reset reference quaternion" + (" and tracker position" if enable_tracker else ""))
    if save_data:
        print("  - Press SPACE to toggle data collection")
    print("  - Press Ctrl+C to stop")
    print("="*80 + "\n")

    start_time = time.monotonic()
    global_step = 0

    try:
        while True:
            # Check for reset request
            if keyboard_listener.get("reset"):
                print("Resetting VRTrix reference quaternions" + (" and tracker positions" if enable_tracker else "") + "...")
                vrtrix_teleop_controller.reset()

            # Handle data collection toggle
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

            # Calculate timing
            end_time = start_time + (global_step + 1) * dt
            global_step += 1

            # Get feedback from VRTrix teleop controller
            vrtrix_feedback = vrtrix_teleop_controller.get_feedback()

            right_q = vrtrix_feedback["right_q"]  # (7,)
            left_q = vrtrix_feedback["left_q"]  # (7,)
            right_hand_cmd = vrtrix_feedback["right_hand_cmd"]  # (6,)
            left_hand_cmd = vrtrix_feedback["left_hand_cmd"]  # (6,)

            # Send commands to robot controllers
            if enable_right:
                right_realman_controller.schedule_joint(right_q, end_time)
                right_realman_controller.schedule_hand_pos(right_hand_cmd, end_time)

            if enable_left:
                left_realman_controller.schedule_joint(left_q, end_time)
                left_realman_controller.schedule_hand_pos(left_hand_cmd, end_time)

            # Save data if collecting
            if save_data and cur_collecting_status:
                if enable_right:
                    right_feedback = right_realman_controller.get_feedback()
                    right_cur_q = right_feedback["cur_q"]  # (7,)
                    right_timestamp = right_feedback["timestamp"]
                    right_hand_angle = right_feedback["cur_hand_angle"]  # (6,)

                    hdf5_saver_controllers["right_q"].send_command({
                        "sample": right_cur_q,
                        "timestamp": right_timestamp,
                    })
                    hdf5_saver_controllers["right_hand_angle"].send_command({
                        "sample": right_hand_angle,
                        "timestamp": right_timestamp,
                    })

                if enable_left:
                    left_feedback = left_realman_controller.get_feedback()
                    left_cur_q = left_feedback["cur_q"]  # (7,)
                    left_timestamp = left_feedback["timestamp"]
                    left_hand_angle = left_feedback["cur_hand_angle"]  # (6,)

                    hdf5_saver_controllers["left_q"].send_command({
                        "sample": left_cur_q,
                        "timestamp": left_timestamp,
                    })
                    hdf5_saver_controllers["left_hand_angle"].send_command({
                        "sample": left_hand_angle,
                        "timestamp": left_timestamp,
                    })

                if enable_osmo:
                    osmo_feedback = osmo_controller.get_transformed_feedback()
                    mkv_saver_controllers[osmo_name].send_command({
                        "img": osmo_feedback["img"],
                        "timestamp": osmo_feedback["timestamp"],
                    })

                collect_step += 1
                if collect_step % 30 == 0:  # Print every 30 frames
                    print(f"Collecting step: {collect_step}, global step: {global_step}, "
                          f"time elapsed: {end_time - start_time:.2f}s")

            # Sleep to maintain target FPS
            time.sleep(max(0, end_time - time.monotonic()))

    except KeyboardInterrupt:
        print("\nStopping teleoperation...")
    finally:
        # Clean up
        vrtrix_teleop_controller.stop()
        vrtrix_teleop_controller.join(timeout=2.0)

        if enable_right:
            right_realman_controller.stop()
        if enable_left:
            left_realman_controller.stop()

        print("Teleoperation stopped.")


if __name__ == '__main__':
    import sys
    sys.argv = [
        sys.argv[0],  # 保留脚本名称
        '--right_ip', '192.168.2.18',
        '--right_port', '8080',
        '--left_ip', '192.168.2.20',
        '--left_port', '8080',
        '--fps', '30',
        '--data_dir', '/nfs_data1/teleop_data/test_vrtrix',
        '--enable_left',
        #'--enable_right',
        '--debug',
        '--enable_tracker',
        # '--enable_osmo',  # 需要时取消注释
    ]
    main()