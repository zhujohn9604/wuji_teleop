import numpy as np
import socket
import json
import time
import sys
from scipy.spatial.transform import Rotation as R
import threading
from pynput import keyboard


class TrackerCalibrationTool:
    """Tool to help calibrate tracker coordinate system alignment"""

    def __init__(self, tracker_ip='192.168.2.68', tracker_port=12345):
        self.tracker_ip = tracker_ip
        self.tracker_port = tracker_port
        self.latest_tracker_data = None
        self.running = True
        self.calibration_poses = []
        self.tracker_buffer = ""

        # Connect to tracker
        self.tracker_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tracker_sock.connect((tracker_ip, tracker_port))

        # Start receiver thread
        self.receiver_thread = threading.Thread(target=self._tracker_receiver_loop, daemon=True)
        self.receiver_thread.start()

        # Setup keyboard listener
        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()

    def _tracker_receiver_loop(self):
        """Receive tracker data continuously"""
        while self.running:
            try:
                data = self.tracker_sock.recv(4096).decode('utf-8')
                if not data:
                    break

                self.tracker_buffer += data
                lines = self.tracker_buffer.split('\n')
                self.tracker_buffer = lines[-1]

                for line in lines[:-1]:
                    if line.strip():
                        try:
                            self.latest_tracker_data = json.loads(line)
                        except json.JSONDecodeError:
                            pass
            except Exception as e:
                print(f"Error in receiver: {e}")
                break

    def on_key_press(self, key):
        """Handle keyboard input"""
        try:
            if key.char == 'c':
                self.capture_calibration_pose()
            elif key.char == 'q':
                self.running = False
        except AttributeError:
            # Special keys
            if key == keyboard.Key.space:
                self.capture_calibration_pose()
            elif key == keyboard.Key.esc:
                self.running = False

    def capture_calibration_pose(self):
        """Capture current tracker pose for calibration"""
        if self.latest_tracker_data:
            tracker_1 = self.latest_tracker_data.get('tracker_1', [])
            tracker_2 = self.latest_tracker_data.get('tracker_2', [])

            if len(tracker_1) == 6:
                self.calibration_poses.append({
                    'hand': 'right',
                    'pose': tracker_1,
                    'timestamp': time.time()
                })
                print(f"Captured right hand pose #{len(self.calibration_poses)}")

            if len(tracker_2) == 6:
                self.calibration_poses.append({
                    'hand': 'left',
                    'pose': tracker_2,
                    'timestamp': time.time()
                })
                print(f"Captured left hand pose #{len(self.calibration_poses)}")

    def analyze_calibration(self):
        """Analyze captured poses to determine coordinate system"""
        print("\n" + "="*60)
        print("Calibration Analysis")
        print("="*60)

        if len(self.calibration_poses) < 3:
            print("Not enough calibration poses. Need at least 3.")
            return

        # Group poses by hand
        right_poses = [p for p in self.calibration_poses if p['hand'] == 'right']
        left_poses = [p for p in self.calibration_poses if p['hand'] == 'left']

        # Analyze movement patterns
        for hand, poses in [('Right', right_poses), ('Left', left_poses)]:
            if len(poses) < 2:
                continue

            print(f"\n{hand} Hand Analysis:")
            print("-"*40)

            # Calculate differences between consecutive poses
            for i in range(1, len(poses)):
                prev_pose = np.array(poses[i-1]['pose'])
                curr_pose = np.array(poses[i]['pose'])

                # Position difference
                pos_diff = curr_pose[:3] - prev_pose[:3]

                # Rotation difference
                rot_diff = curr_pose[3:] - prev_pose[3:]

                print(f"\nMovement {i}:")
                print(f"  Position change: X={pos_diff[0]:.3f}, Y={pos_diff[1]:.3f}, Z={pos_diff[2]:.3f}")
                print(f"  Rotation change: Roll={rot_diff[0]:.1f}°, Pitch={rot_diff[1]:.1f}°, Yaw={rot_diff[2]:.1f}°")

                # Determine dominant axis
                max_pos_idx = np.argmax(np.abs(pos_diff))
                max_rot_idx = np.argmax(np.abs(rot_diff))

                axes = ['X', 'Y', 'Z']
                rots = ['Roll', 'Pitch', 'Yaw']

                if np.abs(pos_diff[max_pos_idx]) > 0.05:  # 5cm threshold
                    print(f"  Dominant position axis: {axes[max_pos_idx]} ({pos_diff[max_pos_idx]:.3f}m)")

                if np.abs(rot_diff[max_rot_idx]) > 10:  # 10 degree threshold
                    print(f"  Dominant rotation axis: {rots[max_rot_idx]} ({rot_diff[max_rot_idx]:.1f}°)")

    def run_calibration(self):
        """Run the calibration process"""
        print("\n" + "="*60)
        print("Tracker Coordinate System Calibration Tool")
        print("="*60)
        print("\nInstructions:")
        print("1. Put on the VRTrix glove with tracker attached")
        print("2. Move your hand in specific directions:")
        print("   - Forward (towards fingers)")
        print("   - Left/Right")
        print("   - Up/Down")
        print("   - Rotate wrist")
        print("3. Press 'C' or SPACE after each movement to capture pose")
        print("4. Press 'Q' or ESC when done")
        print("\nWaiting for tracker data...")

        # Wait for initial data
        while self.latest_tracker_data is None and self.running:
            time.sleep(0.1)

        if not self.running:
            return

        print("Tracker connected! Start calibration movements.")

        # Main loop
        last_print_time = 0
        while self.running:
            current_time = time.time()

            # Print current tracker data periodically
            if current_time - last_print_time > 1.0:
                if self.latest_tracker_data:
                    tracker_1 = self.latest_tracker_data.get('tracker_1', [])
                    tracker_2 = self.latest_tracker_data.get('tracker_2', [])

                    sys.stdout.write('\r')
                    if len(tracker_1) == 6:
                        sys.stdout.write(f"Right: X={tracker_1[0]:6.3f} Y={tracker_1[1]:6.3f} Z={tracker_1[2]:6.3f} | ")
                    if len(tracker_2) == 6:
                        sys.stdout.write(f"Left: X={tracker_2[0]:6.3f} Y={tracker_2[1]:6.3f} Z={tracker_2[2]:6.3f}")
                    sys.stdout.flush()

                last_print_time = current_time

            time.sleep(0.01)

        # Analyze results
        self.analyze_calibration()

        # Generate transformation matrix suggestions
        self.suggest_transformations()

    def suggest_transformations(self):
        """Suggest transformation matrices based on calibration"""
        print("\n" + "="*60)
        print("Suggested Transformation Matrices")
        print("="*60)

        print("\nBased on the calibration, here are suggested transformations:")
        print("\nFor right hand:")
        print("tracker_to_hand_right = np.array([")
        print("    [1, 0, 0, 0],   # Adjust based on forward movement")
        print("    [0, 0, -1, 0],  # Adjust based on left/right movement")
        print("    [0, 1, 0, 0],   # Adjust based on up/down movement")
        print("    [0, 0, 0, 1]")
        print("])")

        print("\nFor left hand:")
        print("tracker_to_hand_left = np.array([")
        print("    [1, 0, 0, 0],   # Adjust based on forward movement")
        print("    [0, 0, 1, 0],   # Adjust based on left/right movement")
        print("    [0, 1, 0, 0],   # Adjust based on up/down movement")
        print("    [0, 0, 0, 1]")
        print("])")

        print("\nAdjust the matrix values based on observed movements:")
        print("- If moving forward increases X in tracker: keep [1,0,0] in first row")
        print("- If moving forward increases Y in tracker: use [0,1,0] in first row")
        print("- If moving forward increases Z in tracker: use [0,0,1] in first row")
        print("- Add negative signs to reverse directions as needed")

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if hasattr(self, 'tracker_sock'):
            self.tracker_sock.close()
        if hasattr(self, 'listener'):
            self.listener.stop()


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Tracker Coordinate System Calibration Tool')
    parser.add_argument('--tracker_ip', default='192.168.2.101', help='Tracker server IP')
    parser.add_argument('--tracker_port', type=int, default=12345, help='Tracker server port')

    args = parser.parse_args()

    calibrator = TrackerCalibrationTool(args.tracker_ip, args.tracker_port)

    try:
        calibrator.run_calibration()
    except KeyboardInterrupt:
        print("\nCalibration interrupted")
    finally:
        calibrator.cleanup()


if __name__ == "__main__":
    main()