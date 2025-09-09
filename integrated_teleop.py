"""
Integrated Teleoperation Class for VisionPro to Robot Control

This module provides a simple, integrated class that processes VisionPro data
and sends commands to both robot arms and hands, replacing the complex logic
in avp_teleop_wuji.py.
"""

import sys
import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from pathlib import Path


from controllers.realman_wuji_avp import (
    RealmanAvpControllerConfig,
    RealmanAvpController,
)
from scripts.common_utils import init_realman_wuji_controller
from scripts.sim.rm75_utils import get_robot_init_pose


@dataclass
class TeleopConfig:
    """Configuration for the integrated teleoperation system."""
    # Robot arm IPs and ports
    right_ip: str = '192.168.2.18'
    right_port: int = 8080
    left_ip: str = '192.168.2.20'
    left_port: int = 8080
    
    # VisionPro settings
    avp_ip: Optional[str] = None
    redis_ip: Optional[str] = None
    
    # Control settings
    fps: int = 30
    hand_type: str = 'wuji'
    hand_cmd_len: int = 20
    
    # Retargeting configs
    retarget_config_path_r: Optional[str] = None
    retarget_config_path_l: Optional[str] = None
    
    # Enable/disable arms
    enable_right: bool = True
    enable_left: bool = True
    
    # Data collection
    data_dir: Optional[str] = None




class IntegratedTeleop:
    """
    Integrated teleoperation class that processes VisionPro data and controls robot arms.
    
    This class simplifies the complex logic in avp_teleop_wuji.py by providing
    a single interface for:
    - VisionPro data processing
    - Robot arm control
    - Hand control
    - Optional data collection
    """
    
    def __init__(self, config: TeleopConfig):
        """
        Initialize the integrated teleoperation system.
        
        Args:
            config: Configuration object containing all necessary parameters
        """
        self.config = config
        self.dt = 1.0 / config.fps
        
        # Initialize robot poses
        self.robot_r_mat_w, self.robot_l_mat_w, self.assume_avp_pos_offset, self.assume_avp_mat_w = get_robot_init_pose()
        
        # Initialize joint positions
        self.right_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.left_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        
        # Initialize controllers
        self.avp_controller = None
        self.right_robot_controller = None
        self.left_robot_controller = None
        
        # State tracking
        self.is_running = False
        self.global_step = 0
        self.start_time = None
        
    def initialize(self):
        """Initialize all controllers and systems."""
        print("Initializing integrated teleoperation system...")
        
        # Initialize VisionPro controller
        self._init_avp_controller()
        
        # Initialize robot controllers
        self._init_robot_controllers()
        
        
        print("Initialization complete!")
    
    def _init_avp_controller(self):
        """Initialize the VisionPro controller."""
        avp_config = RealmanAvpControllerConfig(
            name="realman_avp",
            fps=self.config.fps,
            put_desired_frequency=self.config.fps,
            ip=self.config.avp_ip,
            redis_ip=self.config.redis_ip,
            robot_r_mat_w=self.robot_r_mat_w,
            robot_l_mat_w=self.robot_l_mat_w,
            retarget_config_path_r=self.config.retarget_config_path_r,
            retarget_config_path_l=self.config.retarget_config_path_l,
            assume_avp_pos_offset=self.assume_avp_pos_offset,
            default_right_q=self.right_q,
            default_left_q=self.left_q,
            enable_left=self.config.enable_left,
            enable_right=self.config.enable_right,
            hand_cmd_len=self.config.hand_cmd_len,
        )
        
        self.avp_controller = RealmanAvpController(avp_config)
        self.avp_controller.start()
        self.avp_controller._ready_event.wait()
    
    def _init_robot_controllers(self):
        """Initialize robot arm controllers."""
        self.right_robot_controller, self.left_robot_controller = init_realman_wuji_controller(
            right_ip=self.config.right_ip,
            right_port=self.config.right_port,
            left_ip=self.config.left_ip,
            left_port=self.config.left_port,
            dof=7,
            controlller_fps=500,
            right_joints_init=self.right_q,
            left_joints_init=self.left_q,
            right_hand_angle_init=None,
            left_hand_angle_init=None,
            enable_right=self.config.enable_right,
            enable_left=self.config.enable_left,
            hand_type=self.config.hand_type,
        )
    
    
    
    def _process_vision_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process VisionPro data and return robot commands.
        
        Returns:
            Tuple of (right_q, left_q, right_hand_cmd, left_hand_cmd)
        """
        # Get feedback from VisionPro controller
        avp_feedback = self.avp_controller.get_feedback()
        
        right_q = avp_feedback["right_q"]  # (7,)
        left_q = avp_feedback["left_q"]  # (7,)
        right_hand_cmd = avp_feedback["right_hand_cmd"]  # (20,)
        left_hand_cmd = avp_feedback["left_hand_cmd"]  # (20,)
        
        return right_q, left_q, right_hand_cmd, left_hand_cmd
    
    def _send_robot_commands(self, right_q: np.ndarray, left_q: np.ndarray, 
                           right_hand_cmd: np.ndarray, left_hand_cmd: np.ndarray, 
                           end_time: float):
        """Send commands to robot arms and hands."""
        if self.config.enable_right:
            self.right_robot_controller.schedule_joint(right_q, end_time)
            self.right_robot_controller.schedule_hand_pos(right_hand_cmd, end_time)
        
        if self.config.enable_left:
            self.left_robot_controller.schedule_joint(left_q, end_time)
            self.left_robot_controller.schedule_hand_pos(left_hand_cmd, end_time)
    
    
    def run(self):
        """Run the main teleoperation loop."""
        if self.is_running:
            print("Teleoperation is already running!")
            return
        
        self.is_running = True
        self.start_time = time.monotonic()
        self.global_step = 0
        
        print("Starting teleoperation loop...")
        print("Press Ctrl+C to stop teleoperation")
        
        try:
            while self.is_running:
                # Calculate timing
                end_time = self.start_time + (self.global_step + 1) * self.dt
                
                # Process VisionPro data
                right_q, left_q, right_hand_cmd, left_hand_cmd = self._process_vision_data()
                
                # Send commands to robots
                self._send_robot_commands(right_q, left_q, right_hand_cmd, left_hand_cmd, end_time)
                
                # Update step counter
                self.global_step += 1
                
                # Sleep to maintain timing
                time.sleep(max(0, end_time - time.monotonic()))
                
        except KeyboardInterrupt:
            print("\nTeleoperation stopped by user")
        except Exception as e:
            print(f"Error during teleoperation: {e}")
            raise
        finally:
            self.stop()
    
    def stop(self):
        """Stop the teleoperation system."""
        if not self.is_running:
            return
        
        print("Stopping teleoperation system...")
        self.is_running = False
        
        # Stop controllers
        if self.avp_controller:
            self.avp_controller.stop()
            self.avp_controller.join()
        
        print("Teleoperation system stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            "is_running": self.is_running,
            "global_step": self.global_step,
            "elapsed_time": time.monotonic() - self.start_time if self.start_time else 0,
        }


def create_teleop_from_args(**kwargs) -> IntegratedTeleop:
    """
    Create an IntegratedTeleop instance from command line arguments.
    
    This function provides a convenient way to create a teleoperation system
    with the same parameters as the original avp_teleop_wuji.py script.
    """
    config = TeleopConfig(**kwargs)
    teleop = IntegratedTeleop(config)
    return teleop


# Example usage
if __name__ == "__main__":
    # Create teleop system with default configuration
    config = TeleopConfig(
        right_ip='192.168.2.18',
        right_port=8080,
        left_ip='192.168.2.20',
        left_port=8080,
        avp_ip='192.168.2.13',
        fps=30,
        data_dir='/nfs_data1/teleop_data/test',
        enable_left=True,
        enable_right=True,
        hand_type='wuji',
        hand_cmd_len=20,
        retarget_config_path_r='/home/wuji/code/dex-real-deployment/configs/wujihand_right_dexpilot.yaml',
        retarget_config_path_l='/home/wuji/code/dex-real-deployment/configs/wujihand_left_dexpilot.yaml',
    )
    
    # Initialize and run
    teleop = IntegratedTeleop(config)
    teleop.initialize()
    teleop.run()
