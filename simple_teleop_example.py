#!/usr/bin/env python3
"""
Simple example script demonstrating how to use the IntegratedTeleop class.

This script replaces the complex avp_teleop_wuji.py with a much simpler interface.
"""

import sys
import click
from integrated_teleop import IntegratedTeleop, TeleopConfig


@click.command()
@click.option('--right_ip', default='192.168.2.18', help='IP address of the right robotic arm')
@click.option('--right_port', default=8080, help='Port number of the right robotic arm')
@click.option('--left_ip', default='192.168.2.20', help='IP address of the left robotic arm')
@click.option('--left_port', default=8080, help='Port number of the left robotic arm')
@click.option('--avp_ip', default='192.168.2.13', help='IP address of the VisionPro')
@click.option('--redis_ip', default=None, help='IP address of Redis server (optional)')
@click.option('--fps', default=30, help='Frames per second for teleoperation')
@click.option('--data_dir', default=None, help='Directory to save data (optional)')
@click.option('--enable_left', is_flag=True, help='Enable left arm control')
@click.option('--enable_right', is_flag=True, help='Enable right arm control')
@click.option('--hand_type', default='wuji', help='Type of hand')
@click.option('--hand_cmd_len', default=20, help='Length of hand command')
@click.option('--retarget_config_path_r', default=None, help='Path to right hand retargeting config')
@click.option('--retarget_config_path_l', default=None, help='Path to left hand retargeting config')
def main(right_ip: str, right_port: int, left_ip: str, left_port: int, 
         avp_ip: str, redis_ip: str, fps: int, data_dir: str, 
         enable_left: bool, enable_right: bool, hand_type: str, 
         hand_cmd_len: int, retarget_config_path_r: str, retarget_config_path_l: str):
    """
    Simple teleoperation script using the IntegratedTeleop class.
    
    This replaces the complex avp_teleop_wuji.py with a much cleaner interface.
    The logic is now encapsulated in the IntegratedTeleop class, making it
    easier to understand, maintain, and extend.
    """
    
    # Create configuration
    config = TeleopConfig(
        right_ip=right_ip,
        right_port=right_port,
        left_ip=left_ip,
        left_port=left_port,
        avp_ip=avp_ip,
        redis_ip=redis_ip,
        fps=fps,
        data_dir=data_dir,
        enable_left=enable_left,
        enable_right=enable_right,
        hand_type=hand_type,
        hand_cmd_len=hand_cmd_len,
        retarget_config_path_r=retarget_config_path_r,
        retarget_config_path_l=retarget_config_path_l,
    )
    
    # Create and initialize teleoperation system
    print("Creating integrated teleoperation system...")
    teleop = IntegratedTeleop(config)
    
    try:
        # Initialize all controllers
        teleop.initialize()
        
        # Run the main teleoperation loop
        teleop.run()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        # Clean shutdown
        teleop.stop()


if __name__ == '__main__':
    # You can uncomment these lines to use the same default arguments as the original script
    sys.argv = [
        sys.argv[0],
        '--right_ip', '192.168.2.18',
        '--right_port', '8080',
        '--left_ip', '192.168.2.20',
        '--left_port', '8080',
        '--avp_ip', '192.168.2.13',
        '--fps', '30',
        '--data_dir', '/nfs_data1/teleop_data/test',
        '--enable_left',
        #'--enable_right',
        '--hand_type', 'wuji',
        '--hand_cmd_len', '20',
        '--retarget_config_path_r', '/home/wuji/code/dex-real-deployment/configs/wujihand_right_dexpilot.yaml',
        '--retarget_config_path_l', '/home/wuji/code/dex-real-deployment/configs/wujihand_left_dexpilot.yaml',
    ]
    main()
