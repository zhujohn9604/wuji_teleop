#!/usr/bin/env python3
"""
Test script for the IntegratedTeleop class.

This script demonstrates how to use the integrated teleoperation system
and can be used for testing and validation.
"""

import time
import sys
from integrated_teleop import IntegratedTeleop, TeleopConfig


def test_config_creation():
    """Test that we can create a configuration object."""
    print("Testing configuration creation...")
    
    config = TeleopConfig(
        right_ip='192.168.2.18',
        left_ip='192.168.2.20',
        avp_ip='192.168.2.13',
        fps=30,
        enable_left=True,
        enable_right=True,
    )
    
    print(f"✓ Configuration created successfully")
    print(f"  - Right IP: {config.right_ip}")
    print(f"  - Left IP: {config.left_ip}")
    print(f"  - AVP IP: {config.avp_ip}")
    print(f"  - FPS: {config.fps}")
    print(f"  - Enable Left: {config.enable_left}")
    print(f"  - Enable Right: {config.enable_right}")
    
    return config


def test_teleop_creation(config):
    """Test that we can create a teleoperation object."""
    print("\nTesting teleoperation object creation...")
    
    try:
        teleop = IntegratedTeleop(config)
        print("✓ Teleoperation object created successfully")
        return teleop
    except Exception as e:
        print(f"✗ Failed to create teleoperation object: {e}")
        return None


def test_initialization(teleop):
    """Test that we can initialize the teleoperation system."""
    print("\nTesting system initialization...")
    
    try:
        teleop.initialize()
        print("✓ System initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize system: {e}")
        return False


def test_status_check(teleop):
    """Test that we can check system status."""
    print("\nTesting status check...")
    
    try:
        status = teleop.get_status()
        print("✓ Status check successful")
        print(f"  - Is Running: {status['is_running']}")
        print(f"  - Global Step: {status['global_step']}")
        print(f"  - Elapsed Time: {status['elapsed_time']:.2f}s")
        return True
    except Exception as e:
        print(f"✗ Failed to check status: {e}")
        return False


def test_short_run(teleop, duration=5):
    """Test running the system for a short duration."""
    print(f"\nTesting short run for {duration} seconds...")
    
    try:
        # Start the system in a separate thread
        import threading
        
        def run_teleop():
            teleop.run()
        
        # Start the teleop thread
        teleop_thread = threading.Thread(target=run_teleop)
        teleop_thread.daemon = True
        teleop_thread.start()
        
        # Let it run for the specified duration
        time.sleep(duration)
        
        # Stop the system
        teleop.stop()
        
        # Wait for thread to finish
        teleop_thread.join(timeout=2)
        
        print(f"✓ Short run completed successfully")
        return True
    except Exception as e:
        print(f"✗ Failed during short run: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Integrated Teleoperation System Test")
    print("=" * 60)
    
    # Test 1: Configuration creation
    config = test_config_creation()
    if not config:
        print("✗ Configuration test failed")
        return False
    
    # Test 2: Teleoperation object creation
    teleop = test_teleop_creation(config)
    if not teleop:
        print("✗ Teleoperation object creation test failed")
        return False
    
    # Test 3: System initialization
    if not test_initialization(teleop):
        print("✗ Initialization test failed")
        return False
    
    # Test 4: Status check
    if not test_status_check(teleop):
        print("✗ Status check test failed")
        return False
    
    # Test 5: Short run
    if not test_short_run(teleop, duration=3):
        print("✗ Short run test failed")
        return False
    
    print("\n" + "=" * 60)
    print("All tests completed successfully! ✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
