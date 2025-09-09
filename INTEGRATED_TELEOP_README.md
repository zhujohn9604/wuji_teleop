# Integrated Teleoperation System

This document describes the new integrated teleoperation system that replaces the complex logic in `avp_teleop_wuji.py` with a simple, clean class-based approach.

## Overview

The `IntegratedTeleop` class provides a unified interface for:
- Processing VisionPro data
- Controlling robot arms (left and right)
- Controlling robot hands

## Key Benefits

1. **Simplified Interface**: Single class handles all teleoperation logic
2. **Clean Separation**: Clear separation between data processing and robot control
3. **Easy to Use**: Simple initialization and run methods
4. **Maintainable**: Well-structured code that's easy to understand and modify
5. **Configurable**: All parameters controlled through a single config object

## Files

- `integrated_teleop.py`: Main integrated teleoperation class
- `simple_teleop_example.py`: Example script showing how to use the class
- `INTEGRATED_TELEOP_README.md`: This documentation

## Quick Start

### Basic Usage

```python
from integrated_teleop import IntegratedTeleop, TeleopConfig

# Create configuration
config = TeleopConfig(
    right_ip='192.168.2.18',
    left_ip='192.168.2.20',
    avp_ip='192.168.2.13',
    fps=30,
    enable_left=True,
    enable_right=True,
)

# Create and run teleoperation system
teleop = IntegratedTeleop(config)
teleop.initialize()
teleop.run()
```

### Using the Example Script

```bash
python simple_teleop_example.py --right_ip 192.168.2.18 --left_ip 192.168.2.20 --avp_ip 192.168.2.13 --enable_left --enable_right
```

## Configuration Options

The `TeleopConfig` class supports all the same parameters as the original script:

### Robot Control
- `right_ip`, `right_port`: Right robot arm connection
- `left_ip`, `left_port`: Left robot arm connection
- `enable_right`, `enable_left`: Enable/disable arms

### VisionPro
- `avp_ip`: VisionPro IP address
- `redis_ip`: Redis server IP (alternative to direct VisionPro connection)

### Control Parameters
- `fps`: Control frequency (default: 30)
- `hand_type`: Type of hand ('wuji', 'rohand', etc.)
- `hand_cmd_len`: Length of hand command (default: 20)

### Retargeting
- `retarget_config_path_r`: Right hand retargeting config
- `retarget_config_path_l`: Left hand retargeting config


## Class Structure

### IntegratedTeleop

Main class that orchestrates the entire teleoperation system.

**Key Methods:**
- `__init__(config)`: Initialize with configuration
- `initialize()`: Initialize all controllers and systems
- `run()`: Start the main teleoperation loop
- `stop()`: Stop the system gracefully
- `get_status()`: Get current system status

**Internal Methods:**
- `_init_avp_controller()`: Initialize VisionPro controller
- `_init_robot_controllers()`: Initialize robot arm controllers
- `_process_vision_data()`: Process VisionPro data to robot commands
- `_send_robot_commands()`: Send commands to robot arms and hands
### TeleopConfig

Configuration class containing all system parameters.

## Data Flow

1. **VisionPro Data**: Raw data from VisionPro or Redis
2. **Processing**: Convert to robot joint angles and hand commands
3. **Robot Control**: Send commands to robot arms and hands

## Migration from avp_teleop_wuji.py

The new system provides the same functionality as the original script but with a much cleaner interface:

### Before (avp_teleop_wuji.py)
```python
# Complex initialization with multiple controllers
realman_avp_controller_config = RealmanAvpControllerConfig(...)
realman_avp_controller = RealmanAvpController(realman_avp_controller_config)
right_realman_controller, left_realman_controller = init_realman_wuji_controller(...)

# Complex main loop with manual timing and data collection
while True:
    # Manual timing calculation
    end_time = start_time + (global_step + 1) * dt
    
    # Manual data processing
    realman_avp_feedback_state_dict = realman_avp_controller.get_feedback()
    right_q = realman_avp_feedback_state_dict["right_q"]
    # ... more manual processing
    
    # Manual robot control
    right_realman_controller.schedule_joint(right_q, end_time)
    # ... more manual control
    
```

### After (IntegratedTeleop)
```python
# Simple initialization
config = TeleopConfig(...)
teleop = IntegratedTeleop(config)
teleop.initialize()
teleop.run()
```

## Error Handling

The integrated system includes proper error handling:
- Graceful shutdown on Ctrl+C
- Exception handling in the main loop
- Proper cleanup of resources

## Thread Safety

The system is designed to be thread-safe:
- Controller communication is handled safely

## Testing

You can test the system by running the test script:

```bash
python test_integrated_teleop.py
```

## Future Extensions

The integrated design makes it easy to add new features:
- Additional robot types
- Different control algorithms
- Real-time visualization
- Web-based control interface

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
2. **Connection Issues**: Check IP addresses and ports
3. **Permission Issues**: Ensure proper permissions for data directories
4. **Timing Issues**: Adjust FPS if experiencing timing problems

### Debug Mode

Enable verbose logging by modifying the configuration or adding debug prints in the code.

## Performance

The integrated system maintains the same performance as the original script while providing a cleaner interface. The main loop is optimized for real-time control with precise timing.
