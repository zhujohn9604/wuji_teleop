import sys

try:
    import wujihand_python_binding as wh
    print("模块导入成功!")
except ImportError as e:
    print(f"模块导入失败: {e}")
    print("请检查:")
    print("1. wujihand_python_binding.so 是否与python文件在同一目录")
    print("2. 是否已编译生成 .so 文件")
    print("3. Python版本是否兼容")
    print("4. 系统架构是否匹配 (x86_64 vs arm64)")
    sys.exit(1)
except Exception as e:
    print(f"导入时发生未知错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

class WujiHand:
    def __init__(self, vid: int = 0x0483, pid: int = 0x5740):
        self.vid = vid
        self.pid = pid
        self._enabled = False  # 跟踪手部状态
        
        try:
            self.hand = wh.WujiHand(vid, pid)
            print("设备连接成功!")
        except Exception as e:
            print(f"设备连接失败: {e}")
            sys.exit(1)
            
        # 设置信号处理，确保程序退出时手部失能
        import signal
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGTSTP, self._signal_handler)
        
        self.set_initial_positions()
        self.hand.enable_hand()
        self._enabled = True
        self.hand.set_sdo_mode()
        
        # 注册atexit处理器，确保程序退出时清理
        import atexit
        atexit.register(self.cleanup)
    
    def _signal_handler(self, signum, frame):
        """信号处理函数"""
        print(f"\n收到信号 {signum}，正在停止...")
        self.cleanup()
        sys.exit(0)

    def __del__(self):
        try:
            if hasattr(self, 'hand') and self.hand and self._enabled:
                self.hand.disable_hand()
                print("手部已禁用")
        except Exception as e:
            print(f"清理资源时出错: {e}")

    def cleanup(self):
        """手动清理资源"""
        try:
            if hasattr(self, 'hand') and self.hand and self._enabled:
                self.hand.disable_hand()
                self._enabled = False
                print("手部已禁用")
        except Exception as e:
            print(f"清理资源时出错: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()

    def set_initial_positions(self):
        try:
            positions = [
                0x000000, 0x200000, 0x200000, 0x200000,  # 拇指关节
                0xFFFFFF, 0x8FFFFF, 0x000000, 0x000000,  # 手指1 关节0-3
                0xFFFFFF, 0x8FFFFF, 0x000000, 0x000000,  # 手指2 关节0-3
                0xFFFFFF, 0x8FFFFF, 0x000000, 0x000000,  # 手指3 关节0-3
                0xFFFFFF, 0x8FFFFF, 0x000000, 0x000000   # 手指4 关节0-3
            ]
            # 初始化设置所有关节位置
            self.hand.init_joint_positions(positions)
            print("初始位置设置完成")
        except Exception as e:
            print(f"设置初始位置失败: {e}")

    def set_joint_position_async(self, finger_idx: int, joint_idx: int, position: int):
        """SDO模式：异步设置单个关节位置"""
        try:
            self.hand.set_joint_position_async(finger_idx, joint_idx, position)
        except Exception as e:
            print(f"设置关节位置失败: {e}")

    def trigger_transmission(self):
        """SDO模式：触发传输"""
        try:
            self.hand.trigger_transmission()
        except Exception as e:
            print(f"触发传输失败: {e}")

    def set_joint_positions_async(self, positions: list[int]):
        """SDO模式：异步设置所有关节位置，然后触发传输"""
        try:
            # 设置每个关节位置
            for finger_idx in range(5):
                for joint_idx in range(4):
                    base_idx = finger_idx * 4 + joint_idx
                    self.hand.set_joint_position_async(finger_idx, joint_idx, positions[base_idx])
            
            # 触发传输
            self.hand.trigger_transmission()
        except Exception as e:
            print(f"设置关节位置失败: {e}") 