import subprocess
import os
import signal

def kill_process_by_device(dev_id):
    # 构建 fuser 命令，获取设备正在使用的进程
    try:
        result = subprocess.run(['fuser', '-v', f'/dev/video{dev_id}'], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"No processes found using /dev/video{dev_id}.")
            return
        print(result)

        # 解析输出，提取 PID
        lines = result.stdout.splitlines()
        if len(lines) == 0:
            print(f"No processes found using /dev/video{dev_id}.")
            return
        pid_to_kill = int(lines[0].strip())
        print(f"Found process with PID: {pid_to_kill} using /dev/video{dev_id}")

        # 杀死进程
        try:
            os.kill(int(pid_to_kill), signal.SIGKILL)
            print(f"Successfully killed process {pid_to_kill}.")
        except ProcessLookupError:
            print(f"Process {pid_to_kill} not found.")
        except PermissionError:
            print(f"Permission denied to kill process {pid_to_kill}.")
        except Exception as e:
            print(f"Error killing process {pid_to_kill}: {e}")
    except Exception as e:
        print(f"Error: {e}")