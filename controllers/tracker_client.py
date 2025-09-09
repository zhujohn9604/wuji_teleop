import socket
import json
import time
import sys

# TCP客户端配置
SERVER_HOST = '192.168.2.101'  # 服务器IP地址
SERVER_PORT = 12345

def connect_to_server():
    """创建并连接到服务器"""
    while True:
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((SERVER_HOST, SERVER_PORT))
            print(f"成功连接到服务器 {SERVER_HOST}:{SERVER_PORT}")
            return client_socket
        except Exception as e:
            print(f"连接失败: {e}，5秒后重试...")
            time.sleep(5)

def format_tracker_data(tracker_name, data):
    """格式化tracker数据显示"""
    if data:
        return f"{tracker_name}: " + " ".join([f"{x:7.4f}" for x in data])
    else:
        return f"{tracker_name}: 无数据"

def main():
    """主函数"""
    print("Tracker数据接收客户端")
    print("=" * 60)

    # 连接到服务器
    client_socket = connect_to_server()

    # 接收缓冲区
    buffer = ""

    try:
        while True:
            try:
                # 接收数据
                data = client_socket.recv(4096).decode('utf-8')

                if not data:
                    print("\n服务器断开连接")
                    client_socket.close()
                    client_socket = connect_to_server()
                    continue

                # 处理接收到的数据（可能包含多条JSON消息）
                buffer += data
                lines = buffer.split('\n')
                buffer = lines[-1]  # 保留未完成的行

                for line in lines[:-1]:
                    if line.strip():
                        try:
                            # 解析JSON数据
                            tracker_data = json.loads(line)

                            # 提取时间戳
                            timestamp = tracker_data.get('timestamp', 0)

                            # 提取tracker数据
                            tracker_1_data = tracker_data.get('tracker_1', [])
                            tracker_2_data = tracker_data.get('tracker_2', [])

                            # 清屏并显示数据
                            sys.stdout.write('\033[2J\033[H')  # 清屏并移动光标到顶部
                            print("Tracker实时数据接收客户端")
                            print("=" * 80)
                            print(f"服务器: {SERVER_HOST}:{SERVER_PORT}")
                            print(f"时间戳: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}")
                            print("-" * 80)

                            # 显示tracker数据
                            print(format_tracker_data("tracker_1", tracker_1_data))
                            print(format_tracker_data("tracker_2", tracker_2_data))

                            # 显示数据解释
                            print("-" * 80)
                            print("数据格式: X     Y     Z     Roll  Pitch Yaw  (位置单位:米, 角度单位:度)")

                            # 如果需要更详细的显示
                            if tracker_1_data and len(tracker_1_data) == 6:
                                print(f"\ntracker_1详细:")
                                print(f"  位置: X={tracker_1_data[0]:.4f}m, Y={tracker_1_data[1]:.4f}m, Z={tracker_1_data[2]:.4f}m")
                                print(f"  姿态: Roll={tracker_1_data[3]:.4f}°, Pitch={tracker_1_data[4]:.4f}°, Yaw={tracker_1_data[5]:.4f}°")

                            if tracker_2_data and len(tracker_2_data) == 6:
                                print(f"\ntracker_2详细:")
                                print(f"  位置: X={tracker_2_data[0]:.4f}m, Y={tracker_2_data[1]:.4f}m, Z={tracker_2_data[2]:.4f}m")
                                print(f"  姿态: Roll={tracker_2_data[3]:.4f}°, Pitch={tracker_2_data[4]:.4f}°, Yaw={tracker_2_data[5]:.4f}°")

                            # 显示原始数据（与tracker_test.py格式一致）
                            print(f"\n原始格式:")
                            if tracker_1_data:
                                txt_1 = " ".join([f"{x:.4f}" for x in tracker_1_data])
                                print(f"tracker_1: {txt_1}")
                            if tracker_2_data:
                                txt_2 = " ".join([f"{x:.4f}" for x in tracker_2_data])
                                print(f"tracker_2: {txt_2}")

                        except json.JSONDecodeError as e:
                            print(f"JSON解析错误: {e}")
                        except Exception as e:
                            print(f"处理数据错误: {e}")

            except socket.timeout:
                continue
            except Exception as e:
                print(f"\n接收数据错误: {e}")
                client_socket.close()
                client_socket = connect_to_server()

    except KeyboardInterrupt:
        print("\n\n客户端退出")
        client_socket.close()
        sys.exit(0)

if __name__ == "__main__":
    main()