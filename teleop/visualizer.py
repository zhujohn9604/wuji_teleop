import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R

class RobotArmVisualizer:
    def __init__(self, dh_params=None):
        """
        初始化机器人手臂可视化器

        Args:
            dh_params: DH参数 [[a, alpha, d, theta_offset], ...] 每个关节一行
                      如果为None，将使用默认的7DOF机器人参数
        """
        if dh_params is None:
            # 默认7DOF机器人的DH参数 (需要根据实际机器人调整)
            self.dh_params = np.array([
                [0.0,    np.pi/2,  0.2755, 0.0],      # Joint 1
                [0.0,   -np.pi/2,  0.0,    0.0],      # Joint 2
                [0.0,    np.pi/2,  0.41,   0.0],      # Joint 3
                [0.0,   -np.pi/2,  0.0,    0.0],      # Joint 4
                [0.0,    np.pi/2,  0.3915, 0.0],      # Joint 5
                [0.0,   -np.pi/2,  0.0,    0.0],      # Joint 6
                [0.0,    0.0,      0.078,  0.0]       # Joint 7
            ])
        else:
            self.dh_params = np.array(dh_params)

        self.fig = None
        self.ax = None
        self.lines = []
        self.points = []
        self.is_initialized = False

        # 轨迹历史记录
        self.target_trajectory = []
        self.actual_trajectory = []
        self.max_trajectory_length = 100  # 最多显示100个历史点

        # 设置matplotlib为非阻塞模式
        plt.ion()

    def dh_transform(self, a, alpha, d, theta):
        """计算DH变换矩阵"""
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        T = np.array([
            [ct,    -st*ca,  st*sa,   a*ct],
            [st,     ct*ca, -ct*sa,   a*st],
            [0,      sa,     ca,      d],
            [0,      0,      0,       1]
        ])
        return T

    def forward_kinematics(self, joint_angles):
        """
        计算正向运动学，返回每个关节的位置

        Args:
            joint_angles: 7个关节角度的numpy数组

        Returns:
            positions: (8, 3) 数组，包含基座和7个关节的位置
            transforms: 每个关节的变换矩阵列表
        """
        assert len(joint_angles) == 7, "需要7个关节角度"

        # 基座位置
        positions = [np.array([0.0, 0.0, 0.0])]
        transforms = []

        # 累积变换矩阵
        T_cumulative = np.eye(4)

        for i, (joint_angle) in enumerate(joint_angles):
            a, alpha, d, theta_offset = self.dh_params[i]
            theta = joint_angle + theta_offset

            # 计算当前关节的DH变换
            T_i = self.dh_transform(a, alpha, d, theta)

            # 累积变换
            T_cumulative = T_cumulative @ T_i
            transforms.append(T_cumulative.copy())

            # 提取位置
            position = T_cumulative[:3, 3]
            positions.append(position)

        return np.array(positions), transforms

    def setup_plot(self):
        """设置3D绘图环境"""
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # 设置坐标轴
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Robot Arm Real-time Visualization (Target vs Actual)')

        # 设置坐标轴范围
        limit = 1.0
        self.ax.set_xlim([-limit, limit])
        self.ax.set_ylim([-limit, limit])
        self.ax.set_zlim([0, 1.5])

        # 初始化机器人手臂线条
        self.arm_line, = self.ax.plot([], [], [], 'b-', linewidth=3, marker='o', markersize=8,
                                     markerfacecolor='red', markeredgecolor='black', label='Actual Robot')

        # 初始化目标轨迹线条
        self.target_trajectory_line, = self.ax.plot([], [], [], 'g-', linewidth=2, alpha=0.7, label='Target Trajectory')
        self.actual_trajectory_line, = self.ax.plot([], [], [], 'r-', linewidth=2, alpha=0.7, label='Actual Trajectory')

        # 初始化当前目标点和实际末端点
        self.target_point, = self.ax.plot([], [], [], 'go', markersize=10, label='Current Target')
        self.actual_point, = self.ax.plot([], [], [], 'ro', markersize=8, label='Actual End-effector')

        # 添加图例
        self.ax.legend()

        self.is_initialized = True
        plt.show(block=False)

        return self.fig, self.ax

    def update_plot(self, joint_angles, target_position=None, show_frames=False):
        """
        非阻塞更新机器人手臂可视化

        Args:
            joint_angles: 7个关节角度的numpy数组
            target_position: 目标末端位置 (x, y, z)
            show_frames: 是否显示坐标系
        """
        if not self.is_initialized:
            self.setup_plot()

        # 计算实际位置
        positions, transforms = self.forward_kinematics(joint_angles)
        actual_end_position = positions[-1]  # 末端执行器位置

        # 更新轨迹历史
        if target_position is not None:
            self.target_trajectory.append(target_position)
            if len(self.target_trajectory) > self.max_trajectory_length:
                self.target_trajectory.pop(0)

        self.actual_trajectory.append(actual_end_position)
        if len(self.actual_trajectory) > self.max_trajectory_length:
            self.actual_trajectory.pop(0)

        # 更新机器人手臂
        self.arm_line.set_data_3d(positions[:, 0], positions[:, 1], positions[:, 2])

        # 更新轨迹线条
        if len(self.target_trajectory) > 1:
            target_traj = np.array(self.target_trajectory)
            self.target_trajectory_line.set_data_3d(target_traj[:, 0], target_traj[:, 1], target_traj[:, 2])

        if len(self.actual_trajectory) > 1:
            actual_traj = np.array(self.actual_trajectory)
            self.actual_trajectory_line.set_data_3d(actual_traj[:, 0], actual_traj[:, 1], actual_traj[:, 2])

        # 更新当前点
        if target_position is not None:
            self.target_point.set_data_3d([target_position[0]], [target_position[1]], [target_position[2]])

        self.actual_point.set_data_3d([actual_end_position[0]], [actual_end_position[1]], [actual_end_position[2]])

        # 清除之前的坐标系线条
        if hasattr(self, 'frame_lines'):
            for line in self.frame_lines:
                line.remove()
        self.frame_lines = []

        # 绘制坐标系
        if show_frames:
            frame_size = 0.1
            for i, T in enumerate(transforms):
                origin = T[:3, 3]
                x_axis = T[:3, 0] * frame_size
                y_axis = T[:3, 1] * frame_size
                z_axis = T[:3, 2] * frame_size

                # X轴 (红色)
                line_x, = self.ax.plot([origin[0], origin[0] + x_axis[0]],
                                      [origin[1], origin[1] + x_axis[1]],
                                      [origin[2], origin[2] + z_axis[2]], 'r-', linewidth=2)
                self.frame_lines.append(line_x)

                # Y轴 (绿色)
                line_y, = self.ax.plot([origin[0], origin[0] + y_axis[0]],
                                      [origin[1], origin[1] + y_axis[1]],
                                      [origin[2], origin[2] + y_axis[2]], 'g-', linewidth=2)
                self.frame_lines.append(line_y)

                # Z轴 (蓝色)
                line_z, = self.ax.plot([origin[0], origin[0] + z_axis[0]],
                                      [origin[1], origin[1] + z_axis[1]],
                                      [origin[2], origin[2] + z_axis[2]], 'b-', linewidth=2)
                self.frame_lines.append(line_z)

        # 刷新图形但不阻塞
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def clear_trajectories(self):
        """清除轨迹历史"""
        self.target_trajectory = []
        self.actual_trajectory = []

    def close(self):
        """关闭可视化窗口"""
        if self.fig is not None:
            plt.close(self.fig)
            self.is_initialized = False


class TrajectoryGenerator:
    """轨迹生成器类"""

    @staticmethod
    def circular_trajectory(t, center=[0.5, 0.0, 0.8], radius=0.2, frequency=1.0):
        """生成圆形轨迹"""
        angle = 2 * np.pi * frequency * t
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = center[2]
        return np.array([x, y, z])

    @staticmethod
    def figure_eight_trajectory(t, center=[0.5, 0.0, 0.8], scale=0.15, frequency=0.5):
        """生成8字形轨迹"""
        angle = 2 * np.pi * frequency * t
        x = center[0] + scale * np.sin(angle)
        y = center[1] + scale * np.sin(2 * angle)
        z = center[2]
        return np.array([x, y, z])

    @staticmethod
    def spiral_trajectory(t, center=[0.5, 0.0, 0.8], radius_scale=0.1, height_scale=0.2, frequency=0.3):
        """生成螺旋轨迹"""
        angle = 2 * np.pi * frequency * t
        radius = radius_scale * (1 + 0.5 * t)
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        z = center[2] + height_scale * np.sin(angle * 2)
        return np.array([x, y, z])

    @staticmethod
    def lissajous_trajectory(t, center=[0.5, 0.0, 0.8], amplitude=[0.2, 0.15, 0.1],
                           frequency=[1.0, 1.5, 2.0], phase=[0, np.pi/4, np.pi/2]):
        """生成利萨如曲线轨迹"""
        x = center[0] + amplitude[0] * np.sin(2 * np.pi * frequency[0] * t + phase[0])
        y = center[1] + amplitude[1] * np.sin(2 * np.pi * frequency[1] * t + phase[1])
        z = center[2] + amplitude[2] * np.sin(2 * np.pi * frequency[2] * t + phase[2])
        return np.array([x, y, z])


def main():
    # 示例使用
    dh_params = [
        [0.0, np.pi/2, 0.2755, 0.0],
        [0.0, -np.pi/2, 0.0, 0.0],
        [0.0, np.pi/2, 0.41, 0.0],
        [0.0, -np.pi/2, 0.0, 0.0],
        [0.0, np.pi/2, 0.3915, 0.0],
        [0.0, -np.pi/2, 0.0, 0.0],
        [0.0, 0.0, 0.078, 0.0]
    ]

    visualizer = RobotArmVisualizer(dh_params)
    joint_angles = np.zeros(7)  # 初始关节角度
    trajectory_gen = TrajectoryGenerator()

    # 设置初始图形
    visualizer.setup_plot()

    # 模拟轨迹跟踪
    dt = 0.03  # 30Hz
    total_time = 10  # 10秒
    steps = int(total_time / dt)

    for i in range(steps):
        t = i * dt

        # 生成目标位置（可以切换不同的轨迹）
        target_pos = trajectory_gen.circular_trajectory(t, frequency=0.5)
        # target_pos = trajectory_gen.figure_eight_trajectory(t)
        # target_pos = trajectory_gen.lissajous_trajectory(t)

        # 模拟IK求解（这里用简单的逼近）
        target_error = np.random.normal(0, 0.05, 3)  # 模拟IK误差
        actual_pos = target_pos + target_error

        # 从实际位置反推关节角度（这里用随机变化模拟）
        joint_angles += np.random.normal(0, 0.05, size=7)
        joint_angles = np.clip(joint_angles, -np.pi, np.pi)

        # 更新可视化
        visualizer.update_plot(joint_angles, target_position=target_pos, show_frames=False)

    # 关闭可视化
    input("Press Enter to close...")
    visualizer.close()

if __name__ == "__main__":
    main()
