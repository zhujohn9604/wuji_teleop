
import pinocchio as pin

from pink import solve_ik
from pink.configuration import Configuration
from pinocchio.robot_wrapper import RobotWrapper

from pink.tasks import FrameTask

import torch
import numpy as np

from utils.retargeting_utils import mat_to_pinse3



class PinkIKController:
    def __init__(self, urdf_path: str, mesh_path: str, variable_input_tasks: list[FrameTask], fixed_input_tasks: list[FrameTask], dt: float, init_q: pin.SE3 = None):
        """
        Args:
            init_q: Initial joint positions in pink ordering. If None, uses the default configuration.
        """
        self.urdf_path = urdf_path
        self.mesh_path = mesh_path

        self.robot_wrapper = RobotWrapper.BuildFromURDF(urdf_path, mesh_path, root_joint=None)
        self.pink_configuration = Configuration(
            self.robot_wrapper.model, self.robot_wrapper.data, self.robot_wrapper.q0
        )

        self.pink_joint_names = self.robot_wrapper.model.names.tolist()[1:]  # Skip the root and universal joints

        self.variable_input_tasks = variable_input_tasks
        self.fixed_input_tasks = fixed_input_tasks

        if init_q is None:
            self.q = pin.neutral(self.robot_wrapper.model)
        else:
            self.q = init_q

        self.dt = dt

    def set_target(
        self,
        mat: np.ndarray
    ):
        """
        Args:
            mat: (4, 4), eef_target_pose_base_link
        """
        pinse3 = mat_to_pinse3(mat)
        task = self.variable_input_tasks[0]
        task.set_target(pinse3)

    def compute(
        self,
        q = None,
        # curr_joint_pos: np.ndarray,
    ) -> np.ndarray:
        """Compute the target joint positions based on current state and tasks.

        Args:
            curr_joint_pos: The current joint positions in pink ordering.
            dt: The time step for computing joint position changes.
            q: Optional; if provided, it overrides the current joint positions.

        Returns:
            The target joint positions.
        """
        if q is not None:
            self.q = q

        # Update Pink's robot configuration with the current joint positions
        self.pink_configuration.update(self.q)

        # pink.solve_ik can raise an exception if the solver fails
        try:
            velocity = solve_ik(
                self.pink_configuration, self.variable_input_tasks + self.fixed_input_tasks, self.dt, solver="osqp", safety_break=False
            )
            Delta_q = velocity * self.dt
        except (AssertionError, Exception):
            # Print warning and return the current joint positions as the target
            # Not using omni.log since its not available in CI during docs build
            # if self.cfg.show_ik_warnings:
            print(
                "Warning: IK quadratic solver could not find a solution! Did not update the target joint positions."
            )
            return self.q

        self.q = self.q + Delta_q

        return self.q

def main():
    urdf_path = "/home/wuji/code/dex-real-deployment/urdf/rm75b/rm_75_b_description.urdf"
    mesh_path = "/home/wuji/code/dex-real-deployment/urdf/rm75b/meshes"

    pink_ik_controller = PinkIKController(
        urdf_path=urdf_path,
        mesh_path=mesh_path,
        variable_input_tasks=[
            FrameTask(
                "Link7",
                position_cost=1.0,  # [cost] / [m]
                orientation_cost=1.0,  # [cost] / [rad]
                lm_damping=10,  # dampening for solver for step jumps
                gain=0.1,
            ),
        ],
        fixed_input_tasks=[],
    )



if __name__ == "__main__":
    main()
