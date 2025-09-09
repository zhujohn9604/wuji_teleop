import pinocchio as pin
import numpy as np


class FKController:
    def __init__(self, urdf_path: str, eef_link_name: str):
        """
        Args:
            eef_link_name: Name of the end-effector link in the URDF model.
                e.g. Link7
        """
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.eef_link_id = self.model.getFrameId(eef_link_name, pin.BODY)

    def solve(self, q: np.ndarray) -> np.ndarray:
        """
        Args:
            q: Joint configuration vector of shape (nq,)
        Returns:
            eef_pose: End-effector pose in the world frame as a 4x4 homogeneous transformation matrix.
        """
        pin.forwardKinematics(self.model, self.data, q)
        eef_pose = pin.updateFramePlacement(self.model, self.data, self.eef_link_id).homogeneous # (4, 4)

        return eef_pose
