import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.markers.visualization_markers import VisualizationMarkersCfg, VisualizationMarkers

from .retargeting_utils import np_to_torch, quat_to_sim, mat_to_pose, pose_to_pos_euler
import numpy as np


HUGE_FRAME_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/myMarkers",
    markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.5, 0.5, 0.5),
        )
    }
)

LARGE_FRAME_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/myMarkers",
    markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.1, 0.1, 0.1),
        )
    }
)
MIDDLE_FRAME_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/myMarkers",
    markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.05, 0.05, 0.05),
        )
    }
)
PP_FRAME_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/myMarkers",
    markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.02, 0.02, 0.02),
        )
    }
)
SMALL_FRAME_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/myMarkers",
    markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.01, 0.01, 0.01),
        )
    }
)

class AvpViz:
    def __init__(self):
        self.land_marker = VisualizationMarkers(LARGE_FRAME_MARKER_CFG.replace(prim_path="/Visuals/land_marker"))
        self.land_marker.visualize(
            translations=np_to_torch(np.array([[0.0, 0.0, 0.0]])),
            orientations=np_to_torch(quat_to_sim(np.array([[0.0, 0.0, 0.0, 1.0]]))),
        )

        self.right_wrist_marker = VisualizationMarkers(MIDDLE_FRAME_MARKER_CFG.replace(prim_path="/Visuals/right_wrist_marker"))
        self.left_wrist_marker = VisualizationMarkers(MIDDLE_FRAME_MARKER_CFG.replace(prim_path="/Visuals/left_wrist_marker"))

        self.head_marker = VisualizationMarkers(MIDDLE_FRAME_MARKER_CFG.replace(prim_path="/Visuals/head_marker"))

        self.right_fingers_marker = VisualizationMarkers(SMALL_FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/right_fingers_marker",
            markers={
                f"frame_i": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.01, 0.01, 0.01),
                ) for i in range(25)
            }
        ))
        self.left_fingers_marker = VisualizationMarkers(SMALL_FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/left_fingers_marker",
            markers={
                f"frame_i": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.01, 0.01, 0.01),
                ) for i in range(25)
            }
        ))

    def register(self, marker_name: str, marker_size: str):
        """
        Args:
            marker_name: name of the marker to register
            marker_size: size of the marker, one of ['huge', 'large', 'middle', 'pp', 'small']
        """
        assert marker_size in ['huge', 'large', 'middle', 'pp', 'small'], "Invalid marker size."

        marker_cfg = globals()[f"{marker_size.upper()}_FRAME_MARKER_CFG"]
        assert not hasattr(self, f"{marker_name}_marker"), f"Marker {marker_name} already exists in AvpViz."
        setattr(self, f"{marker_name}_marker", VisualizationMarkers(marker_cfg.replace(prim_path=f"/Visuals/{marker_name}_marker")))

    def viz(self, marker_name: str, pose: np.ndarray, already_sim_format: bool = False):
        """
        Args:
            pose: (x, y, z, qx, qy, qz, qw) in world frame
            (7, ) or (N, 7) numpy array

            or

            mat: (4, 4) or (N, 4, 4) numpy array

            or

            pos_euler: (x, y, z, roll, pitch, yaw) in world frame
            (6, ) or (N, 6) numpy array
        """

        marker_instance = getattr(self, marker_name + "_marker", None)
        assert marker_instance is not None, f"Marker {marker_name} does not exist in AvpViz."

        if pose.shape[-1] == 4:
            pose = mat_to_pose(pose) # convert (4, 4) or (N, 4, 4) to (7, ) or (N, 7)

        if pose.shape[-1] == 6:
            pose = pose_to_pos_euler(pose) # convert (6, ) or (N, 6) to (7, ) or (N, 7)

        if pose.ndim == 1:
            pose = pose[None, :]

        if not already_sim_format:
            translations = np_to_torch(pose[..., :3])
            orientations = np_to_torch(quat_to_sim(pose[..., 3:]))
        else:
            translations = pose[..., :3]
            orientations = pose[..., 3:]

        marker_instance.visualize(
            translations=translations,
            orientations=orientations,
        )
