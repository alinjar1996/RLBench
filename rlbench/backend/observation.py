import numpy as np

from dataclasses import dataclass

@dataclass
class Observation:
    """Storage for both visual and low-dimensional observations."""

    left_shoulder_rgb: np.ndarray
    left_shoulder_depth: np.ndarray
    left_shoulder_mask: np.ndarray
    left_shoulder_point_cloud: np.ndarray
    right_shoulder_rgb: np.ndarray
    right_shoulder_depth: np.ndarray
    right_shoulder_mask: np.ndarray
    right_shoulder_point_cloud: np.ndarray
    overhead_rgb: np.ndarray
    overhead_depth: np.ndarray
    overhead_mask: np.ndarray
    overhead_point_cloud: np.ndarray
    wrist_rgb: np.ndarray
    wrist_depth: np.ndarray
    wrist_mask: np.ndarray
    wrist_point_cloud: np.ndarray
    front_rgb: np.ndarray
    front_depth: np.ndarray
    front_mask: np.ndarray
    front_point_cloud: np.ndarray
    joint_velocities: np.ndarray
    joint_positions: np.ndarray
    joint_forces: np.ndarray
    gripper_open: float
    gripper_pose: np.ndarray
    gripper_matrix: np.ndarray
    gripper_joint_positions: np.ndarray
    gripper_touch_forces: np.ndarray
    task_low_dim_state: np.ndarray
    ignore_collisions: np.ndarray
    misc: dict

    def get_low_dim_data(self) -> np.ndarray:
        """Gets a 1D array of all the low-dimensional obseervations.

        :return: 1D array of observations.
        """
        low_dim_data = [] if self.gripper_open is None else [[self.gripper_open]]
        for data in [self.joint_velocities, self.joint_positions,
                     self.joint_forces,
                     self.gripper_pose, self.gripper_joint_positions,
                     self.gripper_touch_forces, self.task_low_dim_state]:
            if data is not None:
                low_dim_data.append(data)
        return np.concatenate(low_dim_data) if len(low_dim_data) > 0 else np.array([])
