from abc import abstractmethod

import numpy as np

from rlbench.action_modes.arm_action_modes import ArmActionMode
from rlbench.action_modes.gripper_action_modes import GripperActionMode
from rlbench.backend.scene import Scene


class ActionMode(object):

    def __init__(self,
                 arm_action_mode: 'ArmActionMode',
                 gripper_action_mode: 'GripperActionMode'):
        self.arm_action_mode = arm_action_mode
        self.gripper_action_mode = gripper_action_mode

    @abstractmethod
    def action(self, scene: Scene, action: np.ndarray):
        pass

    @abstractmethod
    def action_shape(self, scene: Scene):
        pass




class MoveArmThenGripper(ActionMode):
    """The arm action is first applied, followed by the gripper action. """

    def action(self, scene: Scene, action: np.ndarray):
        assert(len(action) == 9)
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:arm_act_size+1])
        ignore_collisions = bool(action[arm_act_size+1:arm_act_size+2])
        self.arm_action_mode.action(scene, arm_action, ignore_collisions)
        self.gripper_action_mode.action(scene, ee_action)

    def action_shape(self, scene: Scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene))


class BimanualMoveArmThenGripper(MoveArmThenGripper):
    """The arm action is first applied, followed by the gripper action. """

    def action(self, scene: Scene, action: np.ndarray):

        assert(len(action) == 18)

        arm_action_size = np.prod(self.arm_action_mode.unimanual_action_shape(scene))
        ee_action_size = np.prod(self.gripper_action_mode.unimanual_action_shape(scene))
        ignore_collisions_size = 1

        action_size = arm_action_size + ee_action_size + ignore_collisions_size

        assert(action_size == 9)

        right_action = action[:action_size]
        left_action = action[action_size:]

        right_arm_action = np.array(right_action[:arm_action_size])
        left_arm_action = np.array(left_action[:arm_action_size])

        arm_action = np.concatenate([right_arm_action, left_arm_action], axis=0)        

        right_ee_action = np.array(right_action[arm_action_size:arm_action_size+ee_action_size])
        left_ee_action = np.array(left_action[arm_action_size:arm_action_size+ee_action_size])
        ee_action = np.concatenate([right_ee_action, left_ee_action], axis=0)

        right_ignore_collisions = bool(right_action[arm_action_size+ee_action_size:arm_action_size+ee_action_size+1])
        left_ignore_collisions = bool(left_action[arm_action_size+ee_action_size:arm_action_size+ee_action_size+1])
        ignore_collisions = [right_ignore_collisions, left_ignore_collisions]

        self.arm_action_mode.action(scene, arm_action, ignore_collisions)
        self.gripper_action_mode.action(scene, ee_action)
