from pyrep.robots.arms.arm import Arm
from pyrep.robots.end_effectors.gripper import Gripper

from abc import ABC
from abc import abstractmethod

class Robot(ABC):
    """Simple container for the robot components.
    """

    @abstractmethod
    def release_gripper(self):
        pass

    @abstractmethod
    def is_in_collision(self):
        pass
    
    @abstractmethod
    def zero_velocity(self):
        pass

class UnimanualRobot(Robot):

    def __init__(self, arm: Arm, gripper: Gripper):
        self.arm = arm
        self.gripper = gripper

    def release_gripper(self):
        self.gripper.release()

    def initial_state(self):
        return [self.arm.get_configuration_tree(), self.gripper.get_configuration_tree()]

    def is_in_collision(self):
        return self.arm.check_arm_collision()

    def zero_velocity(self):
        self.arm.set_joint_target_velocities([0] * len(self.arm.joints))
        self.gripper.set_joint_target_velocities([0] * len(self.gripper.joints))


        


