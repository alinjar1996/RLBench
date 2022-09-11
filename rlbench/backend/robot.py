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

class UnimanualRobot(Robot):

    def __init__(self, arm: Arm, gripper: Gripper):
        self.arm = arm
        self.gripper = gripper

    def release_gripper(self):
        self.gripper.release()

    def initial_state(self):
        return [self.arm.get_configuration_tree(), self.gripper.get_configuration_tree()]

        


