from collections import defaultdict
from typing import List, Tuple

import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from pyrep.objects.object import Object
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.task import Task
from rlbench.backend.task import BimanualTask
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.conditions import Condition
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.spawn_boundary import SpawnBoundary

from pyrep.const import PrimitiveShape

class LiftedCondition(Condition):

    def __init__(self, item: Shape):
        self.item = item
        # self.min_height = min_height
        self.goal_pos = np.array([0.0, 0.0, 1.0]) # The target goal position


    def condition_met(self):
        # pos = self.item.get_position()
        # goal_pos = [0.25, 0.05, 1.5]
        # # print(pos)
        # return pos[2] >= self.min_height, False

        current_pos = np.array(self.item.get_position())
        
        # Check if the item's current position is close to the goal position
        # A small threshold (e.g., 0.1) can be used to account for floating-point inaccuracies
        distance_to_goal = np.linalg.norm(current_pos - self.goal_pos)

        print("distance_to_goal", distance_to_goal)
        
        # Check if the item has been lifted above the minimum height threshold
        #height_reached = current_pos[2] >= self.min_height

        # Return True if the item is close to the goal position AND above the minimum height
        success = distance_to_goal < 0.04 #and height_reached

        print(success)
        
        return success, False


class BimanualLiftTray(BimanualTask):

    def init_task(self) -> None:
        self.item = Shape('item')
        self.tray = Shape('tray')
        
        # self.obstacle = None
        # # Add a new obstacle in the task initialization
        # self.obstacle = Shape.create(
        #     type=PrimitiveShape.CUBOID,
        #     size=[0.25, 0.25, 0.5],
        #     color=[1.0, 0.0, 0.0],  # Red
        #     position=[0.0, 0.0, 1.0],
        #     mass=600.0,
        #     respondable=True,
        #     renderable=True,
        # )

        self.register_graspable_objects([self.item])

        self.waypoint_mapping = defaultdict(lambda: 'left')
        for i in range(0, 7, 2):
            self.waypoint_mapping.update({f'waypoint{i}': 'right'})

    def init_episode(self, index) -> List[str]:

        # # 1. Get the joint objects for the right and left arms
        # right_arm_joints = [Joint(f'Panda_rightArm_joint{i}') for i in range(1, 8)]
        # left_arm_joints = [Joint(f'Panda_leftArm_joint{i}') for i in range(1, 8)]

        # # 2. Define the desired starting joint positions (in radians)
        # right_arm_start_pos = [0.5, -0.2, 0.0, -1.8, 0.0, 1.6, 0.0]
        # left_arm_start_pos = [0.5, 0.2, 0.0, -1.8, 0.0, 1.6, 0.0]

        # # 3. Use the set_joint_positions method to apply the new configuration
        # Joint.set_joint_position(right_arm_joints, right_arm_start_pos)
        # Joint.set_joint_position(left_arm_joints, left_arm_start_pos)

        # Assuming you have a list of joint objects for each arm
        right_arm_joints = [Joint(f'Panda_rightArm_joint{i}') for i in range(1, 8)]
        left_arm_joints = [Joint(f'Panda_leftArm_joint{i}') for i in range(1, 8)]

        # And a list of corresponding positions
        right_arm_start_pos = [0.5, -0.2, 0.0, -1.8, 0.0, 1.6, 0.0]
        left_arm_start_pos = [0.5, 0.2, 0.0, -1.8, 0.0, 1.6, 0.0]

        # Loop through each joint and set its position individually
        for i in range(len(right_arm_joints)):
            right_arm_joints[i].set_joint_position(right_arm_start_pos[i], disable_dynamics=True)
            left_arm_joints[i].set_joint_position(left_arm_start_pos[i], disable_dynamics=True)


        self._variation_index = index

        right_sensor = ProximitySensor('Panda_rightArm_gripper_attachProxSensor')
        left_sensor = ProximitySensor('Panda_leftArm_gripper_attachProxSensor')

        
        #tray_visual = Shape('tray_visual')
        #print(self.item.get_position())
        #tray_visual.sample(self.item, min_distance=0.1, ignore_collisions=True)
        self.item.set_position([0.0, 0.0, 0.001], relative_to=self.tray, reset_dynamics=False)
        # print(self.item.get_position())

        self.register_success_conditions([
            LiftedCondition(self.tray),
            LiftedCondition(self.item),
            DetectedCondition(self.tray, right_sensor),
            DetectedCondition(self.tray, left_sensor)])

 

        return ['Lift the tray']

    def variation_count(self) -> int:
        return 1 #len(self._options)

    def boundary_root(self) -> Object:
        return Dummy('bimanual_lift_tray')

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]
