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

    def __init__(self, item: Shape, min_height: float):
        self.item = item
        self.min_height = min_height
        # print(self.item)

    def condition_met(self):
        pos = self.item.get_position()
        return pos[2] >= self.min_height, False


class BimanualLiftBall(BimanualTask):

    def init_task(self) -> None:
        self.ball = Shape('ball')
        self.table = Shape('diningTable_visible')
        #self.register_graspable_objects([self.tray])
        self.obstacle = None
        # Add a new obstacle in the task initialization
        self.obstacle = Shape.create(
            type=PrimitiveShape.CUBOID,
            size=[0.25, 0.25, 0.5],
            color=[1.0, 0.0, 0.0],  # Red
            position=[0.0, 0.0, 1.0],
            mass=600.0,
            respondable=True,
            renderable=True,
        )

        print('Position' , self.ball.get_position())
        print('Orientation' ,self.ball.get_orientation())
        print('Pose' ,self.ball.get_pose())
        print('Mass' ,self.ball.get_mass())
        print("Bounding Box", self.ball.get_bounding_box())
        
        print('Table Position' , self.table.get_position())
        print('Table Orientation' ,self.table.get_orientation())
        print('Table Pose' ,self.table.get_pose())
        print('Table Mass' ,self.table.get_mass())
        print("Table Bounding Box", self.table.get_bounding_box())
        
        
        

        # print('Shape', self.ball.get)
        # print("Inertia", self.ball.get)
        # print('Transformation' ,self.ball.get_matrix())


        self.waypoint_mapping = defaultdict(lambda: 'left')
        for i in range(0, 7, 2):
            self.waypoint_mapping.update({f'waypoint{i}': 'right'})

    def init_episode(self, index) -> List[str]:
        self._variation_index = index

        self.register_success_conditions([LiftedCondition(self.ball, 1.2)])

        return ['Lift the ball']

    def variation_count(self) -> int:
        return 1 #len(self._options)

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]
