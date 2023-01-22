from collections import defaultdict
from typing import List, Tuple

import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.task import BimanualTask
from rlbench.backend.spawn_boundary import SpawnBoundary
from pyrep.objects.dummy import Dummy

colors = [
    ('red', (1.0, 0.0, 0.0)),
    ('green', (0.0, 1.0, 0.0)),
    ('blue', (0.0, 0.0, 1.0)),
]

class HandoverItemEasy(BimanualTask):

    def init_task(self) -> None:

        self.items = [Shape(f'item{i}') for i in range(3)]

        for i, (_, color) in enumerate(colors):
            self.items[i].set_color(color)

        self.register_graspable_objects(self.items)

        self.waypoint_mapping = defaultdict(lambda: 'left')
        self.waypoint_mapping.update({'waypoint0': 'right', 'waypoint5': 'right'})

        self.boundaries = Shape('handover_item_boundary')


    def init_episode(self, index: int) -> List[str]:

        self._variation_index = index

        success_sensor = ProximitySensor('Panda_rightArm_gripper_attachProxSensor')

        color_name, _color = colors[index]

        w0 = Dummy('waypoint2')
        w0.set_position([0.0, 0.0, -0.025], relative_to=self.items[index], reset_dynamics=False)
        #w0.set_orientation([-np.pi, 0, -np.pi], relative_to=self.items[index], reset_dynamics=False)

        w1 = Dummy('waypoint1')
        w1.set_position([0.0, 0.0, 0.1], relative_to=self.items[index], reset_dynamics=False)

        w3 = Dummy('waypoint3')
        w3.set_position([0.0, 0.0, 0.1], relative_to=self.items[index], reset_dynamics=False)



        #b = SpawnBoundary([self.boundaries])
        #b.clear()
        #for item in self.items:
        #    b.sample(item, min_distance=0.1)

        self.register_success_conditions(
            [DetectedCondition(self.items[index], success_sensor)])

        return [f'bring me the {color_name} item',
                f'hand over the {color_name} object']

    def variation_count(self) -> int:
        return len(colors)

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]
