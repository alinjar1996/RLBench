from typing import List, Tuple
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from rlbench.backend.conditions import JointCondition
from rlbench.backend.task import Task
from rlbench.const import colors


class OpenDrawerTestColor(Task):

    DRAWER_COLORS = [colors[9], colors[9], colors[9]]

    def init_task(self) -> None:
        self._options = ['bottom', 'middle', 'top']
        self._anchors = [Dummy('waypoint_anchor_%s' % opt)
                         for opt in self._options]
        self._joints = [Joint('drawer_joint_%s' % opt)
                        for opt in self._options]
        self._waypoint1 = Dummy('waypoint1')

    def init_episode(self, index: int) -> List[str]:
        color_name, color_rgb = self.DRAWER_COLORS[index]

        drawer_frame = Shape('drawer_frame')
        drawer_top = Shape('drawer_top')
        drawer_middle = Shape('drawer_middle')
        drawer_bottom = Shape('drawer_bottom')

        drawer_frame.set_color(color_rgb)
        drawer_top.set_color(color_rgb)
        drawer_middle.set_color(color_rgb)
        drawer_bottom.set_color(color_rgb)

        option = self._options[index]
        self._waypoint1.set_position(self._anchors[index].get_position())
        self.register_success_conditions(
            [JointCondition(self._joints[index], 0.15)])

        # color_name = ""
        return ['open the %s %s drawer' % (color_name, option),
                'grip the %s handle and pull the %s %s drawer open' % (
                    option, color_name, option),
                'slide the %s %s drawer open' % (color_name, option)]

    def variation_count(self) -> int:
        return 3

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]
