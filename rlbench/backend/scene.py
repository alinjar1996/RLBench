from typing import List, Callable

import numpy as np
from pyrep import PyRep
from pyrep.const import ObjectType
from pyrep.errors import ConfigurationPathError
from pyrep.objects import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.robots.arms.arm import Arm
from pyrep.robots.arms.dual_panda import PandaLeft, PandaRight
from pyrep.robots.end_effectors.gripper import Gripper

from rlbench.backend.exceptions import (
    WaypointError, BoundaryError, NoWaypointsError, DemoError)
from rlbench.backend.observation import Observation
from rlbench.backend.observation import UnimodalObservationData
from rlbench.backend.observation import UnimodalObservation
from rlbench.backend.observation import BimanualObservation

from rlbench.backend.robot import Robot
from rlbench.backend.robot import UnimanualRobot
from rlbench.backend.robot import BimanualRobot
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.task import Task
from rlbench.backend.utils import rgb_handles_to_mask
from rlbench.demo import Demo
from rlbench.noise_model import NoiseModel
from rlbench.observation_config import ObservationConfig, CameraConfig

STEPS_BEFORE_EPISODE_START = 10

import logging


class Scene(object):
    """Controls what is currently in the vrep scene. This is used for making
    sure that the tasks are easily reachable. This may be just replaced by
    environment. Responsible for moving all the objects. """

    def __init__(self,
                 pyrep: PyRep,
                 robot: Robot,
                 obs_config: ObservationConfig = ObservationConfig(),
                 robot_setup: str = 'panda'):
        self.pyrep = pyrep
        self.robot = robot
        self.robot_setup = robot_setup
        self.task = None
        self._obs_config = obs_config
        self._initial_task_state = None


        if self.robot.is_bimanual:
            self._start_arm_joint_pos = [robot.right_arm.get_joint_positions(), robot.left_arm.get_joint_positions()]
            self._starting_gripper_joint_pos = [robot.right_gripper.get_joint_positions(), robot.left_gripper.get_joint_positions()]
            logging.warning("Only using cameras for left arm")
            name_prefix = 'Panda_leftArm_'
        else:
            self._start_arm_joint_pos = robot.arm.get_joint_positions()
            self._starting_gripper_joint_pos = robot.gripper.get_joint_positions()
            name_prefix = ''
    
        self._workspace = Shape('workspace')
        self._workspace_boundary = SpawnBoundary([self._workspace])
        
        
        self._cam_over_shoulder_left = VisionSensor(f'cam_over_shoulder_left')
        self._cam_over_shoulder_right = VisionSensor(f'cam_over_shoulder_right')
        self._cam_overhead = VisionSensor('cam_overhead')
        self._cam_wrist = VisionSensor(f'{name_prefix}cam_wrist')
        self._cam_front = VisionSensor(f'cam_front')


        self._cam_over_shoulder_left_mask = VisionSensor(
                'cam_over_shoulder_left_mask')
        self._cam_over_shoulder_right_mask = VisionSensor(
                'cam_over_shoulder_right_mask')


        self._cam_overhead_mask = VisionSensor('cam_overhead_mask')
        self._cam_wrist_mask = VisionSensor(f'{name_prefix}cam_wrist_mask')
        self._cam_front_mask = VisionSensor(f'cam_front_mask')
        

        self._has_init_task = self._has_init_episode = False
        self._variation_index = 0

        if self.robot.is_bimanual:
            self._initial_robot_state = [(robot.right_arm.get_configuration_tree(),
                                     robot.right_gripper.get_configuration_tree()),
                                     (robot.left_arm.get_configuration_tree(),
                                     robot.left_gripper.get_configuration_tree())]
        else:
            self._initial_robot_state = (robot.arm.get_configuration_tree(),
                                     robot.gripper.get_configuration_tree())

        self._ignore_collisions_for_current_waypoint = False

        # Set camera properties from observation config
        self._set_camera_properties()

        x, y, z = self._workspace.get_position()
        minx, maxx, miny, maxy, _, _ = self._workspace.get_bounding_box()
        self._workspace_minx = x - np.fabs(minx) - 0.2
        self._workspace_maxx = x + maxx + 0.2
        self._workspace_miny = y - np.fabs(miny) - 0.2
        self._workspace_maxy = y + maxy + 0.2
        self._workspace_minz = z
        self._workspace_maxz = z + 1.0  # 1M above workspace

        self.target_workspace_check = Dummy.create()
        self._step_callback = None

        if self.robot.is_bimanual:
               self._robot_shapes = [self.robot.right_arm.get_objects_in_tree(object_type=ObjectType.SHAPE), 
               self.robot.left_arm.get_objects_in_tree(object_type=ObjectType.SHAPE)]
        else:
            self._robot_shapes = self.robot.arm.get_objects_in_tree(
                object_type=ObjectType.SHAPE)

    def load(self, task: Task) -> None:
        """Loads the task and positions at the centre of the workspace.

        :param task: The task to load in the scene.
        """
        task.load()  # Load the task in to the scene

        # Set at the centre of the workspace
        task.get_base().set_position(self._workspace.get_position())

        self._initial_task_state = task.get_state()
        self.task = task
        self._initial_task_pose = task.boundary_root().get_orientation()
        self._has_init_task = self._has_init_episode = False
        self._variation_index = 0

    def unload(self) -> None:
        """Clears the scene. i.e. removes all tasks. """
        if self.task is not None:
            self.robot.release_gripper()
            if self._has_init_task:
                self.task.cleanup_()
            self.task.unload()
        self.task = None
        self._variation_index = 0

    def init_task(self) -> None:
        self.task.init_task()
        self._initial_task_state = self.task.get_state()
        self._has_init_task = True
        self._variation_index = 0

    def init_episode(self, index: int, randomly_place: bool=True,
                     max_attempts: int = 5) -> List[str]:
        """Calls the task init_episode and puts randomly in the workspace.
        """

        self._variation_index = index

        if not self._has_init_task:
            self.init_task()

        # Try a few times to init and place in the workspace
        attempts = 0
        descriptions = None
        while attempts < max_attempts:
            descriptions = self.task.init_episode(index)
            try:
                if (randomly_place and
                        not self.task.is_static_workspace()):
                    self._place_task()
                    if self.robot.is_in_collision():
                        raise BoundaryError()
                self.task.validate()
                break
            except (BoundaryError, WaypointError) as e:
                self.task.cleanup_()
                self.task.restore_state(self._initial_task_state)
                attempts += 1
                if attempts >= max_attempts:
                    raise e

        # Let objects come to rest
        [self.pyrep.step() for _ in range(STEPS_BEFORE_EPISODE_START)]
        self._has_init_episode = True
        return descriptions

    def reset(self) -> None:
        """Resets the joint angles. """

        self.robot.release_gripper()

        if self.robot.is_bimanual:
            self.reset_bimanual()
        else:
            self.reset_unimanual()

        self.robot.zero_velocity()
        
        if self.task is not None and self._has_init_task:
            self.task.cleanup_()
            self.task.restore_state(self._initial_task_state)
        self.task.set_initial_objects_in_scene()

    def reset_unimanual(self) -> None:
        arm, gripper = self._initial_robot_state   
        self.pyrep.set_configuration_tree(arm)
        self.pyrep.set_configuration_tree(gripper)
        self.robot.arm.set_joint_positions(self._start_arm_joint_pos, disable_dynamics=True)
        self.robot.gripper.set_joint_positions(
            self._starting_gripper_joint_pos, disable_dynamics=True)


    def reset_bimanual(self) -> None:

        for arm, gripper in self._initial_robot_state:        
            self.pyrep.set_configuration_tree(arm)
            self.pyrep.set_configuration_tree(gripper)
        
        self.robot.right_arm.set_joint_positions(self._start_arm_joint_pos[0], disable_dynamics=True)
        self.robot.right_gripper.set_joint_positions(self._starting_gripper_joint_pos[0], disable_dynamics=True)

        self.robot.left_arm.set_joint_positions(self._start_arm_joint_pos[1], disable_dynamics=True)
        self.robot.left_gripper.set_joint_positions(self._starting_gripper_joint_pos[1], disable_dynamics=True)


    def get_observation(self) -> Observation:

        observation_data = {}

        lsc_ob = self._obs_config.left_shoulder_camera
        rsc_ob = self._obs_config.right_shoulder_camera
        oc_ob = self._obs_config.overhead_camera
        wc_ob = self._obs_config.wrist_camera
        fc_ob = self._obs_config.front_camera

        lsc_mask_fn, rsc_mask_fn, oc_mask_fn, wc_mask_fn, fc_mask_fn = [
            (rgb_handles_to_mask if c.masks_as_one_channel else lambda x: x
             ) for c in [lsc_ob, rsc_ob, oc_ob, wc_ob, fc_ob]]

        def get_rgb_depth(sensor: VisionSensor, get_rgb: bool, get_depth: bool,
                          get_pcd: bool, rgb_noise: NoiseModel,
                          depth_noise: NoiseModel, depth_in_meters: bool):
            rgb = depth = pcd = None
            if sensor is not None and (get_rgb or get_depth):
                sensor.handle_explicitly()
                if get_rgb:
                    rgb = sensor.capture_rgb()
                    if rgb_noise is not None:
                        rgb = rgb_noise.apply(rgb)
                    rgb = np.clip((rgb * 255.).astype(np.uint8), 0, 255)
                if get_depth or get_pcd:
                    depth = sensor.capture_depth(depth_in_meters)
                    if depth_noise is not None:
                        depth = depth_noise.apply(depth)
                if get_pcd:
                    depth_m = depth
                    if not depth_in_meters:
                        near = sensor.get_near_clipping_plane()
                        far = sensor.get_far_clipping_plane()
                        depth_m = near + depth * (far - near)
                    pcd = sensor.pointcloud_from_depth(depth_m)
                    if not get_depth:
                        depth = None
            return rgb, depth, pcd

        def get_mask(sensor: VisionSensor, mask_fn):
            mask = None
            if sensor is not None:
                sensor.handle_explicitly()
                mask = mask_fn(sensor.capture_rgb())
            return mask

        left_shoulder_rgb, left_shoulder_depth, left_shoulder_pcd = get_rgb_depth(
            self._cam_over_shoulder_left, lsc_ob.rgb, lsc_ob.depth, lsc_ob.point_cloud,
            lsc_ob.rgb_noise, lsc_ob.depth_noise, lsc_ob.depth_in_meters)
        right_shoulder_rgb, right_shoulder_depth, right_shoulder_pcd = get_rgb_depth(
            self._cam_over_shoulder_right, rsc_ob.rgb, rsc_ob.depth, rsc_ob.point_cloud,
            rsc_ob.rgb_noise, rsc_ob.depth_noise, rsc_ob.depth_in_meters)
        overhead_rgb, overhead_depth, overhead_pcd = get_rgb_depth(
            self._cam_overhead, oc_ob.rgb, oc_ob.depth, oc_ob.point_cloud,
            oc_ob.rgb_noise, oc_ob.depth_noise, oc_ob.depth_in_meters)
        wrist_rgb, wrist_depth, wrist_pcd = get_rgb_depth(
            self._cam_wrist, wc_ob.rgb, wc_ob.depth, wc_ob.point_cloud,
            wc_ob.rgb_noise, wc_ob.depth_noise, wc_ob.depth_in_meters)
        front_rgb, front_depth, front_pcd = get_rgb_depth(
            self._cam_front, fc_ob.rgb, fc_ob.depth, fc_ob.point_cloud,
            fc_ob.rgb_noise, fc_ob.depth_noise, fc_ob.depth_in_meters)

        left_shoulder_mask = get_mask(self._cam_over_shoulder_left_mask,
                                      lsc_mask_fn) if lsc_ob.mask else None
        right_shoulder_mask = get_mask(self._cam_over_shoulder_right_mask,
                                      rsc_mask_fn) if rsc_ob.mask else None
        overhead_mask = get_mask(self._cam_overhead_mask,
                                 oc_mask_fn) if oc_ob.mask else None
        wrist_mask = get_mask(self._cam_wrist_mask,
                              wc_mask_fn) if wc_ob.mask else None
        front_mask = get_mask(self._cam_front_mask,
                              fc_mask_fn) if fc_ob.mask else None
    
        observation_data.update({
            "left_shoulder_rgb": left_shoulder_rgb,
            "left_shoulder_depth": left_shoulder_depth,
            "left_shoulder_point_cloud": left_shoulder_pcd,
            "right_shoulder_rgb": right_shoulder_rgb,
            "right_shoulder_depth": right_shoulder_depth,
            "right_shoulder_point_cloud": right_shoulder_pcd,
            "overhead_rgb": overhead_rgb,
            "overhead_depth": overhead_depth,
            "overhead_point_cloud": overhead_pcd,
            "wrist_rgb": wrist_rgb,
            "wrist_depth": wrist_depth,
            "wrist_point_cloud": wrist_pcd,
            "front_rgb": front_rgb,
            "front_depth": front_depth,
            "front_point_cloud": front_pcd,
            "left_shoulder_mask": left_shoulder_mask,
            "right_shoulder_mask": right_shoulder_mask,
            "overhead_mask": overhead_mask,
            "wrist_mask": wrist_mask,
            "front_mask": front_mask
        })


        def get_proprioception(arm: Arm, gripper: Gripper):
            tip = arm.get_tip()



            if self._obs_config.joint_velocities:
                joint_velocities=np.array(arm.dot_q)
                joint_velocities=self._obs_config.joint_velocities_noise.apply(joint_velocities)
            else:
                joint_velocities=None

            if self._obs_config.joint_positions:
                joint_positions = np.array(arm.q)
                joint_positions = self._obs_config.joint_positions_noise.apply(joint_positions)
            else:
                joint_positions = None
            

            if self._obs_config.joint_forces:
                fs = arm.get_joint_forces()
                vels = arm.get_joint_target_velocities()
                joint_forces = np.array([-f if v < 0 else f for f, v in zip(fs, vels)])
                joint_forces = self._obs_config.joint_forces_noise.apply(joint_forces)
            else:
                joint_forces=None

            if self._obs_config.gripper_open:
                if gripper.get_open_amount()[0] > 0.95:
                    gripper_open = 1.0
                else:
                    gripper_open = 0.0
            else:
                gripper_open = None

            if self._obs_config.gripper_pose:
                gripper_pose = tip.get_pose()
            else:
                gripper_pose = None


            if self._obs_config.gripper_matrix:
                gripper_matrix = tip.get_matrix()
            else:
                gripper_matrix = None

            if self._obs_config.gripper_touch_forces:
                ee_forces = gripper.get_touch_sensor_forces()
                ee_forces_flat = []
                for eef in ee_forces:
                    ee_forces_flat.extend(eef)
                gripper_touch_forces = np.array(ee_forces_flat)
            else:
                gripper_touch_forces =  None


            if self._obs_config.gripper_joint_positions:
                gripper_joint_positions= np.array(gripper.q)
            else:
                gripper_joint_positions = None


            if self._obs_config.record_ignore_collisions:
                if self._ignore_collisions_for_current_waypoint:
                    ignore_collisions = np.array(1.0)
                else:
                    ignore_collisions = np.array(0.0)
            else:
                ignore_collisions = None

            return {"joint_velocities": joint_velocities, 
            "joint_positions": joint_positions,
            "joint_forces": joint_forces, 
            "gripper_open": gripper_open,
            "gripper_pose": gripper_pose,
            "gripper_matrix": gripper_matrix,
            "gripper_touch_forces": gripper_touch_forces,
            "gripper_joint_positions": gripper_joint_positions, 
            "ignore_collisions": ignore_collisions}


        if self.robot.is_bimanual:
            observation_data["right"] = UnimodalObservationData(**get_proprioception(self.robot.right_arm, self.robot.right_gripper))
            observation_data["left"] = UnimodalObservationData(**get_proprioception(self.robot.left_arm, self.robot.left_gripper))
        else:
            observation_data.update(get_proprioception(self.robot.arm, self.robot.gripper))

        task_low_dim_state=(
            self.task.get_low_dim_state() if
            self._obs_config.task_low_dim_state else None),

        observation_data.update({
            "task_low_dim_state": task_low_dim_state,
            "misc": self._get_misc()
        })

        if self.robot.is_bimanual:
            obs = BimanualObservation(**observation_data)
        else:
            obs = UnimodalObservation(**observation_data)

        obs = self.task.decorate_observation(obs)
        return obs

    def step(self):
        self.pyrep.step()
        self.task.step()
        if self._step_callback is not None:
            self._step_callback()

    def register_step_callback(self, func):
        self._step_callback = func

    def get_demo(self, record: bool = True,
                 callable_each_step: Callable[[Observation], None] = None,
                 randomly_place: bool = True) -> Demo:
        """Returns a demo (list of observations)"""

        if not self._has_init_task:
            self.init_task()
        if not self._has_init_episode:
            self.init_episode(self._variation_index,
                              randomly_place=randomly_place)
        self._has_init_episode = False

        waypoints = self.task.get_waypoints()
        if len(waypoints) == 0:
            raise NoWaypointsError(
                'No waypoints were found.', self.task)

        demo = []
        if record:
            self.pyrep.step()  # Need this here or get_force doesn't work...
            demo.append(self.get_observation())
        while True:
            success = False
            self._ignore_collisions_for_current_waypoint = False
            for i, point in enumerate(waypoints):
                self._ignore_collisions_for_current_waypoint = point._ignore_collisions
                point.start_of_path()
                if point.skip:
                    continue

                colliding_shapes = []                
                # ..todo:: check if we can mix colliding shapes
                if not self.robot.is_bimanual:
                    grasped_objects = self.robot.gripper.get_grasped_objects()
                    colliding_shapes = [s for s in self.pyrep.get_objects_in_tree(
                    object_type=ObjectType.SHAPE) if s not in grasped_objects
                                    and s not in self._robot_shapes and s.is_collidable()
                                    and self.robot.arm.check_arm_collision(s)]
                else:
                    grasped_objects = self.robot.right_gripper.get_grasped_objects() + self.robot.left_gripper.get_grasped_objects()
                    colliding_shapes = []
                    for s in self.pyrep.get_objects_in_tree(object_type=ObjectType.SHAPE):
                        if s in grasped_objects:
                            continue
                        if not s.is_collidable():
                            continue
                        if self.robot.right_arm.check_arm_collision(s):
                            colliding_shapes.append(s)
                        elif self.robot.left_arm.check_arm_collision(s):
                            colliding_shapes.append(s)
                
                

                [s.set_collidable(False) for s in colliding_shapes]
                try:
                    path = point.get_path()
                    [s.set_collidable(True) for s in colliding_shapes]
                except ConfigurationPathError as e:
                    [s.set_collidable(True) for s in colliding_shapes]
                    raise DemoError(
                        'Could not get a path for waypoint %d.' % i,
                        self.task) from e
                ext = point.get_ext()
                path.visualize()

                done = False
                success = False
                while not done:
                    done = path.step()
                    self.step()
                    self._demo_record_step(demo, record, callable_each_step)
                    success, term = self.task.success()

                point.end_of_path()

                path.clear_visualization()

                if len(ext) > 0:
                    contains_param = False
                    start_of_bracket = -1
                    name = ext.split('_', maxsplit=1)[0]
                    if 'open_gripper(' in ext:
                        self.robot.release_gripper(name)
                        start_of_bracket = ext.index('open_gripper(') + 13
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = self.robot.actutate_gripper(1.0, 0.04, name)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)
                    elif 'close_gripper(' in ext:
                        start_of_bracket = ext.index('close_gripper(') + 14
                        contains_param = ext[start_of_bracket] != ')'
                        if not contains_param:
                            done = False
                            while not done:
                                done = self.robot.actutate_gripper(0.0, 0.04, name)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing:
                                    self._demo_record_step(
                                        demo, record, callable_each_step)

                    if contains_param:
                        rest = ext[start_of_bracket:]
                        num = float(rest[:rest.index(')')])
                        done = False
                        while not done:
                            done = self.robot.actutate_gripper(num, 0.04, name)
                            self.pyrep.step()
                            self.task.step()
                            if self._obs_config.record_gripper_closing:
                                self._demo_record_step(
                                    demo, record, callable_each_step)

                    if 'close_gripper(' in ext:
                        for g_obj in self.task.get_graspable_objects():
                            self.robot.grasp(g_obj, name)

                    self._demo_record_step(demo, record, callable_each_step)

            if not self.task.should_repeat_waypoints() or success:
                break

        # Some tasks may need additional physics steps
        # (e.g. ball rowling to goal)
        if not success:
            for _ in range(10):
                self.pyrep.step()
                self.task.step()
                self._demo_record_step(demo, record, callable_each_step)
                success, term = self.task.success()
                if success:
                    break

        success, term = self.task.success()
        if not success:
            raise DemoError('Demo was completed, but was not successful.',
                            self.task)
        return Demo(demo)

    def get_observation_config(self) -> ObservationConfig:
        return self._obs_config

    def check_target_in_workspace(self, target_pos: np.ndarray) -> bool:
        x, y, z = target_pos
        return (self._workspace_maxx > x > self._workspace_minx and
                self._workspace_maxy > y > self._workspace_miny and
                self._workspace_maxz > z > self._workspace_minz)

    def _demo_record_step(self, demo_list, record, func):
        if record:
            demo_list.append(self.get_observation())
        if func is not None:
            func(self.get_observation())

    def _set_camera_properties(self) -> None:
        def _set_rgb_props(rgb_cam: VisionSensor,
                           rgb: bool, depth: bool, conf: CameraConfig):
            if not (rgb or depth or conf.point_cloud):
                rgb_cam.remove()
            else:
                rgb_cam.set_explicit_handling(1)
                rgb_cam.set_resolution(conf.image_size)
                rgb_cam.set_render_mode(conf.render_mode)

        def _set_mask_props(mask_cam: VisionSensor, mask: bool,
                            conf: CameraConfig):
                if not mask:
                    mask_cam.remove()
                else:
                    mask_cam.set_explicit_handling(1)
                    mask_cam.set_resolution(conf.image_size)
        _set_rgb_props(
            self._cam_over_shoulder_left,
            self._obs_config.left_shoulder_camera.rgb,
            self._obs_config.left_shoulder_camera.depth,
            self._obs_config.left_shoulder_camera)
        _set_rgb_props(
            self._cam_over_shoulder_right,
            self._obs_config.right_shoulder_camera.rgb,
            self._obs_config.right_shoulder_camera.depth,
            self._obs_config.right_shoulder_camera)
        _set_rgb_props(
            self._cam_overhead,
            self._obs_config.overhead_camera.rgb,
            self._obs_config.overhead_camera.depth,
            self._obs_config.overhead_camera)
        _set_rgb_props(
            self._cam_wrist, self._obs_config.wrist_camera.rgb,
            self._obs_config.wrist_camera.depth,
            self._obs_config.wrist_camera)
        _set_rgb_props(
            self._cam_front, self._obs_config.front_camera.rgb,
            self._obs_config.front_camera.depth,
            self._obs_config.front_camera)
        _set_mask_props(
            self._cam_over_shoulder_left_mask,
            self._obs_config.left_shoulder_camera.mask,
            self._obs_config.left_shoulder_camera)
        _set_mask_props(
            self._cam_over_shoulder_right_mask,
            self._obs_config.right_shoulder_camera.mask,
            self._obs_config.right_shoulder_camera)
        _set_mask_props(
            self._cam_overhead_mask,
            self._obs_config.overhead_camera.mask,
            self._obs_config.overhead_camera)
        _set_mask_props(
            self._cam_wrist_mask, self._obs_config.wrist_camera.mask,
            self._obs_config.wrist_camera)
        _set_mask_props(
            self._cam_front_mask, self._obs_config.front_camera.mask,
            self._obs_config.front_camera)

    def _place_task(self) -> None:
        self._workspace_boundary.clear()
        # Find a place in the robot workspace for task
        self.task.boundary_root().set_orientation(
            self._initial_task_pose)
        min_rot, max_rot = self.task.base_rotation_bounds()
        self._workspace_boundary.sample(
            self.task.boundary_root(),
            min_rotation=min_rot, max_rotation=max_rot)

    def _get_misc(self):
        def _get_cam_data(cam: VisionSensor, name: str):
            d = {}
            if cam.still_exists():
                d = {
                    '%s_extrinsics' % name: cam.get_matrix(),
                    '%s_intrinsics' % name: cam.get_intrinsic_matrix(),
                    '%s_near' % name: cam.get_near_clipping_plane(),
                    '%s_far' % name: cam.get_far_clipping_plane(),
                }
            return d
        misc = _get_cam_data(self._cam_over_shoulder_left, 'left_shoulder_camera')
        misc.update(_get_cam_data(self._cam_over_shoulder_right, 'right_shoulder_camera'))
        misc.update(_get_cam_data(self._cam_overhead, 'overhead_camera'))
        misc.update(_get_cam_data(self._cam_front, 'front_camera'))
        misc.update(_get_cam_data(self._cam_wrist, 'wrist_camera'))
        return misc
