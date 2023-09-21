#!/usr/bin/env python3

import os

from multiprocessing import Process, Manager

from pyrep.const import RenderMode

from rlbench import ObservationConfig
from rlbench.observation_config import CameraConfig

from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
from rlbench.action_modes.arm_action_modes import BimanualJointVelocity
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete

from rlbench.backend.exceptions import BoundaryError, InvalidActionError, TaskEnvironmentError, WaypointError
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
import rlbench.backend.task as task


import pickle
from PIL import Image
from rlbench.backend import utils
from rlbench.backend.const import *
import numpy as np

import logging

import rich_click as click
from click_prompt import choice_option
from click_prompt import filepath_option

IMAGE_SIZE = [128, 128]

camera_names = ["over_shoulder_left", "over_shoulder_right", "overhead", "wrist_right", "wrist_left", "front"]


def save_demo(demo, example_path, variation):
    data_types = ["rgb", "depth", "point_cloud", "mask"]
    #full_camera_names = list(map(lambda x: ('_'.join(x), x[-1]), product(camera_names, data_types)))

    # Save image data first, and then None the image data, and pickle
    for i, obs in enumerate(demo):
        for camera_name in camera_names:
            for dtype in data_types:

                camera_full_name = f"{camera_name}_{dtype}"
                data_path = os.path.join(example_path, camera_full_name)
                os.makedirs(data_path, exist_ok=True)

                data = obs.perception_data.get(camera_full_name, None)

                if data is not None:
                    if dtype == 'rgb':                
                        data = Image.fromarray(data)
                    elif dtype == 'depth':
                        data = utils.float_array_to_rgb_image(data, scale_factor=DEPTH_SCALE)
                    elif dtype == 'point_cloud':
                        continue
                    elif dtype == 'mask':
                        data = Image.fromarray((data * 255).astype(np.uint8))
                    else:
                        raise Exception('Invalid data type')    
                    logging.debug("saving %s", camera_full_name)
                    data.save(os.path.join(data_path, f"{dtype}_{i:04d}.png"))
                    
        # ..why don't we put everything into a pickle file?
        obs.perception_data.clear()

    print(len(demo))

    # Save the low-dimension data
    with open(os.path.join(example_path, LOW_DIM_PICKLE), 'wb') as f:
        pickle.dump(demo, f)

    with open(os.path.join(example_path, VARIATION_NUMBER), 'wb') as f:
        pickle.dump(variation, f)


def run(i, lock, task_index, variation_count, results, file_lock, tasks, headless, save_path, episodes_per_task, variations):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation."""

    # Initialise each thread with random seed
    np.random.seed(None)
    num_tasks = len(tasks)

    img_size = list(map(int, IMAGE_SIZE))
    obs_config = ObservationConfig()
    obs_config.set_all(True)

    default_config_params = {"image_size": img_size, "depth_in_meters": False, "masks_as_one_channel": False}
    camera_configs = {camera_name: CameraConfig(**default_config_params) for camera_name in camera_names}
    obs_config.camera_configs = camera_configs
    
    robot_setup = 'dual_panda'
    rlbench_env = Environment(
        action_mode=BimanualMoveArmThenGripper(BimanualJointVelocity(), BimanualDiscrete()),
        obs_config=obs_config,
        robot_setup=robot_setup,
        headless=headless)
 
    rlbench_env.launch()

    task_env = None

    tasks_with_problems = results[i] = ''

    while True:
        # Figure out what task/variation this thread is going to do
        with lock:

            if task_index.value >= num_tasks:
                print('Process', i, 'finished')
                break

            my_variation_count = variation_count.value
            t = tasks[task_index.value]
            task_env = rlbench_env.get_task(t)
            var_target = task_env.variation_count()
            if variations >= 0:
                var_target = np.minimum(variations, var_target)
            if my_variation_count >= var_target:
                # If we have reached the required number of variations for this
                # task, then move on to the next task.
                variation_count.value = my_variation_count = 0
                task_index.value += 1

            variation_count.value += 1
            if task_index.value >= num_tasks:
                print('Process', i, 'finished')
                break
            t = tasks[task_index.value]

        task_env = rlbench_env.get_task(t)
        task_env.set_variation(my_variation_count)
        descriptions, obs = task_env.reset()

        variation_path = os.path.join(
            save_path, task_env.get_name(),
            VARIATIONS_FOLDER % my_variation_count)

        os.makedirs(variation_path, exist_ok=True)

        with open(os.path.join(
                variation_path, VARIATION_DESCRIPTIONS), 'wb') as f:
            pickle.dump(descriptions, f)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        os.makedirs(episodes_path, exist_ok=True)

        abort_variation = False
        for ex_idx in range(episodes_per_task):
            print('Process', i, '// Task:', task_env.get_name(),
                  '// Variation:', my_variation_count, '// Demo:', ex_idx)
            # ..todo: I think I might need to increase the variable
            attempts = 10
            while attempts > 0:
                try:
                    # TODO: for now we do the explicit looping.
                    demo, = task_env.get_demos(
                        amount=1,
                        live_demos=True)
                except Exception as e:
                    attempts -= 1
                    if attempts > 0:
                        continue
                    problem = (
                        'Process %d failed collecting task %s (variation: %d, '
                        'example: %d). Skipping this task/variation.\n%s\n' % (
                            i, task_env.get_name(), my_variation_count, ex_idx,
                            str(e))
                    )
                    print(problem)
                    tasks_with_problems += problem
                    abort_variation = True
                    break
                episode_path = os.path.join(episodes_path, EPISODE_FOLDER % ex_idx)
                with file_lock:
                    save_demo(demo, episode_path)
                break
            if abort_variation:
                break

    results[i] = tasks_with_problems
    rlbench_env.shutdown()

def run_all_variations(tasks, headless, save_path, episodes_per_task, variations):
    """Each thread will choose one task and variation, and then gather
    all the episodes_per_task for that variation."""

    # Initialise each thread with random seed
    np.random.seed(None)
    num_tasks = len(tasks)

    img_size = list(map(int, IMAGE_SIZE))
    obs_config = ObservationConfig()
    obs_config.set_all(True)

    default_config_params = {"image_size": img_size, "depth_in_meters": False, "masks_as_one_channel": False}
    camera_configs = {camera_name: CameraConfig(**default_config_params) for camera_name in camera_names}
    obs_config.camera_configs = camera_configs

    robot_setup = 'dual_panda'
    rlbench_env = Environment(
        action_mode=BimanualMoveArmThenGripper(BimanualJointVelocity(), BimanualDiscrete()),
        obs_config=obs_config,
        robot_setup=robot_setup,
        headless=headless)

    rlbench_env.launch()

    task_env = None


    for task in tasks:
        
        task_env = rlbench_env.get_task(task)
        possible_variations = task_env.variation_count()

        variation_path = os.path.join(save_path, task_env.get_name(), VARIATIONS_ALL_FOLDER)
        os.makedirs(variation_path, exist_ok=True)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        os.makedirs(episodes_path, exist_ok=True)

        abort_variation = False
        for ex_idx in range(episodes_per_task):
            attempts = 10
            while attempts > 0:
                try:
                    variation = np.random.randint(possible_variations)

                    task_env = rlbench_env.get_task(t)

                    task_env.set_variation(variation)
                    descriptions, obs = task_env.reset()

                    print('Process', i, '// Task:', task_env.get_name(),
                          '// Variation:', variation, '// Demo:', ex_idx)

                    # TODO: for now we do the explicit looping.
                    demo, = task_env.get_demos(
                        amount=1,
                        live_demos=True)
                #  NoWaypointsError, DemoError,
                except (BoundaryError, WaypointError, InvalidActionError, TaskEnvironmentError) as e:
                    logging.error("exception %s", e)
                    attempts -= 1
                    if attempts > 0:
                        continue
                    problem = (
                        'Failed collecting task %s (variation: %d, '
                        'example: %d). Skipping this task/variation.\n%s\n' % (
                            task_env.get_name(), variation, ex_idx,
                            str(e))
                    )
                    print(problem)
                    #tasks_with_problems += problem
                    abort_variation = True
                    break
                episode_path = os.path.join(episodes_path, EPISODE_FOLDER % ex_idx)
               
                save_demo(demo, episode_path, variation)

                with open(os.path.join(
                        episode_path, VARIATION_DESCRIPTIONS), 'wb') as f:
                    pickle.dump(descriptions, f)
                break
            if abort_variation:
                break


    #results[i] = tasks_with_problems
    rlbench_env.shutdown()


from rlbench.backend.task import BIMANUAL_TASKS_PATH

def get_bimanual_tasks():
    return [t.replace('.py', '') for t in
    os.listdir(BIMANUAL_TASKS_PATH) if t != '__init__.py' and t.endswith('.py')]


@click.command()
@filepath_option("--save_path", default="/tmp/rlbench_data/",  help="Where to save the demos.")
@choice_option('--tasks', type=click.Choice(get_bimanual_tasks()), multiple=True, help='The tasks to collect. If empty, all tasks are collected.')
@click.option("--episodes_per_task", default=10, help="The number of episodes to collect per task.")
@click.option("--all_variations", is_flag=True, default=True, help="Include all variations when sampling epsiodes")
@click.option("--variations", default=-1, help="Number of variations to collect per task. -1 for all.")
@click.option("--headless", default=True, is_flag=True, help='Hide the simulator window')
def main(save_path, tasks, episodes_per_task, all_variations, variations, headless):
    
    import logging
    logging.basicConfig()

    tasks = [task_file_to_task_class(t, True) for t in tasks]

    manager = Manager()

    result_dict = manager.dict()
    file_lock = manager.Lock()

    task_index = manager.Value('i', 0)
    variation_count = manager.Value('i', 0)
    lock = manager.Lock()

    os.makedirs(save_path, exist_ok=True)

    # run_all_variations(0, lock, task_index, variation_count, result_dict, file_lock, tasks)

    if all_variations: 
        run_all_variations(0, lock, task_index, variation_count, result_dict, file_lock, tasks, headless, save_path, episodes_per_task, variations)
    else:
        run(0, lock, task_index, variation_count, result_dict, file_lock, tasks, headless, save_path, episodes_per_task, variations)
   
    #print(result_dict)

if __name__ == '__main__':
  main()


#from multiprocessing import Pool
#with Pool(processes=8) as pool:
#        pool.starmap(eval_fn, enumerate(tasks))

