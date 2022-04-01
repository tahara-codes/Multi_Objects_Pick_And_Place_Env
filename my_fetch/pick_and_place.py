import os
import sys
import random
from gym import utils

sys.path.append("./my_fetch")
import fetch_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "pick_and_place.xml")


class FetchPickAndPlaceEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type="sparse"):
        offset = 0.07

        object_initial_pos = [
            [1.25 - 0.12 + offset, 0.53, 0.4, 1.0, 0.0, 0.0, 0.0],
            [1.25 + 0.0 + offset, 0.53, 0.6, 1.0, 0.0, 0.0, 0.0],
            [1.25 + 0.12 + offset, 0.53, 0.8, 1.0, 0.0, 0.0, 0.0],
        ]

        index = [0, 1, 2]
        random_index = random.sample(index, 3)
        print(random_index)

        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            # "object0:joint": object_initial_pos[random_index[0]],
            # "object1:joint": object_initial_pos[random_index[1]],
            # "object2:joint": object_initial_pos[random_index[2]],
            "object0:joint": object_initial_pos[index[0]],
            "object1:joint": object_initial_pos[index[1]],
            "object2:joint": object_initial_pos[index[2]],
        }

        fetch_env.FetchEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=True,
            block_gripper=False,
            n_substeps=100,
            gripper_extra_height=0.4,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.02,
            target_range=0.00001,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)
