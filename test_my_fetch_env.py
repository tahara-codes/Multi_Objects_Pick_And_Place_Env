import os
import copy
import time
import random
import numpy as np
import pickle
import argparse

import pygame
from control_joy import JoyController

import cv2
import gym
import my_fetch


class CollectData:
    def __init__(self):
        self.config_env()
        self.config_joy()

    def config_joy(self):
        pygame.init()
        self.joy = JoyController(0)
        self.deadzone = 0.2

    def config_env(self):
        # Fetch Env
        self.env = gym.make("MyFetchPickAndPlace-v1")
        self.camera_name = [
            None,
            "head_camera_rgb",
            "gripper_camera_rgb",
            "lidar",
            "external_camera_0",
        ]
        # Create folder
        self.env.config_video("./test")

    def main(self):
        observation = self.env.reset()

        while True:

            all_state = {
                "gripper_positon": observation["gripper_position"],
                "gripper_velocity": observation["gripper_velocity"],
                "gripper_joint": observation["gripper_joint"],
                "object0_pos": observation["object0_pos"],
                "object0_rot": observation["object0_rot"],
                "object1_pos": observation["object1_pos"],
                "object1_rot": observation["object1_rot"],
                "object2_pos": observation["object2_pos"],
                "object2_rot": observation["object2_rot"],
            }

            action = self.move_target(self.joy)

            print("action: ", action)

            image = self.env.my_render(
                mode="rgb_array",
                width=1000,
                height=1000,
                camera_name=self.camera_name[4],
                type_policy="None",
            )

            observation, _, _, _ = self.env.step(action)

    def move_target(self, joy):
        eventlist = pygame.event.get()
        joy.get_controller_value(eventlist)

        action = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 1.57079633, 0.0])
        if abs(joy.l_hand_y) > self.deadzone:
            action[0] = joy.l_hand_y * 0.3
        if abs(joy.l_hand_x) > self.deadzone:
            action[1] = joy.l_hand_x * 0.3
        if abs(joy.r_hand_y) > self.deadzone:
            action[2] = -joy.r_hand_y * 0.1
        if abs(joy.r_hand_x) > self.deadzone:
            action[3] = joy.r_hand_x * 0.1
        return action


if __name__ == "__main__":
    collect_data = CollectData()
    collect_data.main()

