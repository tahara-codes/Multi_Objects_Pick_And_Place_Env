import sys
import numpy as np
from gym.envs.robotics import rotations, utils
import cv2
import PIL.Image
from PIL import Image

sys.path.append("./my_fetch")
import robot_env


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments."""

    def __init__(
        self,
        model_path,
        n_substeps,
        gripper_extra_height,
        block_gripper,
        has_object,
        target_in_the_air,
        target_offset,
        obj_range,
        target_range,
        distance_threshold,
        initial_qpos,
        reward_type,
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        super().__init__(
            model_path=model_path,
            n_substeps=n_substeps,
            n_actions=7,
            initial_qpos=initial_qpos,
        )

    def config_video(self, path):
        # 動画ファイル保存用の設定
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        self.video = cv2.VideoWriter(path + "_video.mp4", fourcc, 10, (1000, 1000))

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _step_callback(self):
        if self.block_gripper:
            self.sim.data.set_joint_qpos("robot0:l_gripper_finger_joint", 0.0)
            self.sim.data.set_joint_qpos("robot0:r_gripper_finger_joint", 0.0)
            self.sim.forward()

    def _set_action(self, action):
        assert action.shape == (7,)
        action = (
            action.copy()
        )  # ensure that we don't change the action outside of this scope

        pos_ctrl = action[0:3]
        pos_ctrl *= 0.10  # limit maximum change in position

        # rot_ctrl = [
        #     0.7,
        #     0.0,
        #     0.7,
        #     0.0,
        # ]  # fixed rotation of the end effector, expressed as a quaternion
        euler = [action[4], action[5], action[6]]
        rot_ctrl = rotations.euler2quat(euler)

        gripper_ctrl = action[3] / 100.0
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)

        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation.
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        # Gripper
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)

        # gripper state
        grip_pos = self.sim.data.get_site_xpos("robot0:grip")
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip") * dt * 1000.0
        gripper_joint = robot_qpos[-1:]

        # object state
        if self.has_object:
            object0_pos = self.sim.data.get_site_xpos("object0")
            object0_rot = rotations.mat2euler(self.sim.data.get_site_xmat("object0"))
            object1_pos = self.sim.data.get_site_xpos("object1")
            object1_rot = rotations.mat2euler(self.sim.data.get_site_xmat("object1"))
            object2_pos = self.sim.data.get_site_xpos("object2")
            object2_rot = rotations.mat2euler(self.sim.data.get_site_xmat("object2"))
        else:
            object0_pos = object0_rot = np.zeros(0)
            object1_pos = object1_rot = np.zeros(0)
            object2_pos = object2_rot = np.zeros(0)

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object0_pos.copy())

        obs = np.concatenate([[0]])

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": self.goal.copy(),
            "gripper_position": grip_pos - self.initial_gripper_xpos,
            "gripper_velocity": grip_velp,
            "gripper_joint": gripper_joint,
            "object0_pos": object0_pos.ravel() - self.initial_gripper_xpos,
            "object0_rot": object0_rot.ravel(),
            "object1_pos": object1_pos.ravel() - self.initial_gripper_xpos,
            "object1_rot": object1_rot.ravel(),
            "object2_pos": object2_pos.ravel() - self.initial_gripper_xpos,
            "object2_rot": object2_rot.ravel(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id("robot0:gripper_link")
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.0
        self.viewer.cam.elevation = -14.0

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        # site_id = self.sim.model.site_name2id("target0")
        # self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:

            # Object0
            object_qpos = self.sim.data.get_joint_qpos("object0:joint")
            diff = self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos[0] += diff[0]
            object_qpos[1] += diff[1]
            self.sim.data.set_joint_qpos("object0:joint", object_qpos)

            # Object1
            object_qpos = self.sim.data.get_joint_qpos("object1:joint")
            diff = self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos[0] += diff[0]
            object_qpos[1] += diff[1]
            self.sim.data.set_joint_qpos("object1:joint", object_qpos)

            # Object2
            object_qpos = self.sim.data.get_joint_qpos("object2:joint")
            diff = self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos[0] += diff[0]
            object_qpos[1] += diff[1]
            self.sim.data.set_joint_qpos("object2:joint", object_qpos)

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
                -self.target_range, self.target_range, size=3
            )
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        gripper_target = np.array(
            [-0.498, 0.005, -0.431 + self.gripper_extra_height]
        ) + self.sim.data.get_site_xpos("robot0:grip")

        diff = self.np_random.uniform(-self.target_range, self.target_range, size=2)
        gripper_target[0] += diff[0]
        gripper_target[1] += diff[1]

        gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        self.sim.data.set_mocap_pos("robot0:mocap", gripper_target)
        self.sim.data.set_mocap_quat("robot0:mocap", gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos("robot0:grip").copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos("object0")[2]

    def my_render(
        self, mode="human", width=500, height=500, camera_name=None, type_policy="None"
    ):
        original_image = super().render(mode, width, height, camera_name)

        cvImage = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        # cvImage = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

        cvImage = cv2.resize(cvImage, (50, 50))
        image = Image.fromarray(cvImage)
        image = np.array(image)

        cvImage = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        # cvImage = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

        cv2.putText(
            cvImage,
            text=type_policy,
            org=(400, 800),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2.0,
            color=(0, 0, 0),
            thickness=3,
            lineType=cv2.LINE_4,
        )

        filename = "image" + str(1)
        cv2.imshow(filename, cvImage)
        cv2.moveWindow(filename, 2000, 200)
        self.video.write(cvImage)  # 動画を1フレームずつ保存する
        cv2.waitKey(1)

        return image

