"""Pusher task for the sawyer robot."""

import collections

import gym
import moveit_commander
import numpy as np

from garage.contrib.ros.envs.sawyer.sawyer_env import SawyerEnv
from garage.contrib.ros.robots import Sawyer
from garage.contrib.ros.worlds import BlockWorld
from garage.core import Serializable


class PusherEnv(SawyerEnv, Serializable):
    def __init__(self,
                 initial_goal,
                 initial_joint_pos,
                 sparse_reward=False,
                 simulated=False,
                 distance_threshold=0.05,
                 target_range=0.15,
                 robot_control_mode='position'):
        """
        Pusher task for the sawyer robot.

        :param initial_goal: np.array()
                    the initial goal of pnp task,
                    which is object's target position
        :param initial_joint_pos: dict{string: float}
                    initial joint position
        :param sparse_reward: Bool
                    if use sparse reward
        :param simulated: Bool
                    if use simulator
        :param distance_threshold: float
                    threshold for judging if the episode is done
        :param target_range: float
                    the range within which the new target is randomized
        :param robot_control_mode: string
                    control mode 'position'/'velocity'/'effort'
        """
        Serializable.quick_init(self, locals())

        self._distance_threshold = distance_threshold
        self._target_range = target_range
        self._sparse_reward = sparse_reward
        self.initial_goal = initial_goal
        self.goal = self.initial_goal.copy()
        self.simulated = simulated

        # Initialize moveit to get safety check
        self._moveit_robot = moveit_commander.RobotCommander()
        self._moveit_scene = moveit_commander.PlanningSceneInterface()
        self._moveit_group_name = 'right_arm'
        self._moveit_group = moveit_commander.MoveGroupCommander(
            self._moveit_group_name)

        self._robot = Sawyer(
            initial_joint_pos=initial_joint_pos,
            control_mode=robot_control_mode,
            moveit_group=self._moveit_group_name)
        self._world = BlockWorld(self._moveit_scene,
                                 self._moveit_robot.get_planning_frame(),
                                 simulated)

        SawyerEnv.__init__(self, simulated=simulated)

    @property
    def observation_space(self):
        """
        Returns a Space object.

        :return: observation_space
        """
        return gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=self.get_observation().observation.shape,
            dtype=np.float32)

    def sample_goal(self):
        """
        Sample goals.

        :return: the new sampled goal
        """
        goal = self.initial_goal.copy()

        random_goal_delta = np.random.uniform(
            -self._target_range, self._target_range, size=2)
        goal[:2] += random_goal_delta

        return goal

    def get_observation(self):
        """
        Get Observation.

        :return observation: dict
                    {'observation': obs,
                     'achieved_goal': achieved_goal,
                     'desired_goal': self.goal}
        """
        robot_joint_angles = self._robot.limb_joint_angles()

        robot_obs = np.array(
            [robot_joint_angles['right_j{}'.format(i)] for i in range(7)])

        blocks_position = self._world.get_blocks_position()
        blocks_orientation = self._world.get_blocks_orientation()

        block_pos_obs = np.array([])
        for position in blocks_position:
            block_pos_obs = np.concatenate(
                (block_pos_obs, np.array([position.x, position.y,
                                          position.z])))

        block_ori_obs = np.array([])
        for orientation in blocks_orientation:
            block_ori_obs = np.concatenate(
                (block_ori_obs,
                 np.array([orientation.w, orientation.x, orientation.y, orientation.z])))

        gripper_pos = self._robot.gripper_pose.position
        gripper_pos = np.array([gripper_pos.x, gripper_pos.y, gripper_pos.z])

        obs = np.concatenate((robot_obs, block_pos_obs, block_ori_obs, gripper_pos))

        Observation = collections.namedtuple(
            'Observation', 'observation achieved_goal desired_goal')

        observation = Observation(
            observation=obs,
            achieved_goal=block_pos_obs,
            desired_goal=self.goal)

        return observation

    def reward(self, achieved_goal, goal):
        """
        Compute the reward for current step.

        :param achieved_goal: np.array
                    the current gripper's position or object's
                    position in the current training episode.
        :param goal: np.array
                    the goal of the current training episode, which mostly
                    is the target position of the object or the position.
        :return reward: float
                    if sparse_reward, the reward is -1, else the
                    reward is minus distance from achieved_goal to
                    our goal. And we have completion bonus for two
                    kinds of types.
        """
        d = self._goal_distance(achieved_goal, goal)
        if d < self._distance_threshold:
            return 100
        else:
            if self._sparse_reward:
                return -1.
            else:
                return -d

    def _goal_distance(self, goal_a, goal_b):
        """
        :param goal_a:
        :param goal_b:
        :return distance: distance between goal_a and goal_b
        """
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def done(self, achieved_goal, goal):
        """
        If done.

        :return if_done: bool
                    if current episode is done:
        """
        if not self._robot.safety_check():
            done = True
        else:
            done = self._goal_distance(achieved_goal,
                                       goal) < self._distance_threshold
        return done
