from tf_agents.environments.py_environment import PyEnvironment
from tf_agents.environments import utils
from tf_agents.specs.array_spec import BoundedArraySpec
from tf_agents.trajectories import time_step
from simulation.simulation import AgentState, Simulator
import numpy as np


class CarlaEnvironment(PyEnvironment):
    def __init__(self, log_dir=None):
        # Defining environment variables.
        self._env = Simulator(log_dir)
        self._env.start()

        # Defining action variables.
        self._min_speed = self._env.vehicle_min_speed
        self._max_speed = self._env.vehicle_max_speed

        self._actions_dict = {
            0: AgentState.DANGER,
            1: AgentState.STABILITY,
            2: AgentState.ACCELERATION_LOW,
            3: AgentState.ACCELERATION_MEDIUM,
            4: AgentState.ACCELERATION_HIGH
        }

        # Defining the action spec.
        self._action_spec = BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0, maximum=len(self._actions_dict)-1, name='action'
        )

        # Defining the observation spec.
        self._observation_spec = BoundedArraySpec(
            shape=(self._env.num_of_frames, self._env.image_height, self._env.image_width, 1),
            dtype=np.float32,
            minimum=0.0, maximum=1.0, name='observation'
        )

        self._done = False
        self.discount_rate = 0.99

        # Defining metrics placeholders.
        self._target_speed_sum = 0
        self._num_of_episode_steps = []
        self._episode_mean_target_speed = []

    def get_num_of_episode_steps(self):
        return self._num_of_episode_steps

    def get_episode_mean_target_speed(self):
        return self._episode_mean_target_speed

    # Returns the _action_spec.
    def action_spec(self):
        return self._action_spec

    # Returns the _observation_spec.
    def observation_spec(self):
        return self._observation_spec

    # Resets the environment: Returns the agent back to its initial position.
    def _reset(self):
        self._target_speed_sum = 0
        self._done = False
        observation = self._env.reset()
        return time_step.restart(observation=observation)

    # Executes a step.
    def _step(self, action_spec):
        if self._done:
            return self._reset()

        action = action_spec.item()
        state = self._actions_dict[action]
        observation, reward, self._done = self._env.step(state)

        self._target_speed_sum += self._env.get_speed()
        if self._done:
            step = self._env.get_step()
            self._num_of_episode_steps.append(step)
            self._episode_mean_target_speed.append(self._target_speed_sum / step)
            return time_step.termination(observation=observation, reward=reward)
        else:
            return time_step.transition(observation=observation, reward=reward, discount=self.discount_rate)

    def close(self):
        self._env.close()


def validate_environment(env):
    utils.validate_py_environment(environment=env, episodes=3)
