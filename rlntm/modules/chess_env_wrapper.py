import gym.spaces as spaces
from gym import ObservationWrapper
import numpy as np


class ChessObservationWrapper(ObservationWrapper):
    r"""Observation wrapper that expands the observation space to have a mask listing the valid moves."""
    def __init__(self, env):
        super(ChessObservationWrapper, self).__init__(env)
        self.observation_space = spaces.tuple.Tuple([env.observation_space, env.observation_space])
        self.observation_space = spaces.Box(
            low=0,
            high=np.iinfo(np.int).max,
            shape=(8, 8, 8 * 14 + 7 + 73),
            dtype=np.int
        )

    def observation(self, observation):
        legal_action_mask = np.zeros(8 * 8 * 73)
        legal_action_mask[np.array(self.env.legal_actions)] = 1
        legal_action_mask = legal_action_mask.reshape(8, 8, 73)
        out = np.concatenate([observation, legal_action_mask], axis=2)
        return out
