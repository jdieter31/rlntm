from stable_baselines3.common.policies import ActorCriticPolicy
import collections
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch
from torch import nn

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.preprocessing import get_action_dim, maybe_transpose, preprocess_obs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, MlpExtractor, NatureCNN, create_mlp
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_device, is_vectorized_observation
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper
from .hdtransformer import HDTransformerEncoder, positionalencoding3d

class ChessTransformerEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 256, nhead=4, num_layers=6, dim_feedforward=1024, dropout=0, activation="relu", layer_norm_eps = 1e-5):
        super(ChessTransformerEncoder, self).__init__(observation_space, features_dim)

        self.hdtransformer = HDTransformerEncoder(3, features_dim, nhead, num_layers, dim_feedforward, dropout, activation, layer_norm_eps)
        self.meta_board_encoder = nn.Linear(7, features_dim, bias=True)
        self.board_encoder = nn.Linear(14, features_dim, bias=True)

    def forward(self, observations):
        observations_board = observations[:, :, :, :112].view(-1, 8, 8, 8, 14)
        observations_meta = observations[:, :, :, 112:]
        model_in = self.board_encoder(observations_board) + self.meta_board_encoder(observations_meta).unsqueeze(-2)
        pe = positionalencoding3d(self.features_dim, 8, 8, 8).to(model_in).transpose(0,1).transpose(1,2).transpose(2,3)

        features = self.hdtransformer(model_in + pe.unsqueeze(0)).mean(-2)

        return features


class ChessTransformerPolicy(ActorCriticPolicy):
    """
    Chess transformer policy class for actor-critic algorithms (has both policy and value prediction).
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = ChessTransformerEncoder,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(ChessTransformerPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            sde_net_arch,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def _build(self, lr_schedule):
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        # Separate features extractor for gSDE
        if self.sde_net_arch is not None:
            self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
                self.features_dim, self.sde_net_arch, self.activation_fn
            )

        self.action_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 73)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)
        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        features = self.extract_features(obs[:, :, :, :-73])
        move_mask = obs[:, :, :, -73:]

        # Evaluate the values for the given observations
        values = self.value_net(features).mean(-2).mean(-2)

        # Get action probs and apply mask to ensure valid move is selected
        action_probs = self.action_net(features).view(obs.size(0), -1)
        action_probs = nn.functional.softmax(action_probs)
        action_probs = action_probs * move_mask.reshape(obs.size(0), -1)
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

        distribution = torch.distributions.Categorical(probs=action_probs)
        if deterministic:
           actions = torch.argmax(distribution.probs, dim=1)
        else:
            actions = distribution.sample()

        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        features = self.extract_features(obs[:, :, :, :-73])
        move_mask = obs[:, :, :, -73:]

        # Evaluate the values for the given observations
        values = self.value_net(features).mean(-2).mean(-2)

        # Get action probs and apply mask to ensure valid move is selected
        action_probs = self.action_net(features).view(obs.size(0), -1)
        action_probs = nn.functional.softmax(action_probs)
        action_probs = action_probs * move_mask.reshape(obs.size(0), -1)
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)


        distribution = torch.distributions.Categorical(probs=action_probs)
        log_prob = distribution.log_prob(actions)

        return values, log_prob, distribution.entropy()

    def _predict(self, obs: torch.Tensor, deterministic: bool = False):
        features = self.extract_features(obs[:, :, :, :-73])
        move_mask = obs[:, :, :, -73:]

        # Get action probs and apply mask to ensure valid move is selected
        action_probs = self.action_net(features).view(obs.size(0), -1)
        action_probs = nn.functional.softmax(action_probs)
        action_probs = action_probs * move_mask.reshape(obs.size(0), -1)
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)

        distribution = torch.distributions.Categorical(probs=action_probs)
        if deterministic:
           actions = torch.argmax(distribution.probs, dim=1)
        else:
            actions = distribution.sample()

        return actions



