import argparse
from itertools import count

import gym
import scipy.optimize

import torch
from torch.autograd import Variable
from collections import OrderedDict
from gym.spaces import Box, Discrete
from trntm.trpo.simple_policy import SimplePolicy
from trntm.trpo.trpo import TRPO
import wandb
from stable_baselines3 import PPO

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Swimmer-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--num-env', type=int, default=4)
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=50000, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--render-rate', type=int, default=100)
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--use-cuda', action='store_true')

args = parser.parse_args()

wandb.init(project="trntm")
wandb.gym.monitor()

class MultiEnv:
    def __init__(self, env_id, num_env):
        self.envs = []
        for _ in range(num_env):
            self.envs.append(gym.make(env_id))

    def reset(self):
        obs = []
        for env in self.envs:
            obs.append(env.reset())
        return obs

    def step(self, actions):
        obs = []
        rewards = []
        dones = []
        infos = []

        for env, ac in zip(self.envs, actions):
            ob, rew, done, info = env.step(ac)
            obs.append(ob)
            rewards.append(rew)
            dones.append(done)
            infos.append(info)

            if done:
                obs[-1] = env.reset()

        return obs, rewards, dones, infos

env = MultiEnv(args.env_name, args.num_env)
if args.render:
    env.envs[0] = gym.wrappers.Monitor(env.envs[0], "./vid", video_callable=lambda episode_id: episode_id  % args.render_rate == 0,force=True)

num_inputs = env.envs[0].observation_space.shape[0]
disc_action_sizes = []
cont_action_sizes = []

if isinstance(env.envs[0].action_space, Box):
    cont_action_sizes.append(env.envs[0].action_space.shape[0])

if isinstance(env.envs[0].action_space, Discrete):
    disc_action_sizes.append(env.envs[0].action_space.n)

device = torch.device('cuda' if args.use_cuda
                        and torch.cuda.is_available() else 'cpu')

policy.to(device)
trpo = TRPO(env, policy, device=device)

for i_episode in count(1):
    batch_reward = trpo.train_batch(args.batch_size).cpu().numpy()[0]
    if i_episode % args.log_interval == 0:
        print('Episode {}\t Average reward {:.2f}'.format(
            i_episode, batch_reward))

