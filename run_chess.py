import gym
import gym_chess

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from rlntm.modules.chess_transformer import ChessTransformerPolicy
from rlntm.modules.chess_env_wrapper import ChessObservationWrapper

def main():
    x = lambda : ChessObservationWrapper(gym.make("ChessAlphaZero-v0"))
    env = SubprocVecEnv([x for _ in range(1)])

    model = PPO(ChessTransformerPolicy, env, verbose=1)
    model.learn(total_timesteps=10000)

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

    env.close()

if __name__ == "__main__":
    main()
