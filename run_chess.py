import gym
from gym import Wrapper
import gym_chess

import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from rlntm.modules.chess_transformer import ChessTransformerPolicy
from rlntm.modules.chess_env_wrapper import ChessObservationWrapper, InfoWrapper
from rlntm.chess.ChessPPO import ChessPPO
from gym_chess.alphazero.move_encoding import MoveEncoding

import chess
import chess.svg
import chess.pgn
import chess.engine
import wandb
from pathlib import Path

import pgn2gif



def main():
    wandb.init(project="trntm")
    wandb.tensorboard.patch(root_logdir="./tensorboard/")
    x = lambda : ChessObservationWrapper(InfoWrapper(gym.make("ChessAlphaZero-v0")))
    env = SubprocVecEnv([x for _ in range(64)])
    eval_env = ChessObservationWrapper(InfoWrapper(gym.make("ChessAlphaZero-v0")))
    move_encoding = MoveEncoding(eval_env)
    eval_freq = 100000

    engine = chess.engine.SimpleEngine.popen_uci(
            "/home/justin/stockfish/stockfish_13_linux_x64_avx2/stockfish_13_linux_x64_avx2")

    ppo = ChessPPO(ChessTransformerPolicy, env, verbose=1, batch_size=256, n_epochs=3, learning_rate=1e-5, n_steps=1024, tensorboard_log="./tensorboard/")
    next_eval = -1
    def call_back(local_vals, global_vals):
        nonlocal next_eval
        iterations = local_vals["self"].num_timesteps
        if iterations < next_eval:
            return
        next_eval = next_eval + eval_freq

        obs = eval_env.reset()
        num_games = 0
        move_num = 0
        delta_score = 0
        images = []

        game = chess.pgn.Game()

        node = game

        total_timesteps = 0
        while num_games < 1:
            action, _states = ppo.predict(obs, deterministic=False)
            move = move_encoding.decode(action)

            score = engine.analyse(eval_env.unwrapped._board,chess.engine.Limit(time=0.01), root_moves=[move])["score"].relative.score(mate_score=20000)
            stockfish_score = engine.analyse(eval_env.unwrapped._board,chess.engine.Limit(time=0.01))["score"].relative.score(mate_score=20000)

            delta_score += score - stockfish_score

            obs, reward, done, info = eval_env.step(action)
            if num_games == 0:
                print(f"score {score}")
                print(f"stockfish score {stockfish_score}")
                print(eval_env.unwrapped.render())
                print(f"----- Move {move_num} -----")
                node = node.add_variation(move)
            if done:
                if num_games == 0:
                    Path(f"./game_data/t{iterations}").mkdir(parents=True, exist_ok=True)
                    print(game, file=open(f"./game_data/t{iterations}.pgn", "w"), end="\n\n")
                    wandb.save(f"./game_data/t{iterations}.pgn")
                    creator = pgn2gif.PgnToGifCreator(reverse=False, duration=1, ws_color='white', bs_color='gray')
                    creator.create_gif(f"./game_data/t{iterations}.pgn", out_path=f"./game_data/t{iterations}.gif")
                obs = eval_env.reset()
                num_games += 1

            move_num += 1
        delta_score /= move_num

        wandb.log({"delta_score": delta_score,
            "game_video": wandb.Video(f"./game_data/t{iterations}.gif", fps=1, format="gif"),
            })

        save_checkpoint({
            'state_dict': ppo.policy.state_dict(),
        }, filename=f"./game_data/t{iterations}.pth")

    ppo.learn(total_timesteps=1000000000, callback=call_back)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

if __name__ == "__main__":
    main()

