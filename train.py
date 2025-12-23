"""
train.py
Main training script for RecoveryDagger in maze environment.

Usage:
    python train.py <exp_name> [--seed INT] [--device INT] [--iters N]
                     [--targetrate FLOAT] [--expert_policy_file PATH]
                     [--demonstration_set_file PATH] [--max_expert_query N]
                     [--environment ENV] [--render] [--recovery_type TYPE]
                     [--eval PATH] [--num_test_episodes N]
                     [--fix_thresholds] [--noisy_scale FLOAT]
                     [--bc_checkpoint PATH] [--save_bc_checkpoint PATH]
                     [--skip_bc_pretrain] [--risk-probe]
"""

# --------------------------------------------------------------
# Imports
# --------------------------------------------------------------

# standard libraries
import numpy as np
import torch
import wandb

from recoverydagger.algos.thriftydagger import thrifty
from recoverydagger.utils.run_utils import setup_logger_kwargs
from recoverydagger.utils.wrappers import (
    LunarLanderSuccessWrapper,
    MazeWrapper,
    NoisyActionWrapper,
)
from recoverydagger.algos.recovery import FiveQRecovery, QRecovery, ExpertAsRecovery
from recoverydagger.maze import (
    FOUR_ROOMS_ANGLE,
    FOUR_ROOMS_21_21_REWARD,
    FOUR_ROOMS_ANGLE_SINGLE_START,
    FOUR_ROOMS_ANGLE_REWARD,
    FOUR_ROOMS_21_21,
    FOUR_ROOMS_21_21_LEFT_UP_RANDOM,
    COMPLICATED_MAZE,
    COMPLICATED_MAZE_REWARD,
    FOUR_ROOMS_ANGLE_REWARD,
)
from recoverydagger.algos.rule_expert import RuleBasedExpert


import gymnasium as gym
import gymnasium_robotics
from gymnasium_robotics.envs.maze.maps import U_MAZE
from gymnasium.wrappers import FlattenObservation


class SB3Expert:
    """Wrap a Stable-Baselines3 policy to match the expected expert API."""

    def __init__(self, model):
        self.model = model

    def start_episode(self):
        return

    def __call__(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)


def main(args):
    # ---- load expert policy ----
    # 這裡用你搬到比較短路徑的 expert model
    # 路徑是相對於你執行 python 的地方（目前你是在 thriftydagger/scripts 底下跑）
    device = "cuda" if torch.cuda.is_available() else "cpu"

    render = args.render

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    gym.register_envs(gymnasium_robotics)

    # ---- wandb ----
    # wandb.init(
    #     entity="aawrail-RL2025",
    #     project="final_project_maze",
    #     name=args.exp_name,
    #     config={
    #         "seed": args.seed,
    #         "device": args.device,
    #         "iters": args.iters,
    #         "target_rate": args.targetrate,
    #         "environment": args.environment,
    #         "max_expert_query": args.max_expert_query,
    #         "demonstration_set_file": args.demonstration_set_file,
    #         "recovery_type": args.recovery_type,
    #         "bc_checkpoint": args.bc_checkpoint,
    #         "save_bc_checkpoint": args.save_bc_checkpoint,
    #         "skip_bc_pretrain": args.skip_bc_pretrain,
    #     },
    # )

    # ---- 建 env ----
    env = None
    if args.environment == "PointMaze_4rooms-v3":
        env = gym.make(
            "PointMaze_Medium-v3",
            continuing_task=False,
            reset_target=False,
            render_mode="human" if render else None,
            maze_map=FOUR_ROOMS_21_21_LEFT_UP_RANDOM,
            max_episode_steps=1000,
        )
        env = FlattenObservation(env)
        env = NoisyActionWrapper(env, noise_scale=args.noisy_scale)
        env = MazeWrapper(env, FOUR_ROOMS_21_21_LEFT_UP_RANDOM, touch_wall_distance=0.3)
        expert_pol = RuleBasedExpert(
            FOUR_ROOMS_21_21_LEFT_UP_RANDOM, FOUR_ROOMS_21_21_REWARD
        )

    elif args.environment == "PointMaze_Complicated-v3":
        env = gym.make(
            "PointMaze_Medium-v3",
            continuing_task=False,
            reset_target=False,
            render_mode="human" if render else None,
            maze_map=COMPLICATED_MAZE,
            max_episode_steps=1500,
        )
        env = FlattenObservation(env)
        env = NoisyActionWrapper(env, noise_scale=args.noisy_scale)
        env = MazeWrapper(env, COMPLICATED_MAZE, touch_wall_distance=0.3)
        expert_pol = RuleBasedExpert(COMPLICATED_MAZE, COMPLICATED_MAZE_REWARD)

    elif args.environment == "PointMaze_4rooms-v3-angle-single-start":
        env = gym.make(
            "PointMaze_Medium-v3",
            continuing_task=False,
            reset_target=False,
            render_mode="human" if render else None,
            maze_map=FOUR_ROOMS_ANGLE_SINGLE_START,
            max_episode_steps=1100,
        )
        env = FlattenObservation(env)
        env = NoisyActionWrapper(env, noise_scale=args.noisy_scale)
        env = MazeWrapper(env, FOUR_ROOMS_ANGLE_SINGLE_START)
        expert_pol = RuleBasedExpert(
            FOUR_ROOMS_ANGLE_SINGLE_START, FOUR_ROOMS_ANGLE_REWARD
        )

    elif args.environment == "PointMaze_4rooms-v3-angle":
        env = gym.make(
            "PointMaze_Medium-v3",
            continuing_task=False,
            reset_target=False,
            render_mode="human" if render else None,
            maze_map=FOUR_ROOMS_ANGLE,
            max_episode_steps=1100,
        )
        env = FlattenObservation(env)
        env = NoisyActionWrapper(env, noise_scale=args.noisy_scale)
        env = MazeWrapper(env, FOUR_ROOMS_ANGLE)
        expert_pol = RuleBasedExpert(FOUR_ROOMS_ANGLE, FOUR_ROOMS_ANGLE_REWARD)

    else:
        raise NotImplementedError("This environment is not implemented in this script.")
    max_ep_len = 1000
    gym_cfg = {"MAX_EP_LEN": max_ep_len}

    # ---- 建 recovery policy ----
    recovery_policy = None
    if args.recovery_type == "five_q":
        recovery_policy = FiveQRecovery(env.observation_space, env.action_space)
    elif args.recovery_type == "q":
        recovery_policy = QRecovery(env.observation_space, env.action_space)
    elif args.recovery_type == "expert":
        recovery_policy = ExpertAsRecovery(expert_pol)
    else:
        recovery_policy = QRecovery(env.observation_space, env.action_space)

    # ---- 主訓練流程 ----
    try:
        thrifty(
            env=env,
            iters=args.iters,
            seed=args.seed,
            logger_kwargs=logger_kwargs,
            device_idx=args.device,
            target_rate=args.targetrate,
            expert_policy=expert_pol,
            recovery_policy=recovery_policy,
            input_file=args.demonstration_set_file,
            robosuite=False,
            gym_cfg=gym_cfg,  # 或者直接傳 None
            init_model=args.eval,
            max_expert_query=args.max_expert_query,
            recovery_type=args.recovery_type,
            num_test_episodes=args.num_test_episodes,
            bc_checkpoint=args.bc_checkpoint,
            save_bc_checkpoint=args.save_bc_checkpoint,
            skip_bc_pretrain=args.skip_bc_pretrain,
            fix_thresholds=args.fix_thresholds,
        )
    except Exception:
        # wandb.finish(exit_code=1)
        raise
    else:
        # 正常跑完
        # wandb.finish(exit_code=0)
        pass
    finally:
        env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Main training script for RecoveryDAgger in maze environments.\n\n"
            "This script runs a DAgger-style imitation learning loop with optional "
            "query-efficient expert intervention and a recovery policy."
        )
    )

    # ------------------------
    # Positional arguments
    # ------------------------
    parser.add_argument(
        "exp_name",
        type=str,
        help=(
            "Experiment name. Used as the logging directory name "
            "and as the run identifier."
        ),
    )

    # ------------------------
    # General training setup
    # ------------------------
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=0,
        help="Random seed for environment, policy initialization, and data sampling.",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help=(
            "GPU device index to use. "
            "If CUDA is unavailable, this value is ignored and CPU is used."
        ),
    )

    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of DAgger-style training iterations.",
    )

    parser.add_argument(
        "--environment",
        type=str,
        default="PointMaze_4rooms-v3",
        help="Maze environment name. Must be supported by this training script.",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        dest="render",
        help="Enable environment rendering during training (slows down training).",
    )
    parser.set_defaults(render=False)

    # ------------------------
    # Expert querying / switching
    # ------------------------
    parser.add_argument(
        "--targetrate",
        type=float,
        default=0.01,
        help=(
            "Target expert query (or policy switching) rate. "
            "Used to adapt switching thresholds online unless --fix_thresholds is set."
        ),
    )

    parser.add_argument(
        "--max_expert_query",
        type=int,
        default=50000,
        help="Maximum total number of expert queries allowed during training.",
    )

    parser.add_argument(
        "--fix_thresholds",
        action="store_true",
        dest="fix_thresholds",
        help=(
            "Fix switching thresholds and disable online adaptation. "
            "When enabled, --targetrate is ignored."
        ),
    )
    parser.set_defaults(fix_thresholds=False)

    # ------------------------
    # Demonstration data
    # ------------------------
    parser.add_argument(
        "--demonstration_set_file",
        type=str,
        default="models/offline_dataset_mazeMedium_1000.pkl",
        help="Path to offline expert demonstration dataset (.pkl).",
    )

    # ------------------------
    # Recovery policy
    # ------------------------
    parser.add_argument(
        "--recovery_type",
        type=str,
        default="five_q",
        choices=["five_q", "q", "expert"],
        help=(
            "Type of recovery policy to use:\n"
            "  - five_q : Ensemble-based Q recovery policy (default)\n"
            "  - q      : Single Q-function recovery policy\n"
            "  - expert : Always use expert as recovery policy"
        ),
    )

    parser.add_argument(
        "--noisy_scale",
        type=float,
        default=0.0,
        help=(
            "Scale of Gaussian noise added to actions during recovery policy training. "
            "Set to 0 to disable action noise."
        ),
    )

    # ------------------------
    # Evaluation
    # ------------------------
    parser.add_argument(
        "--eval",
        type=str,
        default=None,
        help=(
            "Path to a saved PyTorch policy checkpoint used to initialize the policy "
            "before training."
        ),
    )

    parser.add_argument(
        "--num_test_episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes run after each training iteration.",
    )

    # ------------------------
    # Behavior Cloning (BC)
    # ------------------------
    parser.add_argument(
        "--bc_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a pretrained behavior cloning (BC) policy checkpoint. "
            "Typically used together with --skip_bc_pretrain."
        ),
    )

    parser.add_argument(
        "--save_bc_checkpoint",
        type=str,
        default=None,
        help=(
            "If set, save the BC policy after the initial BC pretraining step "
            "to this path."
        ),
    )

    parser.add_argument(
        "--skip_bc_pretrain",
        action="store_true",
        dest="skip_bc_pretrain",
        help=(
            "Skip the BC pretraining phase and directly load the policy from "
            "--bc_checkpoint. An error will be raised if the checkpoint is missing."
        ),
    )
    parser.set_defaults(skip_bc_pretrain=False)

    args = parser.parse_args()

    main(args)
