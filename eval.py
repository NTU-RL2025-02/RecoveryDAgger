"""
scripts/eval.py

Evaluation script for RecoveryDAgger policies on Gymnasium-Robotics PointMaze tasks.

This script loads a trained policy checkpoint (.pt), runs evaluation episodes, and
optionally visualizes a Q-value heatmap (max Q + best-action field) over the maze.

Examples:
    # Evaluate a checkpoint for 100 episodes on 4-rooms
    python eval.py checkpoints/final_model.pt --environment PointMaze_4rooms-v3 --iters 100

    # Evaluate with rendering (slower)
    python eval.py checkpoints/final_model.pt --render

    # Draw Q-value heatmap (writes q_heatmap.png)
    python eval.py checkpoints/final_model.pt --q_heatmap

Arguments:
    pt_path (positional):
        Path to the saved PyTorch checkpoint (.pt). The loaded object is expected to
        implement:
            - act(obs) -> action
            - safety(obs, action) -> scalar Q / safety score   (only needed for --q_heatmap)

Notes:
    - --render enables human rendering and will slow down evaluation.
    - --q_heatmap is computationally expensive (iterates over the maze grid).
    - This script registers Gymnasium-Robotics environments via gym.register_envs().
"""

# --- stdlib ---
import argparse
import sys
from typing import Any

# --- third-party ---
import gymnasium as gym
import gymnasium_robotics  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.wrappers import FlattenObservation
from tqdm import trange

# --- local (your project) ---
import recoverydagger

# module alias for backward-compat (old checkpoints / pickles)
sys.modules["thrifty_gym"] = recoverydagger

from recoverydagger.maze import (
    COMPLICATED_MAZE,
    FOUR_ROOMS_21_21,
    FOUR_ROOMS_21_21_LEFT_UP_RANDOM,
    FOUR_ROOMS_ANGLE,
    FOUR_ROOMS_ANGLE_SINGLE_START,
)
from recoverydagger.utils.wrappers import (
    MazeWrapper,
    NoisyActionWrapper,
)


@torch.no_grad()
def eval(
    env: Any,
    ac: Any,
    total_iterations: int,
    act_limit: float,
    horizon: int,
) -> float:

    if total_iterations <= 0:
        return 0.0

    total_success = 0

    for iter in trange(total_iterations):
        np.random.seed(iter)
        torch.manual_seed(iter)
        ball_traj = []
        episode_len = 0.0
        o, _ = env.reset()
        done = False

        while not done and episode_len < horizon:
            ball_x, ball_y = o[0], o[1]
            ball_traj.append([ball_x, ball_y])

            a = ac.act(o)
            a = np.clip(a, -act_limit, act_limit)

            o, _r, terminated, truncated, _info = env.step(a)

            success = env.is_success()
            total_success += success
            done = terminated or truncated or success

            episode_len += 1

        ball_x, ball_y = o[0], o[1]
        ball_traj.append([ball_x, ball_y])

    success_rate = total_success / total_iterations
    return success_rate, np.array(ball_traj)


def ij_to_xy(i, j, maze):
    width = len(maze[0])
    height = len(maze)
    x = j - width / 2
    y = height / 2 - i
    return x, y


def draw_q_heatmap(ac, maze, obs0):
    np.random.seed(0)
    torch.manual_seed(0)
    width = len(maze[0])
    height = len(maze)
    resolution = 20
    actions = []
    slice = 32
    for k in range(slice):
        theta = 2 * np.pi * k / slice
        actions.append((np.cos(theta), np.sin(theta)))
    actions.append((0.0, 0.0))

    # 對應到 quiver 的 (U,V)
    # U 對應 x 方向（jj 增加往右）
    # V 對應 y 方向（ii 增加往下，所以要反過來）
    U = np.array([a[0] for a in actions], dtype=np.float32)
    V = -np.array([a[1] for a in actions], dtype=np.float32)

    q_table = np.zeros((height * resolution, width * resolution, len(actions)))

    for i in trange(height):
        for j in range(width):
            if maze[i][j] == 1 or maze[i][j] == "1":
                continue
            obs = obs0.copy()
            x, y = ij_to_xy(i, j, maze)
            for di in range(resolution):
                for dj in range(resolution):
                    ii = i * resolution + di
                    jj = j * resolution + dj
                    dx = di / resolution
                    dy = dj / resolution

                    obs[4] = x + dx
                    obs[5] = y + dy
                    obs[6] = 0
                    obs[7] = 0
                    for a_idx, a in enumerate(actions):
                        q_value = ac.safety(obs, a)
                        q_table[ii, jj, a_idx] = q_value

    # 合併：每格只留最大值 + 最佳 action index
    q_max = q_table.max(axis=2)  # (H*res, W*res)
    a_best = q_table.argmax(axis=2)  # (H*res, W*res)

    plt.figure(figsize=(12, 10))

    # 你原本用 seaborn heatmap 也行，但我建議用 imshow 更好疊 quiver
    im = plt.imshow(
        q_max,
        origin="upper",
        cmap="viridis",
        vmin=0.3,
        vmax=0.6,
    )
    plt.colorbar(im, label="max Q-value")
    plt.title("Max Q-value heatmap + best action arrows")
    plt.xticks([])
    plt.yticks([])

    # 箭頭太密會變成一坨：做下採樣
    step = 5  # 你可以調大/調小（越大越稀疏）
    ys = np.arange(1, q_max.shape[0], step)
    xs = np.arange(1, q_max.shape[1], step)
    XX, YY = np.meshgrid(xs, ys)

    best_idx_sampled = a_best[YY, XX]  # (len(ys), len(xs))
    UU = U[best_idx_sampled]
    VV = V[best_idx_sampled]

    # (0,0) action 不想畫箭頭：做 mask
    mask = (UU == 0) & (VV == 0)
    UU = UU.astype(np.float32)
    VV = VV.astype(np.float32)
    UU[mask] = np.nan
    VV[mask] = np.nan

    # plt.quiver(
    #     XX,
    #     YY,
    #     UU,
    #     VV,
    #     angles="xy",
    #     scale_units="xy",
    #     scale=0.3,  # 越大箭頭越短；你可以調
    #     width=0.001,
    #     alpha=0.6,
    # )

    plt.tight_layout()
    plt.savefig("q_heatmap.png", dpi=400)
    plt.show()


def make_env(args):
    gym.register_envs(gymnasium_robotics)

    env = None
    rander_mode = "human" if args.render else None
    if args.environment == "PointMaze_4rooms-v3":
        env = gym.make(
            "PointMaze_Medium-v3",
            continuing_task=False,
            reset_target=False,
            render_mode=rander_mode,
            maze_map=FOUR_ROOMS_21_21_LEFT_UP_RANDOM,
            max_episode_steps=1000,
        )
        env = FlattenObservation(env)
        env = NoisyActionWrapper(env, noise_scale=args.noisy_scale)
        env = MazeWrapper(env, FOUR_ROOMS_21_21_LEFT_UP_RANDOM, touch_wall_distance=0.3)

    elif args.environment == "PointMaze_Complicated-v3":
        env = gym.make(
            "PointMaze_Medium-v3",
            continuing_task=False,
            reset_target=False,
            render_mode=rander_mode,
            maze_map=COMPLICATED_MAZE,
            max_episode_steps=1500,
        )
        env = FlattenObservation(env)
        env = NoisyActionWrapper(env, noise_scale=args.noisy_scale)
        env = MazeWrapper(env, COMPLICATED_MAZE, touch_wall_distance=0.3)

    elif args.environment == "PointMaze_4rooms-v3-angle-single-start":
        env = gym.make(
            "PointMaze_Medium-v3",
            continuing_task=False,
            reset_target=False,
            render_mode=rander_mode,
            maze_map=FOUR_ROOMS_ANGLE_SINGLE_START,
            max_episode_steps=1100,
        )
        env = FlattenObservation(env)
        env = NoisyActionWrapper(env, noise_scale=args.noisy_scale)
        env = MazeWrapper(env, FOUR_ROOMS_ANGLE_SINGLE_START)

    elif args.environment == "PointMaze_4rooms-v3-angle":
        env = gym.make(
            "PointMaze_Medium-v3",
            continuing_task=False,
            reset_target=False,
            render_mode=rander_mode,
            maze_map=FOUR_ROOMS_ANGLE,
            max_episode_steps=1100,
        )
        env = FlattenObservation(env)
        env = NoisyActionWrapper(env, noise_scale=args.noisy_scale)
        env = MazeWrapper(env, FOUR_ROOMS_ANGLE)

    else:
        raise NotImplementedError("This environment is not implemented in this script.")

    return env


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained RecoveryDAgger policy on maze environments and optionally "
            "visualize a Q-value heatmap."
        )
    )

    # ------------------------
    # Positional arguments
    # ------------------------
    parser.add_argument(
        "pt_path",
        type=str,
        help=(
            "Path to a PyTorch checkpoint (.pt) to evaluate. "
            "The loaded policy is expected to provide ac.act(obs). "
            "If --q_heatmap is enabled, it must also provide ac.safety(obs, action)."
        ),
    )

    # ------------------------
    # Evaluation configuration
    # ------------------------
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Number of evaluation episodes to run (default: 100).",
    )

    parser.add_argument(
        "--environment",
        type=str,
        default="PointMaze_4rooms-v3",
        help=(
            "Maze environment preset to evaluate on. Supported choices:\n"
            "  - PointMaze_4rooms-v3\n"
            "  - PointMaze_Complicated-v3\n"
            "  - PointMaze_4rooms-v3-angle\n"
            "  - PointMaze_4rooms-v3-angle-single-start"
        ),
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable human rendering during evaluation (slower).",
    )

    # ------------------------
    # Environment / noise settings
    # ------------------------
    parser.add_argument(
        "--noisy_scale",
        type=float,
        default=0.0,
        help=(
            "Standard deviation scale for action noise injected by NoisyActionWrapper. "
            "Set to 0 to disable (default: 0)."
        ),
    )

    # ------------------------
    # Visualization
    # ------------------------
    parser.add_argument(
        "--q_heatmap",
        action="store_true",
        help=("Compute and save a Q-value heatmap (q_heatmap.png). "),
    )

    parser.set_defaults(render=False, q_heatmap=False)
    args = parser.parse_args()

    env = make_env(args)
    max_ep_len = 1000
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with open(args.pt_path, "rb") as f:
        ac = torch.load(f, weights_only=False, map_location=device)
        ac.device = device
        ac.eval()

    success_rate, ball_traj = eval(
        env,
        ac,
        total_iterations=args.iters,
        act_limit=env.action_space.high[0],
        horizon=max_ep_len,
    )
    print(f"Success rate over {args.iters} episodes: {success_rate}")
    if args.q_heatmap:
        draw_q_heatmap(ac, env.maze, env.reset()[0])
