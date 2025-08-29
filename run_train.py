
import os
import argparse
import numpy as np
from rl_face_grid.utils import load_image_grayscale
from rl_face_grid.env import FaceGridEnv
from rl_face_grid.q_learning import QLearningAgent
from rl_face_grid.viz import plot_learning_curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to face image (grayscale is fine).")
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--targets", nargs="+", default=["eyes", "nose"])
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--step-cost", type=float, default=0.01)
    parser.add_argument("--distance-weight", type=float, default=0.1)
    parser.add_argument("--resize", type=int, default=256, help="Resize (square) for simplicity")
    parser.add_argument("--out-dir", type=str, default="out")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    img = load_image_grayscale(args.image, target_size=(args.resize, args.resize))
    env = FaceGridEnv(img, grid_size=args.grid_size, targets=tuple(args.targets),
                      max_steps=args.max_steps, step_cost=args.step_cost,
                      distance_weight=args.distance_weight)

    agent = QLearningAgent(env.nS, env.nA,
                           alpha=args.alpha, gamma=args.gamma,
                           epsilon_start=args.epsilon_start,
                           epsilon_end=args.epsilon_end,
                           epsilon_decay=args.epsilon_decay)

    episode_returns = []
    for ep in range(args.episodes):
        s = env.reset()
        done = False
        total_r = 0.0
        while not done:
            a = agent.select_action(s)
            s_next, r, done, info = env.step(a)
            agent.update(s, a, r, s_next, done)
            s = s_next
            total_r += r
        agent.decay_epsilon()
        episode_returns.append(total_r)

        if (ep + 1) % max(1, args.episodes // 10) == 0:
            print(f"Episode {ep+1}/{args.episodes}  Return={total_r:.3f}  Epsilon={agent.epsilon:.3f}")

    # Save artifacts
    q_path = os.path.join(args.out_dir, "q_table.npy")
    np.save(q_path, agent.Q)
    print(f"Saved Q-table to: {q_path}")

    curve_path = os.path.join(args.out_dir, "learning_curve.png")
    plot_learning_curve(episode_returns, window=max(10, args.episodes//20), save_path=curve_path)
    print(f"Saved learning curve to: {curve_path}")


if __name__ == "__main__":
    main()
