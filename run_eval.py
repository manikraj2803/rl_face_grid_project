
import os
import argparse
import numpy as np
from rl_face_grid.utils import load_image_grayscale, gridify
from rl_face_grid.env import FaceGridEnv
from rl_face_grid.viz import plot_path_over_image


def greedy_rollout(env, Q, max_steps=500):
    """Follow greedy policy wrt Q until done or max_steps. Return path (list of (r,c))."""
    s = env.reset()
    path = []
    done = False
    steps = 0
    while not done and steps < max_steps:
        r, c = env.r, env.c
        path.append((r, c))
        a = int(np.argmax(Q[s]))
        s, rwd, done, info = env.step(a)
        steps += 1
    path.append((env.r, env.c))
    return path, (env.target_r, env.target_c)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--policy", type=str, required=True, help="Path to saved Q-table (.npy).")
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--targets", nargs="+", default=["eyes", "nose"])
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--out-dir", type=str, default="out")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    Q = np.load(args.policy)
    img = load_image_grayscale(args.image, target_size=(args.resize, args.resize))
    env = FaceGridEnv(img, grid_size=args.grid_size, targets=tuple(args.targets))

    path, target = greedy_rollout(env, Q, max_steps=10*args.grid_size)

    _, _, _, _, grid_coords = gridify(img, args.grid_size)
    path_img_path = os.path.join(args.out_dir, "greedy_path.png")
    plot_path_over_image(img, grid_coords, path, target, save_path=path_img_path)
    print(f"Saved greedy path render to: {path_img_path}")


if __name__ == "__main__":
    main()
