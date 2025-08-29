
import numpy as np
import matplotlib.pyplot as plt
from .utils import draw_path_on_image


def plot_learning_curve(returns, window=50, save_path=None):
    """Plot episodic returns and moving average (no color/style specified)."""
    returns = np.asarray(returns, dtype=float)
    plt.figure()
    plt.plot(returns, label="Episode Return")
    if len(returns) >= window:
        cumsum = np.cumsum(np.insert(returns, 0, 0))
        ma = (cumsum[window:] - cumsum[:-window]) / float(window)
        xs = np.arange(window - 1, len(returns))
        plt.plot(xs, ma, label=f"Moving Avg ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_path_over_image(image, grid_coords, path, target, save_path=None):
    """Overlay the path on the image and save."""
    img = draw_path_on_image(image, grid_coords, path)
    # Draw a bounding box for the target cell as well
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    tr, tc = target
    y0, x0, y1, x1 = grid_coords[tr, tc]
    draw.rectangle([x0, y0, x1, y1], outline=128, width=2)
    if save_path is not None:
        img.save(save_path)
