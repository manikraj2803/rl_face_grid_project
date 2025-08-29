
import numpy as np
from PIL import Image, ImageDraw


def load_image_grayscale(path, target_size=None):
    """Load an image in grayscale [0, 1]. Optionally resize to target_size (W, H)."""
    img = Image.open(path).convert("L")
    if target_size is not None:
        img = img.resize(target_size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def gridify(image, grid_size):
    """
    Partition image into a grid_size x grid_size grid of cells.
    Returns:
        H, W = image shape
        cell_h, cell_w
        grid_coords: array shape (grid_size, grid_size, 4) with (y0, x0, y1, x1) per cell
    """
    H, W = image.shape[:2]
    cell_h = H // grid_size
    cell_w = W // grid_size
    coords = np.zeros((grid_size, grid_size, 4), dtype=np.int32)
    for r in range(grid_size):
        for c in range(grid_size):
            y0 = r * cell_h
            x0 = c * cell_w
            y1 = H if r == grid_size - 1 else (r + 1) * cell_h
            x1 = W if c == grid_size - 1 else (c + 1) * cell_w
            coords[r, c] = np.array([y0, x0, y1, x1])
    return H, W, cell_h, cell_w, coords


def make_target_masks(image, grid_size, targets=("eyes", "nose")):
    """
    Create binary masks per target keyword using simple heuristics on face layout:
    - eyes: two blobs around the upper-mid region
    - nose: central blob in the middle
    This is purely for demo/testing with a generic face layout.
    Returns: dict target-> boolean mask with shape (grid_size, grid_size)
    """
    H, W, cell_h, cell_w, coords = gridify(image, grid_size)
    mask_dict = {}
    yy, xx = np.mgrid[0:grid_size, 0:grid_size]
    yy_norm = yy / max(grid_size - 1, 1)
    xx_norm = xx / max(grid_size - 1, 1)

    # Heuristic zones in grid coordinates [0,1]
    # Eyes: left/right around y in [0.25, 0.45], x ~ 0.3 and 0.7
    eyes_mask = ((yy_norm > 0.25) & (yy_norm < 0.45) &
                 (((xx_norm > 0.18) & (xx_norm < 0.42)) |
                  ((xx_norm > 0.58) & (xx_norm < 0.82))))
    # Nose: central vertical band around y in [0.45, 0.65] and x ~ 0.5
    nose_mask = ((yy_norm > 0.45) & (yy_norm < 0.70) &
                 (xx_norm > 0.40) & (xx_norm < 0.60))

    if "eyes" in targets:
        mask_dict["eyes"] = eyes_mask
    if "nose" in targets:
        mask_dict["nose"] = nose_mask

    return mask_dict


def pick_random_target_cell(target_masks, rng):
    """Pick a random cell index (r, c) from the union of all target masks."""    # noqa: E501
    union = None
    for m in target_masks.values():
        union = m if union is None else (union | m)
    if union is None or union.sum() == 0:
        raise ValueError("No target cells available in masks.")
    candidates = np.argwhere(union)
    idx = rng.integers(0, len(candidates))
    r, c = candidates[idx]
    return int(r), int(c)


def draw_path_on_image(image, grid_coords, path):
    """Overlay the traversed path (list of (r,c)) onto the image and return a PIL image."""
    H, W = image.shape[:2]
    # convert to RGB for drawing
    img = Image.fromarray((image * 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    # draw rectangles of path
    for (r, c) in path:
        y0, x0, y1, x1 = grid_coords[r, c]
        draw.rectangle([x0, y0, x1, y1], outline=255, width=1)
    return img
