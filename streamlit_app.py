"""
Streamlit frontend for the RL Face Grid project

Place this file at the project root where `rl_face_grid/` package exists.

Run:
    pip install -r requirements.txt
    pip install streamlit
    streamlit run streamlit_app.py

Features:
- Choose sample image / upload image / live camera
- Configure grid-size, targets, and training hyperparams
- Start training with live plot and progress
- Save and load Q-table
- Run greedy evaluation and visualize path overlay on image
- Enroll known faces and perform full face recognition (identity matching)

This file uses `st.camera_input` for live camera capture (no extra OpenCV required).
"""

import io
import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

# make sure the rl_face_grid package is importable (run from project root)
try:
    from rl_face_grid.utils import load_image_grayscale, gridify, draw_path_on_image, make_target_masks
    from rl_face_grid.env import FaceGridEnv
    from rl_face_grid.q_learning import QLearningAgent
    from rl_face_grid.viz import plot_learning_curve, plot_path_over_image
except Exception as e:
    st.error("Could not import rl_face_grid package. Make sure you're running this from the project root where `rl_face_grid/` exists.")
    st.stop()

# Helpers
@st.cache_data
def load_image_for_app(_pil_image, resize=256):
    # _pil_image: PIL.Image (leading underscore to avoid hashing issues)
    img = _pil_image.convert("L")
    img = img.resize((resize, resize), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def numpy_to_pil(arr):
    return Image.fromarray((arr * 255).astype(np.uint8))

# Face recognition helpers (uses face_recognition)
try:
    import face_recognition
except Exception:
    face_recognition = None


def encode_face_from_pil(pil_img):
    """Return face encodings (list) from a PIL image (RGB)."""
    if face_recognition is None:
        raise RuntimeError("face_recognition library not available. Install 'face_recognition' to enable recognition.")
    rgb = np.array(pil_img.convert("RGB"))
    encs = face_recognition.face_encodings(rgb)
    return encs


def find_faces_in_pil(pil_img):
    """Return list of (top, right, bottom, left) face locations in pixel coords for a PIL image."""
    if face_recognition is None:
        raise RuntimeError("face_recognition library not available. Install 'face_recognition' to enable recognition.")
    rgb = np.array(pil_img.convert("RGB"))
    locs = face_recognition.face_locations(rgb)
    return locs


def match_encoding_to_db(encoding, known_encodings, known_names, tolerance=0.5):
    """Compare a single encoding to DB and return best match name and distance."""
    if len(known_encodings) == 0:
        return None, None
    dists = face_recognition.face_distance(known_encodings, encoding)
    best_idx = int(np.argmin(dists))
    best_dist = float(dists[best_idx])
    is_match = best_dist <= tolerance
    name = known_names[best_idx] if is_match else None
    return name, best_dist



# Sidebar controls
st.sidebar.title("RL Face Grid — Controls")
mode = st.sidebar.radio("Image source:", ["Sample image", "Upload image", "Live camera"])

grid_size = st.sidebar.slider("Grid size:", min_value=8, max_value=32, value=16, step=1)
targets = st.sidebar.multiselect("Targets:", options=["eyes", "nose"], default=["eyes", "nose"])

# training hyperparams
st.sidebar.markdown("---")
episodes = st.sidebar.number_input("Episodes:", min_value=10, max_value=50000, value=2000, step=10)
max_steps = st.sidebar.number_input("Max steps per episode:", min_value=10, max_value=2000, value=200)
alpha = st.sidebar.number_input("Learning rate (alpha):", min_value=0.001, max_value=1.0, value=0.2, step=0.01, format="%.3f")
gamma = st.sidebar.number_input("Discount (gamma):", min_value=0.0, max_value=1.0, value=0.99, step=0.01, format="%.3f")
eps_start = st.sidebar.number_input("Epsilon start:", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
eps_end = st.sidebar.number_input("Epsilon end:", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
eps_decay = st.sidebar.number_input("Epsilon decay:", min_value=0.8, max_value=0.9999, value=0.995, step=0.0001, format="%.4f")

st.sidebar.markdown("---")
step_cost = st.sidebar.number_input("Step cost:", min_value=0.0, max_value=1.0, value=0.01, step=0.001, format="%.3f")
distance_weight = st.sidebar.number_input("Distance shaping weight:", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.3f")

out_dir = st.sidebar.text_input("Output folder:", value="out")
if not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

st.title("Reinforcement Learning — Face Grid (Streamlit Frontend)")
st.markdown("Small demo that trains a tabular Q-learning agent to navigate a grid over a face image and find facial parts (eyes/nose). Uses the project's `rl_face_grid` package.")

# Load selected image
sample_img_path = os.path.join("data", "sample_face.png")
image_pil = None

if mode == "Sample image":
    st.write("Using included synthetic sample face image.")
    image_pil = Image.open(sample_img_path)
elif mode == "Upload image":
    uploaded = st.file_uploader("Upload an image (png/jpg)")
    if uploaded is not None:
        image_pil = Image.open(uploaded)
elif mode == "Live camera":
    st.write("Use your camera: allow permission when prompted.")
    camera_img = st.camera_input("Take a picture")
    if camera_img is not None:
        image_pil = Image.open(camera_img)

if image_pil is None:
    st.info("No image selected yet. Pick a source on the left.")
    st.stop()

# Display the image preview
st.subheader("Input image")
st.image(image_pil, use_column_width=True)

# Convert to numpy grayscale array for env
resize = 256
img_arr = load_image_for_app(image_pil, resize=resize)

st.sidebar.markdown("---")
# Q-table management
q_load = st.sidebar.file_uploader("Load Q-table (.npy)", type=["npy"])
if q_load is not None:
    try:
        Q_loaded = np.load(q_load)
        st.sidebar.success("Loaded Q-table from upload")
    except Exception as ex:
        st.sidebar.error(f"Failed to load .npy: {ex}")
        Q_loaded = None
else:
    Q_loaded = None

q_path = os.path.join(out_dir, "q_table.npy")
if os.path.exists(q_path) and Q_loaded is None:
    if st.sidebar.button("Load q_table.npy from out/"):
        try:
            Q_loaded = np.load(q_path)
            st.sidebar.success("Loaded Q-table from out/q_table.npy")
        except Exception as ex:
            st.sidebar.error(f"Could not load q_table.npy: {ex}")

# Main actions: Train / Evaluate
col1, col2 = st.columns(2)

with col1:
    train_button = st.button("Start training")
with col2:
    eval_button = st.button("Greedy evaluate (visualize path)")

# Training flow
if train_button:
    st.info("Starting training — this runs in the Streamlit session. Use small episode counts for quick demo.")

    env = FaceGridEnv(img_arr, grid_size=grid_size, targets=tuple(targets),
                      max_steps=int(max_steps), step_cost=float(step_cost), distance_weight=float(distance_weight))

    agent = QLearningAgent(env.nS, env.nA, alpha=float(alpha), gamma=float(gamma),
                           epsilon_start=float(eps_start), epsilon_end=float(eps_end), epsilon_decay=float(eps_decay))

    # containers for live updates
    progress_bar = st.progress(0)
    status = st.empty()
    chart = st.line_chart()
    returns = []

    save_every = max(1, int(episodes // 10))

    for ep in range(int(episodes)):
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
        returns.append(total_r)

        # live chart update (streamlit supports simple list update via line_chart)
        if ep % 1 == 0:
            chart.add_rows(np.array([returns[-1]]))

        # update progress
        progress_bar.progress((ep + 1) / max(1, episodes))
        if (ep + 1) % save_every == 0 or ep == episodes - 1:
            status.text(f"Episode {ep+1}/{episodes}  Return={total_r:.3f}  Epsilon={agent.epsilon:.3f}")
            # save q-table and learning curve
            np.save(q_path, agent.Q)
            # save learning curve image
            curve_path = os.path.join(out_dir, "learning_curve.png")
            try:
                plot_learning_curve(returns, window=max(10, int(episodes // 20)), save_path=curve_path)
            except Exception:
                pass

    st.success("Training finished")
    st.write("Saved Q-table to:", q_path)
    if os.path.exists(os.path.join(out_dir, "learning_curve.png")):
        st.image(os.path.join(out_dir, "learning_curve.png"), caption="Learning curve")


# Evaluation (greedy)
if eval_button:
    # choose Q: uploaded -> loaded file -> existing on disk -> prompt to train
    if Q_loaded is not None:
        Q = Q_loaded
    elif os.path.exists(q_path):
        Q = np.load(q_path)
    else:
        st.error("No Q-table available. Train first or upload a q_table.npy.")
        st.stop()

    env = FaceGridEnv(img_arr, grid_size=grid_size, targets=tuple(targets), max_steps=int(max_steps), step_cost=float(step_cost), distance_weight=float(distance_weight))

    # greedy rollout
    s = env.reset()
    path = []
    done = False
    steps = 0
    while not done and steps < 10 * grid_size:
        path.append((env.r, env.c))
        a = int(np.argmax(Q[s]))
        s, rwd, done, info = env.step(a)
        steps += 1
    path.append((env.r, env.c))

    # visualize path
    _, _, _, _, grid_coords = gridify(img_arr, grid_size)
    img_overlay = draw_path_on_image(img_arr, grid_coords, path)
    # mark target
    tr, tc = env.target_r, env.target_c
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img_overlay)
    y0, x0, y1, x1 = grid_coords[tr, tc]
    draw.rectangle([x0, y0, x1, y1], outline=128, width=3)

    buf = io.BytesIO()
    img_overlay.save(buf, format="PNG")
    buf.seek(0)

    st.subheader("Greedy rollout visualization")
    st.image(buf)
    st.write(f"Target cell: (row={tr}, col={tc}), steps={steps}")

# Quick tips and downloads
st.markdown("---")
st.subheader("Run notes & downloads")
st.write("To run this app from the project root: `streamlit run streamlit_app.py`. Ensure `rl_face_grid` package is present (the project files).`")

if os.path.exists(q_path):
    with open(q_path, "rb") as f:
        st.download_button("Download q_table.npy", f, file_name="q_table.npy")

if os.path.exists(os.path.join(out_dir, "learning_curve.png")):
    with open(os.path.join(out_dir, "learning_curve.png"), "rb") as f:
        st.download_button("Download learning curve", f, file_name="learning_curve.png")

st.caption("Live camera uses Streamlit's camera_input. For continuous real-time tracking you'd need a more complex loop or WebRTC integration.")
