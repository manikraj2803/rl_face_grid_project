
# Reinforcement Learning for Image-based Face Identification (Grid Navigation on Facial Parts)

Design a reinforcement learning (RL) agent to navigate a 2D image grid and identify facial landmarks (eyes, nose) by maximizing the reward for reaching target zones. The agent uses tabular Q-learning in a custom Gym-style environment.

## Project Structure
```
rl_face_grid_project/
├── README.md
├── requirements.txt
├── run_train.py                # Train the Q-learning agent
├── run_eval.py                 # Evaluate and visualize a trained policy on an image
├── rl_face_grid/
│   ├── __init__.py
│   ├── env.py                  # Custom environment (Gym-style)
│   ├── q_learning.py           # Tabular Q-learning
│   ├── utils.py                # Image/grid helpers
│   └── viz.py                  # Plot training curves & paths
└── data/
    ├── sample_face.png         # Synthetic face (for quick demo)
    └── masks/                  # Precomputed masks for targets (auto-generated on first run)
```
## Quick Start

1. **Install dependencies** (ideally in a virtual environment):
```bash
pip install -r requirements.txt
```

2. **Train the agent** (uses the synthetic sample face by default):
```bash
python run_train.py --image data/sample_face.png --targets eyes nose --grid-size 16 --episodes 2000
```

3. **Evaluate and visualize** the trained policy:
```bash
python run_eval.py --image data/sample_face.png --targets eyes nose --grid-size 16 --policy out/q_table.npy
```

This will:
- Show the learning curve (episode returns and moving average).
- Render the agent's greedy path over the image toward the selected target(s).
- Save artifacts in the `out/` directory.

## Concepts Covered
- Q-learning and RL fundamentals
- Building a custom OpenAI Gym-style environment
- Grid-based navigation on images
- Reward shaping (distance shaping + target reward + step penalties)
- Policy learning and visualization of both learning curve and agent paths

## Notes
- The environment is deliberately simple and tabular for clarity. You can extend it with state features (e.g., pixel features around the agent, visited flags, multi-target episodes, etc.).
- For real facial landmark detection, you'd typically use function approximation (e.g., Deep Q-Networks) and richer observations (CNN features). This project focuses on the fundamentals with a clean, extensible codebase.
