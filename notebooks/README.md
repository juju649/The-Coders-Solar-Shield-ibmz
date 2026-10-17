# Notebooks

Jupyter notebooks for the solar drop prediction system. The model was run on
**IBM LinuxONE Community Cloud (s390x)**, and outputs are saved so you can see
the results without running anything.

| Notebook | What it does | Owner |
|---|---|---|
| `01_model_training.ipynb` | Loads `features.csv`, splits by date, trains a baseline and a scikit-learn model, compares precision / recall / F1, and times each prediction | Sarosha |
| `02_demo.ipynb` | Replays a cloudy day, marks predicted drops, and prints a battery recommendation for each alert | Naomi |

## Run order

1. `01_model_training.ipynb` creates `predictions.csv`
2. `02_demo.ipynb` reads `predictions.csv` and `src/recommend.py`

See the main [README](../README.md) for the project overview.
