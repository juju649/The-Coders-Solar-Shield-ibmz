# Notebooks

Jupyter notebooks for the solar drop prediction system. All notebooks were
run on **IBM LinuxONE Community Cloud (s390x)**, and outputs are saved so
you can see the results without running anything.

| Notebook | What it does | Owner |
|---|---|---|
| `MUST CHANGE[model_notebook_name].ipynb` | Loads `features.csv`, splits by date, trains a baseline and a scikit-learn model, compares precision / recall / F1, 
and times each prediction | Sarosha | `demo.ipynb` | Replays a cloudy day: plots actual solar output, marks predicted drops, and prints a battery recommendation for each alert | Naomi |

## Run order

1. `[model_notebook_name].ipynb` creates `predictions.csv`
2. `demo.ipynb` reads `predictions.csv` and `src/recommend.py`

See the main [README](../README.md) for the full project overview.
