 # Notebooks

Jupyter notebooks for the solar ramp forecasting system. The model was trained and run on **IBM LinuxONE Community Cloud (s390x)**, and outputs are saved so you can see the results without running anything.

| Notebook | What it does | Owner |
|---|---|---|
| `grid_resilience_model.ipynb` | Loads `features.csv`, builds a clear-sky reference, forecasts solar output 15 minutes ahead with gradient boosting, compares against persistence and smart persistence by time of day, raises ramp-down alerts, measures speed on s390x, and exports `ramp_model.onnx` | Sarosha |
| `02_demo.ipynb` | Replays a test day, shows alerts, and prints a battery recommendation for each alert | Naomi |

## Key results (Aug–Sep 2026 test)
- Forecast error 331 MW vs 465 MW smart persistence (**28.9% better**; 31.3% in the evening)
- Ramp-down alerts (500+ MW): **F1 0.82** vs 0.76 baseline
- **12 µs per forecast** (batch) on s390x

## Run order
1. `grid_resilience_model.ipynb` creates `predictions.csv`, `results_accuracy.csv`, `results_alerts.csv`, and `ramp_model.onnx`
2. `02_demo.ipynb` reads `predictions.csv` and `src/recommend.py`

See the main [README](../README.md) for the project overview.
