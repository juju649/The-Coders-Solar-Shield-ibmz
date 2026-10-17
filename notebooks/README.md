# Notebooks

These notebooks were run on IBM LinuxONE Community Cloud (s390x). Outputs are saved, so you can see the results without running anything.

| Notebook | What it does | Owner |
|---|---|---|
| `solarDataPreparation.ipynb` | Downloads ERCOT 15-minute solar (Apr–Sep 2026) and Open-Meteo weather for 5 Texas sites, cleans and aligns them in UTC, and builds `features.csv` | Aashlesha |
| `grid_resilience_model.ipynb` | Loads `features.csv`, builds a clear-sky reference, forecasts solar 15 minutes ahead with gradient boosting, compares against persistence baselines by time of day, raises ramp-down alerts, measures speed on s390x, and exports `ramp_model.onnx` | Sarosha |
| `ramp_model.onnx` | The trained model in ONNX format for IBM Z AI tooling | Sarosha |

## Key results (Aug–Sep 2026 test)

- Forecast error 331 MW vs 465 MW for smart persistence (28.9% better; 31.3% in the evening)
- Ramp-down alerts (500+ MW): F1 0.82 vs 0.76 for a simple rule
- 12 µs per forecast (batch) on s390x

## Run order

1. `solarDataPreparation.ipynb` creates `features.csv`
2. `grid_resilience_model.ipynb` creates `predictions.csv`, `results_accuracy.csv`, `results_alerts.csv` and `ramp_model.onnx`
3. The web app in [`app/`](../app/) serves live forecasts and battery recommendations on LinuxONE

See the main [README](../README.md) for the project overview.
