# Data

Cleaned data files used by the model and the map. Raw downloads are not included.

| File | Contents | Owner |
|---|---|---|
| `features.csv` | ERCOT solar output + averaged weather (5 Texas sites), 15-minute steps, UTC, Apr–Sep 2026 | Aashlesha |
| `hospitals_tx.csv` | Texas hospital locations (HIFLD Next) | Naomi |
| `batteries_tx.csv` | Texas battery storage sites and capacity (EIA-860) | Naomi |
| `zip_risk_tx.csv` | Patient counts by ZIP code (emPOWER) with ZIP center points (Census) | Naomi |
| `predictions.csv` | Model output (Aug–Sep 2026): actual solar, 15-min-ahead forecast, predicted change (MW), actual ramp-down, model alert | Sarosha |
| `results_accuracy.csv` | Forecast error by time of day vs. persistence baselines | Sarosha |
| `results_alerts.csv` | Ramp-down alert precision / recall / F1 vs. baseline rule | Sarosha |

All patient data is aggregated at ZIP-code level. No personal information is used. Sources and links are in [RESOURCES.md](../RESOURCES.md).
