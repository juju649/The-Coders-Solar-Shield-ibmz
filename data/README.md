# Data

Cleaned data files used by the model and the map. Raw downloads are not included.

| File | Contents | Owner |
|---|---|---|
| `features.csv` | Solar output + weather features and the `target_drop` label, 5-minute steps, UTC | Ashlesh |
| `hospitals_tx.csv` | Texas hospital locations (HIFLD Next) | Naomi |
| `batteries_tx.csv` | Texas battery storage sites and capacity (EIA-860) | Naomi |
| `zip_risk_tx.csv` | Patient counts by ZIP code (emPOWER) with ZIP center points (Census) | Naomi |
| `predictions.csv` | Model output: time, actual solar, real drop, predicted drop, probability | Sarosha |

All patient data is aggregated at ZIP-code level. No personal information is used.
Sources and links are in [RESOURCES.md](../RESOURCES.md).

Make sure this is exact!!!
