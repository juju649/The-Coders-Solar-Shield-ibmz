# Solar Shield

We forecast drops in Texas solar power 15 minutes ahead and suggest which batteries should cover the people who need power most.

IBM Z Datathon 2026 · Wildcard track · Event code SSF2026
Team: Sarosha, Aashlesha, Naomi, Juwairiya

**Live app (running on IBM LinuxONE):** http://148.100.98.89:5000
Demo video: coming soon · Slides: coming soon

## Why we built this

Every evening Texas loses around 30,000 MW of solar as the sun goes down, sometimes over 2,000 MW in 15 minutes. Clouds cause sudden drops too. Batteries can cover the gap, but operators need a heads-up, and they need to know where the power matters most. When the grid struggles, hospitals and people on home oxygen or dialysis are hit first.

## What it does

1. Forecasts ERCOT solar output 15 minutes ahead, using a model that runs on IBM LinuxONE
2. Raises an alert when it expects a drop of 400 MW or more
3. Looks up the ZIP codes with the most electricity-dependent patients, the nearest hospitals, and nearby battery sites
4. Recommends which batteries to use. An operator approves or dismisses every recommendation.

## How it works

```
ERCOT solar + weather → features.csv → model on LinuxONE → alert
hospitals + batteries + patient counts ──────────────────→ battery recommendation → operator
```

## Data

- **ERCOT Fuel Mix:** statewide solar, every 15 minutes, April–September 2026
- **Open-Meteo:** weather averaged over 5 Texas solar regions
- **HHS emPOWER:** electricity-dependent patients by ZIP code (aggregated, no personal data)
- **HIFLD Next:** 876 Texas hospitals
- **EIA-860:** 205 Texas battery sites

We first tried labeling a "drop" as a 20% fall in 15 minutes, but almost every one of those turned out to be sunset. So instead we predict the actual change in MW, and treat a fall of 500 MW or more as a ramp-down.

## Model

Gradient boosting (scikit-learn). It compares today's output to a clear-sky reference built from the past 14 days, so it can tell normal sunset apart from weather. We trained on April–July and tested on August–September, which the model never saw.

## Results (Aug–Sep 2026)

| | Model | Smart persistence | Better by |
|---|---|---|---|
| All daylight | 331 MW error | 465 MW error | 29% |
| Evening | 426 MW | 620 MW | 31% |
| Midday | 290 MW | 326 MW | 11% |

Ramp-down alerts: F1 **0.82**, vs 0.76 for a simple rule. In the evening we catch 99% of ramp-downs.

Midday cloud drops are still hard to catch, because hourly weather data can't see individual clouds. Satellite imagery would be the next step.

## Running on IBM Z

- Trained and served on IBM LinuxONE Community Cloud (s390x)
- Training takes about 7 seconds; each forecast takes about 12 microseconds in batch
- Model exported to ONNX for IBM Z's AI accelerator
- Backend status: http://148.100.98.89:5000/api/status
- Screenshots in [`docs/`](docs/)

## Responsible design

- Only aggregated ZIP-level health data
- A human approves every action, and every decision is logged
- It doesn't control any real equipment
- Recommendations are based on location; we don't know actual grid connections or battery charge

## Repo layout

| Folder | What's in it |
|---|---|
| `notebooks/` | Data prep and model notebooks |
| `app/` | Web app (backend + frontend) |
| `src/` | Battery recommendation and map code |
| `data/` | Cleaned data and results |
| `docs/` | Screenshots |

## How to run

1. Run the data prep notebook to make `features.csv`
2. In LinuxONE Jupyter, put `features.csv` in `shared/` and run `notebooks/grid_resilience_model.ipynb`
3. Start the app with Docker (see [`app/README.md`](app/README.md))
