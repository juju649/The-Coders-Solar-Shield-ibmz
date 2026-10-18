# Solar Shield

We forecast drops in Texas solar power 15 minutes ahead and recommend which batteries should protect the people who need power most.

IBM Z Datathon 2026 · Wildcard track · Event code SSF2026
Team: Sarosha, Aashlesha, Naomi, Juwairiya

**Live app (running on IBM LinuxONE):** http://148.100.98.89:5000

Demo video: coming soon · Slides: coming soon

## Why we built this

Texas solar set a record of 34,665 MW in July 2026, and almost all of it disappears every evening, sometimes more than 5,000 MW in 15 minutes. Batteries can fill the gap, but operators need a heads-up and need to know where power matters most. When the grid fails, people on home oxygen and dialysis are hit first. Over 192,000 Texans rely on electricity-dependent medical equipment.

## What it does

1. Forecasts ERCOT solar output 15 minutes ahead on IBM LinuxONE
2. A second model estimates the chance of a major (500+ MW) ramp-down
3. Raises an alert when it expects a drop of 400 MW or more
4. Finds the ZIP codes with the most electricity-dependent patients, the nearest hospitals, and nearby battery sites
5. IBM Granite writes a plain-English briefing for the operator
6. The operator approves or dismisses every recommendation

## How it works

```
ERCOT solar + weather → features.csv → forecaster + ramp classifier (LinuxONE) → alert
hospitals + batteries + patient counts ─────────────────────────────→ battery plan
                                              IBM Granite briefing → operator approves
```

## Data

- **ERCOT Fuel Mix:** statewide solar, every 15 minutes, April–September 2026
- **Open-Meteo:** weather averaged over 5 Texas solar regions
- **HHS emPOWER:** electricity-dependent patients by ZIP code (aggregated, no personal data)
- **HIFLD Next:** 876 Texas hospitals
- **EIA-860:** 205 Texas battery sites

We first labeled a "drop" as a 20% fall in 15 minutes, but almost every one turned out to be sunset. So we predict the actual change in MW instead, and treat a fall of 500 MW or more as a ramp-down.

## Hybrid AI on IBM Z

| Model | What it does | Result |
|---|---|---|
| Gradient-boosting forecaster | Predicts the solar MW change 15 minutes ahead | 28.9% lower error than smart persistence |
| Gradient-boosting ramp classifier | Probability of a 500+ MW ramp-down | AUC 0.947 on unseen Aug–Sep data |
| IBM Granite 4.0 (llama.cpp, zDNN build) | Writes the operator briefing | ~14 s per briefing on s390x |

All three run live in Docker on IBM LinuxONE. The forecaster compares today's output to a clear-sky reference built from the past 14 days, so it can tell normal sunset apart from weather. Trained on April–July, tested on August–September.

## Results (Aug–Sep 2026)

| | Model | Smart persistence | Better by |
|---|---|---|---|
| All daylight | 331 MW error | 465 MW error | 29% |
| Evening | 426 MW | 620 MW | 31% |
| Midday | 290 MW | 326 MW | 11% |

Ramp-down alerts: F1 **0.82**, vs 0.76 for a simple rule. In the evening we catch 99% of ramp-downs.

Midday cloud drops are still hard to catch, because hourly weather data can't see individual clouds. Satellite imagery would be the next step.

## Running on IBM Z

- Trained and served on IBM LinuxONE Community Cloud (s390x) in Docker containers
- Training takes about 7 seconds; each forecast takes about 12 microseconds in batch
- IBM Granite runs on IBM Z's on-chip AI acceleration (zDNN)
- Forecaster exported to ONNX for IBM Z AI tooling
- Completed both IBM AI/ML labs (multi-model; predictive + generative)
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
| `docs/` | Screenshots, lab evidence, diagrams |

## How to run

1. Run `notebooks/solarDataPreparation.ipynb` to make `features.csv`
2. In LinuxONE Jupyter, put `features.csv` in `shared/` and run `notebooks/grid_resilience_model.ipynb`
3. Start the app with Docker (see [`app/README.md`](app/README.md))
