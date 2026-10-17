# Solar Shield
### Predicting solar drops 15 minutes ahead, and protecting the people who need power most.
**IBM Z Datathon 2026 · Track: Wildcard · Event code: SSF2026**
**Team:** Sarosha · Ashlesh · Naomi · Juwairiya  

🎥 Demo video: *(link coming soon)*  ·  📊 Slide deck: *(coming soon)*  ·  🗺️ Risk map: *(coming soon)*

## The Problem
When a cloud passes over a solar farm, its output can drop by half within
minutes. The power grid has to stay balanced every second, so sudden drops
can cause outages. Batteries can fill the gap instantly, but grid operators
need to know when a drop is coming, and where power matters most.

## Our Idea (Why It's a Wildcard)
Most grid tools focus on cost. We ask: **who needs power most?**

Our system:
1. Watches solar output and weather data for West Texas
2. Uses a machine learning model on **IBM Z (LinuxONE)** to predict a sharp solar drop 15 minutes before it happens
3. Checks a map of hospitals, battery sites, and ZIP codes with many patients on electricity-dependent medical equipment (oxygen, dialysis)
4. Recommends which battery to use to protect the most vulnerable people

**A human operator always approves. The AI only advises.**

## Architecture
*(Diagram coming soon)*

## Proof We Run on IBM Z (LinuxONE)
*(Screenshot of `uname -m` showing `s390x` coming soon)*
- Model trained and tested on IBM LinuxONE Community Cloud
- Prediction latency: **[X] ms per prediction**
- Screenshots: see [docs/linuxone](docs/linuxone)

## Data
*(Ashlesh: sources, date range, how we define a "drop", row count)*

## Model
*(Sarosha: baseline vs. model, how we split by date)*

## Results
*(Baseline vs. model on precision, recall, and F1)*

## Impact & Responsible Design
- Aggregated ZIP-level data only, with no personal information
- A human operator approves every recommendation
- No real equipment is controlled; this is a decision-support prototype

## Why IBM Z
Fast AI predictions right where the data lives, reliability for systems
that can't fail, and strong security for critical infrastructure.

## How to Run
1. Open the LinuxONE Jupyter Lab environment
2. Upload `data/features.csv`
3. Run the model notebook in `notebooks/`, then the demo notebook

## Repository Structure
| Folder | Contents |
|---|---|
| [notebooks](notebooks) | Model and demo notebooks (outputs saved) |
| [src](src) | Battery recommendation function |
| [data](data) | Cleaned data files |
| [docs](docs) | Diagram, screenshots, deck |

## Resources
See [RESOURCES.md](RESOURCES.md)
