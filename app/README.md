# Solar Shield web app

Operator console for Solar Shield. The backend runs on IBM LinuxONE (s390x) using pandas, scikit-learn, joblib and llama.cpp from the LinuxONE Jupyter image.

| Part | File | What it does |
|---|---|---|
| Backend | `app.py` | Loads the forecaster, trains the ramp classifier at startup, runs live forecasts, recommends batteries, asks IBM Granite for a briefing, logs operator decisions |
| Frontend | `static/index.html` | Interactive sky, replay chart, alert card with AI briefing, approve/dismiss, results, map |

## AI models
| Model | Role |
|---|---|
| HistGradientBoostingRegressor | Forecasts the solar MW change 15 minutes ahead |
| HistGradientBoostingClassifier | Probability of a 500+ MW ramp-down (AUC 0.947) |
| IBM Granite 4.0 (GGUF, llama.cpp zDNN build) | Writes the operator briefing; output is validated, with a template fallback |

## API
| Endpoint | Returns |
|---|---|
| `GET /api/status` | Platform (s390x), models, Granite status, results |
| `GET /api/days` | Test days with alert and ramp counts |
| `GET /api/day?date=2026-09-10` | Live forecasts, ramp probabilities and inference time |
| `GET /api/recommend?drop_mw=600` | Battery dispatch plan |
| `GET /api/brief` | IBM Granite operator briefing |
| `GET /api/map` | Hospitals, batteries and ZIP risk points |
| `POST /api/decision` | Logs an approval or dismissal |

## Run on LinuxONE
Put `ramp_model.joblib`, `features.csv`, the three team CSVs and the Granite `.gguf` file in `~/shared/`, then:
```
docker run -d --name solarshield -p 5000:5000 -v /home/linux1/shared:/home/jovyan/shared \
  -w /home/jovyan/shared/app registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest python app.py
sudo iptables -I INPUT -p tcp --dport 5000 -j ACCEPT
```
Open `http://<server-ip>:5000`.
