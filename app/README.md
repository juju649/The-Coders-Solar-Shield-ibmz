# Solar Shield web app

Operator console for the ramp forecasting model. The backend runs on IBM LinuxONE (s390x) and uses only the Python standard library plus pandas, scikit-learn and joblib from the LinuxONE Jupyter image.

| Part | File | What it does |
|---|---|---|
| Backend | `app.py` | Loads `ramp_model.joblib` and `features.csv`, runs live 15-minute forecasts, recommends batteries near the highest-risk ZIP codes, logs operator decisions |
| Frontend | `static/index.html` | Interactive sky that follows the replay clock, scrubbable forecast chart, alert and dispatch approval, results, map |

## API
| Endpoint | Returns |
|---|---|
| `GET /api/status` | Platform (s390x), model, results tables |
| `GET /api/days` | Test days with alert and ramp counts |
| `GET /api/day?date=2026-09-10` | Live forecasts for one day plus inference time |
| `GET /api/recommend?drop_mw=600` | Battery dispatch plan for a forecast drop |
| `GET /api/map` | Hospitals, batteries and ZIP risk points |
| `POST /api/decision` | Logs an operator approval or dismissal |

## Run on LinuxONE
```
docker run -d --name solarshield -p 5000:5000 -v /home/linux1/shared:/home/jovyan/shared \
  -w /home/jovyan/shared/app registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest python app.py
sudo iptables -I INPUT -p tcp --dport 5000 -j ACCEPT
```
Open `http://<server-ip>:5000`. Battery recommendations use `hospitals_tx.csv`, `batteries_tx.csv` and `zip_risk_tx.csv` from the shared folder when present, and clearly labelled sample sites otherwise.
