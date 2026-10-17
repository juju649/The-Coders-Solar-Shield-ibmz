"""Solar Shield backend: 15-minute solar ramp forecasting + battery recommendations.

Runs on IBM LinuxONE (s390x). Uses only the Python standard library for the web
server, plus pandas / numpy / scikit-learn / joblib, which are already in the
LinuxONE Jupyter image.

Start:  python app.py            (serves on port 5000)
Data:   reads files from DATA_DIR (default: the folder above this one, i.e. "shared")
"""
import json
import math
import os
import platform
import threading
import time
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.abspath(os.path.join(HERE, "..")))
PORT = int(os.environ.get("PORT", "5000"))
VALID_END = pd.Timestamp("2026-08-01 05:00", tz="UTC")   # start of the unseen test period
TZ = "America/Chicago"


# ----------------------------------------------------------------------------- model + features
def build_features(df):
    """Same feature engineering as grid_resilience_model.ipynb (section 5). Past-only."""
    df = df.copy()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df = df.sort_values("time_utc").drop_duplicates("time_utc").reset_index(drop=True)
    local = df["time_utc"].dt.tz_convert(TZ)
    df["slot"] = local.dt.hour * 4 + local.dt.minute // 15
    df["date"] = local.dt.date
    grid = df.pivot_table(index="date", columns="slot", values="solar_mw")
    clear = grid.shift(1).rolling(14, min_periods=5).max().stack().rename("clear_sky_mw").reset_index()
    df = df.merge(clear, on=["date", "slot"], how="left")
    df["clear_sky_next"] = df["clear_sky_mw"].shift(-1)
    df["cs_change"] = df["clear_sky_next"] - df["clear_sky_mw"]
    df["kt"] = df["solar_mw"] / df["clear_sky_mw"].clip(lower=1)
    df["kt_lag15"] = df["solar_lag15"] / df["clear_sky_mw"].shift(1).clip(lower=1)
    df["kt_lag30"] = df["solar_lag30"] / df["clear_sky_mw"].shift(2).clip(lower=1)
    df["change_30"] = df["solar_mw"] - df["solar_lag30"]
    df["change_60"] = df["solar_mw"] - df["solar_lag60"]
    return df


class Forecaster:
    def __init__(self):
        bundle = joblib.load(os.path.join(DATA_DIR, "ramp_model.joblib"))
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.alert_mw = float(bundle["alert_mw"])
        self.ramp_mw = float(bundle["ramp_mw"])
        df = build_features(pd.read_csv(os.path.join(DATA_DIR, "features.csv")))
        day = df[df["is_daylight"] == 1].dropna(subset=self.features + ["target_solar_15"])
        self.test = day[day["time_utc"] >= VALID_END].copy()
        self.test["day"] = self.test["time_utc"].dt.tz_convert(TZ).dt.date.astype(str)
        t0 = time.perf_counter()
        self.test["predicted_change"] = self.model.predict(self.test[self.features])
        self.startup_ms = (time.perf_counter() - t0) * 1000
        self.test["predicted_solar_15"] = self.test["solar_mw"] + self.test["predicted_change"]
        self.test["alert"] = (self.test["predicted_change"] <= -self.alert_mw).astype(int)
        self.test["actual_ramp"] = ((self.test["target_solar_15"] - self.test["solar_mw"]) <= -self.ramp_mw).astype(int)

    def days(self):
        g = self.test.groupby("day").agg(alerts=("alert", "sum"), ramps=("actual_ramp", "sum"),
                                         caught=("alert", lambda s: int(((s == 1) & (self.test.loc[s.index, "actual_ramp"] == 1)).sum())))
        g = g.reset_index().sort_values("day")
        return [{"day": r.day, "alerts": int(r.alerts), "ramps": int(r.ramps), "caught": int(r.caught)} for r in g.itertuples()]

    def day(self, day):
        rows = self.test[self.test["day"] == day]
        if rows.empty:
            return None
        t0 = time.perf_counter()
        change = self.model.predict(rows[self.features])            # live inference for this request
        infer_ms = (time.perf_counter() - t0) * 1000
        local = rows["time_utc"].dt.tz_convert(TZ)
        points = []
        for i, (idx, r) in enumerate(rows.iterrows()):
            c = float(change[i])
            points.append({
                "t": local.loc[idx].strftime("%H:%M"),
                "t_next": (local.loc[idx] + pd.Timedelta("15min")).strftime("%H:%M"),
                "actual": round(float(r["solar_mw"]), 1),
                "actual_next": round(float(r["target_solar_15"]), 1),
                "forecast_next": round(float(r["solar_mw"]) + c, 1),
                "predicted_change": round(c, 1),
                "alert": int(c <= -self.alert_mw),
                "actual_ramp": int(r["actual_ramp"]),
            })
        return {"day": day, "points": points, "inference_ms": round(infer_ms, 2),
                "per_row_us": round(infer_ms / len(rows) * 1000, 1), "rows": len(rows)}


# ----------------------------------------------------------------------------- recommendation layer
ALIASES = {
    "name": ["name", "facility", "facility_name", "plant_name", "site", "hospital", "battery"],
    "lat": ["latitude", "lat", "y"],
    "lon": ["longitude", "lon", "lng", "long", "x"],
    "capacity_mw": ["capacity_mw", "capacity", "nameplate_mw", "mw", "power_mw", "nameplate_capacity_mw"],
    "patients": ["patients", "count", "beneficiaries", "electricity_dependent", "power_dependent", "total"],
    "zip": ["zip", "zipcode", "zip_code", "zcta", "zcta5"],
}


def normalize(df, needed):
    cols = {c.lower().strip(): c for c in df.columns}
    out = {}
    for key in needed:
        for alias in ALIASES[key]:
            if alias in cols:
                out[key] = df[cols[alias]]
                break
        else:
            raise ValueError(f"missing column for '{key}' (accepted names: {ALIASES[key]})")
    res = pd.DataFrame(out)
    for k in ("lat", "lon", "capacity_mw", "patients"):
        if k in res:
            res[k] = pd.to_numeric(res[k], errors="coerce")
    return res.dropna(subset=[k for k in ("lat", "lon") if k in res])


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# Clearly-labelled placeholder sites so the app works before the real files arrive.
SAMPLE = {
    "hospitals": pd.DataFrame({"name": ["Sample Hospital 1", "Sample Hospital 2", "Sample Hospital 3", "Sample Hospital 4"],
                               "lat": [29.42, 32.78, 31.85, 30.27], "lon": [-98.49, -96.80, -102.37, -97.74]}),
    "batteries": pd.DataFrame({"name": ["Sample Battery A", "Sample Battery B", "Sample Battery C", "Sample Battery D"],
                               "lat": [29.60, 32.60, 31.70, 30.40], "lon": [-98.30, -96.90, -102.10, -97.60],
                               "capacity_mw": [200.0, 150.0, 300.0, 100.0]}),
    "zips": pd.DataFrame({"zip": ["SAMPLE-1", "SAMPLE-2", "SAMPLE-3", "SAMPLE-4", "SAMPLE-5"],
                          "lat": [29.45, 32.75, 31.88, 30.30, 29.38], "lon": [-98.50, -96.82, -102.30, -97.70, -98.55],
                          "patients": [420.0, 380.0, 150.0, 300.0, 260.0]}),
}


class Recommender:
    def __init__(self):
        self.load()

    def load(self):
        try:
            h = normalize(pd.read_csv(os.path.join(DATA_DIR, "hospitals_tx.csv")), ["name", "lat", "lon"])
            b = normalize(pd.read_csv(os.path.join(DATA_DIR, "batteries_tx.csv")), ["name", "lat", "lon", "capacity_mw"])
            z = normalize(pd.read_csv(os.path.join(DATA_DIR, "zip_risk_tx.csv")), ["zip", "lat", "lon", "patients"])
            self.hospitals, self.batteries, self.zips, self.source, self.note = h, b, z, "real", "Using team data files"
        except Exception as e:                                           # files missing or columns differ
            self.hospitals, self.batteries, self.zips = SAMPLE["hospitals"], SAMPLE["batteries"], SAMPLE["zips"]
            self.source = "sample"
            self.note = f"Using SAMPLE sites (real files not loaded: {type(e).__name__}: {e})"
        z = self.zips.copy()
        if len(self.hospitals):
            d = haversine_km(z["lat"].values[:, None], z["lon"].values[:, None],
                             self.hospitals["lat"].values[None, :], self.hospitals["lon"].values[None, :])
            z["near_hospital"] = (d < 10).any(axis=1)
        else:
            z["near_hospital"] = False
        z["priority"] = z["patients"].fillna(0) * np.where(z["near_hospital"], 1.5, 1.0)
        self.zips = z.sort_values("priority", ascending=False)

    def recommend(self, drop_mw, top_zips=5, radius_km=100, max_sites=3):
        drop_mw = abs(float(drop_mw))
        targets = self.zips.head(top_zips)
        b = self.batteries.dropna(subset=["capacity_mw"]).copy()
        if b.empty or targets.empty:
            return {"text": "No battery or ZIP data available.", "batteries": [], "zips": []}
        d = haversine_km(b["lat"].values[:, None], b["lon"].values[:, None],
                         targets["lat"].values[None, :], targets["lon"].values[None, :])
        b["covered_patients"] = (np.where(d <= radius_km, 1, 0) * targets["patients"].fillna(0).values[None, :]).sum(axis=1)
        b = b[b["covered_patients"] > 0].sort_values(["covered_patients", "capacity_mw"], ascending=False)
        chosen, total = [], 0.0
        for r in b.itertuples():
            chosen.append({"name": str(r.name), "capacity_mw": round(float(r.capacity_mw), 1),
                           "lat": float(r.lat), "lon": float(r.lon), "covered_patients": int(r.covered_patients)})
            total += float(r.capacity_mw)
            if total >= drop_mw or len(chosen) >= max_sites:
                break
        zips = [{"zip": str(r.zip), "patients": int(r.patients), "lat": float(r.lat), "lon": float(r.lon),
                 "near_hospital": bool(r.near_hospital)} for r in targets.itertuples()]
        patients = sum(z["patients"] for z in zips)
        if chosen:
            names = ", ".join(f'{c["name"]} ({c["capacity_mw"]:.0f} MW)' for c in chosen)
            text = (f"Solar drop of ~{drop_mw:,.0f} MW expected in 15 min. Recommend discharging {names} "
                    f"to support the highest-risk ZIP codes (~{patients:,} electricity-dependent patients). "
                    f"Operator approval required.")
        else:
            text = "No battery within range of the highest-risk ZIP codes. Escalate to operator."
        return {"text": text, "batteries": chosen, "zips": zips, "covered_mw": round(total, 1),
                "source": self.source}

    def map_data(self, limit=400):
        return {"source": self.source, "note": self.note,
                "hospitals": self.hospitals.head(limit).to_dict("records"),
                "batteries": self.batteries.head(limit).to_dict("records"),
                "zips": self.zips.head(limit)[["zip", "lat", "lon", "patients", "near_hospital"]].to_dict("records")}


# ----------------------------------------------------------------------------- HTTP layer
FORECASTER = Forecaster()
RECOMMENDER = Recommender()
DECISIONS = []
LOCK = threading.Lock()


def results_summary():
    out = {}
    try:
        acc = pd.read_csv(os.path.join(DATA_DIR, "results_accuracy.csv"), index_col=0)
        out["accuracy"] = acc.reset_index().to_dict("records")
    except Exception:
        pass
    try:
        al = pd.read_csv(os.path.join(DATA_DIR, "results_alerts.csv"))
        out["alerts"] = al.to_dict("records")
    except Exception:
        pass
    return out


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=os.path.join(HERE, "static"), **kw)

    def log_message(self, fmt, *args):
        pass

    def send_json(self, obj, status=200):
        body = json.dumps(obj, default=lambda o: None if (isinstance(o, float) and math.isnan(o)) else str(o)).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/api/status":
            return self.send_json({
                "platform": platform.machine(), "python": platform.python_version(),
                "model": type(FORECASTER.model).__name__, "alert_mw": FORECASTER.alert_mw,
                "ramp_mw": FORECASTER.ramp_mw, "test_rows": len(FORECASTER.test),
                "startup_inference_ms": round(FORECASTER.startup_ms, 2),
                "recommendation_data": RECOMMENDER.source, "recommendation_note": RECOMMENDER.note,
                "results": results_summary()})
        if u.path == "/api/days":
            return self.send_json(FORECASTER.days())
        if u.path == "/api/day":
            data = FORECASTER.day(q.get("date", ""))
            return self.send_json(data if data else {"error": "unknown date"}, 200 if data else 404)
        if u.path == "/api/recommend":
            return self.send_json(RECOMMENDER.recommend(q.get("drop_mw", FORECASTER.alert_mw)))
        if u.path == "/api/map":
            return self.send_json(RECOMMENDER.map_data())
        if u.path == "/api/decisions":
            with LOCK:
                return self.send_json(DECISIONS[-50:])
        if u.path == "/api/reload":
            RECOMMENDER.load()
            return self.send_json({"recommendation_data": RECOMMENDER.source, "note": RECOMMENDER.note})
        return super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path == "/api/decision":
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            body["logged_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            with LOCK:
                DECISIONS.append(body)
                with open(os.path.join(DATA_DIR, "operator_decisions.jsonl"), "a") as f:
                    f.write(json.dumps(body) + "\n")
            return self.send_json({"ok": True})
        self.send_json({"error": "not found"}, 404)


if __name__ == "__main__":
    print(f"Solar Shield on {platform.machine()} | {len(FORECASTER.test)} test rows scored in "
          f"{FORECASTER.startup_ms:.1f} ms | recommendations: {RECOMMENDER.note}")
    print(f"Open http://<server-ip>:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
