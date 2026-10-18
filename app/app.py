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
        self.train_classifier(day)

    def train_classifier(self, day):
        """Second model: a gradient-boosting classifier that estimates the probability of a
        500+ MW ramp-down in the next 15 minutes. Trained on Apr-Jul (same split and purge as
        the notebook), evaluated on the unseen Aug-Sep test set. Runs on LinuxONE at startup."""
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
        train = day[day["time_utc"] + pd.Timedelta("15min") < VALID_END]
        y = ((train["target_solar_15"] - train["solar_mw"]) <= -self.ramp_mw).astype(int)
        t0 = time.perf_counter()
        pos = max(float(y.mean()), 1e-6)
        w = np.where(y == 1, 0.5 / pos, 0.5 / (1 - pos))
        self.clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=42)
        self.clf.fit(train[self.features], y, sample_weight=w)
        self.clf_train_s = time.perf_counter() - t0
        prob = self.clf.predict_proba(self.test[self.features])[:, 1]
        self.test["ramp_prob"] = prob
        yt = self.test["actual_ramp"]
        both = ((self.test["alert"] == 1) & (prob >= 0.5)).astype(int)
        self.clf_metrics = {
            "model": "HistGradientBoostingClassifier",
            "train_seconds": round(self.clf_train_s, 2),
            "test_auc": round(float(roc_auc_score(yt, prob)), 3),
            "classifier_f1_at_0.5": round(float(f1_score(yt, (prob >= 0.5).astype(int))), 2),
            "agreement_alerts_precision": round(float(precision_score(yt, both, zero_division=0)), 2),
            "agreement_alerts_recall": round(float(recall_score(yt, both, zero_division=0)), 2),
        }

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
            pr = float(r["ramp_prob"]) if "ramp_prob" in r else None
            points.append({
                "ramp_prob": round(pr, 3) if pr is not None else None,
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
    "patients": ["patients", "at_risk_count", "count", "beneficiaries", "electricity_dependent", "power_dependent", "total"],
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
            braw = pd.read_csv(os.path.join(DATA_DIR, "batteries_tx.csv"))
            if "technology" in braw.columns:                      # Naomi's filter: battery storage only
                braw = braw[braw["technology"].astype(str).str.strip() == "Batteries"]
            b = normalize(braw, ["name", "lat", "lon", "capacity_mw"])
            zpath = os.path.join(DATA_DIR, "zip_risk_coordinates_tx.csv")
            if not os.path.exists(zpath):
                zpath = os.path.join(DATA_DIR, "zip_risk_tx.csv")
            z = normalize(pd.read_csv(zpath, dtype=str), ["zip", "lat", "lon", "patients"])
            z = z[z["lon"] > -104.6]          # El Paso / far west Texas is on the WECC grid, not ERCOT
            self.hospitals, self.batteries, self.zips, self.source = h, b, z, "real"
            self.note = (f"Team data: {len(h):,} hospitals (HIFLD), {len(b):,} battery sites (EIA-860), "
                         f"{len(z):,} ZIP codes with {int(z['patients'].sum()):,} electricity-dependent patients (emPOWER)")
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
        self.zips = z.sort_values("priority", ascending=False).reset_index(drop=True)

    def recommend(self, drop_mw, top_zips=5, zip_radius_km=50, max_sites=3):
        """Naomi's rule, applied to the statewide drop: start at the highest-risk ZIP codes,
        group at-risk ZIPs within 50 km, find the nearest hospitals and the closest battery
        for each area, and add batteries until the forecast drop is covered."""
        drop_mw = abs(float(drop_mw))
        b = self.batteries.dropna(subset=["capacity_mw"]).reset_index(drop=True)
        if b.empty or self.zips.empty:
            return {"text": "No battery or ZIP data available.", "batteries": [], "zips": [], "hospitals": []}
        chosen, used, zips_out, hosp_out, total, anchors = [], set(), [], [], 0.0, []
        for anchor in self.zips.itertuples():
            if any(haversine_km(anchor.lat, anchor.lon, a[0], a[1]) < zip_radius_km * 2 for a in anchors):
                continue                                          # one recommendation per distinct area
            anchors.append((anchor.lat, anchor.lon))
            dz = haversine_km(anchor.lat, anchor.lon, self.zips["lat"].values, self.zips["lon"].values)
            area = self.zips[dz <= zip_radius_km].sort_values("patients", ascending=False).head(top_zips)
            db = haversine_km(anchor.lat, anchor.lon, b["lat"].values, b["lon"].values)
            free = [j for j in range(len(b)) if j not in used]
            near = [j for j in free if db[j] <= 100]
            if not free:
                break
            j = max(near, key=lambda k: b.iloc[k]["capacity_mw"]) if near else min(free, key=lambda k: db[k])
            used.add(j); r = b.iloc[j]
            people = int(area["patients"].sum())
            chosen.append({"name": str(r["name"]), "capacity_mw": round(float(r["capacity_mw"]), 1),
                           "lat": float(r["lat"]), "lon": float(r["lon"]), "distance_km": round(float(db[j]), 1),
                           "covered_patients": people})
            for zr in area.itertuples():
                zips_out.append({"zip": str(zr.zip), "patients": int(zr.patients), "lat": float(zr.lat),
                                 "lon": float(zr.lon), "near_hospital": bool(zr.near_hospital),
                                 "battery": len(chosen) - 1})
            if len(self.hospitals):
                dh = haversine_km(anchor.lat, anchor.lon, self.hospitals["lat"].values, self.hospitals["lon"].values)
                k = int(np.argmin(dh))
                hosp_out.append({"name": str(self.hospitals.iloc[k]["name"]).title(), "distance_km": round(float(dh[k]), 1)})
            total += float(r["capacity_mw"])
            if total >= drop_mw or len(chosen) >= max_sites:
                break
        patients = sum(z["patients"] for z in zips_out)
        names = ", ".join(f'{c["name"]} ({c["capacity_mw"]:.0f} MW)' for c in chosen)
        text = (f"Solar drop of ~{drop_mw:,.0f} MW expected in 15 min. Recommend discharging {names} "
                f"to support the highest-risk ZIP codes (~{patients:,} electricity-dependent patients) "
                f"and nearby hospitals. Operator approval required. Geographic planning recommendation: "
                f"grid connectivity and battery state of charge are not confirmed.")
        return {"text": text, "batteries": chosen, "zips": zips_out, "hospitals": hosp_out[:4],
                "covered_mw": round(total, 1), "people": patients, "source": self.source}

    def map_data(self, limit=2500):
        return {"source": self.source, "note": self.note,
                "hospitals": self.hospitals.head(limit).to_dict("records"),
                "batteries": self.batteries.head(limit).to_dict("records"),
                "zips": self.zips.head(limit)[["zip", "lat", "lon", "patients", "near_hospital"]].to_dict("records")}


# ----------------------------------------------------------------------------- generative AI layer
class Briefer:
    """Generative AI: IBM Granite (GGUF) served by llama.cpp on LinuxONE, using the zDNN/NNPA
    build that ships in IBM's LinuxONE Jupyter image. Turns the two predictive models' outputs
    and the recommendation into a short operator briefing. Falls back to a template if the
    model is not available."""
    SEARCH = ["/home/ibm-user", "/home/jovyan", "/opt", "/models", "/root"]

    def __init__(self):
        self.llm, self.path, self.error, self.cache = None, None, None, {}
        self.lock = threading.Lock()
        self.path = os.environ.get("LLM_PATH") or self.find_model()
        if not self.path:
            self.error = "no .gguf model found"
            return
        try:
            from llama_cpp import Llama
            self.llm = Llama(model_path=self.path, n_ctx=2048, n_threads=os.cpu_count() or 2, verbose=False)
        except Exception as e:
            self.llm, self.error = None, f"{type(e).__name__}: {e}"

    def find_model(self):
        import glob
        found = []
        for root in self.SEARCH:
            found += glob.glob(os.path.join(root, "**", "*.gguf"), recursive=True)
        found = [f for f in found if os.path.getsize(f) > 50_000_000]
        granite = [f for f in found if "granite" in os.path.basename(f).lower()]
        pick = granite or found
        return min(pick, key=os.path.getsize) if pick else None

    def status(self):
        return {"available": self.llm is not None,
                "model": os.path.basename(self.path) if self.path else None,
                "runtime": "llama.cpp (zDNN/NNPA build)" if self.llm else None,
                "error": self.error}

    def brief(self, ctx):
        key = json.dumps(ctx, sort_keys=True)
        if key in self.cache:
            return self.cache[key]
        if self.llm is None:
            text = (f"Solar output is forecast to fall about {abs(ctx['drop_mw']):,.0f} MW by {ctx['time']}"
                    f" (ramp probability {ctx['ramp_prob']:.0%}). Dispatching {ctx['batteries']} would cover"
                    f" about {ctx['people']:,} electricity-dependent patients near {ctx['hospitals']}."
                    f" Please approve or dismiss.")
            out = {"text": text, "generated_by": "template (language model unavailable)", "seconds": 0}
            self.cache[key] = out
            return out
        # Base (non-instruct) model: use a few-shot completion prompt it can continue,
        # then validate the output and fall back to the template if it is unusable.
        facts = (f"Drop: {abs(ctx['drop_mw']):,.0f} MW by {ctx['time']}. Ramp probability: {ctx['ramp_prob']:.0%}. "
                 f"Batteries: {ctx['batteries']}. Patients covered: {ctx['people']:,}. Hospitals: {ctx['hospitals']}.")
        prompt = ("Grid operator briefings.\n\n"
                  "Facts: Drop: 620 MW by 6:45 PM. Ramp probability: 88%. Batteries: Alpha Storage (150 MW). "
                  "Patients covered: 2,310. Hospitals: Memorial Hospital.\n"
                  "Briefing: Solar output is expected to fall about 620 MW by 6:45 PM, with an 88% chance of a major "
                  "ramp-down. Dispatching Alpha Storage would protect about 2,310 patients near Memorial Hospital. "
                  "Please approve or dismiss.\n\n"
                  "Facts: Drop: 450 MW by 5:15 PM. Ramp probability: 73%. Batteries: Bravo Battery (100 MW), "
                  "Cedar Energy (90 MW). Patients covered: 1,480. Hospitals: St. Luke's Clinic.\n"
                  "Briefing: Solar output is expected to fall about 450 MW by 5:15 PM, with a 73% chance of a major "
                  "ramp-down. Dispatching Bravo Battery and Cedar Energy would protect about 1,480 patients near "
                  "St. Luke's Clinic. Please approve or dismiss.\n\n"
                  f"Facts: {facts}\nBriefing:")
        t0 = time.perf_counter()
        with self.lock:
            r = self.llm.create_completion(prompt, max_tokens=90, temperature=0.1, top_p=0.9,
                                           stop=["\n", "Facts:", "Grid operator"])
        text = r["choices"][0]["text"].strip()
        bad = (len(text) < 40 or "Facts" in text or "You are" in text
               or f"{abs(ctx['drop_mw']):,.0f}" not in text)
        if bad:
            out = {"text": (f"Solar output is expected to fall about {abs(ctx['drop_mw']):,.0f} MW by {ctx['time']}, "
                            f"with a {ctx['ramp_prob']:.0%} chance of a major ramp-down. Dispatching {ctx['batteries']} "
                            f"would protect about {ctx['people']:,} patients near {ctx['hospitals']}. Please approve or dismiss."),
                   "generated_by": "template (model output failed validation)",
                   "seconds": round(time.perf_counter() - t0, 1)}
            self.cache[key] = out
            return out
        if not text.endswith("."):
            text = text.rsplit(".", 1)[0] + "." if "." in text else text
        out = {"text": text, "generated_by": os.path.basename(self.path) + " on " + platform.machine(),
               "seconds": round(time.perf_counter() - t0, 1)}
        self.cache[key] = out
        return out


# ----------------------------------------------------------------------------- HTTP layer
FORECASTER = Forecaster()
RECOMMENDER = Recommender()
BRIEFER = Briefer()
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
                "models": {"forecaster": type(FORECASTER.model).__name__,
                           "ramp_classifier": FORECASTER.clf_metrics,
                           "generative": BRIEFER.status()},
                "recommendation_data": RECOMMENDER.source, "recommendation_note": RECOMMENDER.note,
                "results": results_summary()})
        if u.path == "/api/days":
            return self.send_json(FORECASTER.days())
        if u.path == "/api/day":
            data = FORECASTER.day(q.get("date", ""))
            return self.send_json(data if data else {"error": "unknown date"}, 200 if data else 404)
        if u.path == "/api/recommend":
            return self.send_json(RECOMMENDER.recommend(q.get("drop_mw", FORECASTER.alert_mw)))
        if u.path == "/api/brief":
            ctx = {"drop_mw": float(q.get("drop_mw", 0)), "time": q.get("time", ""),
                   "ramp_prob": float(q.get("ramp_prob", 0) or 0), "batteries": q.get("batteries", ""),
                   "people": int(float(q.get("people", 0) or 0)), "hospitals": q.get("hospitals", "")}
            return self.send_json(BRIEFER.brief(ctx))
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
          f"{FORECASTER.startup_ms:.1f} ms | ramp classifier AUC {FORECASTER.clf_metrics['test_auc']} | "
          f"generative AI: {BRIEFER.status()} | recommendations: {RECOMMENDER.note}")
    print(f"Open http://<server-ip>:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
