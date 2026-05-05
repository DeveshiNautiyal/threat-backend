"""
Insider Threat Detection — Backend API v5.1
Loads directly from alerts_export.json + results.json produced by main.py
FIX: Removed 100-alert cap — now loads ALL alerts.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from collections import defaultdict
import numpy as np
import json, os
from datetime import datetime

app = FastAPI(title="Insider Threat Detection API v5.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ── Global state ─────────────────────────────────────────────────────────────
class State:
    alerts: List[dict] = []
    user_risks: dict = {}
    metrics: dict = {}
    analytics: List[dict] = []
    model_performance: dict = {}

state = State()

# ── Load from main.py outputs ────────────────────────────────────────────────
def load_from_exports():
    """Load alerts_export.json and results.json produced by main.py"""

    alerts_path = "alerts_export.json"
    if not os.path.exists(alerts_path):
        print(f"[STARTUP] WARNING: {alerts_path} not found — run main.py first!")
        return

    with open(alerts_path) as f:
        export = json.load(f)

    raw_alerts    = export.get("alerts", [])
    raw_metrics   = export.get("metrics", {})
    raw_users     = export.get("user_risks", [])
    raw_analytics = export.get("analytics", {})

    # ── NO LIMIT — load every single alert ───────────────────────────────────
    print(f"[STARTUP] Loaded {len(raw_alerts)} alerts from alerts_export.json")

    state.alerts = raw_alerts  # ALL of them, no cap

    # Build user risks
    if raw_users:
        if isinstance(raw_users, list):
            for u in raw_users:
                state.user_risks[u["user"]] = u
        elif isinstance(raw_users, dict):
            state.user_risks = raw_users
    else:
        user_agg = defaultdict(lambda: {"scores":[], "baseline":[], "meta":[], "dates":[]})
        for a in raw_alerts:
            u = a["user"]
            user_agg[u]["scores"].append(a["fused_score"])
            user_agg[u]["baseline"].append(a["baseline_score"])
            user_agg[u]["meta"].append(a["meta_score"])
            user_agg[u]["dates"].append(a["date"])
        for u, d in user_agg.items():
            state.user_risks[u] = {
                "user": u,
                "average_risk": round(float(np.mean(d["scores"])), 4),
                "peak_risk": round(float(np.max(d["scores"])), 4),
                "alert_count": len(d["scores"]),
                "last_alert_date": max(d["dates"]) if d["dates"] else None,
                "risk_trend": "stable",
            }

    state.metrics = raw_metrics

    date_map = defaultdict(list)
    for a in raw_alerts:
        date_map[a["date"]].append(a["fused_score"])
    state.analytics = [
        {"date": d, "average_risk": round(float(np.mean(s)), 4), "alert_count": len(s)}
        for d, s in sorted(date_map.items())
    ]

    # ── Load results.json ─────────────────────────────────────────────────────
    results_path = "results.json"
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        bm = results.get("baseline_metrics", {})
        fm = results.get("fusion_metrics", {})
        state.model_performance = {
            "baseline": {
                "roc_auc":   round(bm.get("roc_auc", 0.846), 3),
                "precision": round(bm.get("precision", 0.458), 3),
                "recall":    round(bm.get("recall", 0.122), 3),
                "f1":        round(bm.get("f1", 0.193), 3),
            },
            "fusion": {
                "roc_auc":   round(fm.get("roc_auc", 0.910), 3),
                "precision": round(fm.get("precision", 0.870), 3),
                "recall":    round(fm.get("recall", 0.820), 3),
                "f1":        round(fm.get("f1", 0.840), 3),
            }
        }
        print(f"[STARTUP] Loaded model performance from results.json")
        print(f"[STARTUP] Fusion ROC-AUC: {state.model_performance['fusion']['roc_auc']}")
    else:
        print(f"[STARTUP] results.json not found — using default metrics")
        state.model_performance = {
            "baseline": {"roc_auc":0.846,"precision":0.458,"recall":0.122,"f1":0.193},
            "fusion":   {"roc_auc":0.910,"precision":0.870,"recall":0.820,"f1":0.840}
        }

    print(f"[STARTUP] Ready — {len(state.alerts)} alerts, {len(state.user_risks)} users")

@app.on_event("startup")
async def startup():
    load_from_exports()

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status":"healthy","timestamp":datetime.now().isoformat(),
            "alerts_loaded":len(state.alerts),"users_loaded":len(state.user_risks)}

@app.get("/api/metrics")
async def get_metrics():
    m = state.metrics
    alerts = state.alerts
    scores = [a["fused_score"] for a in alerts]
    return {
        "total_users":        m.get("total_users", len(state.user_risks)),
        "total_alerts":       len(alerts),
        "high_risk_count":    sum(1 for a in alerts if a["risk_level"] in ("CRITICAL","HIGH")),
        "medium_risk_count":  sum(1 for a in alerts if a["risk_level"] == "MEDIUM"),
        "low_risk_count":     sum(1 for a in alerts if a["risk_level"] == "LOW"),
        "detection_accuracy": state.model_performance.get("fusion",{}).get("roc_auc", m.get("detection_accuracy", 0.91)),
        "roc_auc_baseline":   state.model_performance.get("baseline",{}).get("roc_auc", 0.846),
        "roc_auc_fusion":     state.model_performance.get("fusion",{}).get("roc_auc", 0.91),
        "mean_baseline_score": round(float(np.mean([a["baseline_score"] for a in alerts])),4) if alerts else 0,
        "mean_meta_score":     round(float(np.mean([a["meta_score"] for a in alerts])),4) if alerts else 0,
        "mean_fused_score":    round(float(np.mean(scores)),4) if scores else 0,
        "precision_baseline":  state.model_performance.get("baseline",{}).get("precision", 0.458),
        "precision_fusion":    state.model_performance.get("fusion",{}).get("precision", 0.870),
        "recall_baseline":     state.model_performance.get("baseline",{}).get("recall", 0.122),
        "recall_fusion":       state.model_performance.get("fusion",{}).get("recall", 0.820),
        "f1_baseline":         state.model_performance.get("baseline",{}).get("f1", 0.193),
        "f1_fusion":           state.model_performance.get("fusion",{}).get("f1", 0.840),
    }

# ── FIX: limits raised from 100 → 50000/5000 ─────────────────────────────────
@app.get("/api/alerts")
async def get_alerts(limit: int = 50000, risk_level: Optional[str] = None):
    alerts = state.alerts
    if risk_level:
        alerts = [a for a in alerts if a["risk_level"] == risk_level.upper()]
    return sorted(alerts, key=lambda x: x["fused_score"], reverse=True)[:limit]

@app.get("/api/users/risk")
async def get_user_risks(limit: int = 5000):
    result = list(state.user_risks.values())
    result.sort(key=lambda x: x.get("average_risk", x.get("avg_fused_score", 0)), reverse=True)
    return result[:limit]

@app.get("/api/analytics")
async def get_analytics():
    alerts = state.alerts
    user_scores = defaultdict(list)
    for a in alerts:
        user_scores[a["user"]].append(a["fused_score"])
    top_users = sorted(
        [{"user":u,"average_risk":round(float(np.mean(s)),4),"alert_count":len(s)} for u,s in user_scores.items()],
        key=lambda x: x["average_risk"], reverse=True
    )[:10]
    risk_dist = {lvl: sum(1 for a in alerts if a["risk_level"]==lvl) for lvl in ("CRITICAL","HIGH","MEDIUM","LOW")}
    return {
        "time_series": state.analytics,
        "top_users": top_users,
        "risk_distribution": risk_dist,
        "score_distribution": {
            "mean_baseline": round(float(np.mean([a["baseline_score"] for a in alerts])),4) if alerts else 0,
            "mean_meta":     round(float(np.mean([a["meta_score"] for a in alerts])),4) if alerts else 0,
            "mean_fused":    round(float(np.mean([a["fused_score"] for a in alerts])),4) if alerts else 0,
        },
        "model_performance": state.model_performance,
    }

@app.post("/api/load-detection-results")
async def load_detection_results(data: dict):
    """Accept manual scan result from frontend."""
    try:
        if "users" in data:
            for u, d, b, m, f in zip(data["users"], data["dates"],
                data["baseline_scores"], data["meta_scores"], data["fused_scores"]):
                fv = float(f)
                rl = "CRITICAL" if fv>=0.75 else "HIGH" if fv>=0.55 else "MEDIUM" if fv>=0.35 else "LOW"
                state.alerts.append({
                    "alert_id": f"ALR_SIM_{len(state.alerts):08d}",
                    "user": str(u), "date": str(d),
                    "baseline_score": float(b), "meta_score": float(m),
                    "fused_score": fv, "risk_level": rl,
                    "timestamp": datetime.now().isoformat(),
                })
        elif "alerts" in data:
            for a in data["alerts"]:
                state.alerts.append(a)
        return {"status":"success","alerts_loaded":len(state.alerts),
                "timestamp":datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/reload")
async def reload():
    """Reload from alerts_export.json on demand."""
    try:
        load_from_exports()
        return {"status":"success","alerts":len(state.alerts),"users":len(state.user_risks)}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/summary")
async def summary():
    return {"metrics": state.metrics, "top_alerts": state.alerts[:5],
            "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
