"""
HVAC Lifetime Optimizer Engine
Hybrid ML/surrogate forecasting, retrofit assessment, and S3 optimization.
Author-ready research utility for severity-strategy HVAC datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json
import math
import time

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

RANDOM_STATE = 42
DEFAULT_TARGETS = [
    "annual_energy_MWh",
    "annual_cost_usd",
    "annual_co2_tonne",
    "mean_COP",
    "mean_delta",
    "mean_comfort_dev",
    "occupied_discomfort_days",
]

ALIASES = {
    "energy": ["annual_energy_MWh", "energy_MWh", "energy", "annual_energy", "total_energy_MWh"],
    "cost": ["annual_cost_usd", "cost_usd", "cost", "annual_cost"],
    "co2": ["annual_co2_tonne", "co2_tonne", "co2", "annual_co2"],
    "cop": ["mean_COP", "COP", "cop", "mean_cop"],
    "delta": ["mean_delta", "delta", "degradation", "degradation_index", "mean_degradation_index"],
    "comfort": ["mean_comfort_dev", "comfort", "comfort_dev", "mean_comfort"],
    "discomfort_days": ["occupied_discomfort_days", "discomfort_days", "uncomfortable_days"],
}


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    lower_map = {c.lower(): c for c in out.columns}
    for canon, names in ALIASES.items():
        preferred = names[0]
        if preferred in out.columns:
            continue
        for n in names:
            if n in out.columns:
                rename[n] = preferred
                break
            if n.lower() in lower_map:
                rename[lower_map[n.lower()]] = preferred
                break
    out = out.rename(columns=rename)
    if "strategy" not in out.columns:
        out["strategy"] = "S0"
    if "severity" not in out.columns:
        if "mean_delta" in out.columns:
            out["severity"] = out["mean_delta"]
        else:
            out["severity"] = 0.0
    if "year" not in out.columns:
        out["year"] = np.arange(len(out)) // max(1, len(out)//5) + 1
    if "climate" not in out.columns:
        out["climate"] = "uploaded"
    return out


def load_dataset(files: Iterable[str | Path]) -> pd.DataFrame:
    frames = []
    for f in files:
        p = Path(f)
        if p.suffix.lower() in [".xlsx", ".xls"]:
            frame = pd.read_excel(p)
        else:
            frame = pd.read_csv(p)
        frame["source_file"] = p.name
        frames.append(frame)
    if not frames:
        raise ValueError("No dataset files were supplied.")
    df = pd.concat(frames, ignore_index=True)
    return canonicalize_columns(df)


def detect_targets(df: pd.DataFrame) -> List[str]:
    return [c for c in DEFAULT_TARGETS if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def prepare_features(df: pd.DataFrame, targets: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    drop_cols = set(targets + ["source_file"])
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = df[targets].copy()
    # remove completely empty cols
    X = X.dropna(axis=1, how="all")
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    return X, y, num_cols, cat_cols


def build_model(model_name: str, num_cols: List[str], cat_cols: List[str]) -> Pipeline:
    model_name = model_name.lower()
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ], remainder="drop"
    )
    if model_name == "random forest":
        base = RandomForestRegressor(n_estimators=250, random_state=RANDOM_STATE, n_jobs=-1)
    elif model_name == "extra trees":
        base = ExtraTreesRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    elif model_name == "gradient boosting":
        base = MultiOutputRegressor(GradientBoostingRegressor(random_state=RANDOM_STATE))
    elif model_name == "catboost" and CATBOOST_AVAILABLE:
        base = MultiOutputRegressor(CatBoostRegressor(iterations=700, depth=6, learning_rate=0.04, loss_function="RMSE", verbose=False, random_seed=RANDOM_STATE))
    else:
        base = ExtraTreesRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    return Pipeline([("pre", pre), ("model", base)])


def train_models(df: pd.DataFrame, model_names: List[str], targets: Optional[List[str]] = None, test_size: float = 0.2):
    df = canonicalize_columns(df)
    targets = targets or detect_targets(df)
    if not targets:
        raise ValueError("No recognized KPI target columns found.")
    X, y, num_cols, cat_cols = prepare_features(df, targets)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)
    results = []
    models = {}
    for name in model_names:
        t0 = time.time()
        pipe = build_model(name, num_cols, cat_cols)
        pipe.fit(X_train, y_train)
        pred = pd.DataFrame(pipe.predict(X_test), columns=targets, index=y_test.index)
        elapsed = time.time() - t0
        for target in targets:
            rmse = math.sqrt(mean_squared_error(y_test[target], pred[target]))
            mae = mean_absolute_error(y_test[target], pred[target])
            r2 = r2_score(y_test[target], pred[target]) if len(y_test) > 1 else np.nan
            results.append({"model": name, "target": target, "RMSE": rmse, "MAE": mae, "R2": r2, "train_seconds": elapsed})
        models[name] = pipe
    metrics = pd.DataFrame(results).sort_values(["target", "RMSE"])
    return models, metrics, targets, X.columns.tolist()


def choose_best_model(metrics: pd.DataFrame) -> str:
    rank = metrics.groupby("model")["RMSE"].mean().sort_values()
    return rank.index[0]


def annualize_template(df: pd.DataFrame) -> pd.DataFrame:
    df = canonicalize_columns(df)
    group_cols = [c for c in ["strategy", "severity", "climate", "year"] if c in df.columns]
    if not group_cols:
        return df.copy()
    num = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c not in group_cols]
    cat = [c for c in df.columns if c not in num and c not in group_cols]
    annual = df.groupby(group_cols, dropna=False)[num].mean().reset_index()
    return annual


def project_future_inputs(df: pd.DataFrame, horizons: List[int], degradation_rate: float = 0.018, climate_load_growth: float = 0.004) -> pd.DataFrame:
    base = annualize_template(df)
    rows = []
    max_year = int(pd.to_numeric(base.get("year", pd.Series([1])), errors="coerce").max())
    last = base.sort_values("year").groupby([c for c in ["strategy", "severity", "climate"] if c in base.columns], dropna=False).tail(1)
    for _, r in last.iterrows():
        for horizon in horizons:
            for yr in range(max_year + 1, max_year + horizon + 1):
                rr = r.copy()
                t = yr - max_year
                rr["year"] = yr
                if "severity" in rr.index:
                    rr["severity"] = min(1.0, float(rr["severity"]) + degradation_rate * t)
                if "mean_delta" in rr.index:
                    rr["mean_delta"] = min(1.0, float(rr["mean_delta"]) + degradation_rate * t)
                if "annual_thermal_hvac_MWh" in rr.index:
                    rr["annual_thermal_hvac_MWh"] = float(rr["annual_thermal_hvac_MWh"]) * (1 + climate_load_growth) ** t
                rr["forecast_horizon_years"] = horizon
                rows.append(rr)
    return pd.DataFrame(rows)


def forecast_kpis(model: Pipeline, future_inputs: pd.DataFrame, targets: List[str], feature_columns: List[str]) -> pd.DataFrame:
    Xf = canonicalize_columns(future_inputs)
    for c in feature_columns:
        if c not in Xf.columns:
            Xf[c] = np.nan
    pred = pd.DataFrame(model.predict(Xf[feature_columns]), columns=targets)
    out = future_inputs.reset_index(drop=True).copy()
    for c in targets:
        out[f"pred_{c}"] = pred[c].values
    return out


def retrofit_analysis(forecast: pd.DataFrame, discount_rate: float = 0.08) -> pd.DataFrame:
    f = forecast.copy()
    energy_col = "pred_annual_energy_MWh" if "pred_annual_energy_MWh" in f else None
    delta_col = "pred_mean_delta" if "pred_mean_delta" in f else None
    co2_col = "pred_annual_co2_tonne" if "pred_annual_co2_tonne" in f else None
    cost_col = "pred_annual_cost_usd" if "pred_annual_cost_usd" in f else None
    cases = [
        ("R0_No_Retrofit", 0.00, 0.00, 0.0, 0),
        ("R1_Filter_Coil", 0.06, 0.08, 0.03, 8000),
        ("R2_Chiller_COP", 0.12, 0.05, 0.10, 35000),
        ("R3_AHU_Control", 0.09, 0.06, 0.06, 18000),
        ("R4_Full_S3_Retrofit", 0.18, 0.15, 0.14, 55000),
    ]
    rows = []
    groups = [c for c in ["strategy", "forecast_horizon_years"] if c in f.columns]
    for keys, g in f.groupby(groups, dropna=False) if groups else [((), f)]:
        base_energy = g[energy_col].sum() if energy_col else np.nan
        base_co2 = g[co2_col].sum() if co2_col else np.nan
        base_cost = g[cost_col].sum() if cost_col else np.nan
        base_life = int((g[delta_col] < 0.85).sum()) if delta_col else np.nan
        for name, e_sav, delta_red, opex_sav, capex in cases:
            annual_saving = base_cost * opex_sav / max(1, g["year"].nunique()) if cost_col else np.nan
            npv = -capex + sum(annual_saving / ((1 + discount_rate) ** i) for i in range(1, max(2, g["year"].nunique()+1))) if cost_col else np.nan
            simple_payback = capex / annual_saving if annual_saving and annual_saving > 0 else np.nan
            row = {"retrofit_case": name, "energy_saving_pct": e_sav*100, "degradation_reduction_pct": delta_red*100,
                   "capex_usd": capex, "NPV_usd": npv, "simple_payback_years": simple_payback,
                   "baseline_life_years_before_delta_0_85": base_life,
                   "estimated_life_extension_years": round(base_life * delta_red, 2) if not pd.isna(base_life) else np.nan,
                   "total_energy_MWh_after_retrofit": base_energy*(1-e_sav) if energy_col else np.nan,
                   "total_CO2_tonne_after_retrofit": base_co2*(1-e_sav) if co2_col else np.nan}
            if groups:
                if len(groups)==1: row[groups[0]] = keys
                else: row.update(dict(zip(groups, keys)))
            rows.append(row)
    return pd.DataFrame(rows)


def normalize(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or abs(mx-mn) < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)


def optimize_s3(forecast: pd.DataFrame, weights: Dict[str, float], comfort_limit: float = 1.5, delta_limit: float = 0.85) -> Tuple[pd.DataFrame, pd.DataFrame]:
    f = forecast.copy()
    # candidate table using all forecast rows; strategy comparison and feasible S3 domain
    cols = {
        "energy": "pred_annual_energy_MWh",
        "comfort": "pred_mean_comfort_dev",
        "co2": "pred_annual_co2_tonne",
        "delta": "pred_mean_delta",
        "cost": "pred_annual_cost_usd",
    }
    for k, c in cols.items():
        if c not in f.columns:
            f[c] = np.nan
    score = 0
    for k, c in cols.items():
        score = score + weights.get(k, 0.0) * normalize(f[c])
    f["objective_J"] = score
    f["feasible"] = (f[cols["comfort"]].fillna(0) <= comfort_limit) & (f[cols["delta"]].fillna(0) <= delta_limit)
    group_cols = [c for c in ["forecast_horizon_years", "year", "climate"] if c in f.columns]
    best_rows = []
    for keys, g in f.groupby(group_cols, dropna=False) if group_cols else [((), f)]:
        feasible = g[g["feasible"]]
        selected = feasible.loc[feasible["objective_J"].idxmin()] if len(feasible) else g.loc[g["objective_J"].idxmin()]
        best_rows.append(selected)
    best = pd.DataFrame(best_rows)
    strategy_summary = f.groupby([c for c in ["strategy", "forecast_horizon_years"] if c in f.columns], dropna=False).agg(
        mean_objective_J=("objective_J", "mean"), feasible_ratio=("feasible", "mean"),
        mean_energy=(cols["energy"], "mean"), mean_delta=(cols["delta"], "mean"), mean_comfort=(cols["comfort"], "mean")
    ).reset_index()
    return best, strategy_summary


def limitation_map(forecast: pd.DataFrame, comfort_limit: float = 1.5, delta_limit: float = 0.85) -> pd.DataFrame:
    f = forecast.copy()
    delta = f.get("pred_mean_delta", f.get("mean_delta", f.get("severity", pd.Series(np.nan, index=f.index))))
    comfort = f.get("pred_mean_comfort_dev", pd.Series(np.nan, index=f.index))
    energy = f.get("pred_annual_energy_MWh", pd.Series(np.nan, index=f.index))
    f["severity_bin"] = pd.cut(delta, bins=[-0.001, .25, .45, .65, .85, 1.01], labels=["Very low", "Low", "Moderate", "High", "Critical"])
    f["limitation_region"] = np.where((delta <= .65) & (comfort <= comfort_limit), "Green: S3 recommended",
                                np.where((delta <= delta_limit) & (comfort <= comfort_limit*1.25), "Yellow: S3 conditional", "Red: S3 not recommended"))
    return f.groupby(["severity_bin", "limitation_region"], dropna=False).agg(
        cases=("limitation_region", "size"), mean_delta=(delta.name if hasattr(delta,'name') and delta.name in f.columns else "severity", "mean") if "severity" in f.columns else ("year","count")
    ).reset_index()


def save_outputs(output_dir: str | Path, metrics: pd.DataFrame, forecast: pd.DataFrame, retrofit: pd.DataFrame, opt_best: pd.DataFrame, opt_summary: pd.DataFrame) -> Dict[str, str]:
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    paths = {}
    tables = {"model_metrics": metrics, "future_forecast": forecast, "retrofit_analysis": retrofit, "s3_optimum_rows": opt_best, "strategy_optimization_summary": opt_summary}
    for name, df in tables.items():
        p = out / f"{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = str(p)
    xlsx = out / "hvac_lifetime_optimizer_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    paths["excel"] = str(xlsx)
    return paths
