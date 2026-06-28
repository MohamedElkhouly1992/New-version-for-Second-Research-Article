import io
from pathlib import Path
import tempfile
import zipfile

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from hvac_lifetime_engine import (
    load_dataset, train_models, choose_best_model, project_future_inputs,
    forecast_kpis, retrofit_analysis, optimize_s3, save_outputs, CATBOOST_AVAILABLE
)

st.set_page_config(page_title="HVAC Lifetime Optimizer", layout="wide")
st.title("HVAC Lifetime Optimizer: ML, Surrogate, Retrofit and S3 Optimization")
st.caption("Deployable research app for two-axis severity–strategy HVAC datasets and long-term KPI forecasting.")

with st.expander("Required/recognized dataset columns", expanded=False):
    st.write("Recommended columns: strategy, severity, climate, year, annual_energy_MWh, annual_cost_usd, annual_co2_tonne, mean_COP, mean_delta, mean_comfort_dev, occupied_discomfort_days.")
    st.write("The app accepts CSV/XLSX files and automatically maps common aliases such as energy, CO2, COP, delta, comfort.")

uploads = st.file_uploader("Upload your five-year severity–strategy dataset(s)", type=["csv", "xlsx", "xls"], accept_multiple_files=True)

with st.sidebar:
    st.header("Model setup")
    choices = ["Extra Trees", "Random Forest", "Gradient Boosting"]
    if CATBOOST_AVAILABLE:
        choices.insert(0, "CatBoost")
    models_to_run = st.multiselect("Algorithms", choices, default=choices[:2])
    horizons = st.multiselect("Forecast horizons", [10, 20, 30], default=[10, 20, 30])
    degradation_rate = st.slider("Annual degradation increment", 0.0, 0.06, 0.018, 0.001)
    climate_growth = st.slider("Annual cooling-load growth", 0.0, 0.03, 0.004, 0.001)
    discount_rate = st.slider("Discount rate for retrofit NPV", 0.0, 0.25, 0.08, 0.01)
    st.header("S3 objective weights")
    w_energy = st.slider("Energy", 0.0, 1.0, 0.30, 0.05)
    w_comfort = st.slider("Comfort", 0.0, 1.0, 0.20, 0.05)
    w_co2 = st.slider("CO₂", 0.0, 1.0, 0.15, 0.05)
    w_delta = st.slider("Degradation", 0.0, 1.0, 0.25, 0.05)
    w_cost = st.slider("Cost", 0.0, 1.0, 0.10, 0.05)
    comfort_limit = st.number_input("Comfort limit", value=1.5)
    delta_limit = st.number_input("Failure/limit δ", value=0.85)

if not uploads:
    st.info("Upload your dataset to start. You can also generate a synthetic demo dataset from the README script.")
    st.stop()

with tempfile.TemporaryDirectory() as td:
    paths = []
    for u in uploads:
        p = Path(td) / u.name
        p.write_bytes(u.read())
        paths.append(p)
    df = load_dataset(paths)

st.subheader("1. Uploaded dataset preview")
st.dataframe(df.head(50), use_container_width=True)
st.write(f"Rows: {len(df):,} | Columns: {len(df.columns):,}")

if st.button("Run full analysis", type="primary"):
    if not models_to_run:
        st.error("Select at least one algorithm.")
        st.stop()
    progress = st.progress(0)
    st.write("Training ML/surrogate models...")
    models, metrics, targets, feature_columns = train_models(df, models_to_run)
    best_name = choose_best_model(metrics)
    best_model = models[best_name]
    progress.progress(25)

    st.write(f"Best model selected by average RMSE: **{best_name}**")
    st.dataframe(metrics, use_container_width=True)

    future_inputs = project_future_inputs(df, horizons, degradation_rate, climate_growth)
    forecast = forecast_kpis(best_model, future_inputs, targets, feature_columns)
    progress.progress(50)

    retro = retrofit_analysis(forecast, discount_rate)
    weights = {"energy": w_energy, "comfort": w_comfort, "co2": w_co2, "delta": w_delta, "cost": w_cost}
    opt_best, opt_summary = optimize_s3(forecast, weights, comfort_limit, delta_limit)
    progress.progress(75)

    outdir = Path(tempfile.mkdtemp()) / "hvac_lifetime_results"
    saved = save_outputs(outdir, metrics, forecast, retro, opt_best, opt_summary)
    progress.progress(100)

    st.subheader("2. Long-term KPI forecast")
    st.dataframe(forecast.head(200), use_container_width=True)

    if "pred_annual_energy_MWh" in forecast.columns:
        fig, ax = plt.subplots()
        plot_df = forecast.groupby(["forecast_horizon_years", "year"], as_index=False)["pred_annual_energy_MWh"].mean()
        for h, g in plot_df.groupby("forecast_horizon_years"):
            ax.plot(g["year"], g["pred_annual_energy_MWh"], label=f"{h}-year horizon")
        ax.set_xlabel("Year")
        ax.set_ylabel("Predicted annual energy (MWh)")
        ax.legend()
        st.pyplot(fig)

    st.subheader("3. Retrofit/lifetime assessment")
    st.dataframe(retro, use_container_width=True)

    st.subheader("4. S3 optimization and strategy comparison")
    st.dataframe(opt_summary, use_container_width=True)
    st.dataframe(opt_best.head(100), use_container_width=True)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for p in Path(outdir).glob("*"):
            z.write(p, arcname=p.name)
    st.download_button("Download all results as ZIP", data=zip_buffer.getvalue(), file_name="hvac_lifetime_optimizer_results.zip", mime="application/zip")
