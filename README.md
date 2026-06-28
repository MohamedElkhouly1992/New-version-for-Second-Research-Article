# HVAC Lifetime Optimizer

Deployable Streamlit software for the second-manuscript framework:

1. ML/surrogate learning from five-year HVAC severity-strategy datasets.
2. Long-term KPI forecasting for 10, 20, and 30 years.
3. Retrofit and maintenance lifetime assessment.
4. Multi-objective S3 optimization and comparison against other strategies.

## Expected columns
Recommended dataset columns:

- `strategy`
- `severity`
- `climate`
- `year`
- `annual_energy_MWh`
- `annual_cost_usd`
- `annual_co2_tonne`
- `mean_COP`
- `mean_delta`
- `mean_comfort_dev`
- `occupied_discomfort_days`

The app accepts CSV/XLSX and maps common aliases such as `energy`, `co2`, `delta`, `comfort`, and `COP`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Generate demo dataset

```bash
python generate_demo_dataset.py
streamlit run streamlit_app.py
```

Upload `demo_hvac_5year_severity_strategy.csv` in the app.

## Outputs

The app exports:

- `model_metrics.csv`
- `future_forecast.csv`
- `retrofit_analysis.csv`
- `s3_optimum_rows.csv`
- `strategy_optimization_summary.csv`
- `hvac_lifetime_optimizer_results.xlsx`

## Manuscript interpretation

Use the outputs as follows:

- `model_metrics.csv`: validates the ML/surrogate model accuracy.
- `future_forecast.csv`: supports 10-, 20-, and 30-year KPI projection.
- `retrofit_analysis.csv`: quantifies lifetime extension, energy savings, payback, and NPV.
- `strategy_optimization_summary.csv`: compares S0, S1, S2, and S3 under the objective function.
- `s3_optimum_rows.csv`: identifies feasible optimal S3 operating regions.

