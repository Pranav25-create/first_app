# Sales Forecasting & Demand Intelligence

End-to-end internship project for the Superstore sales dataset. The repository does not include `train.csv`: download the Kaggle dataset yourself and place its `train.csv` in the repository root before running the analysis. The notebook and pipeline do not invent results when the input data is absent.

## Quick start

1. Download `train.csv` from [Kaggle — Sales Forecasting](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting) and put it beside `analysis.py`.
2. Create an environment and install dependencies: `pip install -r requirements.txt`.
3. Run `python analysis.py` or open `analysis.ipynb` in Jupyter/Colab and run the cells. This generates `outputs/` with metrics, forecasts, anomaly and segment tables, charts, and a data-backed `summary.md` report.
4. Run `streamlit run app.py` for the interactive dashboard. The dashboard runs the pipeline on first launch if outputs are missing.

The secondary Video Game Sales dataset listed in the assignment does not share a meaningful sales/date key with the Superstore transactions. The analysis explicitly avoids an invalid row-level merge; the assignment's anomaly analysis uses the Superstore weekly sales series. If using the supplementary dataset for independent practice, keep it as a separate source unless a defensible join key is available.

## Project contents

- `analysis.ipynb` — documented end-to-end project walkthrough.
- `analysis.py` — reusable data preparation, EDA, time-series analysis, model comparison, anomaly detection, clustering, charting, and report generation.
- `app.py` — four-page Streamlit dashboard.
- `requirements.txt` — runtime dependencies.
- `outputs/` — generated locally after running the pipeline; not committed because these are derived from the user-supplied dataset.

## Method notes

Forecast metrics use a chronological final-three-month holdout; no random shuffling is used. XGBoost forecasts recursively, feeding earlier predictions into later lags. The selected model is chosen by holdout MAE. MAPE excludes zero-valued actuals. SARIMA uses a documented monthly seasonal specification, while Prophet and XGBoost are evaluated on the same holdout period. Very short or incomplete input series may make some models unavailable; the pipeline reports errors instead of fabricating forecasts. Clustering features are standardized before K-Means and cluster labels are descriptive interpretations, not fixed business truths.

The supplied Kaggle data is historical and its date range may not match the assignment's claimed coverage. Treat forecasts and anomaly explanations as hypotheses for business review, not causal findings. Verify the app's generated results and add the live Streamlit link after deploying from your own account; this repository cannot deploy to an external account or submit the Google Form on your behalf.
