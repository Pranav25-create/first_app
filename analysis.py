"""End-to-end Superstore sales forecasting analysis pipeline.

Run from the repository root: python analysis.py
Input: train.csv (Kaggle Superstore dataset)
Output: outputs/*.csv, outputs/charts/*.png, outputs/summary.md
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "train.csv"
OUT = ROOT / "outputs"
CHARTS = OUT / "charts"


def load_sales(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.name}. Download the Kaggle dataset and place train.csv in {ROOT}.")
    df = pd.read_csv(path)
    required = {"Order Date", "Ship Date", "Sales", "Category", "Region", "Sub-Category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    # Kaggle dates are commonly M/D/YYYY; infer without forcing day-first interpretation.
    for col in ("Order Date", "Ship Date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales"]).copy()
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Month Name"] = df["Order Date"].dt.month_name()
    df["Week Number"] = df["Order Date"].dt.isocalendar().week.astype(int)
    df["Day of Week"] = df["Order Date"].dt.day_name()
    df["Quarter"] = df["Order Date"].dt.quarter
    df["Season"] = df["Month"].map({12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",6:"Summer",7:"Summer",8:"Summer",9:"Autumn",10:"Autumn",11:"Autumn"})
    return df.sort_values("Order Date")


def monthly_series(frame: pd.DataFrame) -> pd.Series:
    return frame.set_index("Order Date")["Sales"].resample("MS").sum().asfreq("MS", fill_value=0).astype(float)


def metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    a, p = actual[valid], predicted[valid]
    if len(a) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan}
    return {"MAE": float(mean_absolute_error(a, p)), "RMSE": float(np.sqrt(mean_squared_error(a, p))),
            "MAPE": float(np.mean(np.abs((a[a != 0] - p[a != 0]) / a[a != 0])) * 100) if np.any(a != 0) else np.nan}


def seasonal_naive(series: pd.Series, periods: int) -> np.ndarray:
    vals = series.to_numpy(dtype=float)
    return np.array([vals[-12 + (i % 12)] if len(vals) >= 12 else vals[-1] for i in range(periods)])


def xgb_forecast(train: pd.Series, periods: int, future_index: pd.DatetimeIndex):
    from xgboost import XGBRegressor
    def row(s: pd.Series, date: pd.Timestamp):
        return [s.iloc[-1], s.iloc[-2], s.iloc[-3], s.iloc[-3:].mean(), date.month, date.quarter,
                int(date.month in (12,1,2)), int(date.month in (3,4,5)), int(date.month in (6,7,8)), int(date.month in (9,10,11))]
    features, targets = [], []
    for i in range(3, len(train)):
        features.append([train.iloc[i-1], train.iloc[i-2], train.iloc[i-3], train.iloc[i-3:i].mean(),
                         train.index[i].month, train.index[i].quarter,
                         int(train.index[i].month in (12,1,2)), int(train.index[i].month in (3,4,5)),
                         int(train.index[i].month in (6,7,8)), int(train.index[i].month in (9,10,11))])
        targets.append(train.iloc[i])
    if len(targets) < 6:
        raise ValueError("Not enough monthly history for lag-based XGBoost (need at least 9 months).")
    model = XGBRegressor(n_estimators=250, max_depth=3, learning_rate=.03, subsample=.9,
                         colsample_bytree=.9, objective="reg:squarederror", random_state=42, n_jobs=1)
    model.fit(np.asarray(features), np.asarray(targets))
    history = train.copy()
    preds = []
    for date in future_index:
        pred = float(model.predict(np.asarray([row(history, date)]))[0])
        preds.append(max(0.0, pred))
        history.loc[date] = pred
    return np.asarray(preds)


def forecast_models(series: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Evaluate on the last 3 observed months and refit each viable model for 3 future months."""
    series = series.asfreq("MS", fill_value=0).astype(float)
    if len(series) < 18:
        raise ValueError("At least 18 monthly observations are recommended for comparable seasonal forecasts.")
    train, test = series.iloc[:-3], series.iloc[-3:]
    test_idx = test.index
    future_idx = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=3, freq="MS")
    records, forecasts, errors = [], {}, {}
    # SARIMA: d=1 and D=1 handle level/trend and annual seasonality; compact ARMA terms reduce overfit.
    try:
        fit = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12),
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        pred = fit.get_forecast(3).predicted_mean.to_numpy()
        forecasts["SARIMA"] = np.maximum(0, pred)
        val = SARIMAX(series.iloc[:-3], order=(1,1,1), seasonal_order=(1,1,1,12),
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False).get_forecast(3).predicted_mean
        records.append({"Model":"SARIMA", **metrics(test, val)})
    except Exception as exc:
        errors["SARIMA"] = str(exc)
    try:
        from prophet import Prophet
        ptrain = pd.DataFrame({"ds": train.index, "y": train.values})
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(ptrain)
        val = model.predict(pd.DataFrame({"ds": test_idx}))[["yhat"]]["yhat"].to_numpy()
        model_full = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model_full.fit(pd.DataFrame({"ds": series.index, "y": series.values}))
        fc = model_full.predict(pd.DataFrame({"ds": future_idx}))
        forecasts["Prophet"] = np.maximum(0, fc["yhat"].to_numpy())
        records.append({"Model":"Prophet", **metrics(test, val)})
        fc[["ds","yhat","yhat_lower","yhat_upper"]].to_csv(OUT / "prophet_forecast.csv", index=False)
        fig = model_full.plot_components(fc); fig.savefig(CHARTS / "prophet_components.png", bbox_inches="tight"); plt.close(fig)
    except Exception as exc:
        errors["Prophet"] = str(exc)
    try:
        val_pred = xgb_forecast(train, 3, test_idx)
        forecasts["XGBoost"] = xgb_forecast(series, 3, future_idx)
        records.append({"Model":"XGBoost", **metrics(test, val_pred)})
    except Exception as exc:
        errors["XGBoost"] = str(exc)
    comparison = pd.DataFrame(records)
    if comparison.empty:
        raise RuntimeError(f"No forecast model completed. Details: {errors}")
    comparison = comparison.sort_values("MAE").reset_index(drop=True)
    best = str(comparison.iloc[0]["Model"])
    forecast_table = pd.DataFrame({"Date": future_idx, "Best Model": best, "Forecast": forecasts[best]})
    for name, preds in forecasts.items():
        forecast_table[name] = preds
    comparison.to_csv(OUT / "model_comparison.csv", index=False)
    forecast_table.to_csv(OUT / "overall_forecast.csv", index=False)
    (OUT / "model_errors.json").write_text(json.dumps(errors, indent=2))
    # Holdout chart: model forecasts are evaluated on the same final three actual months.
    fig, ax = plt.subplots(figsize=(11,5)); ax.plot(series.index, series.values, label="Actual", color="#315a7d")
    for name in comparison["Model"]:
        try:
            if name == "SARIMA":
                values = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False,
                                 enforce_invertibility=False).fit(disp=False).get_forecast(3).predicted_mean.to_numpy()
            elif name == "Prophet":
                from prophet import Prophet
                m=Prophet(yearly_seasonality=True,weekly_seasonality=False,daily_seasonality=False); m.fit(pd.DataFrame({"ds":train.index,"y":train.values}))
                values=m.predict(pd.DataFrame({"ds":test_idx}))["yhat"].to_numpy()
            else: values=xgb_forecast(train,3,test_idx)
            ax.plot(test_idx, values, marker="o", label=f"{name} holdout")
        except Exception: pass
    ax.set_title("Three-month chronological holdout comparison"); ax.set_ylabel("Sales"); ax.legend(); fig.tight_layout(); fig.savefig(CHARTS/"model_comparison.png",dpi=150); plt.close(fig)
    return comparison, forecast_table, {"best": best, "errors": errors}


def segment_forecasts(df: pd.DataFrame, best_model: str) -> pd.DataFrame:
    parts = [("Category", c, df[df["Category"] == c]) for c in ["Furniture","Technology","Office Supplies"]]
    parts += [("Region", r, df[df["Region"] == r]) for r in ["West","East"]]
    rows=[]
    for kind,name,part in parts:
        s=monthly_series(part)
        if len(s)<18: continue
        idx=pd.date_range(s.index[-1]+pd.offsets.MonthBegin(1),periods=3,freq="MS")
        try:
            if best_model=="SARIMA":
                pred=SARIMAX(s,order=(1,1,1),seasonal_order=(1,1,1,12),enforce_stationarity=False,enforce_invertibility=False).fit(disp=False).get_forecast(3).predicted_mean.to_numpy()
                pred=np.maximum(0,pred)
            elif best_model=="XGBoost": pred=xgb_forecast(s,3,idx)
            else:
                from prophet import Prophet
                m=Prophet(yearly_seasonality=True,weekly_seasonality=False,daily_seasonality=False);m.fit(pd.DataFrame({"ds":s.index,"y":s.values}))
                pred=np.maximum(0,m.predict(pd.DataFrame({"ds":idx}))["yhat"].to_numpy())
            for d,v in zip(idx,pred): rows.append({"Type":kind,"Segment":name,"Date":d,"Forecast":v})
        except Exception as exc: warnings.warn(f"Could not forecast {kind} {name}: {exc}")
    result=pd.DataFrame(rows); result.to_csv(OUT/"segment_forecasts.csv",index=False)
    if not result.empty:
        fig,ax=plt.subplots(figsize=(11,6))
        for (kind,name),g in result.groupby(["Type","Segment"]): ax.plot(g["Date"],g["Forecast"],marker="o",label=f"{name} ({kind})")
        ax.set_title("Three-month category and region forecasts");ax.set_ylabel("Sales");ax.legend();fig.tight_layout();fig.savefig(CHARTS/"segment_forecasts.png",dpi=150);plt.close(fig)
    return result


def run_pipeline() -> dict:
    OUT.mkdir(exist_ok=True); CHARTS.mkdir(exist_ok=True)
    df=load_sales()
    # Core EDA and data-quality outputs.
    quality={"rows":len(df),"duplicate_rows":int(df.duplicated().sum()),"missing_cells":int(df.isna().sum().sum()),
             "start_date":str(df["Order Date"].min().date()),"end_date":str(df["Order Date"].max().date()),
             "unparseable_ship_dates":int(df["Ship Date"].isna().sum())}
    (OUT/"data_quality.json").write_text(json.dumps(quality,indent=2))
    category=df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    region_year=df.groupby(["Region","Year"])["Sales"].sum().unstack(fill_value=0)
    growth=region_year.pct_change(axis=1).iloc[:,1:] if region_year.shape[1]>1 else pd.DataFrame(index=region_year.index)
    consistency=pd.DataFrame({"Average YoY growth":growth.mean(axis=1),"YoY growth volatility":growth.std(axis=1)}).sort_values("YoY growth volatility")
    ship=(df.dropna(subset=["Ship Date"]).assign(ShipDays=lambda x:(x["Ship Date"]-x["Order Date"]).dt.days).groupby("Region")["ShipDays"].agg(["mean","median"]))
    monthly=monthly_series(df); monthly.index.name="Date"; monthly.rename("Sales").to_csv(OUT/"monthly_sales.csv")
    daily=df.set_index("Order Date")["Sales"].resample("D").sum(); daily.rename("Sales").to_csv(OUT/"daily_sales.csv")
    weekly=df.set_index("Order Date")["Sales"].resample("W-SUN").sum(); weekly.rename("Sales").to_csv(OUT/"weekly_sales.csv")
    seasonal=df.groupby("Month")["Sales"].agg(["mean","median","sum"]); seasonal.to_csv(OUT/"monthly_seasonality.csv")
    fig,ax=plt.subplots(figsize=(12,5));ax.plot(monthly.index,monthly.values);ax.set_title("Monthly sales trend");ax.set_ylabel("Sales");fig.tight_layout();fig.savefig(CHARTS/"monthly_trend.png",dpi=150);plt.close(fig)
    decomposition=None
    if len(monthly)>=24:
        dec=seasonal_decompose(monthly,model="additive",period=12,extrapolate_trend="freq")
        fig=dec.plot();fig.set_size_inches(12,9);fig.tight_layout();fig.savefig(CHARTS/"decomposition.png",dpi=150);plt.close(fig)
        residual=dec.resid.dropna().abs().nlargest(3)
        decomposition={"strongest_absolute_residual_months":[{"date":str(i.date()),"residual":float(v)} for i,v in residual.items()]}
    adf_before=adfuller(monthly,autolag="AIC") if len(monthly)>4 else None
    diff=monthly.diff().dropna();adf_after=adfuller(diff,autolag="AIC") if len(diff)>4 else None
    with open(OUT/"stationarity.txt","w") as f:
        f.write("ADF null hypothesis: the series has a unit root (is non-stationary). Reject at p < 0.05.\n")
        f.write(f"Original: statistic={adf_before[0]:.4f}, p={adf_before[1]:.6f}\n" if adf_before else "Original series too short.\n")
        f.write(f"First difference: statistic={adf_after[0]:.4f}, p={adf_after[1]:.6f}\n" if adf_after else "Differenced series too short.\n")
    # Two complementary weekly anomaly detectors.
    w=weekly.rename_axis("Date").reset_index(); w["Isolation Forest"] = IsolationForest(contamination=.05,random_state=42).fit_predict(w[["Sales"]])==-1
    baseline=w["Sales"].rolling(8,min_periods=4,center=True).mean();spread=w["Sales"].rolling(8,min_periods=4,center=True).std()
    w["Rolling Z"] = ((w["Sales"]-baseline).abs()/spread.replace(0,np.nan))>2
    w["Anomaly"] = w["Isolation Forest"]|w["Rolling Z"]
    w.to_csv(OUT/"weekly_anomalies.csv",index=False)
    fig,ax=plt.subplots(figsize=(13,5));ax.plot(w["Date"],w["Sales"],label="Weekly sales");a=w[w["Anomaly"]];ax.scatter(a["Date"],a["Sales"],color="crimson",label="Anomaly",zorder=3);ax.legend();ax.set_title("Weekly sales anomalies (Isolation Forest / rolling z-score)");fig.tight_layout();fig.savefig(CHARTS/"weekly_anomalies.png",dpi=150);plt.close(fig)
    # Demand segmentation for each sub-category.
    sub=df.groupby("Sub-Category")["Sales"].sum().rename("Total Sales")
    msub=df.groupby([pd.Grouper(key="Order Date",freq="MS"),"Sub-Category"])["Sales"].sum().unstack(fill_value=0)
    first_year=msub.index.year.min(); last_year=msub.index.year.max()
    by_year=df.groupby(["Year","Sub-Category"])["Sales"].sum().unstack(fill_value=0)
    if first_year in by_year.index and last_year in by_year.index:
        growth_rate=(by_year.loc[last_year]-by_year.loc[first_year])/by_year.loc[first_year].replace(0,np.nan)*100
    else: growth_rate=pd.Series(0,index=sub.index)
    volatility=msub.std().reindex(sub.index).fillna(0)
    order_col="Order ID" if "Order ID" in df else None
    aov=df.groupby("Sub-Category")["Sales"].mean() if order_col is None else df.groupby("Sub-Category").apply(lambda g:g.groupby(order_col)["Sales"].sum().mean(),include_groups=False)
    features=pd.DataFrame({"Total Sales":sub,"Growth Rate":growth_rate,"Volatility":volatility,"Average Order Value":aov}).replace([np.inf,-np.inf],np.nan).fillna(0)
    scaled=StandardScaler().fit_transform(features)
    scores=[]
    for k in range(2,min(7,len(features))): scores.append({"k":k,"inertia":float(KMeans(n_clusters=k,random_state=42,n_init=20).fit(scaled).inertia_)})
    best_k=min(scores,key=lambda x:x["inertia"])["k"] if scores else 2
    km=KMeans(n_clusters=best_k,random_state=42,n_init=20).fit(scaled);features["Cluster"]=km.labels_
    pcs=PCA(n_components=2).fit_transform(scaled);features["PC1"]=pcs[:,0];features["PC2"]=pcs[:,1]
    # Labels reflect relative cluster profile; growth and scale are used as readable descriptions.
    means=features.groupby("Cluster").mean(numeric_only=True)
    labels={}
    for c,r in means.iterrows():
        if r["Growth Rate"]>5: label="Growing demand"
        elif r["Growth Rate"] < -5: label="Declining demand"
        elif r["Total Sales"]>=features["Total Sales"].median() and r["Volatility"]<features["Volatility"].median(): label="High volume, stable demand"
        elif r["Volatility"]>=features["Volatility"].median(): label="Lower volume, volatile demand"
        else: label="Steady demand"
        labels[c]=label
    features["Demand Segment"]=features["Cluster"].map(labels);features.index.name="Sub-Category";features.reset_index().to_csv(OUT/"demand_segments.csv",index=False)
    pd.DataFrame(scores).to_csv(OUT/"elbow_scores.csv",index=False)
    fig,ax=plt.subplots(figsize=(9,6));scatter=ax.scatter(features["PC1"],features["PC2"],c=features["Cluster"],cmap="tab10",s=90)
    for name,r in features.iterrows():ax.annotate(name,(r["PC1"],r["PC2"]),fontsize=8)
    ax.set_xlabel("Principal component 1");ax.set_ylabel("Principal component 2");ax.set_title("Product sub-category demand clusters");fig.tight_layout();fig.savefig(CHARTS/"demand_clusters.png",dpi=150);plt.close(fig)
    comparison,forecast,best=forecast_models(monthly)
    seg=segment_forecasts(df,best["best"])
    # Meaningful outputs for the dashboard and report.
    category.rename("Sales").to_csv(OUT/"category_sales.csv");region_year.to_csv(OUT/"region_year_sales.csv")
    ship.to_csv(OUT/"shipping_days_by_region.csv")
    year_sales=df.groupby("Year")["Sales"].sum().rename("Sales");year_sales.to_csv(OUT/"yearly_sales.csv")
    (OUT/"eda_summary.json").write_text(json.dumps({"top_category":str(category.index[0]),"top_category_sales":float(category.iloc[0]),
        "most_consistent_region":str(consistency.index[0]) if len(consistency) else "Not enough years",
        "best_model":best["best"],"mean_shipping_days":float((df["Ship Date"]-df["Order Date"]).dt.days.mean()),
        "highest_month_by_average":int(seasonal["mean"].idxmax()),"strongest_growth_segment":str(seg.groupby("Segment")["Forecast"].sum().idxmax()) if not seg.empty else "Unavailable"},indent=2))
    summary_lines=["# Executive business summary", "", "Generated from the supplied Superstore transaction file. Forecasts are planning estimates, not guarantees.", "",
       f"- **Highest-revenue category:** {category.index[0]} (${category.iloc[0]:,.2f}).",
       f"- **Most consistent regional growth:** {consistency.index[0] if len(consistency) else 'Not measurable from the available years'} (lowest variation in observed year-over-year growth).",
       f"- **Average shipping time:** {(df['Ship Date']-df['Order Date']).dt.days.mean():.2f} days.",
       f"- **Best holdout model by MAE:** {best['best']}. See model_comparison.csv for MAE, RMSE and MAPE.",
       f"- **Three-month forecast:** " + "; ".join(f"{r.Date:%b %Y}: ${r.Forecast:,.2f}" for r in forecast.itertuples()) + ".",
       f"- **Highest average sales month:** {pd.Timestamp(2000,int(seasonal['mean'].idxmax()),1).strftime('%B')}.",
       f"- **Detected unusual weeks:** {int(w['Anomaly'].sum())}. Unusual observations are flags for investigation; the data alone does not establish a cause.",
       "- **Stocking approach:** use higher buffers for high-volume/stable items; use smaller, more frequent replenishment for volatile or declining items; review growth signals with buyers before committing inventory.",
       "- **Business recommendations:** (1) align replenishment targets to the winning model's forecast and its holdout error; (2) investigate flagged weeks against promotion, holiday, and supply records; (3) review segment-level inventory policies quarterly, using safety stock for stable high-volume demand and cautious buys for volatile demand.",
       "- **Limitation:** only historical sales are modeled; promotions, stockouts, price changes, and supply constraints are not represented. Forecast intervals and explanations must be reviewed before operational use.",
       "", "## Data caveat", "Historical performance and statistical anomaly flags are not causal evidence. Validate the output with business context before changing stock levels."]
    (OUT/"summary.md").write_text("\n".join(summary_lines))
    return {"rows":len(df),"comparison":comparison,"forecast":forecast,"best":best["best"],"segments":seg,"quality":quality}


if __name__ == "__main__":
    result=run_pipeline()
    print(f"Analysis complete. Best model: {result['best']}. Outputs: {OUT}")
