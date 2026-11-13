"""Streamlit dashboard for the Superstore forecasting project."""
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"outputs"
CHARTS=OUT/"charts"
st.set_page_config(page_title="Sales Forecast & Demand Intelligence",page_icon="📊",layout="wide")
st.title("📊 Sales Forecast & Demand Intelligence")
st.caption("Historical analysis and planning estimates from the supplied Superstore transactions.")


def ensure_outputs():
    if not (OUT/"model_comparison.csv").exists():
        if not (ROOT/"train.csv").exists():
            st.error("Add the Kaggle Superstore `train.csv` to this repository, then rerun the analysis.")
            st.markdown("Download: [Kaggle — Sales Forecasting](https://www.kaggle.com/datasets/rohitsahoo/sales-forecasting)")
            st.stop()
        with st.spinner("Running the analysis pipeline for the first time…"):
            result=subprocess.run([sys.executable,str(ROOT/"analysis.py")],cwd=ROOT,text=True,capture_output=True)
        if result.returncode:
            st.error("Analysis pipeline failed. Check dependencies and the dataset columns.")
            st.code(result.stderr[-4000:])
            st.stop()

ensure_outputs()

def read_csv(name, **kwargs):
    path=OUT/name
    return pd.read_csv(path,**kwargs) if path.exists() else pd.DataFrame()

pages=["Sales Overview","Forecast Explorer","Anomaly Report","Product Demand Segments"]
page=st.sidebar.radio("Navigate",pages)
if page=="Sales Overview":
    st.header("Sales overview")
    yearly=read_csv("yearly_sales.csv")
    monthly=read_csv("monthly_sales.csv")
    categories=read_csv("category_sales.csv")
    regions=read_csv("region_year_sales.csv")
    raw=pd.read_csv(ROOT/"train.csv")
    category_filter=st.sidebar.multiselect("Category",sorted(raw["Category"].dropna().unique()),default=sorted(raw["Category"].dropna().unique()))
    region_filter=st.sidebar.multiselect("Region",sorted(raw["Region"].dropna().unique()),default=sorted(raw["Region"].dropna().unique()))
    filtered=raw[raw["Category"].isin(category_filter)&raw["Region"].isin(region_filter)]
    cols=st.columns(3)
    cols[0].metric("Total sales",f"${filtered['Sales'].sum():,.0f}")
    cols[1].metric("Orders",filtered["Order ID"].nunique() if "Order ID" in filtered else len(filtered))
    cols[2].metric("Transactions",f"{len(filtered):,}")
    left,right=st.columns(2)
    with left:
        st.subheader("Sales by year")
        if not yearly.empty: st.bar_chart(yearly.set_index("Year")["Sales"])
    with right:
        st.subheader("Sales by category (filtered)")
        st.bar_chart(filtered.groupby("Category")["Sales"].sum())
    st.subheader("Monthly sales trend")
    if not monthly.empty:
        monthly["Date"]=pd.to_datetime(monthly["Date"]);st.line_chart(monthly.set_index("Date")["Sales"])
    st.subheader("Regional sales (filtered)")
    st.bar_chart(filtered.groupby("Region")["Sales"].sum())
elif page=="Forecast Explorer":
    st.header("Forecast explorer")
    fc=read_csv("segment_forecasts.csv"); overall=read_csv("overall_forecast.csv"); comp=read_csv("model_comparison.csv")
    if fc.empty:
        st.info("Segment forecasts are unavailable for this dataset length.")
    else:
        options=sorted(fc["Segment"].unique())
        selected=st.selectbox("Category or region",options)
        horizon=st.slider("Forecast horizon (months)",1,3,3)
        chosen=fc[fc["Segment"]==selected].sort_values("Date").head(horizon)
        st.line_chart(chosen.set_index("Date")["Forecast"])
        st.dataframe(chosen[["Date","Type","Segment","Forecast"]],use_container_width=True)
    if not comp.empty:
        best=comp.sort_values("MAE").iloc[0]
        a,b,c=st.columns(3);a.metric("Selected model",best["Model"]);b.metric("Holdout MAE",f"{best['MAE']:,.2f}");c.metric("Holdout RMSE",f"{best['RMSE']:,.2f}")
        st.caption("Metrics are from a chronological three-month holdout, not in-sample fit.")
        st.dataframe(comp,use_container_width=True)
    if not overall.empty: st.download_button("Download overall forecast CSV",overall.to_csv(index=False),"overall_forecast.csv","text/csv")
elif page=="Anomaly Report":
    st.header("Weekly anomaly report")
    anomalies=read_csv("weekly_anomalies.csv");
    if not anomalies.empty:
        anomalies["Date"]=pd.to_datetime(anomalies["Date"])
        st.line_chart(anomalies.set_index("Date")[["Sales"]])
        marked=anomalies[anomalies["Anomaly"]]
        fig,ax=plt.subplots(figsize=(12,4));ax.plot(anomalies["Date"],anomalies["Sales"],label="Weekly sales");ax.scatter(marked["Date"],marked["Sales"],color="red",label="Flagged week");ax.legend();st.pyplot(fig);plt.close(fig)
        st.caption("Flags identify unusual values, not their cause. Compare these dates with promotion, holiday, and inventory records.")
        st.dataframe(marked[["Date","Sales","Isolation Forest","Rolling Z"]].sort_values("Date"),use_container_width=True)
        st.download_button("Download anomaly table",marked.to_csv(index=False),"weekly_anomalies.csv","text/csv")
else:
    st.header("Product demand segments")
    segments=read_csv("demand_segments.csv")
    if not segments.empty:
        st.image(str(CHARTS/"demand_clusters.png"),use_container_width=True)
        st.dataframe(segments[["Sub-Category","Total Sales","Growth Rate","Volatility","Average Order Value","Demand Segment"]].sort_values("Total Sales",ascending=False),use_container_width=True)
        st.info("Suggested policy: protect availability for high-volume/stable items; replenish volatile items in smaller, more frequent lots; review declining items before increasing buys.")
        st.download_button("Download segment table",segments.to_csv(index=False),"demand_segments.csv","text/csv")
