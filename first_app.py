import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Retail Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Retail Sales Forecast Dashboard")
st.markdown("### Sales Analysis using Superstore Dataset")

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month_name()

    return df

df = load_data()

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select Page",

    [

        "Sales Overview",

        "Forecast Explorer",

        "Anomaly Report",

        "Demand Segments"

    ]

)
if page == "Sales Overview":

    st.header("📈 Sales Overview Dashboard")
    total_sales = df["Sales"].sum()

    total_orders = df["Order ID"].nunique()

    total_profit = df["Profit"].sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Sales", f"${total_sales:,.2f}")

    col2.metric("Total Orders", total_orders)

    col3.metric("Total Profit", f"${total_profit:,.2f}")
    yearly_sales = (

        df.groupby("Year")["Sales"]

        .sum()

        .reset_index()

    )

    st.subheader("Total Sales by Year")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(

        yearly_sales["Year"],

        yearly_sales["Sales"]

    )

    ax.set_xlabel("Year")

    ax.set_ylabel("Sales")

    st.pyplot(fig)
    monthly_sales = (

        df.groupby("Order Date")["Sales"]

        .sum()

        .resample("ME")

        .sum()

    )

    st.subheader("Monthly Sales Trend")

    fig, ax = plt.subplots(figsize=(12,5))

    # matplotlib can handle pandas DatetimeIndex directly
    ax.plot(
        monthly_sales.index,
        monthly_sales.to_numpy(dtype=float),
        linewidth=2,
    )

    ax.set_xlabel("Date")

    ax.set_ylabel("Sales")

    st.pyplot(fig)
    st.sidebar.header("Filters")

    selected_region = st.sidebar.selectbox(

        "Select Region",

        ["All"] + list(df["Region"].unique())

    )
    selected_category = st.sidebar.selectbox(

        "Select Category",

        ["All"] + list(df["Category"].unique())

    )
    filtered = df.copy()

    if selected_region != "All":

        filtered = filtered[

            filtered["Region"] == selected_region

        ]

    if selected_category != "All":

        filtered = filtered[

            filtered["Category"] == selected_category

        ]
        st.subheader("Sales by Region")

    region_sales = (

        filtered.groupby("Region")["Sales"]

        .sum()

    )

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(

        region_sales.index,

        region_sales.to_numpy()

    )

    st.pyplot(fig)
    st.subheader("Sales by Category")

    category_sales = (

        filtered.groupby("Category")["Sales"]

        .sum()

    )

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(

        category_sales.index,

        category_sales.to_numpy()

    )

    st.pyplot(fig)


elif page == "Forecast Explorer":

    st.header("📈 Forecast Explorer")

    st.write("View the 3-month sales forecast using the best-performing model.")


    forecast_df = pd.read_csv("Category_Region_Forecast.csv")

    comparison_df = pd.read_csv("Model_Comparison.csv")
    st.subheader("Best Model Performance")

    best_model = comparison_df.sort_values("MAE").iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", best_model["Model"])

    col2.metric("MAE", round(best_model["MAE"],2))

    col3.metric("RMSE", round(best_model["RMSE"],2))
    st.subheader("Forecast Settings")

    segment = st.selectbox(

        "Select Category / Region",

        forecast_df["Segment"]

    )
    horizon = st.slider(

        "Forecast Horizon (Months)",

        min_value=1,

        max_value=3,

        value=3

    )
    selected = forecast_df[
        forecast_df["Segment"]==segment
    ]

    months = []

    values = []

    if horizon >= 1:
        months.append("Month 1")
        values.append(selected["Month 1"].values[0])

    if horizon >= 2:
        months.append("Month 2")
        values.append(selected["Month 2"].values[0])

    if horizon >= 3:
        months.append("Month 3")
        values.append(selected["Month 3"].values[0])
        st.subheader("Forecast Values")

    table = pd.DataFrame({

        "Forecast Month":months,

        "Predicted Sales":values

    })

    st.dataframe(table)
    st.subheader("Forecast Trend")

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(

        months,

        values,

        marker="o",

        linewidth=3

    )

    ax.set_xlabel("Forecast Horizon")

    ax.set_ylabel("Predicted Sales")

    ax.set_title(f"Forecast for {segment}")

    ax.grid(True)

    st.pyplot(fig)
    st.subheader("Forecast Comparison")

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(

        months,

        values

    )

    ax.set_ylabel("Sales")

    ax.set_title(segment)

    st.pyplot(fig)
    st.subheader("Model Evaluation")

    st.dataframe(comparison_df)
    csv = table.to_csv(index=False).encode("utf-8")

    st.download_button(

        label="Download Forecast",

        data=csv,

        file_name="forecast.csv",

        mime="text/csv"

    )
    st.subheader("Forecast Summary")

    st.success(

        f"""
        The selected segment is **{segment}**.

        Forecast horizon: **{horizon} month(s)**.

        Forecast generated using the **{best_model['Model']}** model.

        Model Accuracy:
        • MAE = {best_model['MAE']:.2f}
        • RMSE = {best_model['RMSE']:.2f}
        """
    )
elif page == "Anomaly Report":

    st.header("🚨 Sales Anomaly Report")

    st.write("Weekly sales anomalies detected using Isolation Forest.")

    # Load anomaly file
    anomaly_df = pd.read_csv("IsolationForest_Anomalies.csv")

    anomaly_df["Order Date"] = pd.to_datetime(anomaly_df["Order Date"])

    # Load original dataset
    sales = pd.read_csv("train.csv")

    sales["Order Date"] = pd.to_datetime(sales["Order Date"])

    weekly = (
        sales.groupby("Order Date")["Sales"]
        .sum()
        .resample("W")
        .sum()
        .reset_index()
    )

    st.subheader("Weekly Sales with Detected Anomalies")

    fig, ax = plt.subplots(figsize=(14,6))

    ax.plot(
        weekly["Order Date"],
        weekly["Sales"],
        linewidth=2,
        label="Weekly Sales"
    )

    ax.scatter(
        anomaly_df["Order Date"],
        anomaly_df["Sales"],
        color="red",
        s=80,
        label="Anomaly"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()

    st.pyplot(fig)

    st.subheader("Detected Anomalies")

    st.dataframe(
        anomaly_df[
            ["Order Date","Sales"]
        ]
    )
    st.subheader("Summary")

    col1,col2,col3 = st.columns(3)

    col1.metric(
        "Total Weeks",
        len(weekly)
    )

    col2.metric(
        "Detected Anomalies",
        len(anomaly_df)
    )

    col3.metric(
        "Anomaly %",
        f"{len(anomaly_df)/len(weekly)*100:.2f}%"
    )
    csv = anomaly_df.to_csv(index=False).encode()

    st.download_button(

        "Download Anomaly Report",

        csv,

        "Anomaly_Report.csv",

        "text/csv"

    )
# ==========================================
# PAGE 4 : DEMAND SEGMENTS
# ==========================================

elif page == "Demand Segments":

    st.header("📦 Product Demand Segmentation")
    cluster = pd.read_csv("Demand_Segments.csv")

    st.subheader("Clustered Products")

    st.dataframe(cluster)
    st.subheader("Products per Demand Segment")

    counts = (

        cluster["Demand Segment"]

        .value_counts()

    )

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(
        counts.index,
        counts.to_numpy(dtype=float)
    )

    plt.xticks(rotation=20)

    st.pyplot(fig)
    st.subheader("Average Sales by Demand Segment")

    avg = (

        cluster.groupby(

            "Demand Segment"

        )["Total Sales"]

        .mean()

    )

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(
        avg.index,
        avg.to_numpy(dtype=float)
    )

    plt.xticks(rotation=20)

    st.pyplot(fig)
    st.subheader("Cluster Visualization")

    from sklearn.preprocessing import StandardScaler

    from sklearn.decomposition import PCA

    features = cluster[

        [

            "Total Sales",

            "Growth Rate",

            "Volatility",

            "Average Order Value"

        ]

    ]

    scaler = StandardScaler()

    scaled = scaler.fit_transform(features)

    pca = PCA(n_components=2)

    components = pca.fit_transform(scaled)

    fig, ax = plt.subplots(figsize=(10,6))

    scatter = ax.scatter(

        components[:,0],

        components[:,1],

        c=cluster["Cluster"],

        s=120

    )

    for i in range(len(cluster)):

        ax.text(

            components[i,0],

            components[i,1],

            cluster.iloc[i]["Sub-Category"],

            fontsize=8

        )

    ax.set_xlabel("Principal Component 1")

    ax.set_ylabel("Principal Component 2")

    st.pyplot(fig)
    st.subheader("Filter by Demand Segment")

    option = st.selectbox(

        "Choose Segment",

        cluster["Demand Segment"].unique()

    )

    filtered = cluster[

        cluster["Demand Segment"]==option

    ]

    st.dataframe(filtered)
    csv = cluster.to_csv(index=False).encode()

    st.download_button(

        "Download Cluster Report",

        csv,

        "Demand_Segments.csv",

        "text/csv"

    )
