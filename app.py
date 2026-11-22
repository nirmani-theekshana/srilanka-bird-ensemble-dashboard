"""
Sri Lanka Bird Biodiversity Dashboard - Ensemble Learning Assignment
Student: Indrajith KTN 224078X

AI use declaration: the code of this dashboard (front end and model loading) was
generated with the help of Claude (Anthropic) and reviewed / run by the student.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION AND PATHS
# ==========================================
st.set_page_config(page_title="Sri Lanka Bird Biodiversity Dashboard",
                   page_icon="🐦", layout="wide")

BASE = Path(__file__).resolve().parent
PROC = BASE / "data" / "processed"
MODELS = BASE / "models"

st.title("🐦 Sri Lanka Bird Biodiversity Dashboard")
st.caption("Ensemble Learning: Bagging, Boosting, Stacking and Hybrid models (RF + ANN, RF + ARIMA)")


# ==========================================
# 2. HELPERS
# ==========================================
@st.cache_data
def load_csv(name):
    path = PROC / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_resource
def load_model(file_name):
    path = MODELS / file_name
    return joblib.load(path) if path.exists() else None


def model_file(name):
    """Same file-name rule that the notebook used when saving models."""
    return name.replace(" ", "_").replace("+", "").replace("(", "").replace(")", "") + ".pkl"


def sl_season(m):
    if m in [12, 1, 2]:
        return "NE_Monsoon"
    if m in [3, 4]:
        return "Inter_Monsoon_1"
    if m in [5, 6, 7, 8, 9]:
        return "SW_Monsoon"
    return "Inter_Monsoon_2"


def show_table(df):
    st.dataframe(df, width="stretch", hide_index=True)


data = load_csv("bird_monthly_data.csv")
if data is None:
    st.error("data/processed/bird_monthly_data.csv not found. Run the notebooks first "
             "and keep the folder structure: app.py, data/processed/, models/.")
    st.stop()
data["Season"] = data["Month"].apply(sl_season)

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", [
    "Overview",
    "Biodiversity Analysis",
    "Environmental Analysis",
    "Model Comparison",
    "Actual vs Predicted",
    "Time Series Forecast",
    "Explainability",
    "Prediction",
    "About",
])
st.sidebar.markdown("---")
st.sidebar.caption("Student: Indrajith KTN 224078X")
st.sidebar.caption("Code generated with Claude (Anthropic)")

# ==========================================
# 4. OVERVIEW
# ==========================================
if page == "Overview":
    st.header("Project Overview")
    st.write("This dashboard explores how environmental factors relate to **recorded bird species "
             "richness** in Sri Lankan districts, and compares ensemble learning models that "
             "predict it.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total records", f"{len(data):,}")
    c2.metric("Districts", data["stateProvince"].nunique())
    c3.metric("Years", f"{data['Year'].min()} - {data['Year'].max()}")
    c4.metric("Mean richness", f"{data['species_richness'].mean():.1f}")

    st.subheader("Sample rows")
    st.dataframe(data.head(10), width="stretch", hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Mean species richness by district")
        by_d = (data.groupby("stateProvince")["species_richness"].mean()
                .sort_values().reset_index())
        fig = px.bar(by_d, x="species_richness", y="stateProvince", orientation="h",
                     labels={"species_richness": "Mean richness", "stateProvince": ""})
        st.plotly_chart(fig, width="stretch")
    with col_b:
        st.subheader("Mean species richness by year")
        by_y = data.groupby("Year")["species_richness"].mean().reset_index()
        fig = px.line(by_y, x="Year", y="species_richness", markers=True,
                      labels={"species_richness": "Mean richness"})
        st.plotly_chart(fig, width="stretch")

    st.info("Note: richness is the number of species *recorded*. It depends strongly on how many "
            "observations were made (survey effort), not only on true biodiversity.")

# ==========================================
# 5. BIODIVERSITY ANALYSIS
# ==========================================
elif page == "Biodiversity Analysis":
    st.header("Biodiversity Analysis")
    district = st.selectbox("Select district", sorted(data["stateProvince"].unique()))
    d = data[data["stateProvince"] == district]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Richness by month")
        m = d.groupby("Month")["species_richness"].mean().reset_index()
        st.plotly_chart(px.bar(m, x="Month", y="species_richness",
                               labels={"species_richness": "Mean richness"}), width="stretch")
    with c2:
        st.subheader("Richness by monsoon season")
        s = d.groupby("Season")["species_richness"].mean().reset_index()
        st.plotly_chart(px.bar(s, x="Season", y="species_richness",
                               labels={"species_richness": "Mean richness"}), width="stretch")

    st.subheader(f"Richness over the years - {district}")
    y = d.groupby("Year")["species_richness"].mean().reset_index()
    st.plotly_chart(px.line(y, x="Year", y="species_richness", markers=True), width="stretch")

# ==========================================
# 6. ENVIRONMENTAL ANALYSIS
# ==========================================
elif page == "Environmental Analysis":
    st.header("Environmental Analysis")
    env_cols = {"average_temperature": "Temperature", "average_rainfall": "Rainfall",
                "average_humidity": "Humidity", "average_ndvi": "NDVI (vegetation)",
                "average_wind": "Wind", "average_elevation": "Elevation",
                "average_population_density": "Population density",
                "average_night_light": "Night light"}
    choice = st.selectbox("Environmental factor", list(env_cols.keys()),
                          format_func=lambda c: env_cols[c])
    fig = px.scatter(data, x=choice, y="species_richness", color="Season", opacity=0.6,
                     labels={choice: env_cols[choice], "species_richness": "Species richness"})
    st.plotly_chart(fig, width="stretch")

    st.subheader("Correlation with species richness")
    corr = (data[list(env_cols) + ["species_richness"]].corr()["species_richness"]
            .drop("species_richness").rename(index=env_cols).sort_values().reset_index())
    corr.columns = ["Factor", "Correlation"]
    st.plotly_chart(px.bar(corr, x="Correlation", y="Factor", orientation="h"), width="stretch")
    if "observation_count" in data.columns:
        st.caption("Correlation between species richness and observation_count (survey effort): "
                   f"{data['species_richness'].corr(data['observation_count']):.2f}")

# ==========================================
# 7. MODEL COMPARISON
# ==========================================
elif page == "Model Comparison":
    st.header("Ensemble Model Comparison")
    results = load_csv("model_results.csv")
    if results is None:
        st.warning("model_results.csv not found. Run step1_models.ipynb first.")
        st.stop()

    metric_names = [c for c in ["MAE", "RMSE", "R2", "MAPE", "Train_Time_s"] if c in results.columns]
    best = results.sort_values("MAE").iloc[0]["Model"]
    st.success(f"Best model by MAE: **{best}**")

    st.subheader("Model selector")
    chosen = st.multiselect("Models to show", results["Model"].tolist(), default=results["Model"].tolist())
    view = results[results["Model"].isin(chosen)]

    st.subheader("Metrics table")
    show_table(view.round(3))

    metric = st.selectbox("Metric for the chart", metric_names)
    ascending = metric != "R2"
    plot_df = view.sort_values(metric, ascending=ascending)
    plot_df["Best"] = np.where(plot_df["Model"] == best, "Best (MAE)", "Other")
    fig = px.bar(plot_df, x=metric, y="Model", orientation="h", color="Best",
                 color_discrete_map={"Best (MAE)": "#2ca02c", "Other": "#1f77b4"})
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")

    st.caption("Lower is better for MAE, RMSE, MAPE and training time. Higher is better for R2. "
               "All models use the same time-based split (train: before 2023, test: 2023-2024).")

# ==========================================
# 8. ACTUAL VS PREDICTED
# ==========================================
elif page == "Actual vs Predicted":
    st.header("Actual vs Predicted (test set 2023-2024)")
    preds = load_csv("test_predictions.csv")
    if preds is None:
        st.warning("test_predictions.csv not found. Run step1_models.ipynb first.")
        st.stop()

    model_cols = [c for c in preds.columns if c not in ["Year", "Month", "stateProvince", "Actual"]]
    model = st.selectbox("Select model", model_cols)

    fig = px.scatter(preds, x="Actual", y=model, color="stateProvince", opacity=0.7,
                     labels={model: "Predicted"})
    lim = [0, max(preds["Actual"].max(), preds[model].max())]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Perfect prediction",
                             line=dict(color="black", dash="dash")))
    st.plotly_chart(fig, width="stretch")

    err = (preds[model] - preds["Actual"]).abs()
    c1, c2 = st.columns(2)
    c1.metric("Mean absolute error", f"{err.mean():.2f} species")
    c2.metric("Largest error", f"{err.max():.0f} species")

    st.subheader("Average error by district")
    e = (pd.DataFrame({"District": preds["stateProvince"], "Abs error": err})
         .groupby("District")["Abs error"].mean().sort_values().reset_index())
    st.plotly_chart(px.bar(e, x="Abs error", y="District", orientation="h"), width="stretch")

# ==========================================
# 9. TIME SERIES FORECAST
# ==========================================
elif page == "Time Series Forecast":
    st.header("Time Series: ARIMA, SARIMA and RF + ARIMA hybrid")
    ts_res = load_csv("timeseries_results.csv")
    ts_fc = load_csv("timeseries_forecasts.csv")
    if ts_res is None or ts_fc is None:
        st.warning("Time-series files not found. Run step2_timeseries.ipynb first.")
        st.stop()

    st.success(f"Best time-series model by MAE: **{ts_res.sort_values('MAE').iloc[0]['Model']}**")
    show_table(ts_res.round(3))

    metric = st.selectbox("Metric", [c for c in ["MAE", "RMSE", "MAPE"] if c in ts_res.columns])
    st.plotly_chart(px.bar(ts_res.sort_values(metric), x=metric, y="Model", orientation="h"),
                    width="stretch")

    st.subheader("Forecast vs actual (2023-2024)")
    models = [c for c in ts_fc.columns if c not in ["date", "Actual"]]
    pick = st.multiselect("Models", models,
                          default=[m for m in ["SARIMA", "Hybrid RF + ARIMA", "Hybrid RF + SARIMA"] if m in models])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts_fc["date"], y=ts_fc["Actual"], name="Actual",
                             line=dict(color="black", width=4)))
    for m in pick:
        fig.add_trace(go.Scatter(x=ts_fc["date"], y=ts_fc[m], name=m, line=dict(dash="dash")))
    fig.update_layout(yaxis_title="Mean species richness")
    st.plotly_chart(fig, width="stretch")

    st.caption("Hybrid idea: ARIMA/SARIMA captures trend and seasonality, then a Random Forest "
               "predicts ARIMA's leftover errors (residuals). Final forecast = ARIMA forecast + RF residual forecast.")

# ==========================================
# 10. EXPLAINABILITY
# ==========================================
elif page == "Explainability":
    st.header("Explainability: what drives species richness?")
    imp = load_csv("feature_importance.csv")
    if imp is None:
        st.warning("feature_importance.csv not found. Run step3_explainability.ipynb first.")
        st.stop()

    method = st.selectbox("Importance method", [c for c in imp.columns if c != "Feature"])
    d = imp.sort_values(method)
    st.plotly_chart(px.bar(d, x=method, y="Feature", orientation="h"), width="stretch")

    shap_path = PROC / "shap_summary.png"
    if shap_path.exists():
        st.subheader("SHAP summary (XGBoost)")
        st.image(str(shap_path), width=700,
                 caption="Each dot is one test row. Red = high feature value, blue = low. "
                         "Right of 0 = pushes predicted richness up.")

# ==========================================
# 11. PREDICTION
# ==========================================
elif page == "Prediction":
    st.header("Predict recorded species richness")
    st.write("Choose a district and month, adjust the environmental values, and select a trained model.")

    feature_cols = load_model("feature_columns.pkl")
    results = load_csv("model_results.csv")
    if feature_cols is None or results is None:
        st.warning("Models not found. Run step1_models.ipynb first (it saves models/ and feature_columns.pkl).")
        st.stop()

    available = []
    for name in results["Model"]:
        if name == "Hybrid RF+ANN (average)":
            if (MODELS / model_file("Random Forest")).exists() and (MODELS / model_file("ANN (MLP)")).exists():
                available.append(name)
        elif (MODELS / model_file(name)).exists():
            available.append(name)
    best = results.sort_values("MAE").iloc[0]["Model"]
    default_idx = available.index(best) if best in available else 0

    c1, c2, c3 = st.columns(3)
    district = c1.selectbox("District", sorted(data["stateProvince"].unique()))
    month = c2.selectbox("Month", list(range(1, 13)))
    year = c3.number_input("Year", min_value=2014, max_value=2035, value=2025, step=1)
    model_name = st.selectbox("Model", available, index=default_idx)

    # default values = average of this district in this month (fall back to district average)
    sub = data[(data["stateProvince"] == district) & (data["Month"] == month)]
    ref = sub if len(sub) else data[data["stateProvince"] == district]
    avg = ref.mean(numeric_only=True)
    key = f"{district}_{month}"

    st.subheader("Environmental values (defaults = historical average)")
    a, b, c = st.columns(3)
    temp = a.number_input("Temperature (C)", value=float(round(avg["average_temperature"], 2)), key="t" + key)
    rain = b.number_input("Rainfall (mm)", value=float(round(avg["average_rainfall"], 2)), key="r" + key)
    hum = c.number_input("Humidity (%)", value=float(round(avg["average_humidity"], 2)), key="h" + key)
    d1, d2 = st.columns(2)
    ndvi = d1.number_input("NDVI (vegetation)", value=float(round(avg["average_ndvi"], 3)), key="n" + key)
    wind = d2.number_input("Wind", value=float(round(avg["average_wind"], 2)), key="w" + key)

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([{
            "stateProvince": district, "Season": sl_season(month), "Year": int(year), "Month": month,
            "average_temperature": temp, "average_rainfall": rain, "average_humidity": hum,
            "average_ndvi": ndvi, "average_wind": wind,
            "average_elevation": avg["average_elevation"],
            "average_population_density": avg["average_population_density"],
            "average_night_light": avg["average_night_light"],
        }])
        X = (pd.get_dummies(row, columns=["stateProvince", "Season"])
             .reindex(columns=feature_cols, fill_value=0))

        if model_name == "Hybrid RF+ANN (average)":
            p1 = load_model(model_file("Random Forest")).predict(X)[0]
            p2 = load_model(model_file("ANN (MLP)")).predict(X)[0]
            pred = (p1 + p2) / 2
        else:
            pred = load_model(model_file(model_name)).predict(X)[0]

        pred = max(float(pred), 0.0)
        st.success(f"Predicted recorded species richness: **{pred:.0f} species**")

        hist = data[data["stateProvince"] == district]["species_richness"]
        st.caption(f"For comparison: {district} historical average is {hist.mean():.0f} species "
                   f"(min {hist.min():.0f}, max {hist.max():.0f}).")
        st.warning("The models were trained on environmental data only (not survey effort). "
                   "Typical error is about 25 species (see Model Comparison), so treat this as a rough estimate.")

# ==========================================
# 12. ABOUT
# ==========================================
elif page == "About":
    st.header("About this project")
    st.markdown("""
**Problem:** Predict recorded bird species richness for Sri Lankan districts from environmental
factors, and find which ensemble learning approach works best.

**Data:** Sri Lanka bird records aggregated by district and month (2014-2024), joined with
temperature, rainfall, humidity, NDVI, wind, elevation, population density and night light.

**Ensemble methods**
- **Bagging:** Random Forest, Bagging, ExtraTrees
- **Boosting:** Gradient Boosting, XGBoost, LightGBM, CatBoost
- **Stacking:** RF + XGBoost + LightGBM + CatBoost with a Ridge meta-model
- **Hybrid:** RF + ANN (average and stacking); RF + ARIMA and RF + SARIMA for time series

**AI use declaration:** The code in this dashboard and the notebooks was generated with the help of
Claude (Anthropic), then reviewed, run and interpreted by the student.
""")
