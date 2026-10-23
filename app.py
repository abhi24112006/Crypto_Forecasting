
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent

FILES = {
    "LSTM": {
        "path": ROOT / "LSTM_Light_GBM/results/lstm_predictions.csv",
        "date": "Date",
        "actual": "Actual_Close",
        "prediction": "LSTM_Prediction",
        "color": "#00C2A8",
    },
    "LightGBM": {
        "path": ROOT / "LSTM_Light_GBM/results/lightgbm_predictions.csv",
        "date": "Date",
        "actual": "Actual_Close",
        "prediction": "LightGBM_Prediction",
        "color": "#FFB000",
    },
    "Prophet": {
        "path": ROOT / "Prophet_HistGradientBoosting/outputs/prophet_final_results.csv",
        "date": "ds",
        "actual": "y",
        "prediction": "predicted",
        "color": "#A78BFA",
    },
    "HistGradientBoosting": {
        "path": ROOT / "Prophet_HistGradientBoosting/outputs/histgradientboosting_final_results.csv",
        "date": "Date",
        "actual": "Target",
        "prediction": "Predicted",
        "color": "#F472B6",
    },
    "TFT": {
        "path": ROOT / "TFT_NBEATS/results/tft/final_tft_predictions.csv",
        "date": "Date",
        "actual": "Actual_Close",
        "prediction": "Predicted_Close",
        "color": "#60A5FA",
    },
    "N-BEATS": {
        "path": ROOT / "TFT_NBEATS/results/nbeats/final_nbeats_predictions.csv",
        "date": "Date",
        "actual": "Actual",
        "prediction": "Predicted",
        "color": "#FB7185",
    },
}

MODEL_NAMES = list(FILES.keys())


# =========================================================
# PAGE AND STYLE
# =========================================================

st.set_page_config(
    page_title="Bitcoin Historical Prediction Explorer",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1700px;
        padding: 2rem 3rem 3rem;
    }
    h1 {
        font-size: 2.6rem !important;
        font-weight: 800 !important;
    }
    h2 {
        font-size: 1.8rem !important;
    }
    [data-testid="stMetric"] {
        background: rgba(128,128,128,0.08);
        border: 1px solid rgba(128,128,128,0.20);
        border-radius: 12px;
        padding: 18px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD EXISTING PREDICTIONS
# =========================================================

@st.cache_data
def load_all_predictions():
    predictions = {}
    errors = {}

    for model_name, config in FILES.items():
        path = config["path"]

        if not path.exists():
            errors[model_name] = f"File not found: {path}"
            continue

        try:
            df = pd.read_csv(path)

            required = [
                config["date"],
                config["actual"],
                config["prediction"],
            ]

            missing = [c for c in required if c not in df.columns]

            if missing:
                errors[model_name] = (
                    f"Missing columns in {path.name}: {missing}"
                )
                continue

            clean = pd.DataFrame({
                "Date": pd.to_datetime(
                    df[config["date"]], errors="coerce"
                ),
                "Actual": pd.to_numeric(
                    df[config["actual"]], errors="coerce"
                ),
                "Prediction": pd.to_numeric(
                    df[config["prediction"]], errors="coerce"
                ),
            })

            clean = clean.dropna(
                subset=["Date", "Actual", "Prediction"]
            )

            clean["Date"] = clean["Date"].dt.normalize()
            clean = clean.sort_values("Date")
            clean = clean.drop_duplicates(
                subset=["Date"], keep="last"
            ).reset_index(drop=True)

            clean["Absolute_Error"] = (
                clean["Actual"] - clean["Prediction"]
            ).abs()

            clean["Error"] = (
                clean["Prediction"] - clean["Actual"]
            )

            clean["Percentage_Error"] = np.where(
                clean["Actual"] != 0,
                clean["Absolute_Error"] / clean["Actual"] * 100,
                np.nan,
            )

            predictions[model_name] = clean

        except Exception as exc:
            errors[model_name] = str(exc)

    return predictions, errors


predictions, load_errors = load_all_predictions()

if not predictions:
    st.error(
        "No prediction files could be loaded. Check that the project "
        "folders and CSV files exist."
    )
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Dashboard Controls")

available_models = list(predictions.keys())

selected_models = st.sidebar.multiselect(
    "Models to compare",
    options=available_models,
    default=available_models,
)

if not selected_models:
    st.warning("Select at least one model in the sidebar.")
    st.stop()

st.sidebar.divider()

st.sidebar.caption(
    "This dashboard reads saved historical predictions. "
    "It does not train models or fetch live market prices."
)

if st.sidebar.button("Reload prediction CSVs"):
    load_all_predictions.clear()
    st.rerun()


# =========================================================
# HEADER
# =========================================================

st.title("₿ Bitcoin Historical Prediction Explorer")

st.markdown(
    """
    Explore saved predictions from all six team algorithms.
    Select a historical date, inspect individual errors, and compare
    model performance without retraining anything.
    """
)

st.info(
    "These are previously saved model evaluation results. "
    "They are not new predictions for the current Bitcoin market."
)


# =========================================================
# DATA COVERAGE
# =========================================================

st.header("Prediction Dataset Coverage")

coverage_rows = []

for name, df in predictions.items():
    coverage_rows.append({
        "Model": name,
        "Prediction rows": len(df),
        "First available date": df["Date"].min().date(),
        "Last available date": df["Date"].max().date(),
        "Available dates": df["Date"].nunique(),
    })

coverage = pd.DataFrame(coverage_rows)

st.dataframe(
    coverage,
    use_container_width=True,
    hide_index=True,
)

if load_errors:
    with st.expander("Files that could not be loaded"):
        for name, message in load_errors.items():
            st.warning(f"{name}: {message}")


# =========================================================
# DATE SELECTION
# =========================================================

st.divider()
st.header("Explore Predictions by Date")

all_dates = sorted(
    set(
        date
        for name in selected_models
        for date in predictions[name]["Date"]
    )
)

if not all_dates:
    st.warning("No dates are available for the selected models.")
    st.stop()

min_date = all_dates[0].date()
max_date = all_dates[-1].date()

default_date = max_date

left, right = st.columns([1, 2])

with left:
    selected_date = st.date_input(
        "Select a historical date",
        value=default_date,
        min_value=min_date,
        max_value=max_date,
        help=(
            "Choose a date in the available prediction period. "
            "Individual models may have different date coverage."
        ),
    )

with right:
    date_options = [d.date() for d in all_dates]

    dropdown_date = st.selectbox(
        "Or choose an available prediction date",
        options=date_options,
        index=date_options.index(default_date),
        format_func=lambda d: d.strftime("%d %B %Y"),
    )

use_dropdown = st.checkbox(
    "Use dropdown date instead of calendar",
    value=False,
)

if use_dropdown:
    selected_date = dropdown_date

selected_timestamp = pd.Timestamp(selected_date)

st.caption(
    f"Selected date: {selected_timestamp:%d %B %Y}"
)


# =========================================================
# SELECTED-DATE PREDICTIONS
# =========================================================

st.subheader("Predictions for the Selected Date")

selected_rows = []

for model_name in selected_models:
    df = predictions[model_name]

    row = df.loc[df["Date"] == selected_timestamp]

    if row.empty:
        selected_rows.append({
            "Model": model_name,
            "Actual Close (USD)": np.nan,
            "Predicted Close (USD)": np.nan,
            "Absolute Error (USD)": np.nan,
            "Absolute Percentage Error (%)": np.nan,
            "Status": "No saved prediction for this date",
        })
        continue

    item = row.iloc[0]

    selected_rows.append({
        "Model": model_name,
        "Actual Close (USD)": item["Actual"],
        "Predicted Close (USD)": item["Prediction"],
        "Absolute Error (USD)": item["Absolute_Error"],
        "Absolute Percentage Error (%)": item["Percentage_Error"],
        "Status": "Available",
    })

selected_table = pd.DataFrame(selected_rows)

available_rows = selected_table.dropna(
    subset=["Predicted Close (USD)"]
)

if available_rows.empty:
    st.warning(
        "None of the selected models has a saved prediction for this date. "
        "Choose a date within the available prediction periods."
    )
else:
    # The saved files may contain slightly different actual prices.
    # Display the actual values from each file rather than silently
    # replacing them with a single assumed value.
    actual_values = available_rows["Actual Close (USD)"].dropna()

    if len(actual_values) > 1:
        spread = actual_values.max() - actual_values.min()

        if spread > max(1.0, actual_values.mean() * 0.0001):
            st.warning(
                "The actual closing prices differ between some model files "
                "for this date. The table preserves each file's recorded "
                "value; check dataset alignment before comparing errors."
            )

    metric_cols = st.columns(3)

    # Highlight LSTM and LightGBM where available.
    for i, preferred in enumerate(["LSTM", "LightGBM"]):
        matching = available_rows[
            available_rows["Model"] == preferred
        ]

        if not matching.empty:
            item = matching.iloc[0]
            metric_cols[i].metric(
                f"{preferred} prediction",
                f"${item['Predicted Close (USD)']:,.2f}",
                f"Error: ${item['Absolute Error (USD)']:,.2f}",
            )
        elif i == 0:
            metric_cols[i].metric("LSTM prediction", "Unavailable")
        else:
            metric_cols[i].metric("LightGBM prediction", "Unavailable")

    if not actual_values.empty:
        metric_cols[2].metric(
            "Recorded actual close",
            f"${actual_values.iloc[0]:,.2f}",
        )

    st.dataframe(
        selected_table.style.format({
            "Actual Close (USD)": "${:,.2f}",
            "Predicted Close (USD)": "${:,.2f}",
            "Absolute Error (USD)": "${:,.2f}",
            "Absolute Percentage Error (%)": "{:.3f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download selected-date results",
        data=selected_table.to_csv(index=False).encode("utf-8"),
        file_name=f"bitcoin_predictions_{selected_date}.csv",
        mime="text/csv",
    )


# =========================================================
# ACTUAL VS PREDICTED FOR A DATE RANGE
# =========================================================

st.divider()
st.header("Actual vs Predicted Prices Over Time")

range_dates = sorted(
    set(
        date
        for name in selected_models
        for date in predictions[name]["Date"]
    )
)

range_min = range_dates[0].date()
range_max = range_dates[-1].date()

default_start = max(
    range_min,
    range_max - pd.Timedelta(days=180),
)

range_start, range_end = st.date_input(
    "Choose chart date range",
    value=(default_start, range_max),
    min_value=range_min,
    max_value=range_max,
)

if range_start > range_end:
    st.error("The start date must be before the end date.")
    st.stop()

show_actual = st.checkbox("Show actual closing price", value=True)

fig = go.Figure()

# Plot actual values from the first selected model as a reference,
# and warn if actual series differ between model files.
if show_actual:
    reference_name = selected_models[0]
    reference_df = predictions[reference_name]

    reference_df = reference_df[
        reference_df["Date"].between(
            pd.Timestamp(range_start),
            pd.Timestamp(range_end),
        )
    ]

    fig.add_trace(
        go.Scatter(
            x=reference_df["Date"],
            y=reference_df["Actual"],
            name=f"Actual close ({reference_name} file)",
            mode="lines",
            line=dict(color="#E5E7EB", width=3),
        )
    )

for model_name in selected_models:
    df = predictions[model_name]

    subset = df[
        df["Date"].between(
            pd.Timestamp(range_start),
            pd.Timestamp(range_end),
        )
    ]

    fig.add_trace(
        go.Scatter(
            x=subset["Date"],
            y=subset["Prediction"],
            name=f"{model_name} prediction",
            mode="lines",
            line=dict(
                color=FILES[model_name]["color"],
                width=2,
            ),
        )
    )

fig.update_layout(
    height=600,
    xaxis_title="Date",
    yaxis_title="Bitcoin close (USD)",
    hovermode="x unified",
    legend_title="Series",
    margin=dict(l=20, r=20, t=40, b=20),
)

st.plotly_chart(fig, use_container_width=True)


# =========================================================
# ERROR COMPARISON
# =========================================================

st.divider()
st.header("Historical Model Performance")

st.caption(
    "Metrics are calculated from each model's saved evaluation CSV. "
    "Different models may cover different dates or use different "
    "evaluation setups, so compare metrics cautiously."
)

metric_rows = []

for model_name in selected_models:
    df = predictions[model_name]

    subset = df[
        df["Date"].between(
            pd.Timestamp(range_start),
            pd.Timestamp(range_end),
        )
    ]

    if subset.empty:
        continue

    actual = subset["Actual"].to_numpy()
    predicted = subset["Prediction"].to_numpy()
    valid = np.isfinite(actual) & np.isfinite(predicted)

    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) == 0:
        continue

    errors = predicted - actual

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = (
        np.mean(np.abs(errors / actual)) * 100
        if np.all(actual != 0)
        else np.nan
    )

    metric_rows.append({
        "Model": model_name,
        "Rows evaluated": len(actual),
        "MAE (USD)": mae,
        "RMSE (USD)": rmse,
        "MAPE (%)": mape,
    })

metrics_df = pd.DataFrame(metric_rows)

if not metrics_df.empty:
    st.dataframe(
        metrics_df.sort_values("MAE (USD)").style.format({
            "MAE (USD)": "${:,.2f}",
            "RMSE (USD)": "${:,.2f}",
            "MAPE (%)": "{:.3f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    error_fig = px.bar(
        metrics_df.sort_values("MAE (USD)"),
        x="Model",
        y="MAE (USD)",
        color="Model",
        title="MAE over the selected date range",
    )

    error_fig.update_layout(
        height=400,
        showlegend=False,
    )

    st.plotly_chart(error_fig, use_container_width=True)

    st.download_button(
        "Download model metrics",
        data=metrics_df.to_csv(index=False).encode("utf-8"),
        file_name="bitcoin_model_metrics.csv",
        mime="text/csv",
    )
else:
    st.info("No valid predictions are available for this date range.")


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Bitcoin Historical Prediction Explorer | "
    "Uses existing team model outputs without retraining."
)

st.caption(
    "Historical predictions are not live forecasts or financial advice. "
    "For a fair model comparison, ensure the models were evaluated on "
    "the same dates with the same target definition."
)
