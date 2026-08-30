import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting import TemporalFusionTransformer


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/engineered_data.csv"

MODEL_PATH = (
    "models/tft/"
    "final-tft-epoch=14-train_loss=0.0386.ckpt"
)

RESULTS_DIR = "results/tft"

LOOKBACK = 30

FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Return",
    "SMA_7",
    "SMA_14",
    "Volatility_7",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "Volume_Change"
]

TARGET = "Target"


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(
    df["Date"]
)

df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)


# ============================================================
# CLEAN DATA
# ============================================================

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df["Volume_Change"] = (
    df["Volume_Change"]
    .fillna(0)
)


# ============================================================
# SPLIT DATA
# ============================================================

n = len(df)

train_end = int(n * 0.70)

val_end = int(n * 0.85)


train_df = df.iloc[
    :train_end
].copy()

val_df = df.iloc[
    train_end:val_end
].copy()

test_df = df.iloc[
    val_end:
].copy()


# ============================================================
# SAVE ORIGINAL TEST DATES
# ============================================================

test_dates = test_df["Date"].copy()


# ============================================================
# SCALE FEATURES
# ============================================================

feature_scaler = StandardScaler()

target_scaler = StandardScaler()


train_df[FEATURES] = (
    feature_scaler.fit_transform(
        train_df[FEATURES]
    )
)

val_df[FEATURES] = (
    feature_scaler.transform(
        val_df[FEATURES]
    )
)

test_df[FEATURES] = (
    feature_scaler.transform(
        test_df[FEATURES]
    )
)


# ============================================================
# SCALE TARGET
# ============================================================

train_df[[TARGET]] = (
    target_scaler.fit_transform(
        train_df[[TARGET]]
    )
)

val_df[[TARGET]] = (
    target_scaler.transform(
        val_df[[TARGET]]
    )
)

test_df[[TARGET]] = (
    target_scaler.transform(
        test_df[[TARGET]]
    )
)


# ============================================================
# REMOVE LAG FEATURES
# ============================================================

for data in [
    train_df,
    val_df,
    test_df
]:

    data.drop(
        columns=[
            "Lag_1",
            "Lag_2",
            "Lag_3"
        ],
        inplace=True
    )


# ============================================================
# TIME INDEX
# ============================================================

train_df["time_idx"] = np.arange(
    len(train_df)
)

val_df["time_idx"] = np.arange(
    len(train_df),
    len(train_df) + len(val_df)
)

test_df["time_idx"] = np.arange(
    len(train_df) + len(val_df),
    len(df)
)


# ============================================================
# GROUP
# ============================================================

train_df["group"] = "BTC"

val_df["group"] = "BTC"

test_df["group"] = "BTC"


# ============================================================
# CREATE TRAINING DATASET
# ============================================================

training_dataset = TimeSeriesDataSet(

    train_df,

    time_idx="time_idx",

    target=TARGET,

    group_ids=["group"],

    min_encoder_length=LOOKBACK,

    max_encoder_length=LOOKBACK,

    min_prediction_length=1,

    max_prediction_length=1,

    static_categoricals=[
        "group"
    ],

    time_varying_known_reals=[
        "time_idx"
    ],

    time_varying_unknown_reals=FEATURES,

    allow_missing_timesteps=False
)


# ============================================================
# TEST DATASET
# ============================================================

test_dataset = TimeSeriesDataSet.from_dataset(

    training_dataset,

    test_df,

    predict=False,

    stop_randomization=True
)


# ============================================================
# TEST DATALOADER
# ============================================================

test_dataloader = test_dataset.to_dataloader(

    train=False,

    batch_size=64,

    num_workers=0
)


print(
    "\nTest samples:",
    len(test_dataset)
)


# ============================================================
# LOAD BEST TFT MODEL
# ============================================================

print("\nLoading best TFT checkpoint...")

model = (
    TemporalFusionTransformer
    .load_from_checkpoint(
        MODEL_PATH
    )
)


model.eval()


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    test_dataloader
)


predictions = (
    predictions
    .detach()
    .cpu()
    .numpy()
    .reshape(-1)
)


# ============================================================
# GET ACTUAL TARGETS
# ============================================================

actual_scaled = []

for batch in test_dataloader:

    x, y = batch

    target = y[0]

    actual_scaled.extend(
        target
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )


actual_scaled = np.array(
    actual_scaled
)


# ============================================================
# INVERSE TRANSFORM
# ============================================================

predicted_prices = (
    target_scaler
    .inverse_transform(
        predictions.reshape(-1, 1)
    )
    .reshape(-1)
)


actual_prices = (
    target_scaler
    .inverse_transform(
        actual_scaled.reshape(-1, 1)
    )
    .reshape(-1)
)


# ============================================================
# ALIGN DATES
# ============================================================

# TimeSeriesDataSet removes the first LOOKBACK
# observations from the prediction range.

prediction_dates = test_dates.iloc[
    LOOKBACK:
    LOOKBACK + len(predicted_prices)
].reset_index(drop=True)


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame({

    "Date": prediction_dates,

    "Actual_Close": actual_prices,

    "Predicted_Close": predicted_prices
})


results["Absolute_Error"] = (
    np.abs(
        results["Actual_Close"]
        -
        results["Predicted_Close"]
    )
)


results["Percentage_Error"] = (
    results["Absolute_Error"]
    /
    np.abs(results["Actual_Close"])
) * 100


# ============================================================
# METRICS
# ============================================================

mae = (
    results["Absolute_Error"]
    .mean()
)


rmse = np.sqrt(
    np.mean(
        (
            results["Actual_Close"]
            -
            results["Predicted_Close"]
        ) ** 2
    )
)


mape = (
    results["Percentage_Error"]
    .mean()
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n================================")
print("FINAL TFT TEST RESULTS")
print("================================")

print(
    f"MAE  : {mae:.2f}"
)

print(
    f"RMSE : {rmse:.2f}"
)

print(
    f"MAPE : {mape:.2f}%"
)

print(
    f"Predictions: {len(results)}"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_path = (
    RESULTS_DIR
    +
    "/tft_predictions.csv"
)

results.to_csv(
    prediction_path,
    index=False
)


print(
    "\nPredictions saved to:"
)

print(
    prediction_path
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame({

    "Metric": [
        "MAE",
        "RMSE",
        "MAPE"
    ],

    "Value": [
        mae,
        rmse,
        mape
    ]

})


metrics.to_csv(
    RESULTS_DIR
    +
    "/tft_metrics.csv",
    index=False
)


# ============================================================
# ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    results["Date"],
    results["Actual_Close"],
    label="Actual"
)

plt.plot(
    results["Date"],
    results["Predicted_Close"],
    label="TFT Prediction"
)

plt.title(
    "TFT: Actual vs Predicted Bitcoin Close"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "BTC Closing Price"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()


plot_path = (
    RESULTS_DIR
    +
    "/tft_actual_vs_predicted.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.close()


print(
    "Prediction plot saved to:"
)

print(
    plot_path
)


# ============================================================
# ERROR DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    results["Absolute_Error"],
    bins=30
)

plt.title(
    "TFT Prediction Error Distribution"
)

plt.xlabel(
    "Absolute Error"
)

plt.ylabel(
    "Frequency"
)

plt.grid(
    True
)

plt.tight_layout()


error_plot_path = (
    RESULTS_DIR
    +
    "/tft_error_distribution.png"
)

plt.savefig(
    error_plot_path,
    dpi=300
)

plt.close()


print(
    "Error distribution saved to:"
)

print(
    error_plot_path
)


# ============================================================
# BEST / WORST PREDICTIONS
# ============================================================

best_predictions = (
    results
    .sort_values(
        "Percentage_Error"
    )
    .head(10)
)


worst_predictions = (
    results
    .sort_values(
        "Percentage_Error",
        ascending=False
    )
    .head(10)
)


best_predictions.to_csv(
    RESULTS_DIR
    +
    "/tft_best_predictions.csv",
    index=False
)


worst_predictions.to_csv(
    RESULTS_DIR
    +
    "/tft_worst_predictions.csv",
    index=False
)


print(
    "\nBest and worst prediction files saved."
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

configuration = pd.DataFrame({

    "Parameter": [

        "Model",
        "Asset",
        "Lookback",
        "Forecast Horizon",
        "Features",
        "Hidden Size",
        "Attention Heads",
        "Dropout",
        "Learning Rate",
        "Batch Size",
        "Maximum Epochs",
        "Early Stopping Patience"
    ],

    "Value": [

        "Temporal Fusion Transformer",
        "Bitcoin",
        LOOKBACK,
        1,
        len(FEATURES),
        16,
        4,
        0.1,
        0.001,
        64,
        30,
        5
    ]

})


configuration.to_csv(
    RESULTS_DIR
    +
    "/tft_configuration.csv",
    index=False
)


print(
    "\nTFT evaluation completed."
)

print(
    "================================"
)