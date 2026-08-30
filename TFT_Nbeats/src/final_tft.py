import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import lightning.pytorch as pl

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "engineered_data.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "tft" / "final"

os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET = "Target"

# WINNING CONFIGURATION
HIDDEN_SIZE = 64
ATTENTION_HEADS = 4
LOOKBACK = 30

LEARNING_RATE = 0.001
DROPOUT = 0.1
HIDDEN_CONTINUOUS_SIZE = 8


# ============================================================
# FEATURES
# ============================================================

# IMPORTANT:
# Lag_1, Lag_2, Lag_3 are intentionally NOT included.
#
# Our experiment showed that removing them produced better
# validation performance.

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


# ============================================================
# SEED
# ============================================================

pl.seed_everything(
    42,
    workers=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FINAL TFT MODEL")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])

df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df["Volume_Change"] = (
    df["Volume_Change"]
    .fillna(0)
)

print(
    f"Dataset shape: {df.shape}"
)

print(
    f"Date range: "
    f"{df['Date'].min().date()} → "
    f"{df['Date'].max().date()}"
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

n = len(df)

train_end = int(n * 0.70)

val_end = int(n * 0.85)

train_df = df.iloc[
    :train_end
].copy()

validation_df = df.iloc[
    train_end:val_end
].copy()

test_df = df.iloc[
    val_end:
].copy()


print("\nOriginal split:")

print(
    f"Train      : {len(train_df)}"
)

print(
    f"Validation : {len(validation_df)}"
)

print(
    f"Test       : {len(test_df)}"
)


# ============================================================
# FINAL TRAINING DATA
#
# Hyperparameters are already selected.
#
# Therefore:
#
# TRAIN + VALIDATION → FINAL TRAINING
# TEST                → NEVER TOUCHED
# ============================================================

final_train_df = pd.concat(
    [
        train_df,
        validation_df
    ],
    ignore_index=True
)

print("\nFinal training data:")

print(
    f"Training samples: "
    f"{len(final_train_df)}"
)


# ============================================================
# SCALE FEATURES
#
# Fit scaler ONLY on final training data.
# ============================================================

feature_scaler = StandardScaler()

target_scaler = StandardScaler()


final_train_df[FEATURES] = (
    feature_scaler
    .fit_transform(
        final_train_df[FEATURES]
    )
)

test_df[FEATURES] = (
    feature_scaler
    .transform(
        test_df[FEATURES]
    )
)


# ============================================================
# SCALE TARGET
# ============================================================

final_train_df[[TARGET]] = (
    target_scaler
    .fit_transform(
        final_train_df[[TARGET]]
    )
)

test_df[[TARGET]] = (
    target_scaler
    .transform(
        test_df[[TARGET]]
    )
)


# ============================================================
# TIME INDEX
# ============================================================

final_train_df["time_idx"] = np.arange(
    len(final_train_df)
)

test_start_idx = len(final_train_df)

test_df["time_idx"] = np.arange(
    test_start_idx,
    test_start_idx + len(test_df)
)


# ============================================================
# GROUP
# ============================================================

final_train_df["group"] = "BTC"

test_df["group"] = "BTC"


# ============================================================
# TRAINING DATASET
# ============================================================

training_dataset = TimeSeriesDataSet(

    final_train_df,

    time_idx="time_idx",

    target=TARGET,

    group_ids=[
        "group"
    ],

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
#
# Give TFT the final 30 training observations as historical
# context before the first test prediction.
# ============================================================

encoder_context = final_train_df.iloc[
    -LOOKBACK:
].copy()

test_input = pd.concat(
    [
        encoder_context,
        test_df
    ],
    ignore_index=True
)


test_input["time_idx"] = np.arange(
    test_start_idx - LOOKBACK,
    test_start_idx + len(test_df)
)

test_input["group"] = "BTC"


# ============================================================
# TEST DATASET
# ============================================================

test_dataset = TimeSeriesDataSet.from_dataset(

    training_dataset,

    test_input,

    predict=False,

    stop_randomization=True,

    min_prediction_idx=test_start_idx
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = (
    training_dataset
    .to_dataloader(

        train=True,

        batch_size=64,

        num_workers=0
    )
)

test_loader = (
    test_dataset
    .to_dataloader(

        train=False,

        batch_size=64,

        num_workers=0
    )
)


print("\nTFT dataset:")

print(
    f"Training samples : "
    f"{len(training_dataset)}"
)

print(
    f"Test samples     : "
    f"{len(test_dataset)}"
)


# ============================================================
# MODEL
# ============================================================

print("\nCreating final TFT...")

model = (
    TemporalFusionTransformer
    .from_dataset(

        training_dataset,

        learning_rate=LEARNING_RATE,

        hidden_size=HIDDEN_SIZE,

        attention_head_size=ATTENTION_HEADS,

        dropout=DROPOUT,

        hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,

        output_size=1,

        loss=MAE(),

        log_interval=-1,

        reduce_on_plateau_patience=3
    )
)


# ============================================================
# CHECKPOINT
# ============================================================

checkpoint_callback = (
    pl.callbacks.ModelCheckpoint(

        dirpath=RESULTS_DIR,

        filename="final-tft-{epoch:02d}-{train_loss:.4f}",

        monitor="train_loss",

        mode="min",

        save_top_k=1
    )
)


# ============================================================
# TRAINER
# ============================================================

trainer = pl.Trainer(

    max_epochs=15,

    accelerator="cpu",

    devices=1,

    gradient_clip_val=0.1,

    callbacks=[
        checkpoint_callback
    ],

    enable_progress_bar=True,

    logger=False
)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING FINAL TFT")
print("=" * 70)

trainer.fit(

    model,

    train_dataloaders=train_loader
)


# ============================================================
# BEST CHECKPOINT
# ============================================================

best_checkpoint = (
    checkpoint_callback
    .best_model_path
)

print("\nBest final checkpoint:")

print(
    best_checkpoint
)


# ============================================================
# LOAD FINAL MODEL
# ============================================================

final_model = (
    TemporalFusionTransformer
    .load_from_checkpoint(
        best_checkpoint
    )
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nRunning final test prediction...")

predictions = (
    final_model
    .predict(
        test_loader
    )
)

predictions = (
    predictions
    .detach()
    .cpu()
    .numpy()
    .reshape(-1)
)


# ============================================================
# ACTUAL TARGETS
# ============================================================

actual_scaled = []

for batch in test_loader:

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

        predictions.reshape(
            -1,
            1
        )

    )
    .reshape(-1)
)

actual_prices = (
    target_scaler
    .inverse_transform(

        actual_scaled.reshape(
            -1,
            1
        )

    )
    .reshape(-1)
)


# ============================================================
# ALIGN DATES
# ============================================================

prediction_dates = test_df[
    "Date"
].iloc[
    :len(predicted_prices)
].reset_index(
    drop=True
)


# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    actual_prices,
    predicted_prices
)

rmse = np.sqrt(
    mean_squared_error(
        actual_prices,
        predicted_prices
    )
)

mape = (
    np.mean(
        np.abs(
            (
                actual_prices
                -
                predicted_prices
            )
            /
            np.where(
                actual_prices == 0,
                1,
                actual_prices
            )
        )
    )
    * 100
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame({

    "Date": prediction_dates,

    "Actual_Close": actual_prices,

    "Predicted_Close": predicted_prices,

    "Absolute_Error":
        np.abs(
            actual_prices
            -
            predicted_prices
        ),

    "Percentage_Error":
        np.abs(
            (
                actual_prices
                -
                predicted_prices
            )
            /
            np.where(
                actual_prices == 0,
                1,
                actual_prices
            )
        )
        * 100
})


prediction_path = (
    RESULTS_DIR
    +
    "/final_tft_predictions.csv"
)

predictions_df.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({

    "Model": [
        "TFT"
    ],

    "Hidden_Size": [
        HIDDEN_SIZE
    ],

    "Attention_Heads": [
        ATTENTION_HEADS
    ],

    "Lookback": [
        LOOKBACK
    ],

    "Lag_Features": [
        False
    ],

    "MAE": [
        mae
    ],

    "RMSE": [
        rmse
    ],

    "MAPE": [
        mape
    ]
})


metrics_path = (
    RESULTS_DIR
    +
    "/final_tft_metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TFT TEST RESULTS")
print("=" * 70)

print(
    f"MAE  : {mae:.2f}"
)

print(
    f"RMSE : {rmse:.2f}"
)

print(
    f"MAPE : {mape:.2f}%"
)

print("=" * 70)

print("\nFinal configuration:")

print(
    f"Hidden size     : {HIDDEN_SIZE}"
)

print(
    f"Attention heads : {ATTENTION_HEADS}"
)

print(
    f"Lookback        : {LOOKBACK} days"
)

print(
    "Lag features    : False"
)

print(
    "\nPredictions saved to:"
)

print(
    prediction_path
)

print(
    "\nMetrics saved to:"
)

print(
    metrics_path
)

print(
    "\nFinal TFT completed."
)