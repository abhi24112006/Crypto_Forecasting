import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler

from pytorch_forecasting import (
    TimeSeriesDataSet,
    TemporalFusionTransformer
)

from pytorch_forecasting.metrics import MAE

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "engineered_data.csv"

LOOKBACK = 30

BATCH_SIZE = 64

MAX_EPOCHS = 30

LEARNING_RATE = 0.001

HIDDEN_SIZE = 16

ATTENTION_HEAD_SIZE = 4

DROPOUT = 0.1

HIDDEN_CONTINUOUS_SIZE = 8

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

pl.seed_everything(SEED, workers=True)


# ============================================================
# FEATURES
# ============================================================

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
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])

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


print("Dataset shape:", df.shape)

print(
    "Missing values:",
    df.isnull().sum().sum()
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

val_df = df.iloc[
    train_end:val_end
].copy()

test_df = df.iloc[
    val_end:
].copy()


print("\nDataset split:")
print(
    "Train:",
    len(train_df)
)

print(
    "Validation:",
    len(val_df)
)

print(
    "Test:",
    len(test_df)
)


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
# REMOVE UNUSED LAG FEATURES
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

print("\nCreating TFT dataset...")


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

    time_varying_unknown_reals=[
        *FEATURES
    ],

    allow_missing_timesteps=False
)


# ============================================================
# VALIDATION DATASET
# ============================================================

validation_dataset = (
    TimeSeriesDataSet.from_dataset(

        training_dataset,

        val_df,

        predict=False,

        stop_randomization=True
    )
)


# ============================================================
# TEST DATASET
# ============================================================

test_dataset = (
    TimeSeriesDataSet.from_dataset(

        training_dataset,

        test_df,

        predict=False,

        stop_randomization=True
    )
)


# ============================================================
# DATALOADERS
# ============================================================

train_dataloader = (
    training_dataset.to_dataloader(

        train=True,

        batch_size=BATCH_SIZE,

        num_workers=0
    )
)


val_dataloader = (
    validation_dataset.to_dataloader(

        train=False,

        batch_size=BATCH_SIZE,

        num_workers=0
    )
)


test_dataloader = (
    test_dataset.to_dataloader(

        train=False,

        batch_size=BATCH_SIZE,

        num_workers=0
    )
)


print(
    "\nTraining samples:",
    len(training_dataset)
)

print(
    "Validation samples:",
    len(validation_dataset)
)

print(
    "Test samples:",
    len(test_dataset)
)


# ============================================================
# CREATE TFT MODEL
# ============================================================

print("\nCreating TFT model...")


tft = TemporalFusionTransformer.from_dataset(

    training_dataset,

    learning_rate=LEARNING_RATE,

    hidden_size=HIDDEN_SIZE,

    attention_head_size=ATTENTION_HEAD_SIZE,

    dropout=DROPOUT,

    hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,

    output_size=1,

    loss=MAE(),

    log_interval=10,

    reduce_on_plateau_patience=4
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\nTFT model:")
print(tft)


# ============================================================
# CALLBACKS
# ============================================================

early_stop_callback = EarlyStopping(

    monitor="val_loss",

    min_delta=1e-4,

    patience=5,

    verbose=True,

    mode="min"
)


checkpoint_callback = ModelCheckpoint(

    dirpath="models/tft",

    filename="tft-{epoch:02d}-{val_loss:.4f}",

    monitor="val_loss",

    mode="min",

    save_top_k=1
)


# ============================================================
# TRAINER
# ============================================================

trainer = pl.Trainer(

    max_epochs=MAX_EPOCHS,

    accelerator="cpu",

    devices=1,

    gradient_clip_val=0.1,

    callbacks=[
        early_stop_callback,
        checkpoint_callback
    ],

    enable_model_summary=True,

    log_every_n_steps=10
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting TFT training...")

trainer.fit(

    tft,

    train_dataloaders=train_dataloader,

    val_dataloaders=val_dataloader
)


# ============================================================
# BEST MODEL
# ============================================================

print("\nBest model:")

print(
    checkpoint_callback.best_model_path
)


# ============================================================
# TEST
# ============================================================

print("\nRunning test prediction...")

best_tft = (
    TemporalFusionTransformer.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )
)


predictions = best_tft.predict(
    test_dataloader
)


print(
    "\nPrediction shape:",
    predictions.shape
)


# ============================================================
# CONVERT PREDICTIONS BACK TO BTC PRICE
# ============================================================

predictions = predictions.detach().cpu().numpy()


predicted_prices = (
    target_scaler.inverse_transform(
        predictions.reshape(-1, 1)
    )
    .flatten()
)


# ============================================================
# ACTUAL TARGET VALUES
# ============================================================

actual_targets = []

for batch in test_dataloader:

    x, y = batch

    target = y[0]

    actual_targets.extend(
        target.detach()
        .cpu()
        .numpy()
        .flatten()
    )


actual_targets = np.array(
    actual_targets
)


actual_prices = (
    target_scaler.inverse_transform(
        actual_targets.reshape(-1, 1)
    )
    .flatten()
)


# ============================================================
# EVALUATION
# ============================================================

mae = np.mean(
    np.abs(
        actual_prices -
        predicted_prices
    )
)


rmse = np.sqrt(
    np.mean(
        (
            actual_prices -
            predicted_prices
        ) ** 2
    )
)


mape = np.mean(
    np.abs(
        (
            actual_prices -
            predicted_prices
        )
        /
        actual_prices
    )
) * 100


print("\n==============================")

print("TFT TEST RESULTS")

print("==============================")

print(
    f"MAE  : {mae:.2f}"
)

print(
    f"RMSE : {rmse:.2f}"
)

print(
    f"MAPE : {mape:.2f}%"
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    PROJECT_ROOT / "results",
    exist_ok=True
)


results = pd.DataFrame({

    "Actual": actual_prices,

    "Predicted": predicted_prices

})


results_path = PROJECT_ROOT / "results" / "tft_predictions.csv"
results.to_csv(
    results_path,
    index=False
)


print(
    "\nPredictions saved to:"
)

print(results_path)