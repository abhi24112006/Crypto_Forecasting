from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from pytorch_forecasting import TimeSeriesDataSet


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "engineered_data.csv"

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
# LOAD DATA
# ============================================================

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

df = df.replace([np.inf, -np.inf], np.nan)

df["Volume_Change"] = df["Volume_Change"].fillna(0)

print("Dataset shape:", df.shape)
print("Missing values:", df.isnull().sum().sum())


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()

print("\nSplit sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# ============================================================
# SCALE FEATURES
# ============================================================

feature_scaler = StandardScaler()
target_scaler = StandardScaler()

train_df[FEATURES] = feature_scaler.fit_transform(
    train_df[FEATURES]
)

val_df[FEATURES] = feature_scaler.transform(
    val_df[FEATURES]
)

test_df[FEATURES] = feature_scaler.transform(
    test_df[FEATURES]
)


# ============================================================
# SCALE TARGET
# ============================================================

train_df[[TARGET]] = target_scaler.fit_transform(
    train_df[[TARGET]]
)

val_df[[TARGET]] = target_scaler.transform(
    val_df[[TARGET]]
)

test_df[[TARGET]] = target_scaler.transform(
    test_df[[TARGET]]
)


# ============================================================
# REMOVE UNUSED LAG FEATURES
# ============================================================

DROP_COLUMNS = [
    "Lag_1",
    "Lag_2",
    "Lag_3"
]

train_df = train_df.drop(columns=DROP_COLUMNS)
val_df = val_df.drop(columns=DROP_COLUMNS)
test_df = test_df.drop(columns=DROP_COLUMNS)


# ============================================================
# CREATE TIME INDEX
# ============================================================

train_df["time_idx"] = np.arange(len(train_df))

val_df["time_idx"] = np.arange(
    len(train_df),
    len(train_df) + len(val_df)
)

test_df["time_idx"] = np.arange(
    len(train_df) + len(val_df),
    len(df)
)


# ============================================================
# CREATE GROUP
# ============================================================

train_df["group"] = "BTC"
val_df["group"] = "BTC"
test_df["group"] = "BTC"


# ============================================================
# COMBINE DATA
# ============================================================

all_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)


# ============================================================
# DISPLAY DATA
# ============================================================

print("\nFinal columns:")
print(all_df.columns.tolist())

print("\nFinal shape:")
print(all_df.shape)

print("\nFirst rows:")
print(all_df.head())


# ============================================================
# CREATE TFT DATASET
# ============================================================

training_cutoff = train_df["time_idx"].max()

print("\nTraining cutoff:", training_cutoff)


training_dataset = TimeSeriesDataSet(
    train_df,

    time_idx="time_idx",

    target=TARGET,

    group_ids=["group"],

    min_encoder_length=LOOKBACK,
    max_encoder_length=LOOKBACK,

    min_prediction_length=1,
    max_prediction_length=1,

    static_categoricals=["group"],

    time_varying_known_reals=[
        "time_idx"
    ],

    time_varying_unknown_reals=FEATURES,

    allow_missing_timesteps=False
)


# ============================================================
# CREATE VALIDATION DATASET
# ============================================================

validation_dataset = TimeSeriesDataSet.from_dataset(
    training_dataset,
    val_df,
    predict=False,
    stop_randomization=True
)


# ============================================================
# CREATE TEST DATASET
# ============================================================

test_dataset = TimeSeriesDataSet.from_dataset(
    training_dataset,
    test_df,
    predict=False,
    stop_randomization=True
)


# ============================================================
# CREATE DATALOADERS
# ============================================================

train_dataloader = training_dataset.to_dataloader(
    train=True,
    batch_size=64,
    num_workers=0
)

val_dataloader = validation_dataset.to_dataloader(
    train=False,
    batch_size=64,
    num_workers=0
)

test_dataloader = test_dataset.to_dataloader(
    train=False,
    batch_size=64,
    num_workers=0
)


# ============================================================
# PRINT DATASET INFORMATION
# ============================================================

print("\nTFT dataset created successfully.")

print("Training samples:",
      len(training_dataset))

print("Validation samples:",
      len(validation_dataset))

print("Test samples:",
      len(test_dataset))


# ============================================================
# INSPECT ONE BATCH
# ============================================================

x, y = next(iter(train_dataloader))

print("\nBatch inspection:")
print("------------------")

print("Encoder continuous shape:")
print(x["encoder_cont"].shape)

print("\nDecoder continuous shape:")
print(x["decoder_cont"].shape)

print("\nTarget shape:")
print(y[0].shape)