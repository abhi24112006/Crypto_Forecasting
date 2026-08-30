import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import lightning.pytorch as pl

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import MAE


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "engineered_data.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "tft" / "experiments"

os.makedirs(RESULTS_DIR, exist_ok=True)


TARGET = "Target"

BASE_FEATURES = [
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

LAG_FEATURES = [
    "Lag_1",
    "Lag_2",
    "Lag_3"
]


# ============================================================
# RANDOM SEED
# ============================================================

pl.seed_everything(
    42,
    workers=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

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


# ============================================================
# SPLIT
# ============================================================

n = len(df)

train_end = int(n * 0.70)

val_end = int(n * 0.85)

train_raw = df.iloc[
    :train_end
].copy()

val_raw = df.iloc[
    train_end:val_end
].copy()

test_raw = df.iloc[
    val_end:
].copy()


print("\nSplit sizes:")
print("Train:", len(train_raw))
print("Validation:", len(val_raw))
print("Test:", len(test_raw))


# ============================================================
# EXPERIMENT FUNCTION
# ============================================================

def run_experiment(
    experiment_name,
    hidden_size,
    attention_heads,
    lookback,
    use_lags
):

    print("\n")
    print("=" * 70)
    print(f"RUNNING: {experiment_name}")
    print("=" * 70)

    print(
        f"Hidden size      : {hidden_size}"
    )

    print(
        f"Attention heads  : {attention_heads}"
    )

    print(
        f"Lookback         : {lookback}"
    )

    print(
        f"Lag features     : {use_lags}"
    )


    # ========================================================
    # COPY DATA
    # ========================================================

    train_df = train_raw.copy()

    val_df = val_raw.copy()


    # ========================================================
    # FEATURES
    # ========================================================

    features = BASE_FEATURES.copy()

    if use_lags:

        features.extend(
            LAG_FEATURES
        )


    # ========================================================
    # SCALE FEATURES
    # ========================================================

    feature_scaler = StandardScaler()

    target_scaler = StandardScaler()


    train_df[features] = (
        feature_scaler
        .fit_transform(
            train_df[features]
        )
    )

    val_df[features] = (
        feature_scaler
        .transform(
            val_df[features]
        )
    )


    # ========================================================
    # SCALE TARGET
    # ========================================================

    train_df[[TARGET]] = (
        target_scaler
        .fit_transform(
            train_df[[TARGET]]
        )
    )

    val_df[[TARGET]] = (
        target_scaler
        .transform(
            val_df[[TARGET]]
        )
    )


    # ========================================================
    # TIME INDEX
    # ========================================================

    train_df["time_idx"] = np.arange(
        len(train_df)
    )

    val_df["time_idx"] = np.arange(
        len(train_df),
        len(train_df) + len(val_df)
    )


    # ========================================================
    # GROUP
    # ========================================================

    train_df["group"] = "BTC"

    val_df["group"] = "BTC"


    # ========================================================
    # TRAIN DATASET
    # ========================================================

    training_dataset = TimeSeriesDataSet(

        train_df,

        time_idx="time_idx",

        target=TARGET,

        group_ids=[
            "group"
        ],

        min_encoder_length=lookback,

        max_encoder_length=lookback,

        min_prediction_length=1,

        max_prediction_length=1,

        static_categoricals=[
            "group"
        ],

        time_varying_known_reals=[
            "time_idx"
        ],

        time_varying_unknown_reals=features,

        allow_missing_timesteps=False
    )


    # ========================================================
    # VALIDATION DATA
    #
    # Include the last LOOKBACK training rows as encoder
    # context so validation predictions have historical data.
    # ========================================================

    encoder_context = train_df.iloc[
        -lookback:
    ].copy()

    validation_input = pd.concat(
        [
            encoder_context,
            val_df
        ],
        ignore_index=True
    )


    # Re-create continuous time index

    validation_input["time_idx"] = np.arange(
        train_end - lookback,
        val_end
    )

    validation_input["group"] = "BTC"


    # ========================================================
    # VALIDATION DATASET
    # ========================================================

    validation_dataset = (
        TimeSeriesDataSet.from_dataset(

            training_dataset,

            validation_input,

            predict=False,

            stop_randomization=True,

            min_prediction_idx=train_end
        )
    )


    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = (
        training_dataset
        .to_dataloader(

            train=True,

            batch_size=64,

            num_workers=0
        )
    )


    val_loader = (
        validation_dataset
        .to_dataloader(

            train=False,

            batch_size=64,

            num_workers=0
        )
    )


    print(
        f"\nTraining samples   : {len(training_dataset)}"
    )

    print(
        f"Validation samples : {len(validation_dataset)}"
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = (
        TemporalFusionTransformer
        .from_dataset(

            training_dataset,

            learning_rate=0.001,

            hidden_size=hidden_size,

            attention_head_size=attention_heads,

            dropout=0.1,

            hidden_continuous_size=8,

            output_size=1,

            loss=MAE(),

            log_interval=-1,

            reduce_on_plateau_patience=3
        )
    )


    # ========================================================
    # CHECKPOINT
    # ========================================================

    checkpoint_callback = (
        pl.callbacks.ModelCheckpoint(

            dirpath=RESULTS_DIR,

            filename=(
                experiment_name
                +
                "-{epoch:02d}-{val_loss:.4f}"
            ),

            monitor="val_loss",

            mode="min",

            save_top_k=1
        )
    )


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    early_stopping = (
        pl.callbacks.EarlyStopping(

            monitor="val_loss",

            patience=5,

            mode="min"
        )
    )


    # ========================================================
    # TRAINER
    # ========================================================

    trainer = pl.Trainer(

        max_epochs=30,

        accelerator="cpu",

        devices=1,

        gradient_clip_val=0.1,

        callbacks=[
            checkpoint_callback,
            early_stopping
        ],

        enable_progress_bar=True,

        logger=False
    )


    # ========================================================
    # TRAIN
    # ========================================================

    print("\nStarting training...")

    trainer.fit(

        model,

        train_dataloaders=train_loader,

        val_dataloaders=val_loader
    )


    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    best_path = (
        checkpoint_callback
        .best_model_path
    )

    print(
        "\nBest checkpoint:"
    )

    print(
        best_path
    )


    best_model = (
        TemporalFusionTransformer
        .load_from_checkpoint(
            best_path
        )
    )


    # ========================================================
    # VALIDATION PREDICTIONS
    # ========================================================

    predictions = (
        best_model
        .predict(
            val_loader
        )
    )

    predictions = (
        predictions
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )


    # ========================================================
    # ACTUAL VALUES
    # ========================================================

    actual_scaled = []

    for batch in val_loader:

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


    # ========================================================
    # INVERSE TRANSFORM
    # ========================================================

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


    # ========================================================
    # METRICS
    # ========================================================

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


    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "Experiment": experiment_name,

        "Hidden_Size": hidden_size,

        "Attention_Heads": attention_heads,

        "Lookback": lookback,

        "Use_Lags": use_lags,

        "Validation_MAE": mae,

        "Validation_RMSE": rmse,

        "Validation_MAPE": mape,

        "Best_Checkpoint": best_path
    }


    print("\n")
    print("-" * 70)

    print(
        f"{experiment_name} RESULTS"
    )

    print("-" * 70)

    print(
        f"Validation MAE  : {mae:.2f}"
    )

    print(
        f"Validation RMSE : {rmse:.2f}"
    )

    print(
        f"Validation MAPE : {mape:.2f}%"
    )

    print("-" * 70)


    return result


# ============================================================
# EXPERIMENT 1
# MODEL CAPACITY
# ============================================================

experiment_results = []


experiment_results.append(

    run_experiment(

        "capacity_16",

        hidden_size=16,

        attention_heads=4,

        lookback=30,

        use_lags=False
    )
)


experiment_results.append(

    run_experiment(

        "capacity_32",

        hidden_size=32,

        attention_heads=4,

        lookback=30,

        use_lags=False
    )
)


experiment_results.append(

    run_experiment(

        "capacity_64",

        hidden_size=64,

        attention_heads=4,

        lookback=30,

        use_lags=False
    )
)


# ============================================================
# SAVE EXPERIMENT 1
# ============================================================

results_df = pd.DataFrame(
    experiment_results
)

results_df.to_csv(

    RESULTS_DIR
    +
    "/experiment_1_capacity.csv",

    index=False
)


print("\n")
print("=" * 70)
print("EXPERIMENT 1 COMPLETE")
print("=" * 70)

print(
    results_df[
        [
            "Experiment",
            "Hidden_Size",
            "Validation_MAE",
            "Validation_RMSE",
            "Validation_MAPE"
        ]
    ]
    .sort_values(
        "Validation_MAE"
    )
    .to_string(
        index=False
    )
)


# ============================================================
# FIND BEST CAPACITY
# ============================================================

best_capacity = (
    results_df
    .sort_values(
        "Validation_MAE"
    )
    .iloc[0]
)

best_hidden_size = int(
    best_capacity[
        "Hidden_Size"
    ]
)

best_attention_heads = int(
    best_capacity[
        "Attention_Heads"
    ]
)


print("\nBest capacity:")

print(
    f"Hidden size = {best_hidden_size}"
)


# ============================================================
# EXPERIMENT 2
# LOOKBACK
# ============================================================

lookback_results = []


for lookback in [
    30,
    60,
    90
]:

    lookback_results.append(

        run_experiment(

            f"lookback_{lookback}",

            hidden_size=best_hidden_size,

            attention_heads=best_attention_heads,

            lookback=lookback,

            use_lags=False
        )
    )


lookback_df = pd.DataFrame(
    lookback_results
)


lookback_df.to_csv(

    RESULTS_DIR
    +
    "/experiment_2_lookback.csv",

    index=False
)


print("\n")
print("=" * 70)
print("EXPERIMENT 2 COMPLETE")
print("=" * 70)

print(
    lookback_df[
        [
            "Experiment",
            "Lookback",
            "Validation_MAE",
            "Validation_RMSE",
            "Validation_MAPE"
        ]
    ]
    .sort_values(
        "Validation_MAE"
    )
    .to_string(
        index=False
    )
)


# ============================================================
# FIND BEST LOOKBACK
# ============================================================

best_lookback_row = (
    lookback_df
    .sort_values(
        "Validation_MAE"
    )
    .iloc[0]
)

best_lookback = int(
    best_lookback_row[
        "Lookback"
    ]
)


print("\nBest lookback:")

print(
    f"{best_lookback} days"
)


# ============================================================
# EXPERIMENT 3
# LAG FEATURE ABLATION
# ============================================================

lag_results = []


lag_results.append(

    run_experiment(

        "without_lags",

        hidden_size=best_hidden_size,

        attention_heads=best_attention_heads,

        lookback=best_lookback,

        use_lags=False
    )
)


lag_results.append(

    run_experiment(

        "with_lags",

        hidden_size=best_hidden_size,

        attention_heads=best_attention_heads,

        lookback=best_lookback,

        use_lags=True
    )
)


lag_df = pd.DataFrame(
    lag_results
)


lag_df.to_csv(

    RESULTS_DIR
    +
    "/experiment_3_lags.csv",

    index=False
)


print("\n")
print("=" * 70)
print("EXPERIMENT 3 COMPLETE")
print("=" * 70)

print(
    lag_df[
        [
            "Experiment",
            "Lookback",
            "Use_Lags",
            "Validation_MAE",
            "Validation_RMSE",
            "Validation_MAPE"
        ]
    ]
    .sort_values(
        "Validation_MAE"
    )
    .to_string(
        index=False
    )
)


# ============================================================
# FINAL VALIDATION WINNER
# ============================================================

best_final = (
    lag_df
    .sort_values(
        "Validation_MAE"
    )
    .iloc[0]
)


print("\n")
print("=" * 70)
print("TFT EXPERIMENT SUMMARY")
print("=" * 70)

print(
    f"Best hidden size : {best_final['Hidden_Size']}"
)

print(
    f"Best lookback    : {best_final['Lookback']}"
)

print(
    f"Use lag features : {best_final['Use_Lags']}"
)

print(
    f"Validation MAE   : "
    f"{best_final['Validation_MAE']:.2f}"
)

print(
    f"Validation RMSE  : "
    f"{best_final['Validation_RMSE']:.2f}"
)

print(
    f"Validation MAPE  : "
    f"{best_final['Validation_MAPE']:.2f}%"
)

print("=" * 70)


# ============================================================
# SAVE COMPLETE SUMMARY
# ============================================================

all_results = pd.concat(

    [
        results_df,
        lookback_df,
        lag_df
    ],

    ignore_index=True
)


all_results.to_csv(

    RESULTS_DIR
    +
    "/all_tft_experiments.csv",

    index=False
)


print(
    "\nAll experiment results saved to:"
)

print(
    RESULTS_DIR
    +
    "/all_tft_experiments.csv"
)

print("\nExperiments completed.")