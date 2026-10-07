# src/nbeats_data.py

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/engineered_data.csv"

LOOKBACK = 30

# Options:
#   "close"     -> Close price only
#   "features"  -> Close + technical indicators
FEATURE_MODE = "close"

TARGET = "Target"


# ============================================================
# FEATURES
# ============================================================

CLOSE_FEATURES = [
    "Close"
]

TECHNICAL_FEATURES = [
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
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("LOADING N-BEATS DATA")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Replace infinity values
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # The first Volume_Change value may be missing
    df["Volume_Change"] = (
        df["Volume_Change"]
        .fillna(0)
    )

    print(f"Dataset shape: {df.shape}")

    print(
        f"Date range: "
        f"{df['Date'].min().date()} → "
        f"{df['Date'].max().date()}"
    )

    print(
        f"Missing values: "
        f"{df.isna().sum().sum()}"
    )

    return df


# ============================================================
# SELECT FEATURES
# ============================================================

def get_features():

    if FEATURE_MODE == "close":

        return CLOSE_FEATURES

    elif FEATURE_MODE == "features":

        return TECHNICAL_FEATURES

    else:

        raise ValueError(
            "FEATURE_MODE must be "
            "'close' or 'features'"
        )


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(df):

    n = len(df)

    train_end = int(
        n * 0.70
    )

    validation_end = int(
        n * 0.85
    )

    train_df = df.iloc[
        :train_end
    ].copy()

    validation_df = df.iloc[
        train_end:validation_end
    ].copy()

    test_df = df.iloc[
        validation_end:
    ].copy()

    print("\nSplit sizes:")

    print(
        f"Train      : {len(train_df)}"
    )

    print(
        f"Validation : {len(validation_df)}"
    )

    print(
        f"Test       : {len(test_df)}"
    )

    return (
        train_df,
        validation_df,
        test_df
    )


# ============================================================
# CREATE SEQUENCES
# ============================================================

def create_sequences(
    feature_values,
    target_values,
    lookback
):

    X = []
    y = []

    for i in range(
        lookback,
        len(feature_values)
    ):

        X.append(
            feature_values[
                i - lookback:i
            ]
        )

        y.append(
            target_values[i]
        )

    return (
        np.array(X),
        np.array(y)
    )


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data():

    df = load_data()

    features = get_features()

    print("\nFeature mode:")
    print(FEATURE_MODE)

    print("\nFeatures:")

    for feature in features:

        print(
            f"  - {feature}"
        )

    (
        train_df,
        validation_df,
        test_df
    ) = chronological_split(df)


    # ========================================================
    # SCALERS
    #
    # IMPORTANT:
    # Fit ONLY on training data.
    # ========================================================

    feature_scaler = StandardScaler()

    target_scaler = StandardScaler()


    # ========================================================
    # FIT FEATURE SCALER
    # ========================================================

    feature_scaler.fit(
        train_df[features]
    )


    # ========================================================
    # TRANSFORM FEATURES
    # ========================================================

    train_features = (
        feature_scaler.transform(
            train_df[features]
        )
    )

    validation_features = (
        feature_scaler.transform(
            validation_df[features]
        )
    )

    test_features = (
        feature_scaler.transform(
            test_df[features]
        )
    )


    # ========================================================
    # TARGET SCALER
    # ========================================================

    target_scaler.fit(
        train_df[[TARGET]]
    )


    train_target = (
        target_scaler.transform(
            train_df[[TARGET]]
        )
        .flatten()
    )

    validation_target = (
        target_scaler.transform(
            validation_df[[TARGET]]
        )
        .flatten()
    )

    test_target = (
        target_scaler.transform(
            test_df[[TARGET]]
        )
        .flatten()
    )


    # ========================================================
    # CREATE TRAIN SEQUENCES
    # ========================================================

    X_train, y_train = create_sequences(
        train_features,
        train_target,
        LOOKBACK
    )


    # ========================================================
    # VALIDATION SEQUENCES
    #
    # Give validation its previous LOOKBACK observations
    # from training data as historical context.
    # ========================================================

    validation_context_features = np.concatenate(
        [
            train_features[-LOOKBACK:],
            validation_features
        ],
        axis=0
    )

    validation_context_target = np.concatenate(
        [
            train_target[-LOOKBACK:],
            validation_target
        ],
        axis=0
    )

    X_validation, y_validation = create_sequences(
        validation_context_features,
        validation_context_target,
        LOOKBACK
    )


    # ========================================================
    # TEST SEQUENCES
    #
    # Give test its previous LOOKBACK observations from
    # TRAIN + VALIDATION as historical context.
    # ========================================================

    train_validation_features = np.concatenate(
        [
            train_features,
            validation_features
        ],
        axis=0
    )

    train_validation_target = np.concatenate(
        [
            train_target,
            validation_target
        ],
        axis=0
    )

    test_context_features = np.concatenate(
        [
            train_validation_features[-LOOKBACK:],
            test_features
        ],
        axis=0
    )

    test_context_target = np.concatenate(
        [
            train_validation_target[-LOOKBACK:],
            test_target
        ],
        axis=0
    )

    X_test, y_test = create_sequences(
        test_context_features,
        test_context_target,
        LOOKBACK
    )


    # ========================================================
    # DATES
    # ========================================================

    train_dates = train_df[
        "Date"
    ].iloc[
        LOOKBACK:
    ].reset_index(
        drop=True
    )

    validation_dates = validation_df[
        "Date"
    ].reset_index(
        drop=True
    )

    test_dates = test_df[
        "Date"
    ].reset_index(
        drop=True
    )


    # ========================================================
    # RESULTS
    # ========================================================

    print("\nSequence shapes:")

    print(
        f"X_train      : {X_train.shape}"
    )

    print(
        f"y_train      : {y_train.shape}"
    )

    print(
        f"X_validation : {X_validation.shape}"
    )

    print(
        f"y_validation : {y_validation.shape}"
    )

    print(
        f"X_test       : {X_test.shape}"
    )

    print(
        f"y_test       : {y_test.shape}"
    )


    print("\nExpected structure:")

    print(
        "Samples × Lookback × Features"
    )

    print(
        f"Lookback = {LOOKBACK}"
    )

    print(
        f"Features = {len(features)}"
    )


    return {
        "X_train": X_train,
        "y_train": y_train,

        "X_validation": X_validation,
        "y_validation": y_validation,

        "X_test": X_test,
        "y_test": y_test,

        "train_dates": train_dates,
        "validation_dates": validation_dates,
        "test_dates": test_dates,

        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,

        "features": features,

        "train_df": train_df,
        "validation_df": validation_df,
        "test_df": test_df
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    data = prepare_data()

    print("\n")
    print("=" * 70)
    print("N-BEATS DATA PREPARATION COMPLETE")
    print("=" * 70)