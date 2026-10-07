"""
FINAL N-BEATS MODEL
CryptoPredict - BTC Next-Day Close Forecasting

Final selected configuration:
    Features       : Close only
    Lookback       : 30 days
    Forecast       : 1 day
    Hidden size    : 128
    Blocks         : 4
    Layers/block  : 4
    Batch size     : 64
    Learning rate  : 0.001
    Loss           : MSE
    Optimizer      : Adam
    Max epochs     : 100
    Early stopping : 10
    Random seed    : 42

Final training procedure:
    Train + Validation = 85%
    Test               = 15%

The test set is NEVER used during training or model selection.
"""

import os
import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# FINAL CONFIGURATION
# ============================================================

LOOKBACK = 30
FORECAST_HORIZON = 1

HIDDEN_SIZE = 128
NUM_BLOCKS = 4
NUM_LAYERS = 4

BATCH_SIZE = 64
LEARNING_RATE = 0.001

MAX_EPOCHS = 100
PATIENCE = 10

RANDOM_SEED = 42

DATA_PATH = "data/processed/cleaned_crypto_data.csv"

MODEL_DIR = "models/nbeats"
RESULTS_DIR = "results/nbeats"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "final_nbeats.pt"
)

PREDICTIONS_PATH = os.path.join(
    RESULTS_DIR,
    "final_nbeats_predictions.csv"
)

METRICS_PATH = os.path.join(
    RESULTS_DIR,
    "final_nbeats_metrics.csv"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# N-BEATS BLOCK
# ============================================================

class NBeatsBlock(nn.Module):
    """
    Canonical N-BEATS block.

    Each block produces:
        - backcast: reconstruction of the input
        - forecast: contribution to future prediction
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        forecast_size
    ):
        super().__init__()

        layers = []

        # First hidden layer
        layers.append(
            nn.Linear(
                input_size,
                hidden_size
            )
        )

        layers.append(
            nn.ReLU()
        )

        # Remaining hidden layers
        for _ in range(num_layers - 1):

            layers.append(
                nn.Linear(
                    hidden_size,
                    hidden_size
                )
            )

            layers.append(
                nn.ReLU()
            )

        self.hidden_stack = nn.Sequential(
            *layers
        )

        # Backcast + forecast parameters
        self.theta = nn.Linear(
            hidden_size,
            input_size + forecast_size
        )

        self.input_size = input_size
        self.forecast_size = forecast_size

    def forward(self, x):

        hidden = self.hidden_stack(x)

        theta = self.theta(hidden)

        # Reconstruct the input
        backcast = theta[
            :,
            :self.input_size
        ]

        # Predict future
        forecast = theta[
            :,
            self.input_size:
        ]

        return backcast, forecast


# ============================================================
# CANONICAL N-BEATS
# ============================================================

class NBeats(nn.Module):

    def __init__(
        self,
        input_size,
        forecast_size=1,
        hidden_size=128,
        num_blocks=4,
        num_layers=4
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                NBeatsBlock(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    forecast_size=forecast_size
                )
                for _ in range(num_blocks)
            ]
        )

        self.forecast_size = forecast_size

    def forward(self, x):

        residual = x

        forecast = torch.zeros(
            x.size(0),
            self.forecast_size,
            device=x.device
        )

        for block in self.blocks:

            backcast, block_forecast = (
                block(residual)
            )

            # Canonical residual connection
            residual = (
                residual - backcast
            )

            # Add block forecast
            forecast = (
                forecast + block_forecast
            )

        return forecast


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = np.mean(
        np.abs(
            y_true - y_pred
        )
    )

    rmse = np.sqrt(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )

    non_zero = y_true != 0

    mape = np.mean(
        np.abs(
            (
                y_true[non_zero]
                - y_pred[non_zero]
            )
            / y_true[non_zero]
        )
    ) * 100

    return mae, rmse, mape


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("LOADING FINAL N-BEATS DATA")
    print("=" * 70)

    df = pd.read_csv(
        DATA_PATH
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)

    print(
        f"Dataset shape: {df.shape}"
    )

    print(
        f"Date range: "
        f"{df['Date'].min().date()} → "
        f"{df['Date'].max().date()}"
    )

    print(
        f"Missing values: "
        f"{df.isna().sum().sum()}"
    )

    # --------------------------------------------------------
    # Close-only input
    # --------------------------------------------------------

    close = df[
        "Close"
    ].values.astype(
        np.float32
    )

    dates = df[
        "Date"
    ].values

    n = len(close)

    # Same chronological split
    # 70% / 15% / 15%
    train_end = int(
        n * 0.70
    )

    val_end = int(
        n * 0.85
    )

    train = close[
        :train_end
    ]

    validation = close[
        train_end:val_end
    ]

    test = close[
        val_end:
    ]

    test_dates = dates[
        val_end:
    ]

    print()
    print("Initial split:")

    print(
        f"Train:      {len(train)}"
    )

    print(
        f"Validation: {len(validation)}"
    )

    print(
        f"Test:       {len(test)}"
    )

    # --------------------------------------------------------
    # FINAL TRAINING DATA
    #
    # Combine Train + Validation.
    #
    # The test set remains completely untouched.
    # --------------------------------------------------------

    train_val = np.concatenate(
        [
            train,
            validation
        ]
    )

    print()
    print(
        f"Final training observations: "
        f"{len(train_val)}"
    )

    print(
        f"Final test observations: "
        f"{len(test)}"
    )

    # --------------------------------------------------------
    # Scaling
    #
    # Fit ONLY on Train + Validation.
    # Test is never used to calculate mean/std.
    # --------------------------------------------------------

    train_mean = train_val.mean()
    train_std = train_val.std()

    train_val_scaled = (
        train_val - train_mean
    ) / train_std

    test_scaled = (
        test - train_mean
    ) / train_std

    # --------------------------------------------------------
    # Create sequences
    # --------------------------------------------------------

    def create_sequences(
        data
    ):

        X = []
        y = []

        for i in range(
            LOOKBACK,
            len(data)
        ):

            X.append(
                data[
                    i - LOOKBACK:i
                ]
            )

            y.append(
                data[i]
            )

        return (
            np.array(
                X,
                dtype=np.float32
            ),
            np.array(
                y,
                dtype=np.float32
            )
        )

    # Final training sequences
    X_train, y_train = (
        create_sequences(
            train_val_scaled
        )
    )

    # --------------------------------------------------------
    # Test sequences
    #
    # Include the final 30 observations from
    # Train + Validation as context.
    # --------------------------------------------------------

    test_context = np.concatenate(
        [
            train_val_scaled[
                -LOOKBACK:
            ],
            test_scaled
        ]
    )

    X_test, y_test = (
        create_sequences(
            test_context
        )
    )

    print()
    print("Final sequence shapes:")

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"y_train: {y_train.shape}"
    )

    print(
        f"X_test:  {X_test.shape}"
    )

    print(
        f"y_test:  {y_test.shape}"
    )

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        test_dates,
        train_mean,
        train_std
    )


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model,
    train_loader,
    device
):

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # There is NO validation set here because the model
    # configuration has already been selected.
    #
    # We therefore train on Train + Validation.
    #
    # Early stopping is replaced by a fixed maximum number
    # of epochs based on the selected experimental setup.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL N-BEATS TRAINING")
    print("=" * 70)

    best_state = None
    best_loss = float("inf")

    # We retain the best training-loss state.
    # This is NOT test-based selection.
    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        losses = []

        for (
            X_batch,
            y_batch
        ) in train_loader:

            X_batch = X_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
            )

            optimizer.zero_grad()

            predictions = model(
                X_batch
            )

            loss = criterion(
                predictions.squeeze(-1),
                y_batch
            )

            loss.backward()

            optimizer.step()

            losses.append(
                loss.item()
            )

        epoch_loss = np.mean(
            losses
        )

        print(
            f"Epoch {epoch:03d}/{MAX_EPOCHS} | "
            f"Train Loss: {epoch_loss:.6f}"
        )

        if epoch_loss < best_loss:

            best_loss = epoch_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

    model.load_state_dict(
        best_state
    )

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict(
    model,
    X,
    device
):

    model.eval()

    dataset = TensorDataset(
        torch.tensor(
            X,
            dtype=torch.float32
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    predictions = []

    with torch.no_grad():

        for (X_batch,) in loader:

            X_batch = X_batch.to(
                device
            )

            output = model(
                X_batch
            )

            predictions.extend(
                output.squeeze(-1)
                .cpu()
                .numpy()
            )

    return np.array(
        predictions
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        RANDOM_SEED
    )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 70)
    print("FINAL CANONICAL N-BEATS")
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    print(
        f"Lookback: {LOOKBACK}"
    )

    print(
        f"Forecast horizon: "
        f"{FORECAST_HORIZON}"
    )

    print(
        f"Hidden size: "
        f"{HIDDEN_SIZE}"
    )

    print(
        f"Blocks: {NUM_BLOCKS}"
    )

    print(
        f"Layers per block: "
        f"{NUM_LAYERS}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_test,
        y_test,
        test_dates,
        train_mean,
        train_std
    ) = load_data()

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32
        ),
        torch.tensor(
            y_train,
            dtype=torch.float32
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = NBeats(
        input_size=LOOKBACK,
        forecast_size=FORECAST_HORIZON,
        hidden_size=HIDDEN_SIZE,
        num_blocks=NUM_BLOCKS,
        num_layers=NUM_LAYERS
    ).to(device)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        f"Trainable parameters: "
        f"{total_parameters:,}"
    )

    # --------------------------------------------------------
    # FINAL TRAINING
    # --------------------------------------------------------

    model = train_model(
        model,
        train_loader,
        device
    )

    # --------------------------------------------------------
    # SAVE FINAL MODEL
    # --------------------------------------------------------

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "lookback":
                LOOKBACK,

            "forecast_horizon":
                FORECAST_HORIZON,

            "hidden_size":
                HIDDEN_SIZE,

            "num_blocks":
                NUM_BLOCKS,

            "num_layers":
                NUM_LAYERS,

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "features":
                ["Close"],

            "train_mean":
                train_mean,

            "train_std":
                train_std
        },
        MODEL_PATH
    )

    print()
    print(
        f"Final model saved to:\n"
        f"{MODEL_PATH}"
    )

    # --------------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------------

    test_predictions_scaled = predict(
        model,
        X_test,
        device
    )

    # Inverse scaling
    test_predictions = (
        test_predictions_scaled
        * train_std
        + train_mean
    )

    y_test_actual = (
        y_test
        * train_std
        + train_mean
    )

    # --------------------------------------------------------
    # FINAL TEST METRICS
    # --------------------------------------------------------

    test_mae, test_rmse, test_mape = (
        calculate_metrics(
            y_test_actual,
            test_predictions
        )
    )

    print()
    print("=" * 70)
    print("FINAL N-BEATS TEST RESULTS")
    print("=" * 70)

    print(
        f"MAE:  {test_mae:.2f}"
    )

    print(
        f"RMSE: {test_rmse:.2f}"
    )

    print(
        f"MAPE: {test_mape:.2f}%"
    )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    predictions_df = pd.DataFrame(
        {
            "Date":
                test_dates,

            "Actual":
                y_test_actual,

            "Predicted":
                test_predictions,

            "Error":
                (
                    y_test_actual
                    - test_predictions
                )
        }
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        [
            {
                "Model":
                    "Final Canonical N-BEATS",

                "Features":
                    "Close",

                "Lookback":
                    LOOKBACK,

                "Forecast_Horizon":
                    FORECAST_HORIZON,

                "Hidden_Size":
                    HIDDEN_SIZE,

                "Blocks":
                    NUM_BLOCKS,

                "Layers_Per_Block":
                    NUM_LAYERS,

                "Batch_Size":
                    BATCH_SIZE,

                "Learning_Rate":
                    LEARNING_RATE,

                "Train_Observations":
                    len(X_train),

                "Test_Observations":
                    len(X_test),

                "Test_MAE":
                    test_mae,

                "Test_RMSE":
                    test_rmse,

                "Test_MAPE":
                    test_mape
            }
        ]
    )

    metrics_df.to_csv(
        METRICS_PATH,
        index=False
    )

    print()
    print(
        f"Predictions saved to:\n"
        f"{PREDICTIONS_PATH}"
    )

    print(
        f"\nMetrics saved to:\n"
        f"{METRICS_PATH}"
    )

    print()
    print("=" * 70)
    print("FINAL N-BEATS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()