from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCES = {
    "LSTM": {"path": "LSTM_Light_GBM/results/lstm_predictions.csv", "date": "Date", "actual": "Actual_Close", "prediction": "LSTM_Prediction", "color": "#00C2A8"},
    "LightGBM": {"path": "LSTM_Light_GBM/results/lightgbm_predictions.csv", "date": "Date", "actual": "Actual_Close", "prediction": "LightGBM_Prediction", "color": "#FFB000"},
    "Prophet": {"path": "Prophet_HistGradientBoosting/outputs/prophet_final_results.csv", "date": "ds", "actual": "y", "prediction": "predicted", "color": "#A78BFA"},
    "HistGradientBoosting": {"path": "Prophet_HistGradientBoosting/outputs/histgradientboosting_final_results.csv", "date": "Date", "actual": "Target", "prediction": "Predicted", "color": "#F472B6"},
    "TFT": {"path": "TFT_NBEATS/results/tft/final_tft_predictions.csv", "date": "Date", "actual": "Actual_Close", "prediction": "Predicted_Close", "color": "#60A5FA"},
    "N-BEATS": {"path": "TFT_NBEATS/results/nbeats/final_nbeats_predictions.csv", "date": "Date", "actual": "Actual", "prediction": "Predicted", "color": "#FB7185"},
}
DEFAULT_WEIGHTS = {"mae": .35, "rmse": .25, "mape": .20, "directional_accuracy": .20}
