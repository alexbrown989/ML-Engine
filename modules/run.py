# run.py
import os

def run_all():
    print("🚀 Running all scripts...")

    # Initialize the database schema (tables for signals, features, and labels)
    os.system("python database/db_init.py")

    # Build features (calculation of technical features)
    os.system("python database/build_features.py")

    # Label the data (adding labels for classification)
    os.system("python labeling/label_batch.py")

    # Apply risk filters (filter out weak signals)
    os.system("python database/risk_filter.py")

    # Train the model (XGBoost, hyperparameter tuning)
    os.system("python database/ml_tuner.py")

    # Reflect on model performance (evaluate prediction accuracy)
    os.system("python database/ml_reflector.py")

    # Test the model (evaluate on a test dataset)
    os.system("python database/test_model.py")

    # Make predictions on new data (inference)
    os.system("python database/inference.py")

    # Optionally: Run backtesting (simulate trading with historical data)
    os.system("python database/backtest_model.py")

if __name__ == "__main__":
    run_all()
