# run.py
import os

def run_all():
    print("🚀 Running all scripts...")

    # Initialize database schema
    os.system("python database/db_init.py")

    # Create/Build features
    os.system("python database/build_features.py")

    # Run labeling process
    os.system("python labeling/label_batch.py")

    # Apply risk filters
    os.system("python database/risk_filter.py")

    # Run model training and tuning
    os.system("python database/ml_tuner.py")

    # Run model reflection (evaluate model performance)
    os.system("python database/ml_reflector.py")

    # Test the trained model
    os.system("python database/test_model.py")

    # Run inference on new data (using trained model)
    os.system("python database/inference.py")

    # Optionally: Run the backtest
    os.system("python database/backtest_model.py")

if __name__ == "__main__":
    run_all()
