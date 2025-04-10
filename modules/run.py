# run.py
import os

def run_all():
    print("🚀 Running all scripts...")

    # Initialize database schema
    os.system("python db_init.py")

    # Run labeling process
    os.system("python labeling/label_batch.py")

    # Apply risk filters
    os.system("python risk_filter.py")

if __name__ == "__main__":
    run_all()
