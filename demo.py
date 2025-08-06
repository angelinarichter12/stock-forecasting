#!/usr/bin/env python3
"""
Demo Script for Stock Transformer Predictor
Runs the complete pipeline with a smaller dataset for quick testing
"""

import os
import subprocess
import sys
from datetime import datetime, timedelta

def run_command(command, description):
    """Run a command and print the description"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Error!")
        print(f"Error: {e}")
        if e.stdout:
            print("Stdout:")
            print(e.stdout)
        if e.stderr:
            print("Stderr:")
            print(e.stderr)
        return False

def main():
    print("🚀 Stock Transformer Predictor Demo")
    print("This demo will run the complete pipeline with a smaller dataset")
    
    # Set demo parameters
    symbol = "AAPL"
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")  # 1 year of data
    
    print(f"\n📊 Demo Configuration:")
    print(f"Symbol: {symbol}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Training: 10 epochs (demo mode)")
    
    # Step 1: Fetch data
    fetch_cmd = f"python3 data/fetch_data.py --symbol {symbol} --start_date {start_date} --end_date {end_date} --output data/demo_data.csv"
    if not run_command(fetch_cmd, "Step 1: Fetching stock data"):
        return
    
    # Step 2: Train model (demo mode with fewer epochs)
    train_cmd = f"python3 train.py --symbol {symbol} --data_path data/demo_data.csv --epochs 10 --batch_size 16 --model_save_path models/demo_model.pth"
    if not run_command(train_cmd, "Step 2: Training the transformer model"):
        return
    
    # Step 3: Evaluate model
    eval_cmd = f"python3 evaluate.py --model_path models/demo_model.pth --symbol {symbol} --data_path data/demo_data.csv --output_dir demo_results"
    if not run_command(eval_cmd, "Step 3: Evaluating the model"):
        return
    
    print(f"\n{'='*60}")
    print("🎉 Demo completed successfully!")
    print(f"{'='*60}")
    print("\n📁 Generated files:")
    print("  - data/demo_data.csv (stock data with technical indicators)")
    print("  - models/demo_model.pth (trained model)")
    print("  - demo_results/ (evaluation results and plots)")
    print("  - training_results.json (training metrics)")
    print("  - plots/ (training history and confusion matrix)")
    
    print("\n📈 Next steps:")
    print("  1. Check demo_results/evaluation_results.json for performance metrics")
    print("  2. View plots in demo_results/ and plots/ directories")
    print("  3. Run with more epochs for better performance:")
    print("     python train.py --epochs 100 --batch_size 32")
    print("  4. Try different stocks: python train.py --symbol TSLA")
    
    print("\n⚠️  Disclaimer:")
    print("  This is for educational purposes only. Stock prediction is inherently")
    print("  difficult and past performance does not guarantee future results.")

if __name__ == "__main__":
    main() 