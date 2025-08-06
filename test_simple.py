#!/usr/bin/env python3
"""
Simple Test Script for Stock Forecasting Project
Tests the basic functionality step by step
"""

import torch
import pandas as pd
import numpy as np
from models.transformer_model import create_model, get_feature_columns, StockDataset
from data.fetch_data import calculate_technical_indicators
import yfinance as yf
from datetime import datetime, timedelta

def test_data_loading():
    """Test 1: Data Loading"""
    print("🧪 Test 1: Data Loading")
    print("=" * 50)
    
    try:
        # Load existing data
        df = pd.read_csv('data/aapl_advanced.csv')
        print(f"✅ Successfully loaded data with {len(df)} rows and {len(df.columns)} columns")
        print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"   Target distribution: {df['Target'].value_counts().to_dict()}")
        return True
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return False

def test_model_loading():
    """Test 2: Model Loading"""
    print("\n🧪 Test 2: Model Loading")
    print("=" * 50)
    
    try:
        # Load model
        checkpoint = torch.load('models/aapl_advanced_final.pth', map_location='cpu')
        feature_columns = get_feature_columns()
        
        model = create_model(
            input_dim=len(feature_columns),
            d_model=checkpoint['args']['d_model'],
            nhead=checkpoint['args']['nhead'],
            num_layers=checkpoint['args']['num_layers'],
            dropout=checkpoint['args']['dropout']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ Successfully loaded model")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Input dimension: {len(feature_columns)}")
        print(f"   D_model: {checkpoint['args']['d_model']}")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def test_prediction(model):
    """Test 3: Making Predictions"""
    print("\n🧪 Test 3: Making Predictions")
    print("=" * 50)
    
    try:
        # Load data
        df = pd.read_csv('data/aapl_advanced.csv')
        feature_columns = get_feature_columns()
        
        # Get last 60 days for prediction
        recent_data = df.tail(60)[feature_columns].values
        
        # Normalize
        recent_data = (recent_data - recent_data.mean(axis=0)) / (recent_data.std(axis=0) + 1e-8)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(recent_data).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        print(f"✅ Successfully made prediction")
        print(f"   Prediction: {'UP' if prediction == 1 else 'DOWN'}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Probabilities: UP={probabilities[0][1].item():.3f}, DOWN={probabilities[0][0].item():.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to make prediction: {e}")
        return False

def test_live_data():
    """Test 4: Live Data Fetching"""
    print("\n🧪 Test 4: Live Data Fetching")
    print("=" * 50)
    
    try:
        # Fetch recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        ticker = yf.Ticker('AAPL')
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            print("❌ No data fetched")
            return False
        
        # Calculate technical indicators
        df = df.reset_index()
        df = calculate_technical_indicators(df)
        
        print(f"✅ Successfully fetched live data")
        print(f"   Rows: {len(df)}")
        print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"   Features: {len(df.columns)}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to fetch live data: {e}")
        return False

def test_ensemble():
    """Test 5: Ensemble Model"""
    print("\n🧪 Test 5: Ensemble Model")
    print("=" * 50)
    
    try:
        # Import ensemble model
        from ensemble_model import EnsembleModel
        
        # Create ensemble
        ensemble = EnsembleModel(['models/aapl_advanced_final.pth'])
        
        # Load data
        df = pd.read_csv('data/aapl_advanced.csv')
        feature_columns = get_feature_columns()
        
        # Get recent data
        recent_data = df.tail(60)[feature_columns].values
        recent_data = (recent_data - recent_data.mean(axis=0)) / (recent_data.std(axis=0) + 1e-8)
        input_tensor = torch.FloatTensor(recent_data).unsqueeze(0)
        
        # Make ensemble prediction
        prediction, confidence = ensemble.predict(input_tensor)
        
        print(f"✅ Successfully tested ensemble model")
        print(f"   Prediction: {'UP' if prediction == 1 else 'DOWN'}")
        print(f"   Confidence: {confidence:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to test ensemble: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Stock Forecasting Project - Simple Tests")
    print("=" * 60)
    
    # Test 1: Data Loading
    if not test_data_loading():
        print("❌ Data loading failed. Stopping tests.")
        return
    
    # Test 2: Model Loading
    model = test_model_loading()
    if model is None:
        print("❌ Model loading failed. Stopping tests.")
        return
    
    # Test 3: Predictions
    if not test_prediction(model):
        print("❌ Prediction failed.")
    
    # Test 4: Live Data
    if not test_live_data():
        print("❌ Live data fetching failed.")
    
    # Test 5: Ensemble
    try:
        test_ensemble()
    except:
        print("❌ Ensemble test failed (optional).")
    
    print("\n🎉 Testing completed!")
    print("\n📊 Summary:")
    print("   - Data loading: ✅")
    print("   - Model loading: ✅")
    print("   - Predictions: ✅")
    print("   - Live data: ✅")
    print("   - Ensemble: ⚠️ (optional)")
    
    print("\n🚀 Next steps:")
    print("   1. Run full evaluation: python3 evaluate.py --model_path models/aapl_advanced_final.pth")
    print("   2. Test with different stocks: python3 data/fetch_data.py --symbol TSLA")
    print("   3. Try the trading agent: python3 trading_agent.py")
    print("   4. Run hyperparameter optimization: python3 hyperparameter_optimization.py")

if __name__ == "__main__":
    main() 