#!/usr/bin/env python3
"""
Ensemble Model for Stock Prediction
Combines multiple transformer models for better accuracy
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

from models.transformer_model import StockDataset, create_model, get_feature_columns

class EnsembleModel:
    """Ensemble model combining multiple approaches"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.transformer_models = []
        self.ml_models = []
        self.feature_columns = get_feature_columns()
        
    def add_transformer_model(self, model_path, device='cpu'):
        """Add a transformer model to the ensemble"""
        checkpoint = torch.load(model_path, map_location=device)
        
        model = create_model(
            input_dim=len(self.feature_columns),
            d_model=checkpoint['args']['d_model'],
            nhead=checkpoint['args']['nhead'],
            num_layers=checkpoint['args']['num_layers'],
            dropout=checkpoint['args']['dropout']
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        self.transformer_models.append({
            'model': model,
            'device': device,
            'weight': 1.0
        })
    
    def add_ml_model(self, model, weight=1.0):
        """Add a machine learning model to the ensemble"""
        self.ml_models.append({
            'model': model,
            'weight': weight
        })
    
    def predict_ensemble(self, dataloader, device='cpu'):
        """Make ensemble predictions"""
        all_predictions = []
        all_weights = []
        
        # Get transformer predictions
        for transformer in self.transformer_models:
            predictions = self._predict_transformer(transformer['model'], dataloader, transformer['device'])
            all_predictions.append(predictions)
            all_weights.append(transformer['weight'])
        
        # Get ML model predictions
        for ml_model in self.ml_models:
            predictions = self._predict_ml(ml_model['model'], dataloader)
            all_predictions.append(predictions)
            all_weights.append(ml_model['weight'])
        
        # Weighted voting
        ensemble_predictions = self._weighted_vote(all_predictions, all_weights)
        return ensemble_predictions
    
    def _predict_transformer(self, model, dataloader, device):
        """Get predictions from transformer model"""
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(device)
                output = model(data)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1)
                predictions.extend(pred.cpu().numpy())
        
        return np.array(predictions)
    
    def _predict_ml(self, model, dataloader):
        """Get predictions from ML model"""
        predictions = []
        
        for data, _ in dataloader:
            # Convert to 2D array for ML models
            batch_data = data.reshape(data.shape[0], -1).numpy()
            pred = model.predict(batch_data)
            predictions.extend(pred)
        
        return np.array(predictions)
    
    def _weighted_vote(self, predictions_list, weights):
        """Perform weighted voting"""
        weighted_sum = np.zeros(len(predictions_list[0]))
        
        for predictions, weight in zip(predictions_list, weights):
            weighted_sum += predictions * weight
        
        # Normalize weights
        total_weight = sum(weights)
        weighted_sum /= total_weight
        
        # Convert to binary predictions
        ensemble_predictions = (weighted_sum > 0.5).astype(int)
        return ensemble_predictions

def train_ensemble_models(data_path, output_dir='ensemble_models'):
    """Train multiple models for ensemble"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(data_path)
    feature_columns = get_feature_columns()
    dataset = StockDataset(df, sequence_length=60, feature_columns=feature_columns)
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Prepare data for ML models
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    
    for idx in train_dataset.indices:
        seq, target = dataset[idx]
        X_train.append(seq.flatten().numpy())
        y_train.append(target.item())
    
    for idx in val_dataset.indices:
        seq, target = dataset[idx]
        X_val.append(seq.flatten().numpy())
        y_val.append(target.item())
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    # Train ML models
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, f'{output_dir}/random_forest.pkl')
    
    print("Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42)
    gb_model.fit(X_train, y_train)
    joblib.dump(gb_model, f'{output_dir}/gradient_boosting.pkl')
    
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    joblib.dump(lr_model, f'{output_dir}/logistic_regression.pkl')
    
    # Evaluate individual models
    rf_pred = rf_model.predict(X_val)
    gb_pred = gb_model.predict(X_val)
    lr_pred = lr_model.predict(X_val)
    
    print(f"Random Forest Accuracy: {accuracy_score(y_val, rf_pred):.4f}")
    print(f"Gradient Boosting Accuracy: {accuracy_score(y_val, gb_pred):.4f}")
    print(f"Logistic Regression Accuracy: {accuracy_score(y_val, lr_pred):.4f}")
    
    return output_dir

def create_ensemble_predictions(model_paths, data_path, output_dir='ensemble_results'):
    """Create ensemble predictions"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(data_path)
    feature_columns = get_feature_columns()
    dataset = StockDataset(df, sequence_length=60, feature_columns=feature_columns)
    
    # Use last 20% for evaluation
    eval_size = int(0.2 * len(dataset))
    eval_dataset = torch.utils.data.Subset(dataset, range(len(dataset) - eval_size, len(dataset)))
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=32, shuffle=False)
    
    # Create ensemble
    ensemble = EnsembleModel()
    
    # Add transformer models
    for model_path in model_paths:
        if model_path.endswith('.pth'):
            ensemble.add_transformer_model(model_path)
    
    # Add ML models
    ml_models = ['random_forest.pkl', 'gradient_boosting.pkl', 'logistic_regression.pkl']
    for ml_model in ml_models:
        model_path = f'ensemble_models/{ml_model}'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            ensemble.add_ml_model(model)
    
    # Make ensemble predictions
    predictions = ensemble.predict_ensemble(eval_loader)
    
    # Get actual targets
    targets = []
    for _, target in eval_dataset:
        targets.append(target.item())
    targets = np.array(targets)
    
    # Calculate accuracy
    accuracy = accuracy_score(targets, predictions)
    print(f"Ensemble Accuracy: {accuracy:.4f}")
    
    # Save results
    results = {
        'accuracy': accuracy,
        'predictions': predictions.tolist(),
        'targets': targets.tolist()
    }
    
    import json
    with open(f'{output_dir}/ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return accuracy

if __name__ == "__main__":
    # Example usage
    data_path = "data/aapl_enhanced.csv"
    
    # Train ensemble models
    print("Training ensemble models...")
    train_ensemble_models(data_path)
    
    # Create ensemble predictions
    print("Creating ensemble predictions...")
    model_paths = [
        "models/aapl_enhanced.pth",
        "models/aapl_improved.pth"
    ]
    
    accuracy = create_ensemble_predictions(model_paths, data_path)
    print(f"Final Ensemble Accuracy: {accuracy:.4f}") 