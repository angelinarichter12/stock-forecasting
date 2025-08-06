#!/usr/bin/env python3
"""
Stock Transformer Evaluation Script
Evaluates the trained model with accuracy, confusion matrix, and Sharpe ratio
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import argparse
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score

# Import our modules
from data.fetch_data import fetch_stock_data
from models.transformer_model import StockDataset, create_model, get_feature_columns

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate Sharpe ratio
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate (default: 2%)
    
    Returns:
        float: Sharpe ratio
    """
    if len(returns) == 0:
        return 0
    
    # Convert annual risk-free rate to daily
    daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1
    
    # Calculate excess returns
    excess_returns = returns - daily_rf_rate
    
    # Calculate Sharpe ratio
    if np.std(excess_returns) == 0:
        return 0
    
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    return sharpe_ratio

def calculate_max_drawdown(returns):
    """
    Calculate maximum drawdown
    
    Args:
        returns: Array of returns
    
    Returns:
        float: Maximum drawdown percentage
    """
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)
    return max_drawdown

def calculate_trading_metrics(predictions, actual_returns, confidence_scores=None):
    """
    Calculate trading performance metrics
    
    Args:
        predictions: Array of predictions (0 or 1)
        actual_returns: Array of actual returns
        confidence_scores: Array of confidence scores (optional)
    
    Returns:
        dict: Dictionary of trading metrics
    """
    # Strategy returns: buy when prediction is 1, hold when 0
    strategy_returns = predictions * actual_returns
    
    # Buy and hold returns
    buy_hold_returns = actual_returns
    
    # Calculate metrics
    total_return = np.prod(1 + strategy_returns) - 1
    buy_hold_return = np.prod(1 + buy_hold_returns) - 1
    
    sharpe_ratio = calculate_sharpe_ratio(strategy_returns)
    max_drawdown = calculate_max_drawdown(strategy_returns)
    
    # Win rate
    winning_trades = strategy_returns[strategy_returns > 0]
    total_trades = len(strategy_returns[strategy_returns != 0])
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    
    # Average win/loss
    avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
    losing_trades = strategy_returns[strategy_returns < 0]
    avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0
    
    # Profit factor
    gross_profit = np.sum(winning_trades) if len(winning_trades) > 0 else 0
    gross_loss = abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    metrics = {
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'excess_return': total_return - buy_hold_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor
    }
    
    return metrics

def plot_roc_curve(y_true, y_scores, save_path):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curve(y_true, y_scores, save_path):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_cumulative_returns(strategy_returns, buy_hold_returns, save_path):
    """Plot cumulative returns comparison"""
    strategy_cumulative = np.cumprod(1 + strategy_returns)
    buy_hold_cumulative = np.cumprod(1 + buy_hold_returns)
    
    plt.figure(figsize=(12, 6))
    plt.plot(strategy_cumulative, label='Strategy Returns', linewidth=2)
    plt.plot(buy_hold_cumulative, label='Buy & Hold Returns', linewidth=2)
    plt.xlabel('Trading Days')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Returns Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model, dataloader, device):
    """Evaluate model and return predictions and confidence scores"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_confidence_scores = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Get predictions
            probabilities = torch.softmax(output, dim=1)
            confidence_scores = torch.max(probabilities, dim=1)[0]
            predictions = torch.argmax(output, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_confidence_scores.extend(confidence_scores.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_targets), np.array(all_confidence_scores)

def main():
    parser = argparse.ArgumentParser(description='Evaluate Stock Transformer Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol')
    parser.add_argument('--start_date', type=str, default='2020-01-01', help='Start date')
    parser.add_argument('--end_date', type=str, default='2024-01-01', help='End date')
    parser.add_argument('--data_path', type=str, default='data/stock_data.csv', help='Path to data file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--sequence_length', type=int, default=60, help='Sequence length')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Load or fetch data
    if os.path.exists(args.data_path):
        print(f"Loading data from {args.data_path}")
        df = pd.read_csv(args.data_path)
    else:
        print(f"Fetching data for {args.symbol}")
        df = fetch_stock_data(args.symbol, args.start_date, args.end_date, args.data_path)
        if df is None:
            print("Failed to fetch data")
            return
    
    print(f"Dataset shape: {df.shape}")
    
    # Create dataset
    feature_columns = get_feature_columns()
    dataset = StockDataset(df, args.sequence_length, feature_columns)
    
    # Use last 20% of data for evaluation
    eval_size = int(0.2 * len(dataset))
    eval_dataset = torch.utils.data.Subset(dataset, range(len(dataset) - eval_size, len(dataset)))
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
    
    print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # Create model
    model = create_model(
        input_dim=len(feature_columns),
        d_model=checkpoint['args']['d_model'],
        nhead=checkpoint['args']['nhead'],
        num_layers=checkpoint['args']['num_layers'],
        dropout=checkpoint['args']['dropout']
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate model
    print("Evaluating model...")
    predictions, targets, confidence_scores = evaluate_model(model, eval_loader, device)
    
    # Calculate classification metrics
    accuracy = accuracy_score(targets, predictions)
    classification_rep = classification_report(targets, predictions, target_names=['Down', 'Up'], output_dict=True)
    
    print(f"\nClassification Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Up): {classification_rep['Up']['precision']:.4f}")
    print(f"Recall (Up): {classification_rep['Up']['recall']:.4f}")
    print(f"F1-Score (Up): {classification_rep['Up']['f1-score']:.4f}")
    
    # Calculate trading metrics
    print(f"\nCalculating trading metrics...")
    
    # Get actual returns for the evaluation period
    eval_data = df.iloc[len(df) - len(eval_dataset):]
    actual_returns = eval_data['Returns'].values
    
    # Calculate trading metrics
    trading_metrics = calculate_trading_metrics(predictions, actual_returns, confidence_scores)
    
    print(f"\nTrading Performance:")
    print(f"Total Return: {trading_metrics['total_return']:.4f} ({trading_metrics['total_return']*100:.2f}%)")
    print(f"Buy & Hold Return: {trading_metrics['buy_hold_return']:.4f} ({trading_metrics['buy_hold_return']*100:.2f}%)")
    print(f"Excess Return: {trading_metrics['excess_return']:.4f} ({trading_metrics['excess_return']*100:.2f}%)")
    print(f"Sharpe Ratio: {trading_metrics['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {trading_metrics['max_drawdown']:.4f} ({trading_metrics['max_drawdown']*100:.2f}%)")
    print(f"Win Rate: {trading_metrics['win_rate']:.4f} ({trading_metrics['win_rate']*100:.2f}%)")
    print(f"Total Trades: {trading_metrics['total_trades']}")
    print(f"Average Win: {trading_metrics['avg_win']:.4f}")
    print(f"Average Loss: {trading_metrics['avg_loss']:.4f}")
    print(f"Profit Factor: {trading_metrics['profit_factor']:.4f}")
    
    # Create plots
    print(f"\nCreating plots...")
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Down', 'Up'], 
                yticklabels=['Down', 'Up'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'{args.output_dir}/confusion_matrix.png')
    plt.close()
    
    # ROC curve
    plot_roc_curve(targets, confidence_scores, f'{args.output_dir}/roc_curve.png')
    
    # Precision-Recall curve
    plot_precision_recall_curve(targets, confidence_scores, f'{args.output_dir}/precision_recall_curve.png')
    
    # Cumulative returns
    strategy_returns = predictions * actual_returns
    plot_cumulative_returns(strategy_returns, actual_returns, f'{args.output_dir}/cumulative_returns.png')
    
    # Save results
    results = {
        'classification_metrics': {
            'accuracy': accuracy,
            'classification_report': classification_rep
        },
        'trading_metrics': trading_metrics,
        'model_info': {
            'symbol': args.symbol,
            'date_range': f"{args.start_date} to {args.end_date}",
            'evaluation_samples': len(eval_dataset),
            'model_parameters': checkpoint['args']
        }
    }
    
    with open(f'{args.output_dir}/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to {args.output_dir}/")
    print(f"Plots saved to {args.output_dir}/")

if __name__ == "__main__":
    main() 