# Stock Transformer Predictor - Usage Guide

## Prerequisites

First, make sure you have the required packages installed:

```bash
# If using conda (recommended)
conda install -c conda-forge yfinance
conda install pytorch torchvision torchaudio -c pytorch
conda install pandas numpy scikit-learn matplotlib seaborn

# Or if using pip
pip install torch pandas numpy yfinance scikit-learn matplotlib seaborn
```

## Quick Start (Demo Mode)

Run the complete pipeline with a smaller dataset:

```bash
python demo.py
```

This will:
1. Fetch 1 year of AAPL data
2. Train the model for 10 epochs (demo mode)
3. Evaluate the model and generate results

## Manual Usage (Step by Step)

### Step 1: Fetch Stock Data

Download historical data and calculate technical indicators:

```bash
# Basic usage
python data/fetch_data.py --symbol AAPL --start_date 2020-01-01 --end_date 2024-01-01

# Save to specific file
python data/fetch_data.py --symbol AAPL --start_date 2020-01-01 --end_date 2024-01-01 --output data/aapl_data.csv

# Try different stocks
python data/fetch_data.py --symbol TSLA --start_date 2020-01-01 --end_date 2024-01-01 --output data/tsla_data.csv
python data/fetch_data.py --symbol MSFT --start_date 2020-01-01 --end_date 2024-01-01 --output data/msft_data.csv
```

**What this does:**
- Downloads OHLCV data from Yahoo Finance
- Calculates MACD, RSI, returns, volatility
- Creates binary target (next day up/down)
- Saves processed data to CSV file

### Step 2: Train the Model

Train the transformer model on your data:

```bash
# Basic training
python train.py --symbol AAPL --epochs 100 --batch_size 32

# Use specific data file
python train.py --data_path data/aapl_data.csv --epochs 100 --batch_size 32

# Customize model parameters
python train.py --data_path data/aapl_data.csv --epochs 200 --batch_size 64 --lr 0.0005 --d_model 256

# Quick demo training (fewer epochs)
python train.py --data_path data/aapl_data.csv --epochs 10 --batch_size 16
```

**Training Parameters:**
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--d_model`: Model dimension (default: 128)
- `--nhead`: Number of attention heads (default: 8)
- `--num_layers`: Number of transformer layers (default: 4)

**What this does:**
- Loads and preprocesses the data
- Creates train/validation/test splits
- Trains the transformer model
- Saves the best model based on validation accuracy
- Generates training plots and metrics

### Step 3: Evaluate the Model

Evaluate the trained model and calculate trading performance:

```bash
# Basic evaluation
python evaluate.py --model_path models/stock_transformer.pth --symbol AAPL

# Use specific data file
python evaluate.py --model_path models/stock_transformer.pth --data_path data/aapl_data.csv

# Custom output directory
python evaluate.py --model_path models/stock_transformer.pth --data_path data/aapl_data.csv --output_dir my_results
```

**What this does:**
- Loads the trained model
- Evaluates on test data
- Calculates classification metrics (accuracy, precision, recall)
- Calculates trading metrics (Sharpe ratio, returns, win rate)
- Generates evaluation plots and reports

## Example Workflows

### Workflow 1: Quick Test with AAPL

```bash
# 1. Fetch data
python data/fetch_data.py --symbol AAPL --start_date 2023-01-01 --end_date 2024-01-01 --output data/aapl_test.csv

# 2. Train model (quick demo)
python train.py --data_path data/aapl_test.csv --epochs 20 --batch_size 16 --model_save_path models/aapl_demo.pth

# 3. Evaluate model
python evaluate.py --model_path models/aapl_demo.pth --data_path data/aapl_test.csv --output_dir aapl_results
```

### Workflow 2: Full Training with Multiple Stocks

```bash
# 1. Fetch data for multiple stocks
python data/fetch_data.py --symbol AAPL --start_date 2020-01-01 --end_date 2024-01-01 --output data/aapl_full.csv
python data/fetch_data.py --symbol TSLA --start_date 2020-01-01 --end_date 2024-01-01 --output data/tsla_full.csv
python data/fetch_data.py --symbol MSFT --start_date 2020-01-01 --end_date 2024-01-01 --output data/msft_full.csv

# 2. Train models
python train.py --data_path data/aapl_full.csv --epochs 100 --batch_size 32 --model_save_path models/aapl_model.pth
python train.py --data_path data/tsla_full.csv --epochs 100 --batch_size 32 --model_save_path models/tsla_model.pth
python train.py --data_path data/msft_full.csv --epochs 100 --batch_size 32 --model_save_path models/msft_model.pth

# 3. Evaluate all models
python evaluate.py --model_path models/aapl_model.pth --data_path data/aapl_full.csv --output_dir aapl_eval
python evaluate.py --model_path models/tsla_model.pth --data_path data/tsla_full.csv --output_dir tsla_eval
python evaluate.py --model_path models/msft_model.pth --data_path data/msft_full.csv --output_dir msft_eval
```

## Understanding the Output

### Training Output
- `training_results.json`: Training metrics and history
- `plots/training_history.png`: Training/validation loss and accuracy
- `plots/confusion_matrix.png`: Confusion matrix for test predictions
- `models/stock_transformer.pth`: Saved model file

### Evaluation Output
- `evaluation_results.json`: Classification and trading metrics
- `confusion_matrix.png`: Confusion matrix
- `roc_curve.png`: ROC curve
- `precision_recall_curve.png`: Precision-recall curve
- `cumulative_returns.png`: Strategy vs buy-and-hold returns

### Key Metrics to Look For

**Classification Metrics:**
- Accuracy: Overall prediction accuracy
- Precision: How many predicted ups were actually ups
- Recall: How many actual ups were predicted correctly
- F1-Score: Harmonic mean of precision and recall

**Trading Metrics:**
- Total Return: Strategy's total return
- Sharpe Ratio: Risk-adjusted return (higher is better)
- Max Drawdown: Largest peak-to-trough decline
- Win Rate: Percentage of profitable trades

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'yfinance'**
   ```bash
   conda install -c conda-forge yfinance
   # or
   pip install yfinance
   ```

2. **CUDA out of memory**
   - Reduce batch size: `--batch_size 16`
   - Use CPU: The code automatically uses CPU if CUDA is not available

3. **Not enough data**
   - Use longer date ranges: `--start_date 2019-01-01`
   - Reduce sequence length in the model (requires code modification)

4. **Poor performance**
   - Increase training epochs: `--epochs 200`
   - Try different learning rates: `--lr 0.0005`
   - Use more data (longer date ranges)

### Performance Tips

1. **For faster training:**
   - Use smaller models: `--d_model 64 --num_layers 2`
   - Reduce epochs: `--epochs 50`
   - Use smaller batch size: `--batch_size 16`

2. **For better accuracy:**
   - Use more data (longer date ranges)
   - Increase model size: `--d_model 256 --num_layers 6`
   - Train longer: `--epochs 200`

3. **For different stocks:**
   - Some stocks may need different parameters
   - Try adjusting learning rate and model size
   - Consider the stock's volatility and trading patterns

## Advanced Usage

### Custom Model Parameters

You can modify the model architecture by editing `models/transformer_model.py`:

```python
# Change model dimensions
model = create_model(
    input_dim=15,      # Number of features
    d_model=256,       # Model dimension (default: 128)
    nhead=16,          # Attention heads (default: 8)
    num_layers=6,      # Transformer layers (default: 4)
    dropout=0.2        # Dropout rate (default: 0.1)
)
```

### Adding New Features

To add new technical indicators, modify `data/fetch_data.py`:

```python
def calculate_technical_indicators(df):
    # Add your new indicators here
    df['New_Indicator'] = calculate_new_indicator(df['Close'])
    
    # Don't forget to update the feature list in transformer_model.py
    return df
```

### Custom Trading Strategy

Modify the trading logic in `evaluate.py`:

```python
def calculate_trading_metrics(predictions, actual_returns):
    # Customize your trading strategy here
    # For example, only trade when confidence is high
    strategy_returns = predictions * actual_returns
    return metrics
```

## Disclaimer

This system is for educational purposes only. Stock prediction is inherently difficult and past performance does not guarantee future results. Always do your own research and consider consulting with financial advisors before making investment decisions. 