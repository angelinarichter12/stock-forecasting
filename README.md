# Stock Transformer Predictor

A machine learning system for stock price movement prediction using Transformer architecture with technical indicators.

## Features

- Binary next-day movement prediction (up/down)
- Technical indicators: MACD, RSI, returns as features
- Transformer architecture with positional encoding
- Comprehensive evaluation: Accuracy, confusion matrix, Sharpe ratio
- Data fetching from Yahoo Finance using yfinance

## Project Structure

```
├── data/
│   └── fetch_data.py          # Downloads OHLCV data from yfinance
├── models/
│   └── transformer_model.py   # PyTorch Transformer model
├── train.py                   # Training loop
├── evaluate.py                # Accuracy, confusion matrix, Sharpe ratio
└── README.md                  # Project description and usage
```

## Installation

```bash
pip install torch pandas numpy yfinance scikit-learn matplotlib seaborn
```

## Usage

### 1. Fetch Data
```bash
python data/fetch_data.py --symbol AAPL --start_date 2020-01-01 --end_date 2024-01-01
```

### 2. Train Model
```bash
python train.py --symbol AAPL --epochs 100 --batch_size 32
```

### 3. Evaluate Model
```bash
python evaluate.py --model_path models/stock_transformer.pth --symbol AAPL
```

## Model Architecture

- Transformer: Small transformer with positional encoding
- Features: OHLCV data + MACD + RSI + returns
- Output: Binary classification (next day up/down)
- Sequence Length: 60 days of historical data

## Technical Indicators

- MACD: Moving Average Convergence Divergence
- RSI: Relative Strength Index
- Returns: Daily price returns
- Volume: Trading volume normalization

## Testing

Run the simple test to verify everything works:
```bash
python3 test_simple.py
```

Or run the full demo:
```bash
python3 demo.py
```

## Disclaimer

This project is for educational purposes only. Stock prediction is inherently difficult and past performance does not guarantee future results. Always do your own research and consider consulting with financial advisors before making investment decisions.

## Investment Strategy Insights

Based on the analysis framework discussed:

- Systematic approach to public equity finance analysis
- AI tools and empirical analysis for stock evaluation
- Case studies:
  - Robinhood: 25x revenue valuation analysis
  - Rolls-Royce: Aerospace engines and nuclear reactors focus
- VC insights for identifying emerging opportunities (e.g., quantum computing)

## License

MIT License - feel free to use and modify for educational purposes. 