#!/usr/bin/env python3
"""
Trading Agent Structure for Stock Prediction
Autonomous trading agent that uses the transformer model
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple

from models.transformer_model import create_model, get_feature_columns
from data.fetch_data import calculate_technical_indicators

class TradingAgent:
    """Autonomous trading agent for stock prediction"""
    
    def __init__(self, model_path: str, initial_balance: float = 10000):
        self.model_path = model_path
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0  # 0: no position, 1: long position
        self.entry_price = 0
        self.trades = []
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.load_model()
    
    def load_model(self):
        """Load the trained transformer model"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        feature_columns = get_feature_columns()
        self.model = create_model(
            input_dim=len(feature_columns),
            d_model=checkpoint['args']['d_model'],
            nhead=checkpoint['args']['nhead'],
            num_layers=checkpoint['args']['num_layers'],
            dropout=checkpoint['args']['dropout']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Loaded model from {self.model_path}")
    
    def fetch_latest_data(self, symbol: str, days: int = 60) -> pd.DataFrame:
        """Fetch latest market data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 30)  # Extra buffer
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No data found for {symbol}")
        
        # Reset index and calculate technical indicators
        df = df.reset_index()
        df = calculate_technical_indicators(df)
        
        # Get last 60 days
        df = df.tail(days + 1)  # +1 for target calculation
        return df
    
    def prepare_input(self, df: pd.DataFrame) -> torch.Tensor:
        """Prepare input for the model"""
        feature_columns = get_feature_columns()
        
        # Get last 60 days (excluding today for prediction)
        input_data = df.iloc[:-1][feature_columns].values
        
        # Normalize features
        input_data = (input_data - input_data.mean(axis=0)) / (input_data.std(axis=0) + 1e-8)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(self.device)
        return input_tensor
    
    def predict(self, symbol: str) -> Dict:
        """Make prediction for a given symbol"""
        try:
            # Fetch latest data
            df = self.fetch_latest_data(symbol)
            
            # Prepare input
            input_tensor = self.prepare_input(df)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            # Get current price
            current_price = df.iloc[-1]['Close']
            
            return {
                'symbol': symbol,
                'prediction': 'UP' if prediction == 1 else 'DOWN',
                'confidence': confidence,
                'current_price': current_price,
                'timestamp': datetime.now()
            }
        
        except Exception as e:
            print(f"Error making prediction for {symbol}: {str(e)}")
            return None
    
    def execute_trade(self, symbol: str, action: str, price: float, confidence: float):
        """Execute a trade"""
        trade = {
            'symbol': symbol,
            'action': action,
            'price': price,
            'confidence': confidence,
            'timestamp': datetime.now(),
            'balance_before': self.balance
        }
        
        if action == 'BUY' and self.position == 0:
            # Calculate position size (risk management)
            position_size = min(self.balance * 0.1, self.balance)  # Max 10% of balance
            shares = int(position_size / price)
            
            if shares > 0:
                self.position = shares
                self.entry_price = price
                self.balance -= shares * price
                trade['shares'] = shares
                trade['type'] = 'BUY'
                print(f"BUY {shares} shares of {symbol} at ${price:.2f}")
        
        elif action == 'SELL' and self.position > 0:
            # Sell all shares
            shares = self.position
            self.balance += shares * price
            profit = (price - self.entry_price) * shares
            self.position = 0
            self.entry_price = 0
            
            trade['shares'] = shares
            trade['type'] = 'SELL'
            trade['profit'] = profit
            print(f"SELL {shares} shares of {symbol} at ${price:.2f}, Profit: ${profit:.2f}")
        
        self.trades.append(trade)
        return trade
    
    def get_trading_strategy(self, prediction: Dict) -> str:
        """Determine trading strategy based on prediction"""
        symbol = prediction['symbol']
        pred_direction = prediction['prediction']
        confidence = prediction['confidence']
        current_price = prediction['current_price']
        
        # Strategy logic
        if confidence < 0.6:  # Low confidence
            return 'HOLD'
        
        if pred_direction == 'UP' and self.position == 0:
            return 'BUY'
        elif pred_direction == 'DOWN' and self.position > 0:
            return 'SELL'
        else:
            return 'HOLD'
    
    def run_trading_session(self, symbols: List[str], max_trades: int = 10):
        """Run a trading session for multiple symbols"""
        print(f"Starting trading session with ${self.balance:.2f}")
        
        trades_executed = 0
        
        for symbol in symbols:
            if trades_executed >= max_trades:
                break
            
            # Get prediction
            prediction = self.predict(symbol)
            if prediction is None:
                continue
            
            # Determine action
            action = self.get_trading_strategy(prediction)
            
            if action != 'HOLD':
                # Execute trade
                trade = self.execute_trade(
                    symbol=symbol,
                    action=action,
                    price=prediction['current_price'],
                    confidence=prediction['confidence']
                )
                trades_executed += 1
            
            # Print status
            print(f"{symbol}: {prediction['prediction']} (Confidence: {prediction['confidence']:.2f}) -> {action}")
        
        # Print final status
        total_value = self.balance
        if self.position > 0:
            # Estimate current value of position
            total_value += self.position * prediction['current_price']
        
        print(f"\nTrading session completed!")
        print(f"Final balance: ${self.balance:.2f}")
        print(f"Total value: ${total_value:.2f}")
        print(f"Return: {((total_value - self.initial_balance) / self.initial_balance * 100):.2f}%")
        print(f"Trades executed: {trades_executed}")
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.trades:
            return {}
        
        # Calculate metrics
        total_trades = len(self.trades)
        profitable_trades = len([t for t in self.trades if t.get('profit', 0) > 0])
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        total_profit = sum([t.get('profit', 0) for t in self.trades])
        total_return = (self.balance - self.initial_balance) / self.initial_balance
        
        return {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_return': total_return,
            'final_balance': self.balance
        }

def main():
    # Example usage
    agent = TradingAgent('models/aapl_advanced_final.pth', initial_balance=10000)
    
    # Run trading session
    symbols = ['AAPL', 'TSLA', 'MSFT', 'GOOGL']
    agent.run_trading_session(symbols, max_trades=5)
    
    # Print performance
    metrics = agent.get_performance_metrics()
    print(f"\nPerformance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main() 