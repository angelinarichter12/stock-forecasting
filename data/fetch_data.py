#!/usr/bin/env python3
"""
Stock Data Fetcher
Downloads OHLCV data from Yahoo Finance and calculates technical indicators
"""

import yfinance as yf
import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_rsi(prices, period=14):
    """Calculate RSI (Relative Strength Index)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_returns(prices):
    """Calculate daily returns"""
    return prices.pct_change()

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    bb_position = (prices - lower_band) / (upper_band - lower_band)
    return upper_band, lower_band, bb_position

def calculate_stochastic(prices, period=14):
    """Calculate Stochastic Oscillator"""
    low_min = prices.rolling(window=period).min()
    high_max = prices.rolling(window=period).max()
    k_percent = 100 * ((prices - low_min) / (high_max - low_min))
    d_percent = k_percent.rolling(window=3).mean()
    return k_percent, d_percent

def calculate_williams_r(high, low, close, period=14):
    """Calculate Williams %R"""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return williams_r

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)"""
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = high - high.shift(1)
    dm_minus = low.shift(1) - low
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed values
    tr_smooth = tr.rolling(window=period).mean()
    dm_plus_smooth = dm_plus.rolling(window=period).mean()
    dm_minus_smooth = dm_minus.rolling(window=period).mean()
    
    # DI values
    di_plus = 100 * (dm_plus_smooth / tr_smooth)
    di_minus = 100 * (dm_minus_smooth / tr_smooth)
    
    # ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=period).mean()
    
    return adx, di_plus, di_minus

def calculate_technical_indicators(df):
    """Calculate all technical indicators for the dataset"""
    # Calculate MACD
    macd, signal, histogram = calculate_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Histogram'] = histogram
    
    # Calculate RSI
    df['RSI'] = calculate_rsi(df['Close'])
    
    # Calculate returns
    df['Returns'] = calculate_returns(df['Close'])
    
    # Calculate volume features
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Calculate price features
    df['Price_MA_20'] = df['Close'].rolling(window=20).mean()
    df['Price_MA_50'] = df['Close'].rolling(window=50).mean()
    df['Price_MA_Ratio'] = df['Close'] / df['Price_MA_20']
    
    # Calculate volatility
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # Calculate Bollinger Bands
    upper_bb, lower_bb, bb_position = calculate_bollinger_bands(df['Close'])
    df['BB_Position'] = bb_position
    df['BB_Width'] = (upper_bb - lower_bb) / df['Close']
    
    # Calculate Stochastic Oscillator
    k_percent, d_percent = calculate_stochastic(df['Close'])
    df['Stoch_K'] = k_percent
    df['Stoch_D'] = d_percent
    
    # Calculate Williams %R
    df['Williams_R'] = calculate_williams_r(df['High'], df['Low'], df['Close'])
    
    # Calculate ADX
    adx, di_plus, di_minus = calculate_adx(df['High'], df['Low'], df['Close'])
    df['ADX'] = adx
    df['DI_Plus'] = di_plus
    df['DI_Minus'] = di_minus
    
    # Calculate momentum indicators
    df['Price_Momentum'] = df['Close'] / df['Close'].shift(5) - 1
    df['Volume_Momentum'] = df['Volume'] / df['Volume'].shift(5) - 1
    
    # Calculate support/resistance levels
    df['Support_Level'] = df['Low'].rolling(window=20).min()
    df['Resistance_Level'] = df['High'].rolling(window=20).max()
    df['Price_vs_Support'] = (df['Close'] - df['Support_Level']) / df['Close']
    df['Price_vs_Resistance'] = (df['Resistance_Level'] - df['Close']) / df['Close']
    
    # Create target variable (next day movement: 1 if up, 0 if down)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    return df

def fetch_stock_data(symbol, start_date, end_date, save_path=None):
    """
    Fetch stock data from Yahoo Finance and calculate technical indicators
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL')
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        save_path (str): Path to save the processed data
    
    Returns:
        pd.DataFrame: Processed stock data with technical indicators
    """
    print(f"Fetching data for {symbol} from {start_date} to {end_date}")
    
    try:
        # Fetch data from Yahoo Finance
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        print(f"Downloaded {len(df)} days of data")
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        # Remove rows with NaN values (from technical indicator calculations)
        df = df.dropna()
        
        print(f"Final dataset has {len(df)} rows after removing NaN values")
        
        # Save data if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"Data saved to {save_path}")
        
        return df
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Fetch stock data and calculate technical indicators')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol (default: AAPL)')
    parser.add_argument('--start_date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2024-01-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='data/stock_data.csv', help='Output file path')
    
    args = parser.parse_args()
    
    # Fetch and process data
    df = fetch_stock_data(args.symbol, args.start_date, args.end_date, args.output)
    
    if df is not None:
        print("\nDataset Summary:")
        print(f"Symbol: {args.symbol}")
        print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Total Rows: {len(df)}")
        print(f"Features: {list(df.columns)}")
        print(f"Target Distribution: {df['Target'].value_counts().to_dict()}")
        
        print("\nFirst few rows:")
        print(df.head())
        
        print("\nTechnical Indicators Summary:")
        print(df[['MACD', 'RSI', 'Returns', 'Volatility']].describe())

if __name__ == "__main__":
    main() 