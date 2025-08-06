#!/usr/bin/env python3
"""
Advanced Feature Engineering for Stock Prediction
Adds sophisticated features to improve model accuracy
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def calculate_market_regime_features(df, window=60):
    """Calculate market regime features"""
    # Volatility regime
    df['Volatility_Regime'] = df['Volatility'].rolling(window=window).mean()
    df['Volatility_Regime_Change'] = df['Volatility'] - df['Volatility_Regime']
    
    # Trend regime
    df['Price_Trend'] = df['Close'].rolling(window=window).apply(
        lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1
    )
    
    # Momentum regime
    df['Momentum_Regime'] = df['Returns'].rolling(window=window).sum()
    df['Momentum_Regime_MA'] = df['Momentum_Regime'].rolling(window=20).mean()
    
    # Volume regime
    df['Volume_Regime'] = df['Volume'].rolling(window=window).mean()
    df['Volume_Regime_Ratio'] = df['Volume'] / df['Volume_Regime']
    
    return df

def calculate_sentiment_features(df):
    """Calculate sentiment-like features"""
    # Price momentum indicators
    df['Price_Momentum_1d'] = df['Close'].pct_change(1)
    df['Price_Momentum_3d'] = df['Close'].pct_change(3)
    df['Price_Momentum_5d'] = df['Close'].pct_change(5)
    df['Price_Momentum_10d'] = df['Close'].pct_change(10)
    
    # Volume momentum
    df['Volume_Momentum_1d'] = df['Volume'].pct_change(1)
    df['Volume_Momentum_3d'] = df['Volume'].pct_change(3)
    
    # Volatility momentum
    df['Volatility_Momentum'] = df['Volatility'].pct_change(1)
    
    # RSI momentum
    df['RSI_Momentum'] = df['RSI'].diff()
    
    return df

def calculate_statistical_features(df, window=20):
    """Calculate statistical features"""
    # Z-score features
    df['Price_ZScore'] = (df['Close'] - df['Close'].rolling(window=window).mean()) / df['Close'].rolling(window=window).std()
    df['Volume_ZScore'] = (df['Volume'] - df['Volume'].rolling(window=window).mean()) / df['Volume'].rolling(window=window).std()
    df['RSI_ZScore'] = (df['RSI'] - df['RSI'].rolling(window=window).mean()) / df['RSI'].rolling(window=window).std()
    
    # Percentile features
    df['Price_Percentile'] = df['Close'].rolling(window=window).apply(
        lambda x: stats.percentileofscore(x, x.iloc[-1])
    )
    df['Volume_Percentile'] = df['Volume'].rolling(window=window).apply(
        lambda x: stats.percentileofscore(x, x.iloc[-1])
    )
    
    # Skewness and kurtosis
    df['Returns_Skewness'] = df['Returns'].rolling(window=window).skew()
    df['Returns_Kurtosis'] = df['Returns'].rolling(window=window).kurt()
    
    return df

def calculate_cross_asset_features(df):
    """Calculate cross-asset correlation features"""
    # Price vs moving averages
    df['Price_vs_MA5'] = df['Close'] / df['Close'].rolling(window=5).mean() - 1
    df['Price_vs_MA10'] = df['Close'] / df['Close'].rolling(window=10).mean() - 1
    df['Price_vs_MA20'] = df['Close'] / df['Close'].rolling(window=20).mean() - 1
    df['Price_vs_MA50'] = df['Close'] / df['Close'].rolling(window=50).mean() - 1
    
    # Volume vs price correlation
    df['Volume_Price_Corr'] = df['Volume'].rolling(window=20).corr(df['Close'])
    
    # RSI vs price correlation
    df['RSI_Price_Corr'] = df['RSI'].rolling(window=20).corr(df['Close'])
    
    return df

def calculate_pattern_features(df):
    """Calculate pattern recognition features"""
    # Doji pattern (open ≈ close)
    df['Doji_Pattern'] = ((abs(df['Open'] - df['Close']) / (df['High'] - df['Low'])) < 0.1).astype(int)
    
    # Hammer pattern
    df['Hammer_Pattern'] = (
        (df['Close'] > df['Open']) & 
        ((df['High'] - df['Low']) > 3 * (df['Close'] - df['Open'])) &
        ((df['Close'] - df['Low']) > 0.6 * (df['High'] - df['Low']))
    ).astype(int)
    
    # Engulfing pattern
    df['Bullish_Engulfing'] = (
        (df['Close'] > df['Open']) & 
        (df['Close'].shift(1) < df['Open'].shift(1)) &
        (df['Open'] < df['Close'].shift(1)) &
        (df['Close'] > df['Open'].shift(1))
    ).astype(int)
    
    df['Bearish_Engulfing'] = (
        (df['Close'] < df['Open']) & 
        (df['Close'].shift(1) > df['Open'].shift(1)) &
        (df['Open'] > df['Close'].shift(1)) &
        (df['Close'] < df['Open'].shift(1))
    ).astype(int)
    
    return df

def calculate_advanced_technical_features(df):
    """Calculate advanced technical indicators"""
    # Aroon Oscillator
    def aroon_oscillator(high, low, period=14):
        aroon_up = high.rolling(window=period).apply(lambda x: x.argmax()) / period * 100
        aroon_down = low.rolling(window=period).apply(lambda x: x.argmin()) / period * 100
        return aroon_up - aroon_down
    
    df['Aroon_Oscillator'] = aroon_oscillator(df['High'], df['Low'])
    
    # Commodity Channel Index (CCI)
    def cci(high, low, close, period=20):
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma) / (0.015 * mad)
    
    df['CCI'] = cci(df['High'], df['Low'], df['Close'])
    
    # Money Flow Index (MFI)
    def mfi(high, low, close, volume, period=14):
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi
    
    df['MFI'] = mfi(df['High'], df['Low'], df['Close'], df['Volume'])
    
    return df

def calculate_time_features(df):
    """Calculate time-based features"""
    # Convert date to datetime if needed
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Day of week
            df['Day_of_Week'] = df['Date'].dt.dayofweek
            
            # Month
            df['Month'] = df['Date'].dt.month
            
            # Quarter
            df['Quarter'] = df['Date'].dt.quarter
            
            # Day of year
            df['Day_of_Year'] = df['Date'].dt.dayofyear
            
            # Week of year
            df['Week_of_Year'] = df['Date'].dt.isocalendar().week
            
            # Is month end
            df['Is_Month_End'] = df['Date'].dt.is_month_end.astype(int)
            
            # Is quarter end
            df['Is_Quarter_End'] = df['Date'].dt.is_quarter_end.astype(int)
        except:
            # If date conversion fails, create simple time features
            df['Day_of_Week'] = 0
            df['Month'] = 1
            df['Quarter'] = 1
            df['Day_of_Year'] = 1
            df['Week_of_Year'] = 1
            df['Is_Month_End'] = 0
            df['Is_Quarter_End'] = 0
    
    return df

def create_advanced_features(df):
    """Create all advanced features"""
    print("Creating advanced features...")
    
    # Market regime features
    df = calculate_market_regime_features(df)
    print("✓ Market regime features added")
    
    # Sentiment features
    df = calculate_sentiment_features(df)
    print("✓ Sentiment features added")
    
    # Statistical features
    df = calculate_statistical_features(df)
    print("✓ Statistical features added")
    
    # Cross-asset features
    df = calculate_cross_asset_features(df)
    print("✓ Cross-asset features added")
    
    # Pattern features
    df = calculate_pattern_features(df)
    print("✓ Pattern features added")
    
    # Advanced technical features
    df = calculate_advanced_technical_features(df)
    print("✓ Advanced technical features added")
    
    # Time features
    df = calculate_time_features(df)
    print("✓ Time features added")
    
    # Remove any infinite or NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    print(f"Total features: {len(df.columns)}")
    return df

def select_best_features(df, target_col='Target', n_features=50):
    """Select the best features using correlation and importance"""
    from sklearn.feature_selection import SelectKBest, f_classif
    
    # Remove non-numeric columns and target
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in feature_cols if col != target_col]
    
    # Prepare data
    X = df[feature_cols].fillna(0)
    y = df[target_col]
    
    # Select best features
    selector = SelectKBest(score_func=f_classif, k=min(n_features, len(feature_cols)))
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    selected_features = X.columns[selector.get_support()].tolist()
    
    print(f"Selected {len(selected_features)} best features")
    return selected_features

if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("data/aapl_enhanced.csv")
    df = create_advanced_features(df)
    
    # Select best features
    best_features = select_best_features(df)
    print("Best features:", best_features)
    
    # Save enhanced dataset
    df.to_csv("data/aapl_advanced.csv", index=False)
    print("Advanced features saved to data/aapl_advanced.csv") 