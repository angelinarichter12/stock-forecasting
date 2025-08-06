#!/usr/bin/env python3
"""
Stock Transformer Model
PyTorch Transformer for stock price movement prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class StockTransformer(nn.Module):
    """
    Transformer model for stock prediction
    
    Features:
    - OHLCV data
    - Technical indicators (MACD, RSI, returns, etc.)
    - Positional encoding
    - Binary classification output
    """
    
    def __init__(self, 
                 input_dim=15,  # Number of features
                 d_model=128,   # Model dimension
                 nhead=8,       # Number of attention heads
                 num_layers=4,  # Number of transformer layers
                 dropout=0.1,
                 max_seq_len=60):
        super(StockTransformer, self).__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Feature projection
        self.feature_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 2)  # Binary classification
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        
        Returns:
            logits: Classification logits of shape (batch_size, 2)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project features to d_model
        x = self.feature_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Create attention mask (optional, for variable length sequences)
        # For now, we assume all sequences are the same length
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # Global average pooling across sequence dimension
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        x = self.global_pool(x)  # (batch_size, d_model, 1)
        x = x.squeeze(-1)  # (batch_size, d_model)
        
        # Classification
        logits = self.classifier(x)  # (batch_size, 2)
        
        return logits

class StockDataset(torch.utils.data.Dataset):
    """Dataset for stock prediction"""
    
    def __init__(self, data, sequence_length=60, feature_columns=None):
        """
        Initialize dataset
        
        Args:
            data: DataFrame with stock data
            sequence_length: Length of input sequences
            feature_columns: List of feature column names
        """
        self.data = data
        self.sequence_length = sequence_length
        
        # Default feature columns if not specified
        if feature_columns is None:
            self.feature_columns = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'MACD', 'MACD_Signal', 'MACD_Histogram',
                'RSI', 'Returns', 'Volume_Ratio',
                'Price_MA_Ratio', 'Volatility',
                'BB_Position', 'BB_Width', 'Stoch_K', 'Stoch_D',
                'Williams_R', 'ADX', 'DI_Plus', 'DI_Minus',
                'Price_Momentum', 'Volume_Momentum',
                'Price_vs_Support', 'Price_vs_Resistance'
            ]
        else:
            self.feature_columns = feature_columns
        
        # Normalize features
        self.normalized_data = self._normalize_features()
        
        # Create sequences
        self.sequences, self.targets = self._create_sequences()
    
    def _normalize_features(self):
        """Normalize features using z-score normalization"""
        normalized = self.data[self.feature_columns].copy()
        
        for col in self.feature_columns:
            mean_val = normalized[col].mean()
            std_val = normalized[col].std()
            if std_val > 0:
                normalized[col] = (normalized[col] - mean_val) / std_val
            else:
                normalized[col] = 0
        
        return normalized
    
    def _create_sequences(self):
        """Create sequences and targets"""
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(self.normalized_data)):
            # Get sequence
            seq = self.normalized_data.iloc[i-self.sequence_length:i][self.feature_columns].values
            
            # Get target (next day movement)
            target = self.data.iloc[i-1]['Target']
            
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        target = torch.LongTensor([self.targets[idx]])
        return sequence, target.squeeze()

def create_model(input_dim=15, d_model=128, nhead=8, num_layers=4, dropout=0.1):
    """
    Create and return a StockTransformer model
    
    Args:
        input_dim: Number of input features
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
    
    Returns:
        StockTransformer: Initialized model
    """
    model = StockTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout
    )
    return model

def get_feature_columns():
    """Get the default feature columns for the model"""
    return [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'MACD', 'MACD_Signal', 'MACD_Histogram',
        'RSI', 'Returns', 'Volume_Ratio',
        'Price_MA_Ratio', 'Volatility',
        'BB_Position', 'BB_Width', 'Stoch_K', 'Stoch_D',
        'Williams_R', 'ADX', 'DI_Plus', 'DI_Minus',
        'Price_Momentum', 'Volume_Momentum',
        'Price_vs_Support', 'Price_vs_Resistance'
    ]

if __name__ == "__main__":
    # Test the model
    print("Testing StockTransformer model...")
    
    # Create a small test dataset
    batch_size = 4
    seq_len = 60
    input_dim = 15
    
    # Create random test data
    test_data = torch.randn(batch_size, seq_len, input_dim)
    
    # Create model
    model = StockTransformer(input_dim=input_dim)
    
    # Forward pass
    with torch.no_grad():
        output = model(test_data)
    
    print(f"Input shape: {test_data.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("Model test completed successfully!") 