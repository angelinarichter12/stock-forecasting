#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) for Stock Transformer
Fine-tunes the model on specific market conditions or recent data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import os

from models.transformer_model import StockDataset, create_model, get_feature_columns

class SFTTrainer:
    """Supervised Fine-Tuning trainer for stock prediction"""
    
    def __init__(self, base_model_path, device='cpu'):
        self.device = device
        self.base_model_path = base_model_path
        self.model = None
        
    def load_base_model(self):
        """Load the pre-trained base model"""
        checkpoint = torch.load(self.base_model_path, map_location=self.device)
        
        # Create model with same architecture
        feature_columns = get_feature_columns()
        self.model = create_model(
            input_dim=len(feature_columns),
            d_model=checkpoint['args']['d_model'],
            nhead=checkpoint['args']['nhead'],
            num_layers=checkpoint['args']['num_layers'],
            dropout=checkpoint['args']['dropout']
        ).to(self.device)
        
        # Load pre-trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded base model from {self.base_model_path}")
        
    def prepare_finetuning_data(self, data_path, recent_months=6):
        """Prepare recent data for fine-tuning"""
        df = pd.read_csv(data_path)
        
        # Use only recent data for fine-tuning
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            recent_date = df['Date'].max() - pd.DateOffset(months=recent_months)
            df_recent = df[df['Date'] >= recent_date].copy()
        else:
            # Use last N rows if no date column
            df_recent = df.tail(int(len(df) * 0.3)).copy()
        
        print(f"Fine-tuning on {len(df_recent)} recent samples")
        return df_recent
    
    def finetune(self, data_path, epochs=50, lr=1e-5, batch_size=16):
        """Fine-tune the model on recent data"""
        if self.model is None:
            self.load_base_model()
        
        # Prepare fine-tuning data
        df_finetune = self.prepare_finetuning_data(data_path)
        feature_columns = get_feature_columns()
        dataset = StockDataset(df_finetune, sequence_length=60, feature_columns=feature_columns)
        
        # Split for fine-tuning
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Use smaller learning rate for fine-tuning
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        
        best_val_acc = 0
        
        print(f"Starting fine-tuning for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target.view_as(pred)).sum().item()
                train_total += target.size(0)
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    pred = output.argmax(dim=1, keepdim=True)
                    val_correct += pred.eq(target.view_as(pred)).sum().item()
                    val_total += target.size(0)
            
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            scheduler.step(val_loss)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'finetuned': True
                }, 'models/finetuned_model.pth')
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        print(f"Fine-tuning completed! Best validation accuracy: {best_val_acc:.2f}%")
        return best_val_acc

def main():
    # Example usage
    sft_trainer = SFTTrainer('models/aapl_advanced_final.pth')
    
    # Fine-tune on recent data
    accuracy = sft_trainer.finetune(
        data_path='data/aapl_advanced.csv',
        epochs=50,
        lr=1e-5,
        batch_size=16
    )
    
    print(f"Fine-tuned model accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main() 