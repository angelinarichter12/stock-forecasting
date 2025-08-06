#!/usr/bin/env python3
"""
Reinforcement Learning from Human Feedback (RLHF) for Stock Trading
Uses trading performance as feedback to improve the model
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import os

from models.transformer_model import StockDataset, create_model, get_feature_columns

class TradingEnvironment:
    """Trading environment for RLHF"""
    
    def __init__(self, data_path, initial_balance=10000):
        self.df = pd.read_csv(data_path)
        self.initial_balance = initial_balance
        self.reset()
    
    def reset(self):
        """Reset the environment"""
        self.balance = self.initial_balance
        self.position = 0  # 0: no position, 1: long position
        self.current_step = 60  # Start after sequence length
        self.trades = []
        return self._get_state()
    
    def _get_state(self):
        """Get current state (last 60 days)"""
        if self.current_step >= 60:
            return self.df.iloc[self.current_step-60:self.current_step]
        return None
    
    def step(self, action):
        """Take action and return reward"""
        # action: 0 (hold), 1 (buy), 2 (sell)
        current_price = self.df.iloc[self.current_step]['Close']
        reward = 0
        
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = current_price
            self.trades.append({
                'type': 'buy',
                'price': current_price,
                'step': self.current_step
            })
        
        elif action == 2 and self.position == 1:  # Sell
            self.position = 0
            exit_price = current_price
            trade_return = (exit_price - self.entry_price) / self.entry_price
            reward = trade_return * 100  # Scale reward
            self.balance *= (1 + trade_return)
            
            self.trades.append({
                'type': 'sell',
                'price': current_price,
                'step': self.current_step,
                'return': trade_return
            })
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        return self._get_state(), reward, done

class RLHFTrainer:
    """RLHF trainer for stock trading"""
    
    def __init__(self, base_model_path, device='cpu'):
        self.device = device
        self.base_model_path = base_model_path
        self.model = None
        self.environment = None
        
    def load_base_model(self):
        """Load the pre-trained base model"""
        checkpoint = torch.load(self.base_model_path, map_location=self.device)
        
        feature_columns = get_feature_columns()
        self.model = create_model(
            input_dim=len(feature_columns),
            d_model=checkpoint['args']['d_model'],
            nhead=checkpoint['args']['nhead'],
            num_layers=checkpoint['args']['num_layers'],
            dropout=checkpoint['args']['dropout']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded base model from {self.base_model_path}")
    
    def calculate_trading_reward(self, predictions, actual_returns):
        """Calculate reward based on trading performance"""
        # Convert predictions to trading actions
        actions = (predictions > 0.5).astype(int)
        
        # Calculate cumulative returns
        strategy_returns = actions * actual_returns
        cumulative_return = np.prod(1 + strategy_returns) - 1
        
        # Calculate Sharpe ratio
        if np.std(strategy_returns) > 0:
            sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Combined reward
        reward = cumulative_return * 100 + sharpe_ratio * 10
        return reward
    
    def rlhf_training(self, data_path, episodes=100, lr=1e-5):
        """RLHF training loop"""
        if self.model is None:
            self.load_base_model()
        
        self.environment = TradingEnvironment(data_path)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        
        episode_rewards = []
        
        print(f"Starting RLHF training for {episodes} episodes...")
        
        for episode in range(episodes):
            state = self.environment.reset()
            total_reward = 0
            step_count = 0
            
            while state is not None and step_count < 1000:
                # Convert state to tensor
                feature_columns = get_feature_columns()
                state_tensor = torch.FloatTensor(state[feature_columns].values).unsqueeze(0).to(self.device)
                
                # Get model prediction
                with torch.no_grad():
                    output = self.model(state_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    action_probs = probabilities[0].cpu().numpy()
                
                # Sample action (exploration)
                if np.random.random() < 0.1:  # 10% exploration
                    action = np.random.randint(0, 3)
                else:
                    action = np.argmax(action_probs)
                
                # Take action in environment
                next_state, reward, done = self.environment.step(action)
                total_reward += reward
                
                # Update model based on reward
                if reward != 0:  # Only update on meaningful actions
                    optimizer.zero_grad()
                    
                    # Calculate loss based on reward
                    if reward > 0:
                        # Positive reward: encourage this action
                        target = torch.tensor([action], dtype=torch.long).to(self.device)
                        loss = nn.CrossEntropyLoss()(output, target)
                    else:
                        # Negative reward: discourage this action
                        # Use inverse of the action probabilities
                        inverse_probs = 1 - action_probs
                        inverse_probs = inverse_probs / inverse_probs.sum()
                        target = torch.FloatTensor(inverse_probs).to(self.device)
                        loss = nn.KLDivLoss()(torch.log(probabilities[0]), target)
                    
                    loss.backward()
                    optimizer.step()
                
                state = next_state
                step_count += 1
                
                if done:
                    break
            
            episode_rewards.append(total_reward)
            
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode}: Average Reward: {avg_reward:.2f}")
        
        # Save RLHF-trained model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'episode_rewards': episode_rewards,
            'rlhf_trained': True
        }, 'models/rlhf_model.pth')
        
        print(f"RLHF training completed! Average reward: {np.mean(episode_rewards):.2f}")
        return np.mean(episode_rewards)

def main():
    # Example usage
    rlhf_trainer = RLHFTrainer('models/aapl_advanced_final.pth')
    
    # Train with RLHF
    avg_reward = rlhf_trainer.rlhf_training(
        data_path='data/aapl_advanced.csv',
        episodes=50,
        lr=1e-5
    )
    
    print(f"RLHF training completed with average reward: {avg_reward:.2f}")

if __name__ == "__main__":
    main() 