"""
Complete Advanced Stock Predictor with Sentiment Analysis
Run this file to train on all 200 stocks with real sentiment data
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
import ta
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Optional
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import joblib
import json
import feedparser
import matplotlib.pyplot as plt
import seaborn as sns

# Import components from the other files
from advanced_stock_predictor import (
    AdvancedFeatureEngineering, 
    TransformerPredictor,
    GraphNeuralNetwork,
    MultiModalFusion,
    StockDataset,
    AdvancedStockPredictor,
    RealTimePredictor
)
from top_200_tickers import ALL_TICKERS, TOP_200_TICKERS, get_high_volume_tickers
from sentiment_analysis_implementation import ComprehensiveSentimentAnalyzer

class EnhancedStockPredictor(AdvancedStockPredictor):
    """Enhanced version with real sentiment analysis"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(device)
        # Initialize comprehensive sentiment analyzer
        self.sentiment_analyzer = ComprehensiveSentimentAnalyzer()
        self.sentiment_cache = {}
        self.failed_downloads = []
        
    def fetch_data_with_retry(self, symbols: List[str], start_date: str, end_date: str, 
                             batch_size: int = 20, max_retries: int = 3) -> Dict[str, pd.DataFrame]:
        """Fetch historical data with retry logic and batch processing"""
        data = {}
        failed = []
        
        print(f"\nFetching data for {len(symbols)} stocks...")
        print(f"Date range: {start_date} to {end_date}")
        
        # Process in batches to avoid rate limits
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(symbols) - 1) // batch_size + 1
            
            print(f"\nBatch {batch_num}/{total_batches}:")
            
            for symbol in tqdm(batch, desc=f"Batch {batch_num}"):
                retry_count = 0
                success = False
                
                while retry_count < max_retries and not success:
                    try:
                        df = yf.download(symbol, start=start_date, end=end_date, 
                                       progress=False)
                        
                        if len(df) > 252:  # At least 1 year of data
                            # Handle MultiIndex columns from yfinance
                            if isinstance(df.columns, pd.MultiIndex):
                                # Flatten MultiIndex columns - take the first level (price type)
                                df.columns = df.columns.get_level_values(0)
                            
                            # Reset index properly
                            df = df.reset_index()
                            df.set_index('Date', inplace=True)
                            data[symbol] = df
                            success = True
                        else:
                            retry_count += 1
                            if retry_count >= max_retries:
                                failed.append(symbol)
                                print(f"  ✗ {symbol}: Insufficient data ({len(df)} days)")
                            else:
                                time.sleep(1)  # Wait before retry
                    
                    except Exception as e:
                        # Handle specific yfinance errors that shouldn't be retried
                        error_str = str(e).lower()
                        if 'delisted' in error_str or 'timezone' in error_str or 'not found' in error_str:
                            failed.append(symbol)
                            print(f"  ✗ {symbol}: {str(e)[:70]}")
                            break  # Don't retry for these errors
                        
                        retry_count += 1
                        if retry_count >= max_retries:
                            failed.append(symbol)
                            print(f"  ✗ {symbol}: Failed after {max_retries} attempts - {str(e)[:50]}")
                        else:
                            time.sleep(1)  # Wait before retry
                
            # Rate limiting between batches
            if i + batch_size < len(symbols):
                print("  Waiting 3 seconds before next batch...")
                time.sleep(3)
        
        print(f"\n✓ Successfully downloaded: {len(data)}/{len(symbols)} stocks")
        if failed:
            print(f"✗ Failed stocks ({len(failed)}): {failed[:10]}{'...' if len(failed) > 10 else ''}")
            self.failed_downloads = failed
        
        return data
    
    def get_sentiment_features(self, symbol: str, use_cache: bool = True) -> Dict:
        """Get sentiment features with caching"""
        cache_key = f"{symbol}_{datetime.now().date()}"
        
        if use_cache and cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
        
        try:
            # Get comprehensive sentiment
            sentiment_data = self.sentiment_analyzer.get_combined_sentiment(symbol)
            
            # Standardize the output
            features = {
                'mean_sentiment': sentiment_data.get('overall_sentiment', 0),
                'sentiment_std': 0.1,  # Placeholder
                'positive_ratio': max(0.3, min(0.7, 0.5 + sentiment_data.get('overall_sentiment', 0) * 0.2)),
                'negative_ratio': max(0.3, min(0.7, 0.5 - sentiment_data.get('overall_sentiment', 0) * 0.2)),
                'num_articles': sentiment_data.get('news_articles', 0) + sentiment_data.get('reddit_posts', 0),
                'reddit_sentiment': sentiment_data.get('reddit_sentiment', 0),
                'news_sentiment': sentiment_data.get('news_sentiment', 0)
            }
            
            self.sentiment_cache[cache_key] = features
            return features
            
        except Exception as e:
            print(f"Sentiment error for {symbol}: {e}")
            # Return neutral sentiment on error
            return {
                'mean_sentiment': 0,
                'sentiment_std': 0,
                'positive_ratio': 0.5,
                'negative_ratio': 0.5,
                'num_articles': 0,
                'reddit_sentiment': 0,
                'news_sentiment': 0
            }
    
    def prepare_training_data_enhanced(self, data: Dict[str, pd.DataFrame], 
                                     use_sentiment: bool = True,
                                     sentiment_batch_size: int = 10) -> Dict:
        """Enhanced data preparation with real sentiment analysis"""
        print("\nPreparing training data with sentiment analysis...")
        
        all_features = []
        all_targets = []
        all_symbols = []
        
        # Get market indices for intermarket features
        print("Fetching market indices...")
        market_indices = self.fetch_data_with_retry(
            ['^GSPC', '^DJI', '^IXIC', '^VIX'], 
            list(data.values())[0].index[0].strftime('%Y-%m-%d'),
            list(data.values())[0].index[-1].strftime('%Y-%m-%d')
        )
        
        # Process stocks in batches for sentiment
        symbols_list = list(data.keys())
        
        for i in range(0, len(symbols_list), sentiment_batch_size):
            batch_symbols = symbols_list[i:i + sentiment_batch_size]
            print(f"\nProcessing batch {i//sentiment_batch_size + 1}/{(len(symbols_list)-1)//sentiment_batch_size + 1}")
            
            for symbol in tqdm(batch_symbols, desc="Engineering features"):
                try:
                    df = data[symbol]
                    
                    # Check basic data quality first
                    if len(df) < 100:
                        print(f"Skipping {symbol}: insufficient data ({len(df)} rows)")
                        continue
                    
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    if not all(col in df.columns for col in required_cols):
                        print(f"Skipping {symbol}: missing required columns")
                        continue
                    
                    # Check for valid price data
                    if df['Close'].isna().all() or (df['Close'] <= 0).any():
                        print(f"Skipping {symbol}: invalid price data")
                        continue
                    
                    # Engineer technical features
                    features_df = self.feature_engineer.engineer_features(df, market_indices)
                    
                    # Add sentiment features
                    if use_sentiment:
                        sentiment = self.get_sentiment_features(symbol)
                        for key, value in sentiment.items():
                            features_df[f'sentiment_{key}'] = value
                    else:
                        # Use neutral sentiment
                        for key in ['mean_sentiment', 'sentiment_std', 'positive_ratio', 
                                   'negative_ratio', 'num_articles', 'reddit_sentiment', 'news_sentiment']:
                            features_df[f'sentiment_{key}'] = 0 if 'sentiment' in key else 0.5
                    
                    # Create target (next week's return direction)
                    future_returns = features_df['Close'].pct_change(5).shift(-5)
                    features_df['target'] = np.where(
                        future_returns > 0.02, 2,  # Up
                        np.where(future_returns < -0.02, 0,  # Down
                                1))  # Neutral
                    
                    # Drop NaN values
                    features_df = features_df.dropna()
                    
                    if len(features_df) > 100 and len(self.feature_engineer.feature_names) > 0:
                        available_features = [f for f in self.feature_engineer.feature_names 
                                           if f in features_df.columns]
                        if len(available_features) > 10:
                            all_features.append(features_df[available_features].values)
                            all_targets.append(features_df['target'].values)
                            all_symbols.extend([symbol] * len(features_df))
                            print(f"Successfully processed {symbol}: {len(available_features)} features")
                        else:
                            print(f"Skipping {symbol}: insufficient features ({len(available_features)})")
                    else:
                        print(f"Skipping {symbol}: insufficient processed data ({len(features_df)} rows)")
                        
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
                    continue
            
            # Small delay between sentiment batches
            if use_sentiment and i + sentiment_batch_size < len(symbols_list):
                time.sleep(2)
        
        if not all_features:
            raise ValueError(f"No valid features extracted from {len(data)} symbols. "
                           f"Check data quality and ensure stocks have sufficient historical data (>100 days) "
                           f"with valid OHLCV columns.")
        
        print(f"\nPrepared data for {len(set(all_symbols))} stocks")
        print(f"Total samples: {sum(len(f) for f in all_features)}")
        
        return {
            'features': np.vstack(all_features),
            'targets': np.hstack(all_targets),
            'symbols': all_symbols,
            'feature_names': self.feature_engineer.feature_names
        }

def run_advanced_training(ticker_selection='all', use_sentiment=True, quick_mode=False):
    """
    Main function to run advanced training
    
    Args:
        ticker_selection: 'all', 'high_volume', 'tech', or list of tickers
        use_sentiment: Whether to use real sentiment analysis
        quick_mode: Use less data for faster testing
    """
    
    print("="*60)
    print("Advanced Stock Predictor with Sentiment Analysis")
    print("="*60)
    
    # Initialize enhanced predictor
    predictor = EnhancedStockPredictor()
    
    # Select tickers
    if ticker_selection == 'all':
        tickers = ALL_TICKERS
        print(f"Using all {len(tickers)} tickers")
    elif ticker_selection == 'high_volume':
        tickers = get_high_volume_tickers()
        print(f"Using {len(tickers)} high-volume tickers")
    elif ticker_selection == 'tech':
        tickers = TOP_200_TICKERS['mega_cap_tech'] + TOP_200_TICKERS['large_cap_tech']
        print(f"Using {len(tickers)} tech stocks")
    elif isinstance(ticker_selection, list):
        tickers = ticker_selection
        print(f"Using {len(tickers)} custom tickers")
    else:
        raise ValueError("Invalid ticker selection")
    
    # Set date range
    if quick_mode:
        years = 1
        tickers = tickers[:20]  # Use fewer stocks in quick mode
        print("Quick mode: Using 1 year of data and 20 stocks")
    else:
        years = 2
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
    
    # Fetch data
    print(f"\nDate range: {start_date} to {end_date}")
    data = predictor.fetch_data_with_retry(tickers, start_date, end_date)
    
    if len(data) < 10:
        print("Error: Insufficient data collected. Exiting.")
        return None
    
    # Prepare training data
    train_data = predictor.prepare_training_data_enhanced(data, use_sentiment=use_sentiment)
    
    # Train ensemble
    print("\n" + "="*60)
    print("Training ensemble models...")
    print("="*60)
    
    predictor.train_ensemble(train_data)
    
    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'advanced_model_{timestamp}.pth'
    predictor.save_model(model_path)
    
    # Test predictions
    print("\n" + "="*60)
    print("Sample Predictions")
    print("="*60)
    
    test_symbols = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL']
    test_symbols = [s for s in test_symbols if s in data][:3]
    
    for symbol in test_symbols:
        try:
            prediction = predictor.predict(symbol)
            
            print(f"\n{symbol}:")
            print(f"  Prediction: {prediction['prediction']}")
            print(f"  Confidence: {prediction['confidence']:.1%}")
            print(f"  Probabilities: Up={prediction['ensemble_probabilities']['up']:.1%}, "
                  f"Down={prediction['ensemble_probabilities']['down']:.1%}")
            print(f"  Sentiment: {prediction['sentiment_analysis']['mean_sentiment']:.2f}")
            print(f"  Recommendation: {prediction['recommendation']['action']}")
        except Exception as e:
            print(f"\n{symbol}: Prediction failed - {e}")
    
    # Run backtest
    print("\n" + "="*60)
    print("Running Backtest")
    print("="*60)
    
    backtest_start = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    backtest_data = {k: v for k, v in data.items() if k in test_symbols}
    
    backtest_results = predictor.backtest(backtest_data, backtest_start, end_date)
    
    print(f"\nBacktest Results:")
    print(f"  Total Return: {backtest_results['metrics']['total_return']:.2%}")
    print(f"  Sharpe Ratio: {backtest_results['metrics']['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {backtest_results['metrics']['max_drawdown']:.2%}")
    print(f"  Win Rate: {backtest_results['metrics']['win_rate']:.2%}")
    print(f"  Total Trades: {backtest_results['metrics']['total_trades']}")
    
    # Save results
    results = {
        'model_path': model_path,
        'training_date': datetime.now().isoformat(),
        'tickers_used': len(data),
        'failed_tickers': predictor.failed_downloads,
        'backtest_metrics': backtest_results['metrics'],
        'sentiment_enabled': use_sentiment
    }
    
    with open(f'training_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Model saved to: {model_path}")
    print(f"✓ Results saved to: training_results_{timestamp}.json")
    
    return predictor, results

def main():
    """Main entry point - automatically runs option 4 (all 200 stocks with sentiment)"""
    
    print("\n⚠️  Running Advanced Stock Predictor on all 200 stocks with sentiment analysis")
    print("This will take 4-8 hours to complete and will fetch sentiment data for 200 stocks.")
    
    predictor, results = run_advanced_training('all', use_sentiment=True, quick_mode=False)
    
    print(f"\n✅ Training completed successfully!")
    print(f"Model saved to: {results['model_path']}")
    print(f"Results saved to: training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

if __name__ == "__main__":
    # Check for required packages
    required_packages = ['yfinance', 'torch', 'ta', 'feedparser', 'bs4', 'sklearn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:", missing_packages)
        print("Install with: pip install", ' '.join(missing_packages))
        sys.exit(1)
    
    # Run main program
    main()