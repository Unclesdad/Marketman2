"""
Advanced Stock Prediction System
Combines multiple cutting-edge approaches for weekly stock prediction
Designed for individual researchers with standard GPU/CPU resources
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import ta
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Optional
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import networkx as nx
from scipy import stats
import joblib
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Download required NLTK data
try:
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment import SentimentIntensityAnalyzer
except:
    pass

class AdvancedFeatureEngineering:
    """Implements novel feature engineering techniques from recent papers"""
    
    def __init__(self):
        self.feature_names = []
        self.scaler = RobustScaler()
        
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced price-based features including microstructure indicators"""
        features = df.copy()
        
        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in features.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for sufficient data
        if len(features) < 100:
            raise ValueError(f"Insufficient data: {len(features)} rows (need at least 100)")
        
        # Basic returns at multiple scales
        for period in [1, 5, 10, 20, 60]:
            features[f'return_{period}d'] = features['Close'].pct_change(period)
            features[f'volume_return_{period}d'] = features['Volume'].pct_change(period)
        
        # Check if return_1d was created successfully
        if 'return_1d' not in features.columns or features['return_1d'].isna().all():
            raise ValueError("Failed to create return_1d column - check Close price data")
        
        # Volatility features
        for period in [5, 10, 20, 40]:
            features[f'volatility_{period}d'] = features['return_1d'].rolling(period).std()
        
        # Volatility ratios (after all volatilities are calculated)
        for period in [5, 10, 20]:
            if f'volatility_{period*2}d' in features.columns:
                denominator = features[f'volatility_{period*2}d'].replace(0, np.nan)
                features[f'volatility_ratio_{period}d'] = features[f'volatility_{period}d'] / denominator
            else:
                features[f'volatility_ratio_{period}d'] = 1.0  # Default ratio
        
        # Price position features
        for period in [10, 20, 50, 200]:
            rolling_min = features['Close'].rolling(period).min()
            rolling_max = features['Close'].rolling(period).max()
            denominator = (rolling_max - rolling_min).replace(0, np.nan)
            features[f'price_position_{period}d'] = (features['Close'] - rolling_min) / denominator
        
        # Volume-price divergence
        features['volume_price_divergence'] = (features['Volume'].pct_change() - features['Close'].pct_change()).rolling(10).mean()
        
        # Microstructure features
        features['high_low_ratio'] = features['High'] / features['Low'].replace(0, np.nan)
        features['close_to_high'] = features['Close'] / features['High'].replace(0, np.nan)
        features['intraday_volatility'] = (features['High'] - features['Low']) / features['Open'].replace(0, np.nan)
        
        # Order flow imbalance proxy
        denominator = (features['High'] - features['Low']).replace(0, np.nan)
        features['order_flow_imbalance'] = ((features['Close'] - features['Low']) - (features['High'] - features['Close'])) / denominator
        
        # Efficiency ratio
        for period in [10, 20]:
            price_change = abs(features['Close'] - features['Close'].shift(period))
            path_length = features['Close'].diff().abs().rolling(period).sum().replace(0, np.nan)
            features[f'efficiency_ratio_{period}d'] = price_change / path_length
        
        return features
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced technical indicators beyond standard TA-lib"""
        features = df.copy()
        
        # Ensure we're working with Series, not DataFrame columns
        close_series = features['Close'].squeeze() if isinstance(features['Close'], pd.DataFrame) else features['Close']
        high_series = features['High'].squeeze() if isinstance(features['High'], pd.DataFrame) else features['High']
        low_series = features['Low'].squeeze() if isinstance(features['Low'], pd.DataFrame) else features['Low']
        volume_series = features['Volume'].squeeze() if isinstance(features['Volume'], pd.DataFrame) else features['Volume']
        
        # Advanced momentum indicators
        rsi = ta.momentum.RSIIndicator(close_series)
        features['rsi'] = rsi.rsi()
        features['rsi_signal'] = features['rsi'].rolling(5).mean()
        
        # MACD with signal
        macd = ta.trend.MACD(close_series)
        features['macd'] = macd.macd()
        features['macd_signal'] = macd.macd_signal()
        features['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands features
        bb = ta.volatility.BollingerBands(close_series)
        features['bb_width'] = bb.bollinger_wband()
        features['bb_position'] = (close_series - bb.bollinger_lband()) / \
                                 (bb.bollinger_hband() - bb.bollinger_lband() + 1e-10)
        
        # Keltner Channel
        kc = ta.volatility.KeltnerChannel(high_series, low_series, close_series)
        kc_upper = kc.keltner_channel_hband()
        kc_lower = kc.keltner_channel_lband()
        features['kc_position'] = (close_series - kc_lower) / \
                                 (kc_upper - kc_lower + 1e-10)
        
        # Volume indicators
        features['vwap'] = (close_series * volume_series).cumsum() / volume_series.cumsum().replace(0, np.nan)
        features['volume_sma_ratio'] = volume_series / volume_series.rolling(20).mean().replace(0, np.nan)
        
        # Custom indicators from research
        features['price_acceleration'] = close_series.pct_change().diff()
        features['volume_acceleration'] = volume_series.pct_change().diff()
        
        return features
    
    def create_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market regime identification features"""
        features = df.copy()
        
        # Trend strength
        for period in [20, 50]:
            sma = features['Close'].rolling(period).mean()
            features[f'trend_strength_{period}d'] = (features['Close'] - sma) / sma.replace(0, np.nan)
            # Avoid boolean Series ambiguity by using where
            features[f'above_sma_{period}d'] = pd.Series(
                np.where(features['Close'] > sma, 1, 0), 
                index=features.index
            )
        
        # Volatility regime (simplified to avoid qcut issues)
        try:
            realized_vol = features['return_1d'].rolling(20).std() * np.sqrt(252)
            
            # Use percentiles instead of qcut to avoid duplicates issue
            vol_33 = realized_vol.quantile(0.33)
            vol_67 = realized_vol.quantile(0.67)
            
            # Create regime categories manually using numpy.where to avoid boolean indexing issues
            vol_regime = np.where(realized_vol <= vol_33, 'low',
                                np.where(realized_vol >= vol_67, 'high', 'medium'))
            vol_regime = pd.Series(vol_regime, index=realized_vol.index)
            
            features['volatility_regime'] = vol_regime
            features = pd.get_dummies(features, columns=['volatility_regime'], prefix='vol_regime')
        except Exception:
            # Fallback: create default regime columns
            features['vol_regime_low'] = 0
            features['vol_regime_medium'] = 1
            features['vol_regime_high'] = 0
        
        # Market phase detection
        features['market_phase'] = self._detect_market_phase(features)
        
        return features
    
    def _detect_market_phase(self, df: pd.DataFrame) -> pd.Series:
        """Detect market phases using simplified logic"""
        try:
            returns = df['return_1d'].dropna()
            
            if len(returns) < 20:
                # Not enough data for meaningful analysis
                return pd.Series(['sideways'] * len(df), index=df.index)
            
            # Simple regime detection based on returns and volatility
            vol = returns.rolling(20).std()
            trend = returns.rolling(20).mean()
            
            # Initialize result array
            phases = ['sideways'] * len(df)
            
            # Vectorized approach to avoid boolean ambiguity
            vol_median = vol.median()
            
            for i in range(len(trend)):
                if pd.isna(trend.iloc[i]) or pd.isna(vol.iloc[i]):
                    continue
                    
                t_val = trend.iloc[i]
                v_val = vol.iloc[i]
                
                if t_val > 0.001:
                    if v_val < vol_median:
                        phases[i] = 'bull_quiet'
                    else:
                        phases[i] = 'bull_volatile'
                elif t_val < -0.001:
                    if v_val < vol_median:
                        phases[i] = 'bear_quiet'
                    else:
                        phases[i] = 'bear_volatile'
                else:
                    phases[i] = 'sideways'
            
            return pd.Series(phases, index=df.index)
            
        except Exception:
            # Fallback to all sideways if any error occurs
            return pd.Series(['sideways'] * len(df), index=df.index)
    
    def create_intermarket_features(self, df: pd.DataFrame, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create features from related markets (indices, commodities, etc.)"""
        features = df.copy()
        
        # Add correlation features
        for symbol, data in market_data.items():
            if len(data) >= len(df):
                aligned_data = data.reindex(df.index, method='ffill')
                
                # Rolling correlation
                for period in [20, 60]:
                    features[f'corr_{symbol}_{period}d'] = \
                        features['return_1d'].rolling(period).corr(aligned_data['return_1d'])
                
                # Relative strength
                features[f'relative_strength_{symbol}'] = \
                    features['Close'].pct_change(20) - aligned_data['Close'].pct_change(20)
        
        return features
    
    def engineer_features(self, df: pd.DataFrame, market_data: Optional[Dict] = None) -> pd.DataFrame:
        """Main feature engineering pipeline"""
        
        try:
            # Apply all feature engineering steps
            features = self.create_price_features(df)
            features = self.create_technical_features(features)
            features = self.create_market_regime_features(features)
        except Exception as e:
            # If any step fails, provide detailed error info
            raise ValueError(f"Feature engineering failed: {str(e)}. Data shape: {df.shape}, columns: {list(df.columns)}")
        
        if market_data:
            features = self.create_intermarket_features(features, market_data)
        
        # Create lag features for time series
        feature_cols = [col for col in features.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']]
        
        for col in feature_cols[:20]:  # Limit to prevent explosion
            for lag in [1, 5]:
                features[f'{col}_lag{lag}'] = features[col].shift(lag)
        
        # Store feature names
        self.feature_names = [col for col in features.columns if col not in 
                            ['Open', 'High', 'Low', 'Close', 'Volume', 'Date', 'target']]
        
        return features

class TransformerPredictor(nn.Module):
    """Transformer architecture adapted for stock prediction"""
    
    def __init__(self, input_dim, d_model=128, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = self._create_positional_encoding(1000, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu'
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3)  # 3 classes: down, neutral, up
        )
        
    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0).transpose(0, 1)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        seq_len = x.size(1)
        
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:seq_len, :].to(x.device)
        
        # Transformer expects (seq_len, batch, features)
        x = x.transpose(0, 1)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Use last timestep for prediction
        x = x[-1]
        
        return self.output_layers(x)

class GraphNeuralNetwork(nn.Module):
    """Graph neural network for capturing inter-stock relationships"""
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=32):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Graph attention layers
        self.attention_weights = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.attention_activation = nn.LeakyReLU(0.2)
        
        self.gnn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(3)
        ])
        
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index, edge_weight=None):
        # x shape: (num_nodes, features)
        x = self.input_projection(x)
        
        for layer in self.gnn_layers:
            # Simple graph convolution with attention
            x_neighbors = self._aggregate_neighbors(x, edge_index, edge_weight)
            x = torch.relu(layer(x + x_neighbors))
        
        return self.output_projection(x)
    
    def _aggregate_neighbors(self, x, edge_index, edge_weight):
        # Simplified neighbor aggregation
        num_nodes = x.size(0)
        aggregated = torch.zeros_like(x)
        
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1)).to(x.device)
        
        for i in range(edge_index.size(1)):
            src, dst = edge_index[:, i]
            aggregated[dst] += x[src] * edge_weight[i]
        
        return aggregated

class MultiModalFusion(nn.Module):
    """Fuses price data, technical indicators, and sentiment"""
    
    def __init__(self, price_dim, sentiment_dim, hidden_dim=128):
        super().__init__()
        
        self.price_encoder = nn.LSTM(price_dim, hidden_dim, num_layers=2, 
                                     batch_first=True, dropout=0.1)
        
        self.sentiment_encoder = nn.Sequential(
            nn.Linear(sentiment_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Gated fusion mechanism
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3)
        )
        
    def forward(self, price_features, sentiment_features):
        # Encode price series
        price_out, (h_n, _) = self.price_encoder(price_features)
        price_repr = h_n[-1]  # Last hidden state
        
        # Encode sentiment
        sentiment_repr = self.sentiment_encoder(sentiment_features)
        
        # Gated fusion
        combined = torch.cat([price_repr, sentiment_repr], dim=1)
        gate = self.fusion_gate(combined)
        
        fused = gate * price_repr + (1 - gate) * sentiment_repr
        
        return self.output_layers(fused)

class StockDataset(Dataset):
    """Custom dataset for stock prediction"""
    
    def __init__(self, features, targets, seq_length=60):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.features) - self.seq_length
    
    def __getitem__(self, idx):
        X = self.features[idx:idx+self.seq_length]
        y = self.targets[idx+self.seq_length]
        
        return torch.FloatTensor(X), torch.LongTensor([y])

class SentimentAnalyzer:
    """Analyzes sentiment from multiple sources"""
    
    def __init__(self):
        try:
            self.finbert = pipeline("sentiment-analysis", 
                                  model="ProsusAI/finbert",
                                  device=0 if torch.cuda.is_available() else -1)
        except:
            print("FinBERT not available, using VADER")
            self.finbert = None
            self.vader = SentimentIntensityAnalyzer()
    
    def get_news_sentiment(self, symbol: str, days_back: int = 7) -> Dict:
        """Scrape and analyze news sentiment"""
        # Create data directory if it doesn't exist
        os.makedirs('data/sentiment', exist_ok=True)
        
        # Check if sentiment data exists
        sentiment_file = f"data/sentiment/{symbol}_sentiment_{datetime.now().strftime('%Y-%m-%d')}.json"
        
        if os.path.exists(sentiment_file):
            try:
                with open(sentiment_file, 'r') as f:
                    sentiment_data = json.load(f)
                print(f"Loaded sentiment data for {symbol} from cache")
                return sentiment_data
            except Exception as e:
                print(f"Error loading cached sentiment for {symbol}: {e}")
        
        sentiments = []
        headlines_data = []
        
        # Multiple news sources
        sources = [
            f"https://finviz.com/quote.ashx?t={symbol}",
            f"https://finance.yahoo.com/quote/{symbol}/news"
        ]
        
        for source in sources:
            try:
                response = requests.get(source, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract news headlines (simplified)
                    headlines = []
                    for tag in ['h3', 'h4', 'a']:
                        for element in soup.find_all(tag)[:20]:
                            text = element.get_text().strip()
                            if len(text) > 20 and len(text) < 200:
                                headlines.append(text)
                    
                    # Analyze sentiment
                    for headline in headlines:
                        sentiment = self._analyze_text(headline)
                        sentiments.append(sentiment)
                        headlines_data.append({
                            'text': headline,
                            'sentiment': sentiment,
                            'source': source,
                            'timestamp': datetime.now().isoformat()
                        })
            except:
                continue
        
        if sentiments:
            result = {
                'mean_sentiment': np.mean(sentiments),
                'sentiment_std': np.std(sentiments),
                'positive_ratio': sum(1 for s in sentiments if s > 0.1) / len(sentiments),
                'negative_ratio': sum(1 for s in sentiments if s < -0.1) / len(sentiments),
                'num_articles': len(sentiments),
                'headlines': headlines_data
            }
        else:
            result = {
                'mean_sentiment': 0,
                'sentiment_std': 0,
                'positive_ratio': 0.5,
                'negative_ratio': 0.5,
                'num_articles': 0,
                'headlines': []
            }
        
        # Save sentiment data
        try:
            with open(sentiment_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Saved sentiment data for {symbol} to {sentiment_file}")
        except Exception as e:
            print(f"Error saving sentiment data: {e}")
        
        return result
    
    def _analyze_text(self, text: str) -> float:
        """Analyze sentiment of text"""
        if self.finbert:
            result = self.finbert(text[:512])[0]
            if result['label'] == 'positive':
                return result['score']
            elif result['label'] == 'negative':
                return -result['score']
            else:
                return 0
        else:
            scores = self.vader.polarity_scores(text)
            return scores['compound']

class AdvancedStockPredictor:
    """Main prediction system combining all components"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.feature_engineer = AdvancedFeatureEngineering()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.models = {}
        self.scalers = {}
        self.performance_history = []
        
    def fetch_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for symbols"""
        data = {}
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        for symbol in tqdm(symbols, desc="Fetching data"):
            # Check if data already exists locally
            data_file = f"data/{symbol}_{start_date}_{end_date}.csv"
            
            if os.path.exists(data_file):
                try:
                    # Load from saved file
                    df = pd.read_csv(data_file, index_col='Date', parse_dates=True)
                    if len(df) > 100 and not df.empty:
                        # Ensure columns are proper strings (not MultiIndex)
                        expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                        if all(col in df.columns for col in expected_cols):
                            data[symbol] = df
                            print(f"Loaded {symbol} from saved data")
                            continue
                except Exception as e:
                    print(f"Error loading saved data for {symbol}: {e}")
            
            # Fetch from internet if not available locally
            try:
                df = yf.download(symbol, start=start_date, end=end_date, progress=False)
                if len(df) > 100 and not df.empty:
                    # Handle MultiIndex columns from yfinance
                    if isinstance(df.columns, pd.MultiIndex):
                        # Flatten MultiIndex columns - take the first level (price type)
                        df.columns = df.columns.get_level_values(0)
                    
                    # Ensure all columns are properly formatted
                    df = df.reset_index()
                    df.set_index('Date', inplace=True)
                    
                    # Save to data folder
                    df.to_csv(data_file)
                    print(f"Saved {symbol} data to {data_file}")
                    
                    data[symbol] = df
            except Exception as e:
                print(f"Failed to fetch {symbol}: {e}")
        
        return data
    
    def create_correlation_graph(self, returns_data: pd.DataFrame, threshold: float = 0.5) -> Tuple:
        """Create correlation-based graph structure"""
        corr_matrix = returns_data.corr()
        
        # Create edges where correlation exceeds threshold
        edges = []
        edge_weights = []
        
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > threshold:
                    edges.append([i, j])
                    edges.append([j, i])  # Undirected
                    edge_weights.extend([abs(corr), abs(corr)])
        
        edge_index = torch.LongTensor(edges).t()
        edge_weight = torch.FloatTensor(edge_weights)
        
        return edge_index, edge_weight
    
    def prepare_training_data(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Prepare all training data with features and targets"""
        print("Preparing training data...")
        
        all_features = []
        all_targets = []
        all_symbols = []
        
        # Get market indices for intermarket features (cached)
        market_indices = self.fetch_data(['^GSPC', '^DJI', '^IXIC', '^VIX'], 
                                       list(data.values())[0].index[0].strftime('%Y-%m-%d'),
                                       list(data.values())[0].index[-1].strftime('%Y-%m-%d'))
        
        for symbol, df in tqdm(data.items(), desc="Engineering features"):
            try:
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
                
                # Engineer features
                features_df = self.feature_engineer.engineer_features(df, market_indices)
                
                # Add sentiment features (simplified for speed)
                sentiment = {'mean_sentiment': 0, 'sentiment_std': 0, 
                           'positive_ratio': 0.5, 'negative_ratio': 0.5, 'num_articles': 0}
                
                for key, value in sentiment.items():
                    features_df[f'sentiment_{key}'] = value
                
                # Create target (next week's return direction)
                future_returns = features_df['Close'].pct_change(5).shift(-5)
                features_df['target'] = np.where(future_returns > 0.02, 2,  # Up
                                               np.where(future_returns < -0.02, 0,  # Down
                                                       1))  # Neutral
                
                # Drop NaN values
                features_df = features_df.dropna()
                
                if len(features_df) > 100 and len(self.feature_engineer.feature_names) > 0:
                    # Ensure all feature columns exist
                    available_features = [f for f in self.feature_engineer.feature_names if f in features_df.columns]
                    if len(available_features) > 10:  # Need minimum features
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
        
        if not all_features:
            raise ValueError(f"No valid features extracted from {len(data)} symbols. "
                           f"Check data quality and ensure stocks have sufficient historical data (>100 days) "
                           f"with valid OHLCV columns.")
        
        return {
            'features': np.vstack(all_features),
            'targets': np.hstack(all_targets),
            'symbols': all_symbols,
            'feature_names': self.feature_engineer.feature_names
        }
    
    def train_ensemble(self, train_data: Dict, val_data: Optional[Dict] = None):
        """Train ensemble of models"""
        print("Training ensemble models...")
        
        X_train = train_data['features']
        y_train = train_data['targets']
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        
        # Train multiple models
        self._train_transformer(X_train_scaled, y_train)
        self._train_lstm(X_train_scaled, y_train)
        self._train_multimodal(train_data)
        
        print("Training complete!")
    
    def _train_transformer(self, X_train, y_train, epochs=50):
        """Train transformer model"""
        print("Training Transformer...")
        
        model = TransformerPredictor(
            input_dim=X_train.shape[1],
            d_model=128,
            n_heads=8,
            n_layers=4
        ).to(self.device)
        
        dataset = StockDataset(X_train, y_train, seq_length=60)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.squeeze())
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
        
        self.models['transformer'] = model
    
    def _train_lstm(self, X_train, y_train, epochs=30):
        """Train LSTM model"""
        print("Training LSTM...")
        
        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_dim=128):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3, 
                                   batch_first=True, dropout=0.2)
                self.output = nn.Linear(hidden_dim, 3)
                
            def forward(self, x):
                out, (h_n, _) = self.lstm(x)
                return self.output(h_n[-1])
        
        model = LSTMModel(X_train.shape[1]).to(self.device)
        
        dataset = StockDataset(X_train, y_train, seq_length=40)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.squeeze())
                loss.backward()
                optimizer.step()
        
        self.models['lstm'] = model
    
    def _train_multimodal(self, train_data: Dict):
        """Train multimodal fusion model"""
        # Simplified for demonstration - would need proper sentiment feature extraction
        print("Training Multimodal Fusion...")
        
        # Extract price and sentiment features separately
        price_features = train_data['features'][:, :50]  # First 50 features as price
        sentiment_features = train_data['features'][:, -5:]  # Last 5 as sentiment
        
        model = MultiModalFusion(
            price_dim=price_features.shape[1],
            sentiment_dim=sentiment_features.shape[1]
        ).to(self.device)
        
        # Training loop would go here
        self.models['multimodal'] = model
    
    def predict(self, symbol: str, current_date: str = None) -> Dict:
        """Make prediction for a symbol"""
        if current_date is None:
            current_date = datetime.now().strftime('%Y-%m-%d')
        
        # Fetch recent data
        end_date = datetime.strptime(current_date, '%Y-%m-%d')
        start_date = end_date - timedelta(days=365)
        
        df = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), 
                        end=end_date.strftime('%Y-%m-%d'), progress=False)
        
        if len(df) < 100:
            return {'error': 'Insufficient data'}
        
        # Engineer features (using cached data)
        market_indices = self.fetch_data(['^GSPC', '^VIX'], 
                                       start_date.strftime('%Y-%m-%d'),
                                       end_date.strftime('%Y-%m-%d'))
        
        features_df = self.feature_engineer.engineer_features(df, market_indices)
        
        # Add sentiment
        sentiment = self.sentiment_analyzer.get_news_sentiment(symbol)
        for key, value in sentiment.items():
            features_df[f'sentiment_{key}'] = value
        
        # Prepare features
        features_df = features_df.dropna()
        X = features_df[self.feature_engineer.feature_names].values[-60:]
        X_scaled = self.scalers['standard'].transform(X)
        
        # Get predictions from all models
        predictions = {}
        
        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device)
                
                if name == 'multimodal':
                    # Split features for multimodal
                    price_feat = X_tensor[:, :, :50]
                    sent_feat = X_tensor[:, -1, -5:]
                    output = model(price_feat, sent_feat)
                else:
                    output = model(X_tensor)
                
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                predictions[name] = {
                    'down': probs[0],
                    'neutral': probs[1],
                    'up': probs[2],
                    'prediction': ['down', 'neutral', 'up'][np.argmax(probs)]
                }
        
        # Ensemble prediction
        ensemble_probs = np.mean([
            [pred['down'], pred['neutral'], pred['up']] 
            for pred in predictions.values()
        ], axis=0)
        
        final_prediction = ['down', 'neutral', 'up'][np.argmax(ensemble_probs)]
        
        # Calculate confidence
        confidence = np.max(ensemble_probs)
        
        # Technical analysis summary
        current_price = df['Close'].iloc[-1]
        sma_20 = df['Close'].rolling(20).mean().iloc[-1]
        sma_50 = df['Close'].rolling(50).mean().iloc[-1]
        rsi = ta.momentum.RSIIndicator(df['Close']).rsi().iloc[-1]
        
        return {
            'symbol': symbol,
            'date': current_date,
            'prediction': final_prediction,
            'confidence': float(confidence),
            'ensemble_probabilities': {
                'down': float(ensemble_probs[0]),
                'neutral': float(ensemble_probs[1]),
                'up': float(ensemble_probs[2])
            },
            'individual_predictions': predictions,
            'technical_indicators': {
                'current_price': float(current_price),
                'sma_20': float(sma_20),
                'sma_50': float(sma_50),
                'rsi': float(rsi),
                'above_sma_20': bool(current_price > sma_20),
                'above_sma_50': bool(current_price > sma_50)
            },
            'sentiment_analysis': sentiment,
            'recommendation': self._generate_recommendation(final_prediction, confidence, ensemble_probs)
        }
    
    def _generate_recommendation(self, prediction: str, confidence: float, probs: np.ndarray) -> Dict:
        """Generate trading recommendation based on prediction"""
        if confidence < 0.6:
            return {
                'action': 'HOLD',
                'reason': 'Low confidence in prediction',
                'suggested_position': 0
            }
        
        if prediction == 'up' and probs[2] > 0.7:
            return {
                'action': 'BUY',
                'reason': 'Strong bullish signal',
                'suggested_position': 0.1,  # 10% of portfolio
                'stop_loss': -0.05,  # 5% stop loss
                'take_profit': 0.15  # 15% take profit
            }
        elif prediction == 'down' and probs[0] > 0.7:
            return {
                'action': 'SELL/SHORT',
                'reason': 'Strong bearish signal',
                'suggested_position': -0.05,  # 5% short position
                'stop_loss': 0.05,
                'take_profit': -0.10
            }
        else:
            return {
                'action': 'HOLD',
                'reason': 'Neutral market conditions',
                'suggested_position': 0
            }
    
    def backtest(self, test_data: Dict[str, pd.DataFrame], start_date: str, end_date: str) -> Dict:
        """Backtest the strategy on historical data"""
        print("Running backtest...")
        
        results = {
            'trades': [],
            'equity_curve': [],
            'metrics': {}
        }
        
        initial_capital = 100000
        capital = initial_capital
        position = 0
        
        for symbol, df in test_data.items():
            df_test = df[start_date:end_date]
            
            for i in range(60, len(df_test), 5):  # Weekly predictions
                current_date = df_test.index[i]
                
                # Make prediction
                pred_result = self.predict(symbol, current_date.strftime('%Y-%m-%d'))
                
                if 'error' not in pred_result:
                    recommendation = pred_result['recommendation']
                    
                    if recommendation['action'] == 'BUY' and position == 0:
                        # Enter long position
                        position = int(capital * recommendation['suggested_position'] / df_test['Close'].iloc[i])
                        entry_price = df_test['Close'].iloc[i]
                        
                        results['trades'].append({
                            'date': current_date,
                            'symbol': symbol,
                            'action': 'BUY',
                            'price': entry_price,
                            'shares': position,
                            'confidence': pred_result['confidence']
                        })
                    
                    elif recommendation['action'] == 'SELL/SHORT' and position > 0:
                        # Close long position
                        exit_price = df_test['Close'].iloc[i]
                        pnl = position * (exit_price - entry_price)
                        capital += pnl
                        
                        results['trades'].append({
                            'date': current_date,
                            'symbol': symbol,
                            'action': 'SELL',
                            'price': exit_price,
                            'shares': position,
                            'pnl': pnl
                        })
                        
                        position = 0
                
                results['equity_curve'].append({
                    'date': current_date,
                    'capital': capital,
                    'return': (capital - initial_capital) / initial_capital
                })
        
        # Calculate metrics
        equity_df = pd.DataFrame(results['equity_curve'])
        returns = equity_df['return'].pct_change().dropna()
        
        results['metrics'] = {
            'total_return': (capital - initial_capital) / initial_capital,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'max_drawdown': (equity_df['capital'] / equity_df['capital'].cummax() - 1).min(),
            'win_rate': sum(1 for t in results['trades'] if t.get('pnl', 0) > 0) / len(results['trades']) if results['trades'] else 0,
            'total_trades': len(results['trades'])
        }
        
        return results
    
    def plot_performance(self, backtest_results: Dict):
        """Plot backtest performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Equity curve
        equity_df = pd.DataFrame(backtest_results['equity_curve'])
        axes[0, 0].plot(equity_df['date'], equity_df['capital'])
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Capital ($)')
        
        # Returns distribution
        trades_df = pd.DataFrame(backtest_results['trades'])
        if 'pnl' in trades_df.columns:
            axes[0, 1].hist(trades_df['pnl'].dropna(), bins=30, alpha=0.7)
            axes[0, 1].set_title('PnL Distribution')
            axes[0, 1].set_xlabel('Profit/Loss ($)')
            axes[0, 1].set_ylabel('Frequency')
        
        # Confidence vs accuracy
        if 'confidence' in trades_df.columns:
            axes[1, 0].scatter(trades_df['confidence'], trades_df['pnl'])
            axes[1, 0].set_title('Confidence vs PnL')
            axes[1, 0].set_xlabel('Prediction Confidence')
            axes[1, 0].set_ylabel('PnL ($)')
        
        # Metrics summary
        metrics_text = '\n'.join([f'{k}: {v:.3f}' for k, v in backtest_results['metrics'].items()])
        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes, 
                       fontsize=12, verticalalignment='center')
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Performance Metrics')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, path: str):
        """Save all models and scalers"""
        # Ensure data directory exists
        os.makedirs('data/models', exist_ok=True)
        
        model_path = f"data/models/{path}"
        
        save_dict = {
            'models': {name: model.state_dict() for name, model in self.models.items()},
            'scalers': self.scalers,
            'feature_names': self.feature_engineer.feature_names
        }
        
        torch.save(save_dict, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, path: str):
        """Load saved models"""
        # Check data directory first
        if not path.startswith('data/'):
            model_path = f"data/models/{path}"
        else:
            model_path = path
            
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return
            
        save_dict = torch.load(model_path, map_location=self.device)
        
        # Reconstruct models (you'd need to store architecture info too)
        # This is simplified - in practice you'd save model configs
        
        self.scalers = save_dict['scalers']
        self.feature_engineer.feature_names = save_dict['feature_names']
        print(f"Model loaded from {model_path}")

class RealTimePredictor:
    """Real-time prediction system with live data feeds"""
    
    def __init__(self, model_path: str):
        self.predictor = AdvancedStockPredictor()
        self.predictor.load_model(model_path)
        self.active_positions = {}
        
    def monitor_symbols(self, symbols: List[str], interval: int = 300):
        """Monitor symbols and generate alerts"""
        print(f"Monitoring {len(symbols)} symbols every {interval} seconds...")
        
        while True:
            for symbol in symbols:
                try:
                    # Get current prediction
                    result = self.predictor.predict(symbol)
                    
                    if result['confidence'] > 0.7:
                        self._check_alert_conditions(symbol, result)
                    
                    # Update positions
                    self._update_positions(symbol, result)
                    
                except Exception as e:
                    print(f"Error monitoring {symbol}: {e}")
            
            time.sleep(interval)
    
    def _check_alert_conditions(self, symbol: str, prediction: Dict):
        """Check if alert conditions are met"""
        if prediction['recommendation']['action'] in ['BUY', 'SELL/SHORT']:
            print(f"\n🚨 ALERT for {symbol}:")
            print(f"Action: {prediction['recommendation']['action']}")
            print(f"Confidence: {prediction['confidence']:.2%}")
            print(f"Prediction: {prediction['prediction']}")
            print(f"Probabilities: Up={prediction['ensemble_probabilities']['up']:.2%}, "
                  f"Down={prediction['ensemble_probabilities']['down']:.2%}")
            print(f"Current Price: ${prediction['technical_indicators']['current_price']:.2f}")
            print(f"Sentiment: {prediction['sentiment_analysis']['mean_sentiment']:.2f}")
            print("-" * 50)
    
    def _update_positions(self, symbol: str, prediction: Dict):
        """Update position tracking"""
        if symbol in self.active_positions:
            position = self.active_positions[symbol]
            current_price = prediction['technical_indicators']['current_price']
            
            # Check stop loss / take profit
            if position['type'] == 'long':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                
                if pnl_pct <= position['stop_loss'] or pnl_pct >= position['take_profit']:
                    print(f"📊 Closing {symbol} position: PnL = {pnl_pct:.2%}")
                    del self.active_positions[symbol]


# Main execution example
if __name__ == "__main__":
    # Initialize predictor
    predictor = AdvancedStockPredictor()
    
    # Define symbols to analyze
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'SPY', 'QQQ']
    
    # Fetch historical data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    print("Fetching historical data...")
    data = predictor.fetch_data(symbols, start_date, end_date)
    
    # Prepare training data
    train_data = predictor.prepare_training_data(data)
    
    # Train ensemble
    predictor.train_ensemble(train_data)
    
    # Make prediction for a symbol
    print("\nMaking prediction for AAPL...")
    prediction = predictor.predict('AAPL')
    
    print(f"\nPrediction Results:")
    print(f"Symbol: {prediction['symbol']}")
    print(f"Prediction: {prediction['prediction']}")
    print(f"Confidence: {prediction['confidence']:.2%}")
    print(f"Probabilities: Up={prediction['ensemble_probabilities']['up']:.2%}, "
          f"Neutral={prediction['ensemble_probabilities']['neutral']:.2%}, "
          f"Down={prediction['ensemble_probabilities']['down']:.2%}")
    print(f"\nRecommendation: {prediction['recommendation']['action']}")
    print(f"Reason: {prediction['recommendation']['reason']}")
    
    # Run backtest
    print("\nRunning backtest...")
    backtest_start = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    backtest_results = predictor.backtest(data, backtest_start, end_date)
    
    print(f"\nBacktest Results:")
    print(f"Total Return: {backtest_results['metrics']['total_return']:.2%}")
    print(f"Sharpe Ratio: {backtest_results['metrics']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {backtest_results['metrics']['max_drawdown']:.2%}")
    print(f"Win Rate: {backtest_results['metrics']['win_rate']:.2%}")
    print(f"Total Trades: {backtest_results['metrics']['total_trades']}")
    
    # Save model (will be saved to data/models/)
    predictor.save_model('advanced_stock_predictor.pth')
    
    # Plot results
    predictor.plot_performance(backtest_results)
    
    # Optional: Start real-time monitoring
    # real_time = RealTimePredictor('advanced_stock_predictor.pth')
    # real_time.monitor_symbols(['AAPL', 'GOOGL', 'TSLA'], interval=300)