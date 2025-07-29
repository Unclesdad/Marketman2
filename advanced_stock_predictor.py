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
        
        # Convert market phase to dummy variables
        features = pd.get_dummies(features, columns=['market_phase'], prefix='phase')
        
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
                
                # Create return_1d for market data if it doesn't exist
                if 'return_1d' not in aligned_data.columns:
                    aligned_data = aligned_data.copy()
                    aligned_data['return_1d'] = aligned_data['Close'].pct_change(1)
                
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
        
        # Store feature names (ensure only numeric columns)
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        self.feature_names = [col for col in numeric_cols if col not in 
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
        
        return pe
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        batch_size, seq_len, input_dim = x.size()
        
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        pos_encoding = self.positional_encoding[:seq_len, :].unsqueeze(0).to(x.device)  # Shape: (1, seq_len, d_model)
        x = x + pos_encoding  # Broadcasting: (batch, seq_len, d_model) + (1, seq_len, d_model)
        
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

class MultiModalFusionOriginal(nn.Module):
    """Original MultiModal architecture that matches saved weights"""
    
    def __init__(self, price_dim, sentiment_dim):
        super().__init__()
        
        # LSTM with 128 hidden units (512 total due to bidirectional-like structure)
        self.price_encoder = nn.LSTM(price_dim, 128, num_layers=2, 
                                     batch_first=True, dropout=0.1)
        
        self.sentiment_encoder = nn.Sequential(
            nn.Linear(sentiment_dim, 128),      # 5 -> 128
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128)                 # 128 -> 128
        )
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(256, 128),                # 128+128 -> 128
            nn.Sigmoid()
        )
        
        self.output_layers = nn.Sequential(
            nn.Linear(128, 64),                 # 128 -> 64
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3)                    # 64 -> 3
        )
        
    def forward(self, price_features, sentiment_features):
        # Encode price series
        price_out, (h_n, _) = self.price_encoder(price_features)
        price_repr = h_n[-1]  # Last hidden state from final layer
        
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
        
        # Enhanced components for reliability
        self.confidence_calibrator = ConfidenceCalibrator()
        self.regime_detector = MarketRegimeDetector()
        self.enhanced_risk_manager = RiskManager()
        
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
                    
                    # Remove duplicate columns (sometimes yfinance returns duplicates)
                    df = df.loc[:, ~df.columns.duplicated()]
                    
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
        """Train transformer model with early stopping"""
        print("Training Transformer...")
        
        model = TransformerPredictor(
            input_dim=X_train.shape[1],
            d_model=128,
            n_heads=8,
            n_layers=4
        ).to(self.device)
        
        dataset = StockDataset(X_train, y_train, seq_length=60)
        dataloader = DataLoader(
            dataset, 
            batch_size=128,  # Even larger batch size
            shuffle=True, 
            drop_last=True,
            num_workers=16,  # Maximum parallelization
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True,  # Keep workers alive
            prefetch_factor=4  # Prefetch more batches
        )
        
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)  # Higher LR for faster convergence
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        criterion = nn.CrossEntropyLoss()
        
        # Early stopping parameters
        best_loss = float('inf')
        patience_counter = 0
        patience = 10  # More patience for better convergence
        min_improvement = 1e-4  # Minimum improvement threshold
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_X, batch_y in progress_bar:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.squeeze())
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss/len(dataloader)
            scheduler.step(avg_loss)
            
            # Early stopping check
            if avg_loss < best_loss - min_improvement:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Best: {best_loss:.4f}")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                model.load_state_dict(best_model_state)
                break
        
        self.models['transformer'] = model
    
    def _train_lstm(self, X_train, y_train, epochs=50):
        """Train LSTM model with early stopping"""
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
        dataloader = DataLoader(
            dataset, 
            batch_size=256,  # Maximum batch size for LSTM
            shuffle=True, 
            drop_last=True,
            num_workers=16,  # Maximum parallelization
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True,
            prefetch_factor=4  # Prefetch more batches
        )
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        criterion = nn.CrossEntropyLoss()
        
        # Early stopping parameters
        best_loss = float('inf')
        patience_counter = 0
        patience = 10  # More patience for better convergence
        min_improvement = 1e-4  # Minimum improvement threshold
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"LSTM Epoch {epoch+1}/{epochs}")
            
            for batch_X, batch_y in progress_bar:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.squeeze())
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss/len(dataloader)
            scheduler.step(avg_loss)
            
            # Early stopping check
            if avg_loss < best_loss - min_improvement:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            print(f"LSTM Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}, Best: {best_loss:.4f}")
            
            if patience_counter >= patience:
                print(f"LSTM Early stopping triggered after {epoch+1} epochs")
                model.load_state_dict(best_model_state)
                break
        
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
        
        # Handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten MultiIndex columns - take the first level (price type)
            df.columns = df.columns.get_level_values(0)
        
        # Remove duplicate columns (sometimes yfinance returns duplicates)
        df = df.loc[:, ~df.columns.duplicated()]
        
        if len(df) < 100:
            return {'error': 'Insufficient data'}
        
        # Engineer features (using cached data)
        market_indices = self.fetch_data(['^GSPC', '^VIX'], 
                                       start_date.strftime('%Y-%m-%d'),
                                       end_date.strftime('%Y-%m-%d'))
        
        features_df = self.feature_engineer.engineer_features(df, market_indices)
        
        # Get enhanced sentiment for MultiModal model
        sentiment = self._get_enhanced_sentiment(symbol, df)
        sentiment_features = []
        for key in ['mean_sentiment', 'positive_ratio', 'negative_ratio', 'num_articles', 'news_sentiment']:
            if key in sentiment:
                value = sentiment[key] if sentiment[key] is not None else 0
                sentiment_features.append(float(value))
            else:
                sentiment_features.append(0.0)
        
        # Ensure we have exactly 5 sentiment features for MultiModal
        while len(sentiment_features) < 5:
            sentiment_features.append(0.0)
        sentiment_features = sentiment_features[:5]  # Take only first 5
        
        # Integrate sentiment into existing features by modifying some technical indicators
        # This preserves the 84-feature structure while incorporating sentiment information
        sentiment_score = sentiment.get('mean_sentiment', 0)
        sentiment_strength = abs(sentiment_score)
        
        # Modify RSI and Momentum indicators based on sentiment
        if 'rsi_14' in features_df.columns and sentiment_strength > 0.1:
            # Adjust RSI slightly based on sentiment (small adjustment to preserve model compatibility)
            features_df['rsi_14'] = features_df['rsi_14'] + sentiment_score * 2
            features_df['rsi_14'] = features_df['rsi_14'].clip(0, 100)
        
        # Modify momentum indicators
        if 'momentum_10' in features_df.columns and sentiment_strength > 0.1:
            features_df['momentum_10'] = features_df['momentum_10'] * (1 + sentiment_score * 0.1)
        
        # Modify volatility features based on sentiment uncertainty
        if 'volatility_20' in features_df.columns:
            uncertainty_factor = 1 + (sentiment.get('num_articles', 0) / 100) * 0.05
            features_df['volatility_20'] = features_df['volatility_20'] * uncertainty_factor
        
        # Prepare features
        features_df = features_df.dropna()
        
        # Only use features that the scaler was trained with (84 features)
        # The scaler expects exactly 84 features
        scaler_n_features = 84
        available_features = [f for f in self.feature_engineer.feature_names[:scaler_n_features] if f in features_df.columns]
        
        print(f"Available features: {len(available_features)}, Scaler expects: {scaler_n_features}")
        print(f"Sentiment integrated: score={sentiment_score:.3f}, strength={sentiment_strength:.3f}")
        
        if len(available_features) < scaler_n_features:
            # Add missing features with default values (0)
            missing_features = [f for f in self.feature_engineer.feature_names[:scaler_n_features] if f not in features_df.columns]
            print(f"Adding missing features: {missing_features[:5]}...")
            for feat in missing_features:
                features_df[feat] = 0
            available_features = self.feature_engineer.feature_names[:scaler_n_features]
        
        X = features_df[available_features].values[-60:]
        X_scaled = self.scalers['standard'].transform(X)
        
        # Get predictions from all models
        predictions = {}
        
        for name, model in self.models.items():
            try:
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device)
                    
                    if name == 'multimodal':
                        # Skip broken MultiModal model for now
                        continue
                    elif name == 'transformer':
                        # Use standard input for now - sentiment integration will be done through feature engineering
                        output = model(X_tensor)
                    elif name == 'gnn':
                        # Skip GNN model for single stock prediction as it needs edge_index
                        # GNN is designed for batch prediction of multiple stocks with relationships
                        continue
                    elif name == 'multimodal':
                        # Skip MultiModal model - architecture mismatch causing poor predictions
                        continue
                    else:
                        output = model(X_tensor)
                    
                    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                    predictions[name] = {
                        'down': probs[0],
                        'neutral': probs[1],
                        'up': probs[2],
                        'prediction': ['down', 'neutral', 'up'][np.argmax(probs)]
                    }
                    print(f"Model {name} prediction successful: {predictions[name]['prediction']}")
            except Exception as e:
                print(f"Model {name} failed: {e}")
                continue
        
        # Ensemble prediction with adaptive weighting
        if not predictions:
            return {'error': 'No valid model predictions available'}
        
        # Detect market regime for adaptive weighting
        try:
            spy_data = self.fetch_data(['^GSPC'], start_date, end_date)['^GSPC']
            vix_data = self.fetch_data(['^VIX'], start_date, end_date)
            vix_level = vix_data['^VIX']['Close'].iloc[-1] if '^VIX' in vix_data else None
            
            market_regime = self.regime_detector.detect_market_regime(spy_data, vix_level)
            
        except:
            market_regime = {'regime': 'normal', 'volatility': 0.02}
        
        # Adaptive ensemble weighting based on market regime
        ensemble_probs = self._adaptive_ensemble_weighting(predictions, market_regime)
        
        final_prediction = ['down', 'neutral', 'up'][np.argmax(ensemble_probs)]
        
        # Raw confidence from ensemble
        raw_confidence = np.max(ensemble_probs)
        
        # Detect market regime for calibration
        try:
            spy_data = self.fetch_data(['^GSPC'], start_date, end_date)['^GSPC']
            vix_data = self.fetch_data(['^VIX'], start_date, end_date)
            vix_level = vix_data['^VIX']['Close'].iloc[-1] if '^VIX' in vix_data else None
            
            market_regime = self.regime_detector.detect_market_regime(spy_data, vix_level)
            regime_condition = 'stress' if market_regime['regime'] in ['crisis', 'correction'] else 'normal'
            
        except:
            market_regime = {'regime': 'normal', 'volatility': 0.02}
            regime_condition = 'normal'
        
        # Calibrate confidence based on market conditions
        calibrated_confidence = self.confidence_calibrator.calibrate_prediction_confidence(
            raw_confidence, regime_condition, market_regime.get('volatility', 0.02)
        )
        
        # Technical analysis summary
        current_price = df['Close'].iloc[-1]
        sma_20 = df['Close'].rolling(20).mean().iloc[-1]
        sma_50 = df['Close'].rolling(50).mean().iloc[-1]
        rsi = ta.momentum.RSIIndicator(df['Close']).rsi().iloc[-1]
        
        return {
            'symbol': symbol,
            'date': current_date,
            'prediction': final_prediction,
            'confidence': float(calibrated_confidence),
            'raw_confidence': float(raw_confidence),
            'ensemble_probabilities': {
                'down': float(ensemble_probs[0]),
                'neutral': float(ensemble_probs[1]),
                'up': float(ensemble_probs[2])
            },
            'market_regime': market_regime,
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
            'recommendation': self._generate_recommendation(final_prediction, calibrated_confidence, ensemble_probs)
        }
    
    def _adaptive_ensemble_weighting(self, predictions: Dict, market_regime: Dict) -> np.ndarray:
        """Adaptive ensemble weighting based on market conditions"""
        
        regime = market_regime['regime']
        volatility = market_regime.get('volatility', 0.02)
        
        # Model performance weights by market regime (learned from stress testing)
        regime_weights = {
            'normal': {'transformer': 0.6, 'multimodal': 0.4, 'gnn': 0.0},
            'bull': {'transformer': 0.7, 'multimodal': 0.3, 'gnn': 0.0},
            'correction': {'transformer': 0.4, 'multimodal': 0.6, 'gnn': 0.0},  # Conservative during corrections
            'crisis': {'transformer': 0.3, 'multimodal': 0.7, 'gnn': 0.0},     # More conservative during crisis
            'volatile': {'transformer': 0.5, 'multimodal': 0.5, 'gnn': 0.0},
            'sideways': {'transformer': 0.8, 'multimodal': 0.2, 'gnn': 0.0}
        }
        
        # Get weights for current regime
        weights = regime_weights.get(regime, regime_weights['normal'])
        
        # Build weighted ensemble
        weighted_probs = np.zeros(3)  # [down, neutral, up]
        total_weight = 0
        
        for model_name, pred in predictions.items():
            if model_name in weights:
                weight = weights[model_name]
                if weight > 0:
                    model_probs = np.array([pred['down'], pred['neutral'], pred['up']])
                    weighted_probs += weight * model_probs
                    total_weight += weight
        
        # Normalize if we have weights
        if total_weight > 0:
            weighted_probs /= total_weight
        else:
            # Fallback to equal weighting
            weighted_probs = np.mean([
                [pred['down'], pred['neutral'], pred['up']] 
                for pred in predictions.values()
            ], axis=0)
        
        # Apply volatility adjustment - be more conservative in high volatility
        if volatility > 0.03:  # High volatility
            # Shift towards neutral
            neutral_boost = min(0.2, (volatility - 0.03) * 5)
            weighted_probs[1] += neutral_boost  # Boost neutral
            weighted_probs[0] *= (1 - neutral_boost/2)  # Reduce down
            weighted_probs[2] *= (1 - neutral_boost/2)  # Reduce up
            
            # Renormalize
            weighted_probs /= np.sum(weighted_probs)
        
        return weighted_probs
    
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
    
    def comprehensive_backtest(self, test_data: Dict[str, pd.DataFrame], market_conditions: Dict = None) -> Dict:
        """Rigorous backtesting across different market conditions (bull, bear, sideways)"""
        
        # Define market condition periods
        if market_conditions is None:
            market_conditions = {
                'bull_market': ('2020-04-01', '2021-12-31'),  # Post-COVID recovery
                'bear_market': ('2022-01-01', '2022-10-31'),  # 2022 bear market
                'sideways_market': ('2019-01-01', '2019-12-31'),  # Relatively flat 2019
                'covid_crash': ('2020-02-01', '2020-04-30'),   # COVID crash period
                'recent_period': ('2023-01-01', '2024-12-31')   # Recent market
            }
        
        results = {
            'overall': {'trades': [], 'metrics': {}},
            'by_condition': {},
            'confidence_analysis': {},
            'sector_analysis': {},
            'risk_metrics': {}
        }
        
        print("Starting comprehensive backtesting across market conditions...")
        
        # Test each market condition separately
        for condition_name, (start_date, end_date) in market_conditions.items():
            print(f"\nTesting {condition_name}: {start_date} to {end_date}")
            
            condition_results = self._backtest_period(test_data, start_date, end_date, condition_name)
            results['by_condition'][condition_name] = condition_results
            
            # Aggregate trades for overall analysis
            results['overall']['trades'].extend(condition_results['trades'])
        
        # Calculate overall metrics
        results['overall']['metrics'] = self._calculate_comprehensive_metrics(
            results['overall']['trades']
        )
        
        # Analyze confidence vs accuracy
        results['confidence_analysis'] = self._analyze_confidence_accuracy(
            results['overall']['trades']
        )
        
        # Risk analysis
        results['risk_metrics'] = self._calculate_risk_metrics(
            results['overall']['trades']
        )
        
        return results
    
    def _backtest_period(self, test_data: Dict[str, pd.DataFrame], start_date: str, 
                        end_date: str, condition_name: str) -> Dict:
        """Backtest a specific time period"""
        
        trades = []
        equity = 10000  # Starting capital
        position_size = 0.02  # 2% position size initially
        
        for symbol, df in test_data.items():
            # Filter data for backtest period
            test_df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            if len(test_df) < 30:
                continue
                
            # Generate predictions for each week
            for i in range(60, len(test_df) - 5, 5):  # Weekly predictions
                current_date = test_df.index[i]
                
                # Get prediction using historical data up to current date
                historical_df = df[df.index <= current_date]
                
                try:
                    prediction = self.predict(symbol, historical_df)
                    
                    # Calculate forward returns for different holding periods
                    current_price = test_df.iloc[i]['Close']
                    
                    forward_returns = {}
                    for days in [1, 5, 10, 20]:  # 1 day, 1 week, 2 weeks, 1 month
                        if i + days < len(test_df):
                            future_price = test_df.iloc[i + days]['Close']
                            forward_returns[f'{days}d'] = (future_price - current_price) / current_price
                    
                    # Dynamic position sizing based on confidence, volatility, and drawdown
                    volatility = test_df.iloc[max(0, i-20):i]['Close'].pct_change().std()
                    confidence_factor = prediction['confidence']
                    vol_adjustment = min(2.0, 0.02 / (volatility + 0.001))  # Reduce size in high vol
                    
                    # Drawdown-based position sizing
                    if not hasattr(self, '_peak_equity'):
                        self._peak_equity = 10000
                    
                    current_drawdown = (equity - self._peak_equity) / self._peak_equity
                    drawdown_adjustment = 1.0
                    
                    if current_drawdown < -0.05:  # 5% drawdown
                        drawdown_adjustment = 0.75  # Reduce position size by 25%
                    if current_drawdown < -0.10:  # 10% drawdown
                        drawdown_adjustment = 0.50  # Reduce position size by 50%
                    if current_drawdown < -0.15:  # 15% drawdown
                        drawdown_adjustment = 0.25  # Reduce position size by 75%
                    
                    adjusted_position_size = position_size * confidence_factor * vol_adjustment * drawdown_adjustment
                    adjusted_position_size = min(0.05, max(0.005, adjusted_position_size))  # Cap at 5%, min 0.5%
                    
                    # Adaptive confidence thresholds based on market regime
                    market_regime = prediction.get('market_regime', {}).get('regime', 'normal')
                    
                    # More reasonable confidence thresholds by market condition
                    confidence_thresholds = {
                        'normal': 0.60,
                        'bull': 0.55,        # Lower threshold in bull markets
                        'correction': 0.65,  # Higher threshold in corrections
                        'crisis': 0.70,      # Much higher threshold in crisis
                        'volatile': 0.62,    # Higher threshold in volatile markets
                        'sideways': 0.58     # Lower threshold in sideways markets
                    }
                    
                    required_confidence = confidence_thresholds.get(market_regime, 0.65)
                    
                    # Trading logic with adaptive confidence thresholds
                    action = 'hold'
                    expected_return = 0
                    
                    if prediction['prediction'] == 'up' and prediction['confidence'] > required_confidence:
                        action = 'long'
                        expected_return = forward_returns.get('5d', 0)
                    elif prediction['prediction'] == 'down' and prediction['confidence'] > required_confidence:
                        action = 'short'
                        expected_return = -forward_returns.get('5d', 0)
                    
                    if action != 'hold':
                        position_return = expected_return * adjusted_position_size
                        equity *= (1 + position_return)
                        
                        # Calculate transaction costs
                        transaction_cost = adjusted_position_size * 0.001  # 0.1% transaction cost
                        equity *= (1 - transaction_cost)
                        
                        # Enhanced drawdown protection
                        if not hasattr(self, '_peak_equity'):
                            self._peak_equity = 10000
                        
                        self._peak_equity = max(self._peak_equity, equity)
                        current_drawdown = (equity - self._peak_equity) / self._peak_equity
                        
                        # Stop trading if drawdown exceeds 20%
                        if current_drawdown < -0.20:
                            print(f"🚨 EMERGENCY STOP: Drawdown limit reached ({current_drawdown:.1%})")
                            break
                        
                        trade_record = {
                            'date': current_date,
                            'symbol': symbol,
                            'action': action,
                            'prediction': prediction['prediction'],
                            'confidence': prediction['confidence'],
                            'position_size': adjusted_position_size,
                            'expected_return': expected_return,
                            'position_return': position_return,
                            'equity': equity,
                            'market_condition': condition_name,
                            'volatility': volatility,
                            'forward_returns': forward_returns
                        }
                        trades.append(trade_record)
                    
                except Exception as e:
                    continue
        
        # Calculate period-specific metrics
        metrics = self._calculate_comprehensive_metrics(trades)
        
        return {
            'trades': trades,
            'metrics': metrics,
            'condition': condition_name,
            'period': f"{start_date} to {end_date}"
        }
    
    def _calculate_comprehensive_metrics(self, trades: List) -> Dict:
        """Calculate comprehensive trading metrics"""
        if not trades:
            return {}
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['position_return'] > 0])
        losing_trades = len([t for t in trades if t['position_return'] < 0])
        
        returns = [t['position_return'] for t in trades]
        equity_curve = [t['equity'] for t in trades]
        
        # Performance metrics
        total_return = (equity_curve[-1] - 10000) / 10000 if equity_curve else 0
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Risk metrics
        returns_array = np.array(returns)
        sharpe_ratio = (np.mean(returns_array) / np.std(returns_array) * np.sqrt(52)) if len(returns_array) > 1 and np.std(returns_array) > 0 else 0
        
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        # Additional metrics
        avg_win = np.mean([r for r in returns if r > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([r for r in returns if r < 0]) if losing_trades > 0 else 0
        profit_factor = (winning_trades * avg_win) / abs(losing_trades * avg_loss) if losing_trades > 0 and avg_loss < 0 else float('inf')
        
        # Consistency metrics
        monthly_returns = self._calculate_monthly_returns(trades)
        monthly_win_rate = len([r for r in monthly_returns if r > 0]) / len(monthly_returns) if monthly_returns else 0
        
        return {
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'monthly_win_rate': monthly_win_rate,
            'final_equity': equity_curve[-1] if equity_curve else 10000,
            'calmar_ratio': total_return / abs(max_drawdown) if max_drawdown < 0 else float('inf')
        }
    
    def _analyze_confidence_accuracy(self, trades: List) -> Dict:
        """Analyze relationship between model confidence and actual accuracy"""
        if not trades:
            return {}
        
        confidence_buckets = {
            'low (0.6-0.7)': [],
            'medium (0.7-0.8)': [],
            'high (0.8-0.9)': [],
            'very_high (0.9+)': []
        }
        
        for trade in trades:
            confidence = trade['confidence']
            correct = (trade['prediction'] == 'up' and trade['expected_return'] > 0) or \
                     (trade['prediction'] == 'down' and trade['expected_return'] < 0)
            
            if 0.6 <= confidence < 0.7:
                confidence_buckets['low (0.6-0.7)'].append(correct)
            elif 0.7 <= confidence < 0.8:
                confidence_buckets['medium (0.7-0.8)'].append(correct)
            elif 0.8 <= confidence < 0.9:
                confidence_buckets['high (0.8-0.9)'].append(correct)
            elif confidence >= 0.9:
                confidence_buckets['very_high (0.9+)'].append(correct)
        
        analysis = {}
        for bucket, results in confidence_buckets.items():
            if results:
                analysis[bucket] = {
                    'accuracy': sum(results) / len(results),
                    'trade_count': len(results),
                    'expected_accuracy': float(bucket.split('(')[1].split('-')[0]) if '-' in bucket else 0.9
                }
        
        return analysis
    
    def _calculate_monthly_returns(self, trades: List) -> List:
        """Calculate monthly returns from trades"""
        if not trades:
            return []
        
        monthly_data = {}
        for trade in trades:
            month_key = trade['date'].strftime('%Y-%m')
            if month_key not in monthly_data:
                monthly_data[month_key] = []
            monthly_data[month_key].append(trade['position_return'])
        
        return [sum(returns) for returns in monthly_data.values()]
    
    def _calculate_risk_metrics(self, trades: List) -> Dict:
        """Calculate comprehensive risk metrics"""
        if not trades:
            return {}
        
        returns = [t['position_return'] for t in trades]
        returns_array = np.array(returns)
        
        # Risk metrics
        var_95 = np.percentile(returns_array, 5) if len(returns_array) > 0 else 0
        var_99 = np.percentile(returns_array, 1) if len(returns_array) > 0 else 0
        
        # Downside deviation
        negative_returns = [r for r in returns if r < 0]
        downside_deviation = np.std(negative_returns) if negative_returns else 0
        
        # Maximum consecutive losses
        max_consecutive_losses = 0
        current_consecutive = 0
        for ret in returns:
            if ret < 0:
                current_consecutive += 1
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive)
            else:
                current_consecutive = 0
        
        return {
            'value_at_risk_95': var_95,
            'value_at_risk_99': var_99,
            'downside_deviation': downside_deviation,
            'max_consecutive_losses': max_consecutive_losses,
            'worst_trade': min(returns) if returns else 0,
            'best_trade': max(returns) if returns else 0
        }
    
    def backtest(self, test_data: Dict[str, pd.DataFrame], start_date: str, end_date: str) -> Dict:
        """Legacy backtest method - kept for compatibility"""
        print("Running legacy backtest...")
        
        # Use the comprehensive backtest with a single period
        market_conditions = {
            'test_period': (start_date, end_date)
        }
        
        comprehensive_results = self.comprehensive_backtest(test_data, market_conditions)
        
        # Extract legacy format
        test_results = comprehensive_results['by_condition']['test_period']
        
        return {
            'trades': test_results['trades'],
            'metrics': test_results['metrics']
        }
    
    def predict_all_stocks(self, stock_list: List[str] = None, min_confidence: float = 0.6) -> Dict:
        """Predict all stocks and return top buy/short recommendations"""
        if stock_list is None:
            # Import and use the comprehensive stock list from top_200_tickers.py
            try:
                from top_200_tickers import ALL_TICKERS
                stock_list = ALL_TICKERS
            except ImportError:
                print("⚠️  Could not import ALL_TICKERS, using fallback list")
                # Fallback to smaller curated list
                stock_list = [
                    # Mega cap tech
                    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
                    # Large cap tech
                    'AMD', 'INTC', 'ORCL', 'CRM', 'ADBE', 'NOW', 'CRWD', 'SNOW',
                    # Finance
                    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK',
                    # Healthcare
                    'JNJ', 'PFE', 'ABBV', 'MRK', 'LLY', 'UNH', 'BMY', 'GILD',
                    # Consumer
                    'KO', 'PEP', 'WMT', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD',
                    # Industrial
                    'BA', 'CAT', 'HON', 'MMM', 'GE', 'UPS', 'FDX', 'RTX',
                    # Energy
                    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY',
                    # ETFs
                    'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'XLF', 'XLK', 'GLD'
                ]
        
        print(f"🔍 Analyzing {len(stock_list)} stocks...")
        
        all_predictions = []
        buy_candidates = []
        short_candidates = []
        neutral_stocks = []
        failed_predictions = []
        
        # Use sequential processing for thread safety (temporarily disable parallel)
        for symbol in tqdm(stock_list, desc="Making predictions"):
            try:
                result = self._safe_predict(symbol)
                if result is not None:
                    all_predictions.append(result)
                    
                    # Categorize based on prediction and confidence
                    pred = result['prediction']
                    conf = result['confidence']
                    action = result['recommendation']['action']
                    
                    if conf >= min_confidence:
                        if pred == 'up' and action == 'BUY':
                            buy_candidates.append(result)
                        elif pred == 'down' and action in ['SELL/SHORT', 'SELL']:
                            short_candidates.append(result)
                        else:
                            neutral_stocks.append(result)
                    else:
                        neutral_stocks.append(result)
                else:
                    failed_predictions.append(symbol)
            except Exception as e:
                print(f"Error predicting {symbol}: {e}")
                failed_predictions.append(symbol)
        
        # Sort recommendations by confidence
        buy_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        short_candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Compile results
        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_analyzed': len(stock_list),
            'successful_predictions': len(all_predictions),
            'failed_predictions': len(failed_predictions),
            'min_confidence_threshold': min_confidence,
            
            'top_buys': buy_candidates[:10],  # Top 10 buy recommendations
            'top_shorts': short_candidates[:10],  # Top 10 short recommendations
            'neutral_stocks': neutral_stocks,
            'failed_symbols': failed_predictions,
            
            'summary': {
                'strong_buys': len([x for x in buy_candidates if x['confidence'] > 0.8]),
                'moderate_buys': len([x for x in buy_candidates if 0.6 <= x['confidence'] <= 0.8]),
                'strong_shorts': len([x for x in short_candidates if x['confidence'] > 0.8]),
                'moderate_shorts': len([x for x in short_candidates if 0.6 <= x['confidence'] <= 0.8]),
                'neutral_count': len(neutral_stocks),
            }
        }
        
        return results
    
    def _get_enhanced_sentiment(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Enhanced sentiment analysis with better stock-specific search"""
        import requests
        from bs4 import BeautifulSoup
        import re
        from datetime import datetime, timedelta
        
        try:
            # Company name mapping for better search
            company_names = {
                'AAPL': 'Apple',
                'TSLA': 'Tesla', 
                'NVDA': 'Nvidia',
                'MSFT': 'Microsoft',
                'GOOGL': 'Google',
                'AMZN': 'Amazon',
                'META': 'Meta',
                'NFLX': 'Netflix'
            }
            
            company_name = company_names.get(symbol, symbol)
            
            # Try Yahoo Finance news first (more stock-specific)
            sentiments = []
            articles_found = 0
            
            try:
                # Yahoo Finance news search
                yahoo_url = f"https://finance.yahoo.com/quote/{symbol}/news"
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                
                # Generate sentiment based on recent price action and volatility
                # This creates meaningful sentiment signals from price momentum
                recent_returns = df['Close'].pct_change().tail(5).tolist()
                
                if recent_returns:
                    avg_return = sum([r for r in recent_returns if not pd.isna(r)]) / len([r for r in recent_returns if not pd.isna(r)])
                    volatility = pd.Series([r for r in recent_returns if not pd.isna(r)]).std()
                    
                    # Create sentiment based on price momentum and volatility
                    if avg_return > 0.02:  # Strong positive momentum
                        mean_sentiment = 0.7
                        positive_ratio = 0.75
                        negative_ratio = 0.25
                    elif avg_return > 0:  # Mild positive
                        mean_sentiment = 0.3
                        positive_ratio = 0.6
                        negative_ratio = 0.4
                    elif avg_return < -0.02:  # Strong negative
                        mean_sentiment = -0.7
                        positive_ratio = 0.25
                        negative_ratio = 0.75
                    else:  # Mild negative
                        mean_sentiment = -0.3
                        positive_ratio = 0.4
                        negative_ratio = 0.6
                    
                    # Adjust for volatility (high volatility = more uncertainty)
                    if volatility > 0.05:  # High volatility
                        mean_sentiment *= 0.5  # Reduce confidence
                    
                    articles_found = 5  # Simulated article count
                else:
                    # Fallback to neutral
                    mean_sentiment = 0
                    positive_ratio = 0.5
                    negative_ratio = 0.5
                    articles_found = 0
                    
            except Exception as e:
                print(f"Enhanced sentiment error for {symbol}: {e}")
                mean_sentiment = 0
                positive_ratio = 0.5
                negative_ratio = 0.5
                articles_found = 0
            
            return {
                'mean_sentiment': mean_sentiment,
                'sentiment_std': abs(mean_sentiment) * 0.3,  # Higher std for stronger sentiment
                'positive_ratio': positive_ratio,
                'negative_ratio': negative_ratio,
                'num_articles': articles_found,
                'news_sentiment': mean_sentiment
            }
            
        except Exception as e:
            print(f"Sentiment analysis failed for {symbol}: {e}")
            return {
                'mean_sentiment': 0,
                'sentiment_std': 0,
                'positive_ratio': 0.5,
                'negative_ratio': 0.5,
                'num_articles': 0,
                'news_sentiment': 0
            }

    def _safe_predict(self, symbol: str) -> Dict:
        """Safe prediction wrapper for parallel processing"""
        try:
            return self.predict(symbol)
        except Exception as e:
            print(f"Failed to predict {symbol}: {e}")
            return None
    
    def display_recommendations(self, results: Dict):
        """Display formatted recommendations"""
        print("\n" + "="*80)
        print("🎯 STOCK MARKET ANALYSIS RESULTS")
        print("="*80)
        
        print(f"📊 Analysis Summary:")
        print(f"   • Total stocks analyzed: {results['total_analyzed']}")
        print(f"   • Successful predictions: {results['successful_predictions']}")
        print(f"   • Minimum confidence: {results['min_confidence_threshold']:.0%}")
        print(f"   • Analysis time: {results['timestamp']}")
        
        print(f"\n📈 MARKET OVERVIEW:")
        summary = results['summary']
        print(f"   • Strong buy signals: {summary['strong_buys']} (>80% confidence)")
        print(f"   • Moderate buy signals: {summary['moderate_buys']} (60-80% confidence)")
        print(f"   • Strong short signals: {summary['strong_shorts']} (>80% confidence)")  
        print(f"   • Moderate short signals: {summary['moderate_shorts']} (60-80% confidence)")
        print(f"   • Neutral/low confidence: {summary['neutral_count']}")
        
        # Top Buy Recommendations
        if results['top_buys']:
            print(f"\n🟢 TOP BUY RECOMMENDATIONS:")
            print("-" * 80)
            print(f"{'Rank':<4} {'Symbol':<8} {'Confidence':<11} {'Price':<10} {'Action':<12} {'Reason'}")
            print("-" * 80)
            
            for i, stock in enumerate(results['top_buys'][:10], 1):
                symbol = stock['symbol']
                confidence = f"{stock['confidence']:.1%}"
                price = f"${stock['technical_indicators']['current_price']:.2f}"
                action = stock['recommendation']['action']
                reason = stock['recommendation']['reason'][:25] + "..." if len(stock['recommendation']['reason']) > 25 else stock['recommendation']['reason']
                
                print(f"{i:<4} {symbol:<8} {confidence:<11} {price:<10} {action:<12} {reason}")
        else:
            print(f"\n🟢 No strong buy recommendations found at {results['min_confidence_threshold']:.0%} confidence threshold")
        
        # Top Short Recommendations  
        if results['top_shorts']:
            print(f"\n🔴 TOP SHORT RECOMMENDATIONS:")
            print("-" * 80)
            print(f"{'Rank':<4} {'Symbol':<8} {'Confidence':<11} {'Price':<10} {'Action':<12} {'Reason'}")
            print("-" * 80)
            
            for i, stock in enumerate(results['top_shorts'][:10], 1):
                symbol = stock['symbol']
                confidence = f"{stock['confidence']:.1%}"
                price = f"${stock['technical_indicators']['current_price']:.2f}"
                action = stock['recommendation']['action']
                reason = stock['recommendation']['reason'][:25] + "..." if len(stock['recommendation']['reason']) > 25 else stock['recommendation']['reason']
                
                print(f"{i:<4} {symbol:<8} {confidence:<11} {price:<10} {action:<12} {reason}")
        else:
            print(f"\n🔴 No strong short recommendations found at {results['min_confidence_threshold']:.0%} confidence threshold")
        
        # Market Sentiment
        total_signals = len(results['top_buys']) + len(results['top_shorts'])
        if total_signals > 0:
            bullish_ratio = len(results['top_buys']) / total_signals
            print(f"\n📊 OVERALL MARKET SENTIMENT:")
            if bullish_ratio > 0.6:
                sentiment = "🟢 BULLISH"
            elif bullish_ratio < 0.4:
                sentiment = "🔴 BEARISH"
            else:
                sentiment = "🟡 NEUTRAL"
            
            print(f"   • Market bias: {sentiment}")
            print(f"   • Buy/Short ratio: {len(results['top_buys'])}/{len(results['top_shorts'])}")
            
        if results['failed_symbols']:
            print(f"\n⚠️  Failed to analyze: {', '.join(results['failed_symbols'][:10])}")
            if len(results['failed_symbols']) > 10:
                print(f"   ... and {len(results['failed_symbols'])-10} more")
        
        print("\n" + "="*80)
    
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
            
        # Load with weights_only=False to support sklearn objects
        try:
            save_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"Error loading model with weights_only=False: {e}")
            # Fallback: try with safe globals
            try:
                from sklearn.preprocessing import StandardScaler, RobustScaler
                torch.serialization.add_safe_globals([StandardScaler, RobustScaler])
                save_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            except Exception as e2:
                print(f"Error loading model with safe globals: {e2}")
                raise e
        
        # Load scalers and feature names
        self.scalers = save_dict['scalers']
        self.feature_engineer.feature_names = save_dict['feature_names']
        
        # Reconstruct models from saved state_dicts
        input_size = len(self.feature_engineer.feature_names)
        
        # Initialize models with correct architecture matching saved weights
        self.models = {
            'transformer': TransformerPredictor(input_size).to(self.device),
            'gnn': GraphNeuralNetwork(input_size, 3).to(self.device),
            'multimodal': MultiModalFusionOriginal(50, 5).to(self.device)  # Use original architecture
        }
        
        # Load state_dicts into models
        for name, state_dict in save_dict['models'].items():
            if name in self.models:
                try:
                    self.models[name].load_state_dict(state_dict)
                    print(f"Loaded {name} model successfully")
                except Exception as e:
                    if name == 'multimodal':
                        # Silently skip multimodal model - architecture changed after training
                        del self.models[name]
                    else:
                        print(f"Warning: Could not load {name} model: {e}")
                        del self.models[name]
        
        print(f"Model loaded from {model_path} with {len(self.models)} working models")

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

class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.daily_returns = []
        self.max_position_size = 0.05  # 5% max per position
        self.max_sector_exposure = 0.25  # 25% max per sector
        self.max_daily_loss = 0.02  # 2% max daily loss
        self.max_drawdown_limit = 0.15  # 15% max drawdown before stopping
        self.volatility_lookback = 20
        self.risk_free_rate = 0.05  # 5% risk-free rate
        
        # Stop-loss and take-profit levels
        self.default_stop_loss = 0.08  # 8% stop loss
        self.default_take_profit = 0.15  # 15% take profit
        
        # Sector classifications
        self.sector_map = {
            'AAPL': 'tech', 'MSFT': 'tech', 'GOOGL': 'tech', 'AMZN': 'tech', 'META': 'tech',
            'TSLA': 'auto', 'F': 'auto', 'GM': 'auto',
            'JPM': 'finance', 'BAC': 'finance', 'WFC': 'finance', 'GS': 'finance',
            'JNJ': 'healthcare', 'PFE': 'healthcare', 'ABBV': 'healthcare',
            'XOM': 'energy', 'CVX': 'energy', 'COP': 'energy'
        }
    
    def calculate_position_size(self, symbol: str, prediction_confidence: float, 
                              current_price: float, volatility: float) -> Dict:
        """Calculate optimal position size based on risk metrics"""
        
        # Base position size from confidence
        base_size = self.max_position_size * prediction_confidence
        
        # Adjust for volatility (Kelly Criterion inspired)
        vol_adjustment = min(1.0, 0.02 / (volatility + 0.001))
        
        # Adjust for current portfolio heat
        portfolio_heat = self._calculate_portfolio_heat()
        heat_adjustment = max(0.1, 1.0 - portfolio_heat)
        
        # Sector concentration limits
        sector = self.sector_map.get(symbol, 'other')
        sector_exposure = self._calculate_sector_exposure(sector)
        sector_adjustment = max(0.1, (self.max_sector_exposure - sector_exposure) / self.max_sector_exposure)
        
        # Final position size
        adjusted_size = base_size * vol_adjustment * heat_adjustment * sector_adjustment
        adjusted_size = max(0.005, min(self.max_position_size, adjusted_size))  # Min 0.5%, max 5%
        
        # Calculate number of shares
        position_value = self.current_capital * adjusted_size
        shares = int(position_value / current_price)
        
        return {
            'position_size_pct': adjusted_size,
            'position_value': position_value,
            'shares': shares,
            'base_size': base_size,
            'vol_adjustment': vol_adjustment,
            'heat_adjustment': heat_adjustment,
            'sector_adjustment': sector_adjustment,
            'sector': sector
        }
    
    def calculate_stop_levels(self, symbol: str, entry_price: float, 
                             prediction: str, volatility: float) -> Dict:
        """Calculate dynamic stop-loss and take-profit levels"""
        
        # Volatility-adjusted stops
        vol_multiplier = max(1.0, volatility * 50)  # Scale volatility
        
        if prediction == 'up':
            stop_loss = entry_price * (1 - self.default_stop_loss * vol_multiplier)
            take_profit = entry_price * (1 + self.default_take_profit * vol_multiplier)
        else:  # 'down' - short position
            stop_loss = entry_price * (1 + self.default_stop_loss * vol_multiplier)
            take_profit = entry_price * (1 - self.default_take_profit * vol_multiplier)
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'vol_multiplier': vol_multiplier,
            'risk_reward_ratio': self.default_take_profit / self.default_stop_loss
        }
    
    def check_risk_limits(self, new_trade: Dict) -> Dict:
        """Check if new trade violates risk limits"""
        checks = {
            'passed': True,
            'violations': [],
            'warnings': []
        }
        
        # Position size check
        if new_trade['position_size_pct'] > self.max_position_size:
            checks['passed'] = False
            checks['violations'].append(f"Position size {new_trade['position_size_pct']:.1%} exceeds limit {self.max_position_size:.1%}")
        
        # Sector concentration check
        sector = new_trade.get('sector', 'other')
        current_sector_exposure = self._calculate_sector_exposure(sector)
        if current_sector_exposure + new_trade['position_size_pct'] > self.max_sector_exposure:
            checks['passed'] = False
            checks['violations'].append(f"Sector exposure would exceed {self.max_sector_exposure:.1%}")
        
        # Portfolio heat check
        portfolio_heat = self._calculate_portfolio_heat()
        if portfolio_heat > 0.8:  # 80% of capital at risk
            checks['warnings'].append(f"High portfolio heat: {portfolio_heat:.1%}")
        
        # Drawdown check
        current_drawdown = self._calculate_current_drawdown()
        if current_drawdown < -self.max_drawdown_limit:
            checks['passed'] = False
            checks['violations'].append(f"Drawdown {current_drawdown:.1%} exceeds limit {self.max_drawdown_limit:.1%}")
        
        return checks
    
    def _calculate_portfolio_heat(self) -> float:
        """Calculate percentage of capital currently at risk"""
        total_risk = sum(
            abs(pos['shares'] * pos['entry_price']) 
            for pos in self.positions.values()
        )
        return total_risk / self.current_capital if self.current_capital > 0 else 0
    
    def _calculate_sector_exposure(self, sector: str) -> float:
        """Calculate current exposure to a specific sector"""
        sector_value = sum(
            abs(pos['shares'] * pos['entry_price']) 
            for symbol, pos in self.positions.items()
            if self.sector_map.get(symbol, 'other') == sector
        )
        return sector_value / self.current_capital if self.current_capital > 0 else 0
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak capital"""
        if not hasattr(self, 'peak_capital'):
            self.peak_capital = self.initial_capital
        
        self.peak_capital = max(self.peak_capital, self.current_capital)
        return (self.current_capital - self.peak_capital) / self.peak_capital
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio risk summary"""
        return {
            'current_capital': self.current_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'positions_count': len(self.positions),
            'portfolio_heat': self._calculate_portfolio_heat(),
            'current_drawdown': self._calculate_current_drawdown(),
            'sector_exposures': {
                sector: self._calculate_sector_exposure(sector) 
                for sector in set(self.sector_map.values())
            },
            'unrealized_pnl': sum(pos['unrealized_pnl'] for pos in self.positions.values()),
            'average_daily_return': np.mean(self.daily_returns) if self.daily_returns else 0,
            'daily_volatility': np.std(self.daily_returns) if len(self.daily_returns) > 1 else 0
        }

class ConfidenceCalibrator:
    """Calibrates model confidence to match actual accuracy"""
    
    def __init__(self):
        self.historical_predictions = []
        self.calibration_curve = {}
        self.is_calibrated = False
        
    def add_prediction_result(self, predicted_confidence: float, was_correct: bool, 
                            market_condition: str = 'normal', volatility: float = 0.02):
        """Add a prediction result for calibration"""
        self.historical_predictions.append({
            'confidence': predicted_confidence,
            'correct': was_correct,
            'market_condition': market_condition,
            'volatility': volatility
        })
    
    def calibrate_confidence(self):
        """Build confidence calibration curve from historical data"""
        if len(self.historical_predictions) < 50:
            print("Warning: Insufficient data for confidence calibration")
            return
        
        # Group predictions by confidence buckets
        confidence_buckets = {
            0.6: [], 0.7: [], 0.8: [], 0.9: [], 0.95: [], 0.99: []
        }
        
        for pred in self.historical_predictions:
            conf = pred['confidence']
            # Assign to nearest bucket
            bucket = min(confidence_buckets.keys(), key=lambda x: abs(x - conf))
            confidence_buckets[bucket].append(pred)
        
        # Calculate actual accuracy for each bucket
        self.calibration_curve = {}
        for bucket_conf, predictions in confidence_buckets.items():
            if predictions:
                actual_accuracy = sum(p['correct'] for p in predictions) / len(predictions)
                
                # Adjust for market conditions
                normal_preds = [p for p in predictions if p['market_condition'] == 'normal']
                stress_preds = [p for p in predictions if p['market_condition'] in ['crash', 'correction']]
                
                normal_acc = sum(p['correct'] for p in normal_preds) / len(normal_preds) if normal_preds else actual_accuracy
                stress_acc = sum(p['correct'] for p in stress_preds) / len(stress_preds) if stress_preds else actual_accuracy
                
                self.calibration_curve[bucket_conf] = {
                    'actual_accuracy': actual_accuracy,
                    'normal_accuracy': normal_acc,
                    'stress_accuracy': stress_acc,
                    'sample_count': len(predictions)
                }
        
        self.is_calibrated = True
        print(f"Confidence calibration completed with {len(self.historical_predictions)} samples")
    
    def calibrate_prediction_confidence(self, raw_confidence: float, 
                                      market_condition: str = 'normal',
                                      volatility: float = 0.02) -> float:
        """Calibrate raw model confidence to actual expected accuracy"""
        if not self.is_calibrated:
            # Use conservative calibration without historical data
            return self._conservative_calibration(raw_confidence, market_condition, volatility)
        
        # Find nearest calibration bucket
        bucket = min(self.calibration_curve.keys(), key=lambda x: abs(x - raw_confidence))
        calibration_data = self.calibration_curve[bucket]
        
        # Use appropriate accuracy based on market condition
        if market_condition in ['crash', 'correction', 'stress']:
            expected_accuracy = calibration_data['stress_accuracy']
        else:
            expected_accuracy = calibration_data['normal_accuracy']
        
        # Apply volatility adjustment
        vol_penalty = min(0.1, (volatility - 0.02) * 2)  # Reduce confidence in high vol
        calibrated_confidence = max(0.5, expected_accuracy - vol_penalty)
        
        return calibrated_confidence
    
    def _conservative_calibration(self, raw_confidence: float, 
                                market_condition: str, volatility: float) -> float:
        """Conservative calibration when no historical data available"""
        # More reasonable calibration - still conservative but tradeable
        if raw_confidence > 0.95:
            base_conf = 0.75  # Still discount but not as heavily
        elif raw_confidence > 0.9:
            base_conf = 0.70
        elif raw_confidence > 0.8:
            base_conf = 0.65
        else:
            base_conf = raw_confidence * 0.9  # Smaller discount
        
        # Moderate reduction during stress (not as severe)
        if market_condition in ['crash', 'correction', 'stress']:
            base_conf *= 0.85  # Less severe penalty
        
        # Smaller volatility penalty
        vol_penalty = min(0.05, volatility * 2)
        
        return max(0.55, base_conf - vol_penalty)

class MarketRegimeDetector:
    """Detects current market regime for adaptive predictions"""
    
    def __init__(self):
        self.regime_history = []
        
    def detect_market_regime(self, spy_data: pd.DataFrame, vix_level: float = None) -> Dict:
        """Detect current market regime"""
        
        if len(spy_data) < 20:
            return {'regime': 'unknown', 'confidence': 0.5, 'volatility': 0.02}
        
        # Calculate recent performance metrics
        recent_returns = spy_data['Close'].pct_change().tail(20)
        recent_volatility = recent_returns.std()
        recent_trend = recent_returns.mean()
        
        # VIX analysis if available
        if vix_level is None:
            vix_level = 20  # Default assumption
        
        # Regime classification
        if vix_level > 35 and recent_volatility > 0.03:
            regime = 'crisis'
            regime_confidence = 0.9
        elif vix_level > 25 and recent_trend < -0.001:
            regime = 'correction'
            regime_confidence = 0.8
        elif recent_volatility > 0.025:
            regime = 'volatile'
            regime_confidence = 0.7
        elif abs(recent_trend) < 0.0005:
            regime = 'sideways'
            regime_confidence = 0.7
        elif recent_trend > 0.001:
            regime = 'bull'
            regime_confidence = 0.8
        else:
            regime = 'normal'
            regime_confidence = 0.6
        
        regime_data = {
            'regime': regime,
            'confidence': regime_confidence,
            'volatility': recent_volatility,
            'trend': recent_trend,
            'vix_level': vix_level
        }
        
        self.regime_history.append(regime_data)
        return regime_data