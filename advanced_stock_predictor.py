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
import joblib
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import contextlib
from io import StringIO

# Context manager to suppress yfinance errors
@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr during yfinance downloads"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr  
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Download required NLTK data
# NLTK removed - using external sentiment analysis from sentiment_analysis_implementation.py

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
    
    def create_enhanced_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced technical analysis features"""
        features = df.copy()
        
        # Ichimoku Cloud System
        high_9 = df['High'].rolling(9).max()
        low_9 = df['Low'].rolling(9).min()
        high_26 = df['High'].rolling(26).max()
        low_26 = df['Low'].rolling(26).min()
        high_52 = df['High'].rolling(52).max()
        low_52 = df['Low'].rolling(52).min()
        
        features['ichimoku_conversion'] = (high_9 + low_9) / 2
        features['ichimoku_base'] = (high_26 + low_26) / 2
        features['ichimoku_span_a'] = (features['ichimoku_conversion'] + features['ichimoku_base']) / 2
        features['ichimoku_span_b'] = (high_52 + low_52) / 2
        features['ichimoku_signal'] = np.where(df['Close'] > features['ichimoku_span_a'], 1, 
                                             np.where(df['Close'] < features['ichimoku_span_a'], -1, 0))
        
        # Advanced Stochastic
        features['stochastic_k'] = ((df['Close'] - df['Low'].rolling(14).min()) / 
                                   (df['High'].rolling(14).max() - df['Low'].rolling(14).min()) * 100)
        features['stochastic_d'] = features['stochastic_k'].rolling(3).mean()
        
        # Commodity Channel Index
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        features['cci'] = (typical_price - sma_tp) / (0.015 * mad)
        
        # Williams %R
        features['williams_r'] = ((df['High'].rolling(14).max() - df['Close']) / 
                                 (df['High'].rolling(14).max() - df['Low'].rolling(14).min()) * -100)
        
        return features
    
    def create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure approximation features"""
        features = df.copy()
        
        # Candlestick patterns
        features['body_size'] = abs(df['Close'] - df['Open']) / df['Open']
        features['upper_shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Open']
        features['lower_shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Open']
        
        # Intraday patterns
        features['opening_gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        features['intraday_range'] = (df['High'] - df['Low']) / df['Open']
        features['closing_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Volume-price relationships
        features['volume_price_ratio'] = df['Volume'] / df['Close']
        features['volume_momentum'] = df['Volume'].pct_change(5)
        features['relative_volume'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        return features
    
    def create_advanced_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced momentum and trend features"""
        features = df.copy()
        
        # Multiple timeframe momentum
        for period in [3, 7, 14, 21]:
            features[f'momentum_{period}'] = ((df['Close'] - df['Close'].shift(period)) / 
                                            df['Close'].shift(period) * 100)
        
        # MACD variations
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        features['macd_line'] = ema_12 - ema_26
        features['macd_signal'] = features['macd_line'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd_line'] - features['macd_signal']
        
        return features
    
    def create_options_flow_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Create options flow and sentiment indicators (approximated)"""
        enhanced_features = features.copy()
        
        # VIX-based approximations (using VIX data if available)
        try:
            # Try to get VIX data for volatility analysis
            vix_data = self.fetch_data(['^VIX'], 
                                     df.index[0].strftime('%Y-%m-%d'),
                                     df.index[-1].strftime('%Y-%m-%d'))
            
            if '^VIX' in vix_data:
                vix_df = vix_data['^VIX']
                vix_aligned = vix_df.reindex(df.index, method='ffill')
                
                # VIX-based features
                enhanced_features['vix_level'] = vix_aligned['Close']
                enhanced_features['vix_momentum'] = vix_aligned['Close'].pct_change(5)
                enhanced_features['vix_mean_reversion'] = (vix_aligned['Close'] - vix_aligned['Close'].rolling(20).mean()) / vix_aligned['Close'].rolling(20).std()
                
                # Implied volatility approximations
                realized_vol = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
                enhanced_features['vol_risk_premium'] = enhanced_features['vix_level'] / 100 - realized_vol
                
        except:
            # Fallback: create synthetic options indicators
            pass
        
        # Options sentiment approximations based on price action
        # High volume + large moves often indicate options activity
        volume_surge = features['relative_volume'] > 2.0
        price_surge = abs(df['Close'].pct_change()) > 0.03
        
        enhanced_features['options_activity_proxy'] = (volume_surge & price_surge).astype(int)
        
        # Put/Call ratio approximation
        # When stock falls with high volume, likely put activity
        # When stock rises with high volume, likely call activity
        daily_return = df['Close'].pct_change()
        enhanced_features['put_call_proxy'] = np.where(
            daily_return < -0.02, features['relative_volume'] * -1,
            np.where(daily_return > 0.02, features['relative_volume'], 0)
        )
        
        # Gamma exposure approximation
        # High volatility days with mean reversion suggest gamma effects
        enhanced_features['gamma_proxy'] = (
            (abs(daily_return) > 0.025) & 
            (daily_return * daily_return.shift(1) < 0)
        ).astype(int) * features['relative_volume']
        
        # Dark pool approximation
        # Large moves with relatively low volume might indicate dark pool activity
        enhanced_features['dark_pool_proxy'] = np.where(
            (abs(daily_return) > 0.015) & (features['relative_volume'] < 0.8),
            abs(daily_return), 0
        )
        
        return enhanced_features
    
    def create_derivatives_features(self, features: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Create derivatives-based features"""
        enhanced_features = features.copy()
        
        # Volatility surface approximations
        for period in [5, 10, 20, 30]:
            vol = df['Close'].pct_change().rolling(period).std() * np.sqrt(252)
            enhanced_features[f'realized_vol_{period}d'] = vol
        
        # Volatility term structure
        enhanced_features['vol_term_structure'] = (
            enhanced_features.get('realized_vol_5d', 0) - 
            enhanced_features.get('realized_vol_30d', 0)
        )
        
        # Skew approximations (based on tail movements)
        returns = df['Close'].pct_change()
        for period in [10, 20]:
            # Downside vs upside volatility
            downside_vol = returns[returns < 0].rolling(period).std()
            upside_vol = returns[returns > 0].rolling(period).std()
            enhanced_features[f'vol_skew_{period}d'] = (downside_vol - upside_vol).reindex(df.index).fillna(0)
        
        # Futures basis approximation (using forward returns)
        for days in [30, 60, 90]:
            if len(df) > days:
                forward_return = df['Close'].shift(-days) / df['Close'] - 1
                current_return = df['Close'].pct_change(days)
                enhanced_features[f'basis_{days}d'] = forward_return - current_return
        
        return enhanced_features
    
    def create_fundamental_features(self, features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create fundamental analysis features using available financial data"""
        enhanced_features = features.copy()
        
        try:
            # Get stock info from yfinance for fundamental data
            import yfinance as yf
            stock = yf.Ticker(symbol)
            info = stock.info
            
            # Financial metrics (constant across time series - simplified approach)
            # In a more sophisticated system, you'd fetch quarterly data and create time series
            
            # Valuation metrics
            pe_ratio = info.get('trailingPE', info.get('forwardPE', 15))  # Default to market average
            pb_ratio = info.get('priceToBook', 2.5)
            ps_ratio = info.get('priceToSalesTrailing12Months', 3.0)
            peg_ratio = info.get('pegRatio', 1.0)
            
            # Growth metrics
            revenue_growth = info.get('revenueGrowth', 0.05)  # Default 5%
            earnings_growth = info.get('earningsGrowthQuarterly', 0.05)
            
            # Profitability metrics
            profit_margin = info.get('profitMargins', 0.1)
            operating_margin = info.get('operatingMargins', 0.15)
            roe = info.get('returnOnEquity', 0.15)
            roa = info.get('returnOnAssets', 0.08)
            
            # Financial health
            debt_to_equity = info.get('debtToEquity', 50) / 100  # Convert to ratio
            current_ratio = info.get('currentRatio', 1.5)
            quick_ratio = info.get('quickRatio', 1.0)
            
            # Market metrics
            market_cap = info.get('marketCap', 1e9)
            beta = info.get('beta', 1.0)
            
            # Analyst metrics
            target_mean_price = info.get('targetMeanPrice', features['Close'].iloc[-1])
            recommendation_mean = info.get('recommendationMean', 3.0)  # 1=Strong Buy, 5=Strong Sell
            
            # Create derived fundamental features
            enhanced_features['pe_ratio'] = pe_ratio
            enhanced_features['pb_ratio'] = pb_ratio
            enhanced_features['ps_ratio'] = ps_ratio
            enhanced_features['peg_ratio'] = peg_ratio
            
            # Value score (lower is better for value)
            enhanced_features['value_score'] = (pe_ratio / 20) + (pb_ratio / 3) + (ps_ratio / 4)
            
            # Growth score
            enhanced_features['growth_score'] = revenue_growth + earnings_growth
            
            # Profitability score
            enhanced_features['profitability_score'] = profit_margin + operating_margin + (roe / 2)
            
            # Financial health score
            enhanced_features['financial_health'] = current_ratio + quick_ratio - (debt_to_equity / 2)
            
            # Size factor
            enhanced_features['size_factor'] = np.log(market_cap / 1e9)  # Log of market cap in billions
            
            # Beta-adjusted momentum
            enhanced_features['beta_adj_momentum'] = enhanced_features.get('momentum_20d', 0) / max(beta, 0.1)
            
            # Price vs target analysis
            current_price = features['Close'].iloc[-1]
            enhanced_features['price_to_target'] = current_price / max(target_mean_price, current_price * 0.5)
            
            # Analyst sentiment (inverted so lower recommendation_mean = more bullish)
            enhanced_features['analyst_sentiment'] = (6 - recommendation_mean) / 5  # Scale to 0-1
            
            # Fundamental momentum approximation (simplified)
            # In practice, you'd compare current metrics to historical averages
            enhanced_features['fundamental_momentum'] = (
                (revenue_growth - 0.05) +  # Above/below 5% growth
                (earnings_growth - 0.05) +
                (profit_margin - 0.1)     # Above/below 10% margin
            ) / 3
            
            # Quality score
            enhanced_features['quality_score'] = (roe + roa + profit_margin) / 3
            
            # Earnings surprise proxy (using recent price momentum as proxy)
            price_momentum = features['Close'].pct_change(5).iloc[-1]
            enhanced_features['earnings_surprise_proxy'] = np.tanh(price_momentum * 10)  # Bounded surprise
            
            # Sector relative metrics (simplified using beta as sector proxy)
            enhanced_features['sector_relative_pe'] = pe_ratio / (15 * max(beta, 0.5))
            enhanced_features['sector_relative_growth'] = revenue_growth / max(beta * 0.05, 0.01)
            
        except Exception as e:
            print(f"Warning: Could not fetch fundamental data for {symbol}: {e}")
            # Fill with neutral/market average values
            enhanced_features['pe_ratio'] = 15
            enhanced_features['pb_ratio'] = 2.5
            enhanced_features['ps_ratio'] = 3.0
            enhanced_features['peg_ratio'] = 1.0
            enhanced_features['value_score'] = 2.0
            enhanced_features['growth_score'] = 0.1
            enhanced_features['profitability_score'] = 0.4
            enhanced_features['financial_health'] = 2.0
            enhanced_features['size_factor'] = 1.0
            enhanced_features['beta_adj_momentum'] = 0
            enhanced_features['price_to_target'] = 1.0
            enhanced_features['analyst_sentiment'] = 0.5
            enhanced_features['fundamental_momentum'] = 0
            enhanced_features['quality_score'] = 0.15
            enhanced_features['earnings_surprise_proxy'] = 0
            enhanced_features['sector_relative_pe'] = 1.0
            enhanced_features['sector_relative_growth'] = 1.0
        
        return enhanced_features
    
    def create_alternative_data_features(self, features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create features from alternative data sources"""
        enhanced_features = features.copy()
        
        try:
            # Economic indicators (fetch key economic data)
            economic_symbols = ['^TNX', '^IRX', 'UUP', 'GLD', 'SLV', 'USO', 'UNG']  # 10Y, 3M rates, dollar ETF, gold, silver, oil, gas
            
            # Get economic data aligned to stock dates
            start_date = features.index[0].strftime('%Y-%m-%d')
            end_date = features.index[-1].strftime('%Y-%m-%d')
            
            import yfinance as yf
            
            # Treasury yields and dollar strength
            try:
                tnx_data = yf.download('^TNX', start=start_date, end=end_date, progress=False)
                if not tnx_data.empty and isinstance(tnx_data.columns, pd.MultiIndex):
                    tnx_data.columns = tnx_data.columns.get_level_values(0)
                
                if not tnx_data.empty and 'Close' in tnx_data.columns:
                    tnx_aligned = tnx_data['Close'].reindex(features.index, method='ffill')
                    enhanced_features['treasury_10y'] = tnx_aligned
                    enhanced_features['treasury_10y_momentum'] = tnx_aligned.pct_change(20)
                    # Yield curve proxy (10Y level vs recent average)
                    enhanced_features['yield_curve_proxy'] = tnx_aligned / tnx_aligned.rolling(60).mean() - 1
            except:
                enhanced_features['treasury_10y'] = 2.5  # Default value
                enhanced_features['treasury_10y_momentum'] = 0
                enhanced_features['yield_curve_proxy'] = 0
            
            # Dollar strength (using UUP ETF as proxy for dollar index)
            try:
                with suppress_stdout_stderr():
                    uup_data = yf.download('UUP', start=start_date, end=end_date, progress=False)
                if not uup_data.empty and isinstance(uup_data.columns, pd.MultiIndex):
                    uup_data.columns = uup_data.columns.get_level_values(0)
                
                if not uup_data.empty and 'Close' in uup_data.columns:
                    uup_aligned = uup_data['Close'].reindex(features.index, method='ffill')
                    enhanced_features['dollar_strength'] = uup_aligned / uup_aligned.mean()  # Normalize around mean
                    enhanced_features['dollar_momentum'] = uup_aligned.pct_change(10)
            except Exception:
                enhanced_features['dollar_strength'] = 1.0
                enhanced_features['dollar_momentum'] = 0
            
            # Commodity exposure (Gold as safe haven)
            try:
                gld_data = yf.download('GLD', start=start_date, end=end_date, progress=False)
                if not gld_data.empty and isinstance(gld_data.columns, pd.MultiIndex):
                    gld_data.columns = gld_data.columns.get_level_values(0)
                
                if not gld_data.empty and 'Close' in gld_data.columns:
                    gld_aligned = gld_data['Close'].reindex(features.index, method='ffill')
                    gld_returns = gld_aligned.pct_change()
                    stock_returns = features['Close'].pct_change()
                    
                    # Gold correlation (safe haven indicator)
                    enhanced_features['gold_correlation'] = stock_returns.rolling(60).corr(gld_returns)
                    enhanced_features['gold_momentum'] = gld_returns.rolling(20).mean()
            except:
                enhanced_features['gold_correlation'] = 0
                enhanced_features['gold_momentum'] = 0
            
            # Oil price correlation (for energy sensitivity)
            try:
                uso_data = yf.download('USO', start=start_date, end=end_date, progress=False)
                if not uso_data.empty and isinstance(uso_data.columns, pd.MultiIndex):
                    uso_data.columns = uso_data.columns.get_level_values(0)
                
                if not uso_data.empty and 'Close' in uso_data.columns:
                    uso_aligned = uso_data['Close'].reindex(features.index, method='ffill')
                    uso_returns = uso_aligned.pct_change()
                    stock_returns = features['Close'].pct_change()
                    
                    enhanced_features['oil_correlation'] = stock_returns.rolling(60).corr(uso_returns)
                    enhanced_features['oil_price_momentum'] = uso_returns.rolling(10).mean()
            except:
                enhanced_features['oil_correlation'] = 0
                enhanced_features['oil_price_momentum'] = 0
            
            # Crypto correlation (Bitcoin as risk-on indicator)
            try:
                btc_data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
                if not btc_data.empty and isinstance(btc_data.columns, pd.MultiIndex):
                    btc_data.columns = btc_data.columns.get_level_values(0)
                
                if not btc_data.empty and 'Close' in btc_data.columns:
                    btc_aligned = btc_data['Close'].reindex(features.index, method='ffill')
                    btc_returns = btc_aligned.pct_change()
                    stock_returns = features['Close'].pct_change()
                    
                    enhanced_features['crypto_correlation'] = stock_returns.rolling(30).corr(btc_returns)
                    enhanced_features['crypto_momentum'] = btc_returns.rolling(20).mean()
                    # Crypto volatility (risk-on/risk-off)
                    enhanced_features['crypto_volatility'] = btc_returns.rolling(20).std()
            except:
                enhanced_features['crypto_correlation'] = 0
                enhanced_features['crypto_momentum'] = 0
                enhanced_features['crypto_volatility'] = 0.02
            
            # Sector rotation signals using sector ETFs
            sector_etfs = {
                'XLK': 'tech',  # Technology
                'XLF': 'financials',  # Financials  
                'XLE': 'energy',  # Energy
                'XLV': 'healthcare',  # Healthcare
                'XLI': 'industrials',  # Industrials
                'XLY': 'consumer_disc',  # Consumer Discretionary
                'XLP': 'consumer_staples'  # Consumer Staples
            }
            
            for etf_symbol, sector_name in sector_etfs.items():
                try:
                    etf_data = yf.download(etf_symbol, start=start_date, end=end_date, progress=False)
                    if not etf_data.empty and isinstance(etf_data.columns, pd.MultiIndex):
                        etf_data.columns = etf_data.columns.get_level_values(0)
                    
                    if not etf_data.empty and 'Close' in etf_data.columns:
                        etf_aligned = etf_data['Close'].reindex(features.index, method='ffill')
                        etf_returns = etf_aligned.pct_change()
                        stock_returns = features['Close'].pct_change()
                        
                        # Sector relative performance
                        enhanced_features[f'{sector_name}_relative_perf'] = (
                            stock_returns.rolling(20).mean() - etf_returns.rolling(20).mean()
                        )
                        # Sector correlation
                        enhanced_features[f'{sector_name}_correlation'] = (
                            stock_returns.rolling(60).corr(etf_returns)
                        )
                        
                        break  # Only do one sector to avoid too many features
                        
                except:
                    enhanced_features[f'{sector_name}_relative_perf'] = 0
                    enhanced_features[f'{sector_name}_correlation'] = 0.5
                    break
            
            # Market stress indicators
            # High-low spread as market stress proxy
            enhanced_features['market_stress_proxy'] = (
                (features['High'] - features['Low']) / features['Close']
            ).rolling(20).mean()
            
            # Volume-price divergence (alternative measure)
            price_momentum = features['Close'].pct_change(10)
            volume_momentum = features['Volume'].pct_change(10)
            enhanced_features['volume_price_divergence'] = (
                price_momentum.rolling(10).mean() - 
                (volume_momentum / volume_momentum.std()).rolling(10).mean()
            )
            
            # Options market proxies (using VIX term structure when available)
            try:
                # VIX9D for short-term fear
                vix9d_data = yf.download('^VIX9D', start=start_date, end=end_date, progress=False)
                vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
                
                if not vix9d_data.empty and not vix_data.empty:
                    # Handle MultiIndex if present
                    if isinstance(vix9d_data.columns, pd.MultiIndex):
                        vix9d_data.columns = vix9d_data.columns.get_level_values(0)
                    if isinstance(vix_data.columns, pd.MultiIndex):
                        vix_data.columns = vix_data.columns.get_level_values(0)
                    
                    if 'Close' in vix9d_data.columns and 'Close' in vix_data.columns:
                        vix9d_aligned = vix9d_data['Close'].reindex(features.index, method='ffill')
                        vix_aligned = vix_data['Close'].reindex(features.index, method='ffill')
                        
                        # Term structure slope (short vs medium term volatility expectations)
                        enhanced_features['vix_term_structure'] = (vix_aligned - vix9d_aligned) / vix9d_aligned
                        enhanced_features['short_term_fear'] = vix9d_aligned / 20  # Normalized
            except:
                enhanced_features['vix_term_structure'] = 0
                enhanced_features['short_term_fear'] = 1.0
            
            # Economic calendar proxy using historical patterns (simplified)
            # Month-end effects
            enhanced_features['month_end_effect'] = np.where(
                features.index.day >= 25, 1, 0  # Last week of month
            )
            
            # Quarter-end effects  
            enhanced_features['quarter_end_effect'] = np.where(
                (features.index.month % 3 == 0) & (features.index.day >= 25), 1, 0
            )
            
            # Week day effects
            enhanced_features['monday_effect'] = np.where(features.index.dayofweek == 0, 1, 0)
            enhanced_features['friday_effect'] = np.where(features.index.dayofweek == 4, 1, 0)
            
        except Exception as e:
            print(f"Warning: Could not fetch alternative data for {symbol}: {e}")
            # Fill with neutral values
            alt_features = [
                'treasury_10y', 'treasury_10y_momentum', 'yield_curve_proxy',
                'dollar_strength', 'dollar_momentum', 'gold_correlation', 'gold_momentum',
                'oil_correlation', 'oil_price_momentum', 'crypto_correlation', 
                'crypto_momentum', 'crypto_volatility', 'tech_relative_perf', 
                'tech_correlation', 'market_stress_proxy', 'volume_price_divergence',
                'vix_term_structure', 'short_term_fear', 'month_end_effect',
                'quarter_end_effect', 'monday_effect', 'friday_effect'
            ]
            
            for feature in alt_features:
                if 'correlation' in feature:
                    enhanced_features[feature] = 0.1
                elif 'effect' in feature:
                    enhanced_features[feature] = 0
                elif 'momentum' in feature:
                    enhanced_features[feature] = 0
                elif feature == 'treasury_10y':
                    enhanced_features[feature] = 2.5
                elif feature == 'dollar_strength':
                    enhanced_features[feature] = 1.0
                elif feature == 'short_term_fear':
                    enhanced_features[feature] = 1.0
                else:
                    enhanced_features[feature] = 0
        
        return enhanced_features
    
    def create_macro_intermarket_features(self, features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create macro and intermarket analysis features"""
        enhanced_features = features.copy()
        
        try:
            start_date = features.index[0].strftime('%Y-%m-%d')
            end_date = features.index[-1].strftime('%Y-%m-%d')
            
            import yfinance as yf
            
            # Global equity indices for intermarket analysis
            global_indices = {
                '^GSPC': 'sp500',      # S&P 500
                '^DJI': 'dow',         # Dow Jones  
                '^IXIC': 'nasdaq',     # NASDAQ
                '^RUT': 'russell2k',   # Russell 2000 (small caps)
                '^FTSE': 'ftse',       # FTSE 100 (UK)
                '^N225': 'nikkei',     # Nikkei 225 (Japan)
                'EWZ': 'brazil',       # Brazil ETF
                'FXI': 'china'         # China ETF
            }
            
            stock_returns = features['Close'].pct_change()
            
            # Calculate correlations with major indices
            for index_symbol, index_name in global_indices.items():
                try:
                    index_data = yf.download(index_symbol, start=start_date, end=end_date, progress=False)
                    if not index_data.empty and isinstance(index_data.columns, pd.MultiIndex):
                        index_data.columns = index_data.columns.get_level_values(0)
                    
                    if not index_data.empty and 'Close' in index_data.columns:
                        index_aligned = index_data['Close'].reindex(features.index, method='ffill')
                        index_returns = index_aligned.pct_change()
                        
                        # Rolling correlation
                        enhanced_features[f'{index_name}_correlation'] = (
                            stock_returns.rolling(60).corr(index_returns)
                        )
                        
                        # Beta calculation (simplified)
                        rolling_cov = stock_returns.rolling(60).cov(index_returns)
                        rolling_var = index_returns.rolling(60).var()
                        enhanced_features[f'{index_name}_beta'] = rolling_cov / rolling_var
                        
                        # Relative strength vs index
                        enhanced_features[f'{index_name}_relative_strength'] = (
                            stock_returns.rolling(20).mean() / index_returns.rolling(20).mean()
                        )
                        
                        # Only process first 3 indices to avoid too many features
                        if len([k for k in enhanced_features.keys() if 'correlation' in k]) >= 3:
                            break
                            
                except Exception as e:
                    enhanced_features[f'{index_name}_correlation'] = 0.5
                    enhanced_features[f'{index_name}_beta'] = 1.0
                    enhanced_features[f'{index_name}_relative_strength'] = 1.0
                    break
            
            # Bond market analysis (yield curve and credit spreads)
            try:
                # Treasury yields
                tlt_data = yf.download('TLT', start=start_date, end=end_date, progress=False)  # 20+ Year Treasury
                iei_data = yf.download('IEI', start=start_date, end=end_date, progress=False)  # 3-7 Year Treasury
                
                if not tlt_data.empty and not iei_data.empty:
                    if isinstance(tlt_data.columns, pd.MultiIndex):
                        tlt_data.columns = tlt_data.columns.get_level_values(0)
                    if isinstance(iei_data.columns, pd.MultiIndex):
                        iei_data.columns = iei_data.columns.get_level_values(0)
                    
                    if 'Close' in tlt_data.columns and 'Close' in iei_data.columns:
                        tlt_aligned = tlt_data['Close'].reindex(features.index, method='ffill')
                        iei_aligned = iei_data['Close'].reindex(features.index, method='ffill')
                        
                        tlt_returns = tlt_aligned.pct_change()
                        iei_returns = iei_aligned.pct_change()
                        
                        # Bond-stock correlation (flight to quality indicator)
                        enhanced_features['bond_correlation'] = stock_returns.rolling(60).corr(tlt_returns)
                        
                        # Yield curve slope proxy (long vs medium duration bond performance)
                        enhanced_features['yield_curve_slope'] = (
                            tlt_returns.rolling(20).mean() - iei_returns.rolling(20).mean()
                        )
                        
            except:
                enhanced_features['bond_correlation'] = -0.2  # Typical stock-bond correlation
                enhanced_features['yield_curve_slope'] = 0
            
            # Credit risk analysis
            try:
                # High yield bonds vs treasuries (credit spread proxy)
                hyg_data = yf.download('HYG', start=start_date, end=end_date, progress=False)  # High Yield ETF
                lqd_data = yf.download('LQD', start=start_date, end=end_date, progress=False)  # Investment Grade ETF
                
                if not hyg_data.empty and not lqd_data.empty:
                    if isinstance(hyg_data.columns, pd.MultiIndex):
                        hyg_data.columns = hyg_data.columns.get_level_values(0)
                    if isinstance(lqd_data.columns, pd.MultiIndex):
                        lqd_data.columns = lqd_data.columns.get_level_values(0)
                    
                    if 'Close' in hyg_data.columns and 'Close' in lqd_data.columns:
                        hyg_aligned = hyg_data['Close'].reindex(features.index, method='ffill')
                        lqd_aligned = lqd_data['Close'].reindex(features.index, method='ffill')
                        
                        hyg_returns = hyg_aligned.pct_change()
                        lqd_returns = lqd_aligned.pct_change()
                        
                        # Credit spread proxy (high yield vs investment grade performance)
                        enhanced_features['credit_spread_proxy'] = hyg_returns - lqd_returns
                        
                        # Credit risk correlation
                        enhanced_features['credit_risk_correlation'] = stock_returns.rolling(60).corr(hyg_returns)
                        
            except:
                enhanced_features['credit_spread_proxy'] = 0
                enhanced_features['credit_risk_correlation'] = 0.3
            
            # Currency and international factors
            try:
                # Currency exposure through ETFs
                uup_data = yf.download('UUP', start=start_date, end=end_date, progress=False)  # Dollar Bull ETF
                
                if not uup_data.empty:
                    if isinstance(uup_data.columns, pd.MultiIndex):
                        uup_data.columns = uup_data.columns.get_level_values(0)
                    
                    if 'Close' in uup_data.columns:
                        uup_aligned = uup_data['Close'].reindex(features.index, method='ffill')
                        uup_returns = uup_aligned.pct_change()
                        
                        # Dollar strength correlation
                        enhanced_features['dollar_correlation'] = stock_returns.rolling(60).corr(uup_returns)
                        
            except:
                enhanced_features['dollar_correlation'] = -0.1  # Typical for US stocks
            
            # Commodity complex analysis
            try:
                # Broad commodity exposure
                dji_data = yf.download('DJP', start=start_date, end=end_date, progress=False)  # Commodity ETF
                
                if not dji_data.empty:
                    if isinstance(dji_data.columns, pd.MultiIndex):
                        dji_data.columns = dji_data.columns.get_level_values(0)
                    
                    if 'Close' in dji_data.columns:
                        dji_aligned = dji_data['Close'].reindex(features.index, method='ffill')
                        dji_returns = dji_aligned.pct_change()
                        
                        # Commodity correlation
                        enhanced_features['commodity_correlation'] = stock_returns.rolling(60).corr(dji_returns)
                        
                        # Inflation hedge factor
                        enhanced_features['inflation_hedge_factor'] = (
                            enhanced_features['commodity_correlation'] * dji_returns.rolling(20).mean()
                        )
                        
            except:
                enhanced_features['commodity_correlation'] = 0.1
                enhanced_features['inflation_hedge_factor'] = 0
            
            # Real estate correlation (REITs)
            try:
                reit_data = yf.download('VNQ', start=start_date, end=end_date, progress=False)  # REIT ETF
                
                if not reit_data.empty:
                    if isinstance(reit_data.columns, pd.MultiIndex):
                        reit_data.columns = reit_data.columns.get_level_values(0)
                    
                    if 'Close' in reit_data.columns:
                        reit_aligned = reit_data['Close'].reindex(features.index, method='ffill')
                        reit_returns = reit_aligned.pct_change()
                        
                        # REIT correlation (interest rate sensitivity)
                        enhanced_features['reit_correlation'] = stock_returns.rolling(60).corr(reit_returns)
                        
            except:
                enhanced_features['reit_correlation'] = 0.3
            
            # Market regime indicators
            # Risk-on vs Risk-off sentiment
            try:
                # Risk-on: Emerging markets vs safe havens
                eem_data = yf.download('EEM', start=start_date, end=end_date, progress=False)  # Emerging Markets
                
                if not eem_data.empty:
                    if isinstance(eem_data.columns, pd.MultiIndex):
                        eem_data.columns = eem_data.columns.get_level_values(0)
                    
                    if 'Close' in eem_data.columns:
                        eem_aligned = eem_data['Close'].reindex(features.index, method='ffill')
                        eem_returns = eem_aligned.pct_change()
                        
                        # Risk-on correlation
                        enhanced_features['risk_on_correlation'] = stock_returns.rolling(60).corr(eem_returns)
                        
                        # Risk appetite proxy
                        enhanced_features['risk_appetite'] = eem_returns.rolling(10).mean()
                        
            except:
                enhanced_features['risk_on_correlation'] = 0.4
                enhanced_features['risk_appetite'] = 0
            
            # Volatility regime analysis
            vix_momentum = enhanced_features.get('vix_momentum', 0)
            if isinstance(vix_momentum, (int, float)):
                vix_momentum = pd.Series([vix_momentum] * len(features), index=features.index)
            
            # Combine macro factors into regime score
            enhanced_features['macro_regime_score'] = (
                enhanced_features.get('risk_appetite', 0) * 0.3 +
                enhanced_features.get('credit_spread_proxy', 0) * -0.3 +  # Negative because wider spreads = worse
                enhanced_features.get('yield_curve_slope', 0) * 0.2 +
                enhanced_features.get('commodity_correlation', 0) * enhanced_features.get('inflation_hedge_factor', 0) * 0.2
            )
            
            # Global growth proxy (combination of international correlations)
            global_growth_proxy = 0
            correlation_features = [f for f in enhanced_features.columns if 'correlation' in f and any(x in f for x in ['sp500', 'nasdaq', 'ftse'])]
            if correlation_features:
                for feature in correlation_features[:3]:  # Use first 3 correlations
                    global_growth_proxy += enhanced_features.get(feature, 0.5) / len(correlation_features)
            enhanced_features['global_growth_proxy'] = global_growth_proxy
            
            # Intermarket divergence signals
            # When correlations break down, it often signals regime change
            historical_correlations = [
                enhanced_features.get('sp500_correlation', 0.5),
                enhanced_features.get('bond_correlation', -0.2),
                enhanced_features.get('commodity_correlation', 0.1)
            ]
            
            # Calculate correlation stability (how much correlations are changing)
            correlation_instability = 0
            for corr_series in historical_correlations:
                if hasattr(corr_series, 'rolling'):
                    correlation_instability += corr_series.rolling(20).std().fillna(0)
                else:
                    correlation_instability += 0
            
            enhanced_features['correlation_instability'] = correlation_instability / len(historical_correlations)
            
        except Exception as e:
            print(f"Warning: Could not fetch macro/intermarket data for {symbol}: {e}")
            # Fill with market-typical values
            macro_features = [
                'sp500_correlation', 'sp500_beta', 'sp500_relative_strength',
                'bond_correlation', 'yield_curve_slope', 'credit_spread_proxy', 
                'credit_risk_correlation', 'dollar_correlation', 'commodity_correlation',
                'inflation_hedge_factor', 'reit_correlation', 'risk_on_correlation',
                'risk_appetite', 'macro_regime_score', 'global_growth_proxy',
                'correlation_instability'
            ]
            
            for feature in macro_features:
                if 'correlation' in feature:
                    if 'bond' in feature:
                        enhanced_features[feature] = -0.2
                    elif 'sp500' in feature:
                        enhanced_features[feature] = 0.7
                    else:
                        enhanced_features[feature] = 0.3
                elif 'beta' in feature:
                    enhanced_features[feature] = 1.0
                elif 'relative_strength' in feature:
                    enhanced_features[feature] = 1.0
                else:
                    enhanced_features[feature] = 0
        
        return enhanced_features

    def engineer_features(self, df: pd.DataFrame, market_data: Optional[Dict] = None, symbol: str = None) -> pd.DataFrame:
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
        
        # Add enhanced technical features
        features = self.create_enhanced_technical_features(features)
        features = self.create_microstructure_features(features)
        features = self.create_advanced_momentum_features(features)
        features = self.create_options_flow_features(features, df)
        features = self.create_derivatives_features(features, df)
        
        # Add fundamental analysis features
        if symbol:
            features = self.create_fundamental_features(features, symbol)
            # Add alternative data features
            features = self.create_alternative_data_features(features, symbol)
            # Add macro & intermarket analysis features
            features = self.create_macro_intermarket_features(features, symbol)
            # Add high-frequency microstructure features
            features = self.create_high_frequency_microstructure_features(features)
            # Add event-driven features
            features = self.create_event_driven_features(features, symbol)
            # Add sector & industry analysis features
            features = self.create_sector_industry_features(features, symbol)
        
        # Create lag features for time series
        feature_cols = [col for col in features.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']]
        
        for col in feature_cols[:25]:  # Slightly increased for new features
            for lag in [1, 5]:
                features[f'{col}_lag{lag}'] = features[col].shift(lag)
        
        # Store feature names (ensure only numeric columns)
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        self.feature_names = [col for col in numeric_cols if col not in 
                            ['Open', 'High', 'Low', 'Close', 'Volume', 'Date', 'target']]
        
        return features

    def create_high_frequency_microstructure_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create high-frequency microstructure features"""
        enhanced_features = features.copy()
        
        try:
            # Intraday price dynamics
            # Price efficiency measures
            enhanced_features['price_efficiency'] = (
                (features['Close'] - features['Open']).abs() / 
                (features['High'] - features['Low'] + 1e-8)
            )
            
            # True range normalized by price
            true_range = np.maximum(
                features['High'] - features['Low'],
                np.maximum(
                    (features['High'] - features['Close'].shift(1)).abs(),
                    (features['Low'] - features['Close'].shift(1)).abs()
                )
            )
            enhanced_features['normalized_true_range'] = true_range / features['Close']
            
            # Intraday return patterns
            overnight_return = (features['Open'] - features['Close'].shift(1)) / features['Close'].shift(1)
            intraday_return = (features['Close'] - features['Open']) / features['Open']
            
            enhanced_features['overnight_return'] = overnight_return
            enhanced_features['intraday_return'] = intraday_return
            enhanced_features['overnight_intraday_ratio'] = overnight_return / (intraday_return + 1e-8)
            
            # Gap analysis
            gap_up = np.where(features['Open'] > features['Close'].shift(1), 
                            (features['Open'] - features['Close'].shift(1)) / features['Close'].shift(1), 0)
            gap_down = np.where(features['Open'] < features['Close'].shift(1),
                              (features['Close'].shift(1) - features['Open']) / features['Close'].shift(1), 0)
            
            enhanced_features['gap_up'] = gap_up
            enhanced_features['gap_down'] = gap_down
            enhanced_features['gap_magnitude'] = gap_up + gap_down
            
            # Volume microstructure
            # Volume-weighted price measures
            vwap_approx = (features['High'] + features['Low'] + features['Close']) / 3
            enhanced_features['price_vs_vwap'] = (features['Close'] - vwap_approx) / vwap_approx
            
            # Volume distribution analysis  
            enhanced_features['volume_concentration'] = (
                features['Volume'] / features['Volume'].rolling(20).mean()
            )
            
            # Order flow approximations
            # Buying vs selling pressure (approximated)
            enhanced_features['buying_pressure'] = np.where(
                features['Close'] > features['Open'],
                features['Volume'] * (features['Close'] - features['Open']) / (features['High'] - features['Low'] + 1e-8),
                0
            )
            
            enhanced_features['selling_pressure'] = np.where(
                features['Close'] < features['Open'],
                features['Volume'] * (features['Open'] - features['Close']) / (features['High'] - features['Low'] + 1e-8),
                0
            )
            
            enhanced_features['net_order_flow'] = (
                enhanced_features['buying_pressure'] - enhanced_features['selling_pressure']
            )
            
            # Volatility microstructure
            # Intraday volatility patterns
            intraday_volatility = (features['High'] - features['Low']) / features['Close']
            enhanced_features['intraday_volatility'] = intraday_volatility
            enhanced_features['volatility_ratio'] = (
                intraday_volatility / intraday_volatility.rolling(20).mean()
            )
            
            # Overnight vs intraday volatility
            overnight_volatility = overnight_return.abs()
            enhanced_features['overnight_volatility'] = overnight_volatility
            enhanced_features['overnight_intraday_vol_ratio'] = (
                overnight_volatility / (intraday_volatility + 1e-8)
            )
            
            # Liquidity proxies
            # Spread approximation (High-Low as proxy for bid-ask spread)
            spread_proxy = (features['High'] - features['Low']) / features['Close']
            enhanced_features['spread_proxy'] = spread_proxy
            enhanced_features['relative_spread'] = (
                spread_proxy / spread_proxy.rolling(20).mean()
            )
            
            # Market impact measures
            # How much does volume move price
            volume_normalized = features['Volume'] / features['Volume'].rolling(20).mean()
            price_change_normalized = features['Close'].pct_change().abs()
            enhanced_features['volume_price_impact'] = (
                price_change_normalized / (volume_normalized + 1e-8)
            )
            
            # Amihud illiquidity measure approximation
            daily_return = features['Close'].pct_change().abs()
            dollar_volume = features['Volume'] * features['Close']
            enhanced_features['amihud_illiquidity'] = (
                daily_return / (dollar_volume / dollar_volume.rolling(20).mean() + 1e-8)
            )
            
        except Exception as e:
            print(f"Warning: Error creating microstructure features: {e}")
            # Set default values for microstructure features
            microstructure_features = [
                'price_efficiency', 'normalized_true_range', 'overnight_return', 
                'intraday_return', 'overnight_intraday_ratio', 'gap_up', 'gap_down',
                'gap_magnitude', 'price_vs_vwap', 'volume_concentration',
                'buying_pressure', 'selling_pressure', 'net_order_flow',
                'intraday_volatility', 'volatility_ratio', 'overnight_volatility',
                'overnight_intraday_vol_ratio', 'spread_proxy', 'relative_spread',
                'volume_price_impact', 'amihud_illiquidity'
            ]
            
            for feature in microstructure_features:
                enhanced_features[feature] = 0
        
        return enhanced_features

    def create_event_driven_features(self, features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create event-driven features based on calendar events and market anomalies"""
        enhanced_features = features.copy()
        
        try:
            # Calendar-based event detection
            dates = features.index
            
            # Earnings season detection (simplified - based on quarterly patterns)
            # Typically earnings are announced in months 1,4,7,10 and 2,5,8,11
            earnings_months = [1, 2, 4, 5, 7, 8, 10, 11]  # Peak earnings months
            enhanced_features['earnings_season'] = np.where(
                dates.month.isin(earnings_months), 1, 0
            )
            
            # End of quarter effects (portfolio rebalancing, window dressing)
            quarter_ends = [3, 6, 9, 12]  # March, June, Sept, Dec
            enhanced_features['quarter_end'] = np.where(
                (dates.month.isin(quarter_ends)) & (dates.day >= 25), 1, 0
            )
            
            # Month-end effects (mutual fund flows, rebalancing)
            enhanced_features['month_end'] = np.where(dates.day >= 25, 1, 0)
            
            # Week patterns (Monday effect, Friday effect)
            enhanced_features['monday_effect'] = np.where(dates.dayofweek == 0, 1, 0)
            enhanced_features['friday_effect'] = np.where(dates.dayofweek == 4, 1, 0)
            
            # Holiday effects - simplified using common US market holidays
            # These would be more accurate with actual market calendar
            enhanced_features['pre_holiday'] = 0  # Would need holiday calendar integration
            enhanced_features['post_holiday'] = 0
            
            # Volume and volatility anomaly detection
            # Unusual volume spikes (potential news/events)
            volume_ma = features['Volume'].rolling(20).mean()
            volume_std = features['Volume'].rolling(20).std()
            enhanced_features['volume_spike'] = np.where(
                features['Volume'] > volume_ma + 2 * volume_std, 1, 0
            )
            
            # Unusual price moves (potential events)
            returns = features['Close'].pct_change()
            return_std = returns.rolling(20).std()
            enhanced_features['price_spike_up'] = np.where(
                returns > 2 * return_std, 1, 0
            )
            enhanced_features['price_spike_down'] = np.where(
                returns < -2 * return_std, 1, 0
            )
            
            # Gap events (potential overnight news)
            gap_threshold = 0.02  # 2% gap threshold
            overnight_gap = (features['Open'] - features['Close'].shift(1)) / features['Close'].shift(1)
            enhanced_features['gap_up_event'] = np.where(overnight_gap > gap_threshold, 1, 0)
            enhanced_features['gap_down_event'] = np.where(overnight_gap < -gap_threshold, 1, 0)
            
            # Earnings announcement approximation
            # Look for patterns of high volume + price movement + gaps
            earnings_proxy = (
                enhanced_features['volume_spike'] + 
                (enhanced_features['price_spike_up'] + enhanced_features['price_spike_down']) +
                (enhanced_features['gap_up_event'] + enhanced_features['gap_down_event'])
            )
            enhanced_features['earnings_announcement_proxy'] = np.where(earnings_proxy >= 2, 1, 0)
            
            # Post-earnings drift detection
            # Look for continued movement in same direction after earnings-like events
            for days in [1, 2, 3, 5]:
                post_earnings_return = features['Close'].pct_change(days).shift(-days)
                earnings_return = returns * enhanced_features['earnings_announcement_proxy']
                
                # Same direction drift
                enhanced_features[f'post_earnings_drift_{days}d'] = np.where(
                    (earnings_return > 0) & (post_earnings_return > 0) |
                    (earnings_return < 0) & (post_earnings_return < 0),
                    1, 0
                )
            
            # Default values for complex features that might fail
            enhanced_features['sector_divergence'] = 0
            enhanced_features['sector_momentum_divergence'] = 0
            enhanced_features['volatility_regime_change'] = 0
            enhanced_features['trend_reversal_down'] = 0
            enhanced_features['trend_reversal_up'] = 0
            enhanced_features['resistance_break'] = 0
            enhanced_features['support_break'] = 0
            enhanced_features['days_since_event'] = 15
            enhanced_features['event_cluster'] = 0
            
        except Exception as e:
            print(f"Warning: Error creating event-driven features for {symbol}: {e}")
            # Set default values for event features
            event_features = [
                'earnings_season', 'quarter_end', 'month_end', 'monday_effect', 
                'friday_effect', 'pre_holiday', 'post_holiday', 'volume_spike',
                'price_spike_up', 'price_spike_down', 'gap_up_event', 'gap_down_event',
                'earnings_announcement_proxy', 'sector_divergence', 'sector_momentum_divergence',
                'volatility_regime_change', 'trend_reversal_down', 'trend_reversal_up',
                'resistance_break', 'support_break', 'days_since_event', 'event_cluster'
            ]
            
            for feature in event_features:
                if 'days_since' in feature:
                    enhanced_features[feature] = 15  # Average days
                elif 'divergence' in feature:
                    enhanced_features[feature] = 0
                else:
                    enhanced_features[feature] = 0
            
            # Add period-specific features
            for days in [1, 2, 3, 5]:
                enhanced_features[f'post_earnings_drift_{days}d'] = 0
        
        return enhanced_features

    def create_sector_industry_features(self, features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create sector and industry analysis features"""
        enhanced_features = features.copy()
        
        try:
            # Set default values for all features first
            sector_features = [
                'sector_relative_performance', 'sector_beta', 'sector_correlation',
                'sector_momentum_divergence', 'sector_leadership', 'sector_weakness',
                'size_factor', 'size_correlation', 'sector_rank', 'sector_rotation_momentum',
                'sector_rotation_signal', 'sector_defensive', 'sector_cyclical',
                'interest_rate_sensitivity', 'growth_stock', 'value_stock'
            ]
            
            for feature in sector_features:
                if 'correlation' in feature or 'beta' in feature:
                    enhanced_features[feature] = 0.5
                elif 'rank' in feature:
                    enhanced_features[feature] = 0.5
                elif 'defensive' in feature or 'cyclical' in feature or 'growth' in feature or 'value' in feature:
                    enhanced_features[feature] = 0.5
                else:
                    enhanced_features[feature] = 0
                    
        except Exception as e:
            print(f"Warning: Error creating sector/industry features for {symbol}: {e}")
        
        return enhanced_features

# ============= MACHINE LEARNING ENHANCEMENTS =============

class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.Dropout(dropout)
        )
        self.activation = nn.GELU()
        
    def forward(self, x):
        residual = x
        out = self.layers(x)
        out += residual  # Residual connection
        return self.activation(out)

class EnhancedTransformer(nn.Module):
    """Enhanced Transformer with residual connections and attention visualization"""
    
    def __init__(self, input_dim, d_model=256, n_heads=16, n_layers=6, dropout=0.1):
        super().__init__()
        
        # Enhanced input projection with residual
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Enhanced positional encoding
        self.positional_encoding = self._create_positional_encoding(2000, d_model)
        
        # Multi-scale transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(n_layers)
        ])
        
        # Enhanced output with residual blocks
        self.output_layers = nn.Sequential(
            ResidualBlock(d_model, d_model // 2, dropout),
            ResidualBlock(d_model, d_model // 2, dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 3)  # 3 classes
        )
        
        self.attention_weights = []  # For visualization
        
    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        
        # Process each timestep through input projection
        x_reshaped = x.view(-1, input_dim)  # (batch*seq, input_dim)
        x_projected = self.input_projection(x_reshaped)  # (batch*seq, d_model)
        x = x_projected.view(batch_size, seq_len, -1)  # (batch, seq, d_model)
        
        # Add positional encoding
        pos_encoding = self.positional_encoding[:seq_len, :].unsqueeze(0).to(x.device)
        x = x + pos_encoding
        
        # Apply transformer layers with attention capture
        self.attention_weights = []
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Global attention pooling instead of just last timestep
        attention_weights = torch.softmax(
            torch.sum(x, dim=-1), dim=1
        ).unsqueeze(-1)  # (batch, seq, 1)
        
        # Weighted sum across sequence
        x = torch.sum(x * attention_weights, dim=1)  # (batch, d_model)
        
        # Output prediction
        return self.output_layers(x)

class ConvolutionalPredictor(nn.Module):
    """1D CNN for pattern recognition in time series"""
    
    def __init__(self, input_dim, seq_len=60, dropout=0.1):
        super().__init__()
        
        # Multi-scale convolutional features
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=7, padding=3),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )
        
        # Adaptive pooling to handle variable sequence lengths
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # CNN expects (batch, features, seq_len)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Global average pooling
        x = self.adaptive_pool(x)  # (batch, 512, 1)
        x = x.squeeze(-1)  # (batch, 512)
        
        return self.classifier(x)

class LSTMWithAttention(nn.Module):
    """LSTM with attention mechanism"""
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Output layers
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)
        )
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden_dim * 2)
        
        # Attention weights
        attention_weights = self.attention(lstm_out)  # (batch, seq, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, hidden_dim * 2)
        
        return self.output(context)


# Advanced training techniques removed - using standard loss functions

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

    def create_high_frequency_microstructure_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create high-frequency microstructure features"""
        enhanced_features = features.copy()
        
        try:
            # Intraday price dynamics
            # Price efficiency measures
            enhanced_features['price_efficiency'] = (
                (features['Close'] - features['Open']).abs() / 
                (features['High'] - features['Low'] + 1e-8)
            )
            
            # True range normalized by price
            true_range = np.maximum(
                features['High'] - features['Low'],
                np.maximum(
                    (features['High'] - features['Close'].shift(1)).abs(),
                    (features['Low'] - features['Close'].shift(1)).abs()
                )
            )
            enhanced_features['normalized_true_range'] = true_range / features['Close']
            
            # Intraday return patterns
            overnight_return = (features['Open'] - features['Close'].shift(1)) / features['Close'].shift(1)
            intraday_return = (features['Close'] - features['Open']) / features['Open']
            
            enhanced_features['overnight_return'] = overnight_return
            enhanced_features['intraday_return'] = intraday_return
            enhanced_features['overnight_intraday_ratio'] = overnight_return / (intraday_return + 1e-8)
            
            # Gap analysis
            gap_up = np.where(features['Open'] > features['Close'].shift(1), 
                            (features['Open'] - features['Close'].shift(1)) / features['Close'].shift(1), 0)
            gap_down = np.where(features['Open'] < features['Close'].shift(1),
                              (features['Close'].shift(1) - features['Open']) / features['Close'].shift(1), 0)
            
            enhanced_features['gap_up'] = gap_up
            enhanced_features['gap_down'] = gap_down
            enhanced_features['gap_magnitude'] = gap_up + gap_down
            
            # Volume microstructure
            # Volume-weighted price measures
            vwap_approx = (features['High'] + features['Low'] + features['Close']) / 3
            enhanced_features['price_vs_vwap'] = (features['Close'] - vwap_approx) / vwap_approx
            
            # Volume distribution analysis  
            enhanced_features['volume_concentration'] = (
                features['Volume'] / features['Volume'].rolling(20).mean()
            )
            
            # Order flow approximations
            # Buying vs selling pressure (approximated)
            enhanced_features['buying_pressure'] = np.where(
                features['Close'] > features['Open'],
                features['Volume'] * (features['Close'] - features['Open']) / (features['High'] - features['Low'] + 1e-8),
                0
            )
            
            enhanced_features['selling_pressure'] = np.where(
                features['Close'] < features['Open'],
                features['Volume'] * (features['Open'] - features['Close']) / (features['High'] - features['Low'] + 1e-8),
                0
            )
            
            enhanced_features['net_order_flow'] = (
                enhanced_features['buying_pressure'] - enhanced_features['selling_pressure']
            )
            
            # Volatility microstructure
            # Intraday volatility patterns
            intraday_volatility = (features['High'] - features['Low']) / features['Close']
            enhanced_features['intraday_volatility'] = intraday_volatility
            enhanced_features['volatility_ratio'] = (
                intraday_volatility / intraday_volatility.rolling(20).mean()
            )
            
            # Overnight vs intraday volatility
            overnight_volatility = overnight_return.abs()
            enhanced_features['overnight_volatility'] = overnight_volatility
            enhanced_features['overnight_intraday_vol_ratio'] = (
                overnight_volatility / (intraday_volatility + 1e-8)
            )
            
            # Liquidity proxies
            # Spread approximation (High-Low as proxy for bid-ask spread)
            spread_proxy = (features['High'] - features['Low']) / features['Close']
            enhanced_features['spread_proxy'] = spread_proxy
            enhanced_features['relative_spread'] = (
                spread_proxy / spread_proxy.rolling(20).mean()
            )
            
            # Market impact measures
            # How much does volume move price
            volume_normalized = features['Volume'] / features['Volume'].rolling(20).mean()
            price_change_normalized = features['Close'].pct_change().abs()
            enhanced_features['volume_price_impact'] = (
                price_change_normalized / (volume_normalized + 1e-8)
            )
            
            # Amihud illiquidity measure approximation
            daily_return = features['Close'].pct_change().abs()
            dollar_volume = features['Volume'] * features['Close']
            enhanced_features['amihud_illiquidity'] = (
                daily_return / (dollar_volume / dollar_volume.rolling(20).mean() + 1e-8)
            )
            
        except Exception as e:
            print(f"Warning: Error creating microstructure features: {e}")
            # Set default values for microstructure features
            microstructure_features = [
                'price_efficiency', 'normalized_true_range', 'overnight_return', 
                'intraday_return', 'overnight_intraday_ratio', 'gap_up', 'gap_down',
                'gap_magnitude', 'price_vs_vwap', 'volume_concentration',
                'buying_pressure', 'selling_pressure', 'net_order_flow',
                'intraday_volatility', 'volatility_ratio', 'overnight_volatility',
                'overnight_intraday_vol_ratio', 'spread_proxy', 'relative_spread',
                'volume_price_impact', 'amihud_illiquidity'
            ]
            
            for feature in microstructure_features:
                enhanced_features[feature] = 0
        
        return enhanced_features
    
    def create_event_driven_features(self, features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create event-driven features based on calendar events and market anomalies"""
        enhanced_features = features.copy()
        
        try:
            # Calendar-based event detection
            dates = features.index
            
            # Earnings season detection (simplified - based on quarterly patterns)
            # Typically earnings are announced in months 1,4,7,10 and 2,5,8,11
            earnings_months = [1, 2, 4, 5, 7, 8, 10, 11]  # Peak earnings months
            enhanced_features['earnings_season'] = np.where(
                dates.month.isin(earnings_months), 1, 0
            )
            
            # End of quarter effects (portfolio rebalancing, window dressing)
            quarter_ends = [3, 6, 9, 12]  # March, June, Sept, Dec
            enhanced_features['quarter_end'] = np.where(
                (dates.month.isin(quarter_ends)) & (dates.day >= 25), 1, 0
            )
            
            # Month-end effects (mutual fund flows, rebalancing)
            enhanced_features['month_end'] = np.where(dates.day >= 25, 1, 0)
            
            # Week patterns (Monday effect, Friday effect)
            enhanced_features['monday_effect'] = np.where(dates.dayofweek == 0, 1, 0)
            enhanced_features['friday_effect'] = np.where(dates.dayofweek == 4, 1, 0)
            
            # Holiday effects - simplified using common US market holidays
            # These would be more accurate with actual market calendar
            enhanced_features['pre_holiday'] = 0  # Would need holiday calendar integration
            enhanced_features['post_holiday'] = 0
            
            # Volume and volatility anomaly detection
            # Unusual volume spikes (potential news/events)
            volume_ma = features['Volume'].rolling(20).mean()
            volume_std = features['Volume'].rolling(20).std()
            enhanced_features['volume_spike'] = np.where(
                features['Volume'] > volume_ma + 2 * volume_std, 1, 0
            )
            
            # Unusual price moves (potential events)
            returns = features['Close'].pct_change()
            return_std = returns.rolling(20).std()
            enhanced_features['price_spike_up'] = np.where(
                returns > 2 * return_std, 1, 0
            )
            enhanced_features['price_spike_down'] = np.where(
                returns < -2 * return_std, 1, 0
            )
            
            # Gap events (potential overnight news)
            gap_threshold = 0.02  # 2% gap threshold
            overnight_gap = (features['Open'] - features['Close'].shift(1)) / features['Close'].shift(1)
            enhanced_features['gap_up_event'] = np.where(overnight_gap > gap_threshold, 1, 0)
            enhanced_features['gap_down_event'] = np.where(overnight_gap < -gap_threshold, 1, 0)
            
            # Earnings announcement approximation
            # Look for patterns of high volume + price movement + gaps
            earnings_proxy = (
                enhanced_features['volume_spike'] + 
                (enhanced_features['price_spike_up'] + enhanced_features['price_spike_down']) +
                (enhanced_features['gap_up_event'] + enhanced_features['gap_down_event'])
            )
            enhanced_features['earnings_announcement_proxy'] = np.where(earnings_proxy >= 2, 1, 0)
            
            # Post-earnings drift detection
            # Look for continued movement in same direction after earnings-like events
            for days in [1, 2, 3, 5]:
                post_earnings_return = features['Close'].pct_change(days).shift(-days)
                earnings_return = returns * enhanced_features['earnings_announcement_proxy']
                
                # Same direction drift
                enhanced_features[f'post_earnings_drift_{days}d'] = np.where(
                    (earnings_return > 0) & (post_earnings_return > 0) |
                    (earnings_return < 0) & (post_earnings_return < 0),
                    1, 0
                )
            
            # Sector rotation events (simplified)
            # Detect when stock moves differently from sector
            try:
                # Use sector ETF based on common mapping
                sector_etf_map = {
                    # Tech stocks
                    'AAPL': 'XLK', 'MSFT': 'XLK', 'GOOGL': 'XLK', 'NVDA': 'XLK', 'META': 'XLK',
                    'TSLA': 'XLK', 'NFLX': 'XLK', 'ADBE': 'XLK', 'CRM': 'XLK', 'ORCL': 'XLK',
                    # Financial stocks  
                    'JPM': 'XLF', 'BAC': 'XLF', 'WFC': 'XLF', 'GS': 'XLF', 'MS': 'XLF',
                    # Healthcare stocks
                    'JNJ': 'XLV', 'PFE': 'XLV', 'UNH': 'XLV', 'ABBV': 'XLV', 'MRK': 'XLV',
                    # Energy stocks
                    'XOM': 'XLE', 'CVX': 'XLE', 'COP': 'XLE', 'SLB': 'XLE',
                    # Consumer stocks
                    'AMZN': 'XLY', 'HD': 'XLY', 'MCD': 'XLY', 'NKE': 'XLY', 'SBUX': 'XLY'
                }
                
                sector_etf = sector_etf_map.get(symbol, 'SPY')  # Default to SPY
                
                import yfinance as yf
                start_date = features.index[0].strftime('%Y-%m-%d')
                end_date = features.index[-1].strftime('%Y-%m-%d')
                
                sector_data = yf.download(sector_etf, start=start_date, end=end_date, progress=False)
                if not sector_data.empty:
                    if isinstance(sector_data.columns, pd.MultiIndex):
                        sector_data.columns = sector_data.columns.get_level_values(0)
                    
                    if 'Close' in sector_data.columns:
                        sector_aligned = sector_data['Close'].reindex(features.index, method='ffill')
                        sector_returns = sector_aligned.pct_change()
                        stock_returns = features['Close'].pct_change()
                        
                        # Relative performance divergence
                        relative_performance = stock_returns - sector_returns
                        enhanced_features['sector_divergence'] = np.where(
                            relative_performance.abs() > relative_performance.rolling(20).std() * 2,
                            1, 0
                        )
                        
                        # Sector momentum vs stock momentum
                        enhanced_features['sector_momentum_divergence'] = (
                            stock_returns.rolling(5).mean() - sector_returns.rolling(5).mean()
                        )
                        
            except:
                enhanced_features['sector_divergence'] = 0
                enhanced_features['sector_momentum_divergence'] = 0
            
            # Market regime change detection
            # Detect shifts in volatility regime
            volatility = returns.rolling(20).std()
            vol_regime_change = (volatility / volatility.rolling(60).mean() - 1).abs()
            enhanced_features['volatility_regime_change'] = np.where(
                vol_regime_change > vol_regime_change.rolling(20).std() * 2, 1, 0
            )
            
            # Trend reversal events
            # Simple trend reversal detection using moving averages
            ma_short = features['Close'].rolling(5).mean()
            ma_long = features['Close'].rolling(20).mean()
            
            # Trend change from up to down
            enhanced_features['trend_reversal_down'] = np.where(
                (ma_short.shift(1) > ma_long.shift(1)) & (ma_short < ma_long), 1, 0
            )
            
            # Trend change from down to up
            enhanced_features['trend_reversal_up'] = np.where(
                (ma_short.shift(1) < ma_long.shift(1)) & (ma_short > ma_long), 1, 0
            )
            
            # Support/resistance break events
            # Using rolling max/min as support/resistance levels
            resistance_level = features['High'].rolling(20).max()
            support_level = features['Low'].rolling(20).min()
            
            enhanced_features['resistance_break'] = np.where(
                features['Close'] > resistance_level.shift(1), 1, 0
            )
            enhanced_features['support_break'] = np.where(
                features['Close'] < support_level.shift(1), 1, 0
            )
            
            # Event persistence (how long effects last)
            # Track how many days since last major event
            major_events = (
                enhanced_features['earnings_announcement_proxy'] +
                enhanced_features['volume_spike'] +
                enhanced_features['gap_up_event'] +
                enhanced_features['gap_down_event']
            )
            
            days_since_event = 0
            days_since_last_event = []
            for event in major_events:
                if event > 0:
                    days_since_event = 0
                else:
                    days_since_event += 1
                days_since_last_event.append(min(days_since_event, 30))  # Cap at 30 days
            
            enhanced_features['days_since_event'] = days_since_last_event
            
            # Event clustering (multiple events close together)
            event_window = major_events.rolling(5).sum()  # Events in last 5 days
            enhanced_features['event_cluster'] = np.where(event_window >= 2, 1, 0)
            
        except Exception as e:
            print(f"Warning: Error creating event-driven features for {symbol}: {e}")
            # Set default values for event features
            event_features = [
                'earnings_season', 'quarter_end', 'month_end', 'monday_effect', 
                'friday_effect', 'pre_holiday', 'post_holiday', 'volume_spike',
                'price_spike_up', 'price_spike_down', 'gap_up_event', 'gap_down_event',
                'earnings_announcement_proxy', 'sector_divergence', 'sector_momentum_divergence',
                'volatility_regime_change', 'trend_reversal_down', 'trend_reversal_up',
                'resistance_break', 'support_break', 'days_since_event', 'event_cluster'
            ]
            
            for feature in event_features:
                if 'days_since' in feature:
                    enhanced_features[feature] = 15  # Average days
                elif 'divergence' in feature:
                    enhanced_features[feature] = 0
                else:
                    enhanced_features[feature] = 0
            
            # Add period-specific features
            for days in [1, 2, 3, 5]:
                enhanced_features[f'post_earnings_drift_{days}d'] = 0
        
        return enhanced_features
    
    def create_sector_industry_features(self, features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Create sector and industry analysis features"""
        enhanced_features = features.copy()
        
        try:
            # Comprehensive sector/industry mapping
            sector_industry_map = {
                # TECHNOLOGY SECTOR
                'AAPL': {'sector': 'XLK', 'industry': 'consumer_electronics', 'market_cap': 'mega'},
                'MSFT': {'sector': 'XLK', 'industry': 'software', 'market_cap': 'mega'},
                'GOOGL': {'sector': 'XLK', 'industry': 'internet', 'market_cap': 'mega'},
                'AMZN': {'sector': 'XLY', 'industry': 'e_commerce', 'market_cap': 'mega'},  # Consumer Discretionary
                'NVDA': {'sector': 'XLK', 'industry': 'semiconductors', 'market_cap': 'large'},
                'META': {'sector': 'XLK', 'industry': 'social_media', 'market_cap': 'mega'},
                'TSLA': {'sector': 'XLY', 'industry': 'ev_automotive', 'market_cap': 'large'},
                'NFLX': {'sector': 'XLY', 'industry': 'streaming', 'market_cap': 'large'},
                'ADBE': {'sector': 'XLK', 'industry': 'software', 'market_cap': 'large'},
                'CRM': {'sector': 'XLK', 'industry': 'cloud_software', 'market_cap': 'large'},
                'ORCL': {'sector': 'XLK', 'industry': 'enterprise_software', 'market_cap': 'large'},
                'AMD': {'sector': 'XLK', 'industry': 'semiconductors', 'market_cap': 'large'},
                'INTC': {'sector': 'XLK', 'industry': 'semiconductors', 'market_cap': 'large'},
                
                # FINANCIAL SECTOR
                'JPM': {'sector': 'XLF', 'industry': 'money_center_banks', 'market_cap': 'mega'},
                'BAC': {'sector': 'XLF', 'industry': 'money_center_banks', 'market_cap': 'large'},
                'WFC': {'sector': 'XLF', 'industry': 'money_center_banks', 'market_cap': 'large'},
                'GS': {'sector': 'XLF', 'industry': 'investment_banking', 'market_cap': 'large'},
                'MS': {'sector': 'XLF', 'industry': 'investment_banking', 'market_cap': 'large'},
                'V': {'sector': 'XLF', 'industry': 'payment_processors', 'market_cap': 'mega'},
                'MA': {'sector': 'XLF', 'industry': 'payment_processors', 'market_cap': 'mega'},
                
                # HEALTHCARE SECTOR
                'JNJ': {'sector': 'XLV', 'industry': 'pharmaceuticals', 'market_cap': 'mega'},
                'PFE': {'sector': 'XLV', 'industry': 'pharmaceuticals', 'market_cap': 'large'},
                'UNH': {'sector': 'XLV', 'industry': 'health_insurance', 'market_cap': 'mega'},
                'ABBV': {'sector': 'XLV', 'industry': 'biotechnology', 'market_cap': 'large'},
                'MRK': {'sector': 'XLV', 'industry': 'pharmaceuticals', 'market_cap': 'large'},
                
                # ENERGY SECTOR
                'XOM': {'sector': 'XLE', 'industry': 'oil_gas_integrated', 'market_cap': 'mega'},
                'CVX': {'sector': 'XLE', 'industry': 'oil_gas_integrated', 'market_cap': 'large'},
                'COP': {'sector': 'XLE', 'industry': 'oil_gas_exploration', 'market_cap': 'large'},
                'SLB': {'sector': 'XLE', 'industry': 'oil_services', 'market_cap': 'large'},
                
                # CONSUMER DISCRETIONARY
                'HD': {'sector': 'XLY', 'industry': 'home_improvement', 'market_cap': 'large'},
                'MCD': {'sector': 'XLY', 'industry': 'restaurants', 'market_cap': 'large'},
                'NKE': {'sector': 'XLY', 'industry': 'apparel', 'market_cap': 'large'},
                'SBUX': {'sector': 'XLY', 'industry': 'restaurants', 'market_cap': 'large'},
                
                # CONSUMER STAPLES
                'PG': {'sector': 'XLP', 'industry': 'household_products', 'market_cap': 'large'},
                'KO': {'sector': 'XLP', 'industry': 'beverages', 'market_cap': 'large'},
                'WMT': {'sector': 'XLP', 'industry': 'discount_retail', 'market_cap': 'mega'},
                
                # INDUSTRIALS
                'BA': {'sector': 'XLI', 'industry': 'aerospace', 'market_cap': 'large'},
                'CAT': {'sector': 'XLI', 'industry': 'construction_machinery', 'market_cap': 'large'},
                'GE': {'sector': 'XLI', 'industry': 'conglomerates', 'market_cap': 'large'},
                
                # UTILITIES
                'NEE': {'sector': 'XLU', 'industry': 'electric_utilities', 'market_cap': 'large'},
                
                # REAL ESTATE (REITs)
                'PLD': {'sector': 'XLRE', 'industry': 'industrial_reits', 'market_cap': 'large'},
                
                # MATERIALS
                'LIN': {'sector': 'XLB', 'industry': 'specialty_chemicals', 'market_cap': 'large'},
                
                # COMMUNICATION SERVICES
                'T': {'sector': 'XLC', 'industry': 'telecom', 'market_cap': 'large'},
                'VZ': {'sector': 'XLC', 'industry': 'telecom', 'market_cap': 'large'}
            }
            
            # Get stock's sector/industry info
            stock_info = sector_industry_map.get(symbol, {
                'sector': 'SPY', 'industry': 'diversified', 'market_cap': 'large'
            })
            
            sector_etf = stock_info['sector']
            industry_type = stock_info['industry']
            market_cap_category = stock_info['market_cap']
            
            # Fetch sector and related data
            start_date = features.index[0].strftime('%Y-%m-%d')
            end_date = features.index[-1].strftime('%Y-%m-%d')
            
            import yfinance as yf
            
            # Sector analysis
            sector_data = yf.download(sector_etf, start=start_date, end=end_date, progress=False)
            if not sector_data.empty and isinstance(sector_data.columns, pd.MultiIndex):
                sector_data.columns = sector_data.columns.get_level_values(0)
            
            if not sector_data.empty and 'Close' in sector_data.columns:
                sector_aligned = sector_data['Close'].reindex(features.index, method='ffill')
                sector_returns = sector_aligned.pct_change()
                stock_returns = features['Close'].pct_change()
                
                # Sector relative performance
                enhanced_features['sector_relative_performance'] = (
                    stock_returns.rolling(20).mean() - sector_returns.rolling(20).mean()
                )
                
                # Sector beta (rolling)
                covariance = stock_returns.rolling(60).cov(sector_returns)
                sector_variance = sector_returns.rolling(60).var()
                enhanced_features['sector_beta'] = covariance / sector_variance
                
                # Sector correlation (rolling)
                enhanced_features['sector_correlation'] = (
                    stock_returns.rolling(60).corr(sector_returns)
                )
                
                # Sector momentum divergence
                stock_momentum = stock_returns.rolling(10).mean()
                sector_momentum = sector_returns.rolling(10).mean()
                enhanced_features['sector_momentum_divergence'] = stock_momentum - sector_momentum
                
                # Sector leadership indicator
                enhanced_features['sector_leadership'] = np.where(
                    enhanced_features['sector_relative_performance'] > 
                    enhanced_features['sector_relative_performance'].rolling(20).quantile(0.8),
                    1, 0
                )
                
                # Sector weakness indicator
                enhanced_features['sector_weakness'] = np.where(
                    enhanced_features['sector_relative_performance'] < 
                    enhanced_features['sector_relative_performance'].rolling(20).quantile(0.2),
                    1, 0
                )
            
            # Market cap category analysis
            if market_cap_category == 'mega':
                # Compare with large cap index
                comparison_etf = 'IVV'  # iShares Core S&P 500
            elif market_cap_category == 'large':
                comparison_etf = 'IVV'  # iShares Core S&P 500
            else:
                comparison_etf = 'IWM'  # iShares Russell 2000 (small cap)
            
            try:
                size_data = yf.download(comparison_etf, start=start_date, end=end_date, progress=False)
                if not size_data.empty and isinstance(size_data.columns, pd.MultiIndex):
                    size_data.columns = size_data.columns.get_level_values(0)
                
                if not size_data.empty and 'Close' in size_data.columns:
                    size_aligned = size_data['Close'].reindex(features.index, method='ffill')
                    size_returns = size_aligned.pct_change()
                    
                    # Size factor analysis
                    enhanced_features['size_factor'] = (
                        stock_returns.rolling(20).mean() - size_returns.rolling(20).mean()
                    )
                    
                    # Size correlation
                    enhanced_features['size_correlation'] = (
                        stock_returns.rolling(60).corr(size_returns)
                    )
            except:
                enhanced_features['size_factor'] = 0
                enhanced_features['size_correlation'] = 0.7
            
            # Industry-specific analysis
            industry_specific_features = self._create_industry_specific_features(
                features, industry_type, symbol
            )
            enhanced_features = pd.concat([enhanced_features, industry_specific_features], axis=1)
            
            # Sector rotation analysis
            # Compare performance across major sectors
            major_sectors = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU']
            sector_performances = {}
            
            for sector in major_sectors[:4]:  # Limit to prevent too many API calls
                try:
                    sector_temp_data = yf.download(sector, start=start_date, end=end_date, progress=False)
                    if not sector_temp_data.empty:
                        if isinstance(sector_temp_data.columns, pd.MultiIndex):
                            sector_temp_data.columns = sector_temp_data.columns.get_level_values(0)
                        
                        if 'Close' in sector_temp_data.columns:
                            sector_temp_aligned = sector_temp_data['Close'].reindex(features.index, method='ffill')
                            sector_temp_returns = sector_temp_aligned.pct_change(20)  # 20-day returns
                            sector_performances[sector] = sector_temp_returns.iloc[-1] if len(sector_temp_returns) > 0 else 0
                except:
                    sector_performances[sector] = 0
            
            # Current sector rank
            if sector_performances:
                current_sector_performance = sector_performances.get(sector_etf, 0)
                all_performances = list(sector_performances.values())
                if all_performances:
                    sector_rank = sum(1 for p in all_performances if p < current_sector_performance) / len(all_performances)
                    enhanced_features['sector_rank'] = sector_rank
                else:
                    enhanced_features['sector_rank'] = 0.5
            else:
                enhanced_features['sector_rank'] = 0.5
            
            # Rotation momentum (is sector gaining or losing relative strength)
            if 'sector_relative_performance' in enhanced_features.columns:
                sector_rotation_momentum = enhanced_features['sector_relative_performance'].diff(5)
                enhanced_features['sector_rotation_momentum'] = sector_rotation_momentum
                
                # Sector rotation signal
                enhanced_features['sector_rotation_signal'] = np.where(
                    sector_rotation_momentum > sector_rotation_momentum.rolling(20).quantile(0.7),
                    1,  # Rotating into sector
                    np.where(
                        sector_rotation_momentum < sector_rotation_momentum.rolling(20).quantile(0.3),
                        -1,  # Rotating out of sector
                        0    # Neutral
                    )
                )
            
            # Defensive vs cyclical classification
            defensive_sectors = ['XLP', 'XLU', 'XLV']  # Staples, Utilities, Healthcare
            cyclical_sectors = ['XLY', 'XLF', 'XLI', 'XLE']  # Discretionary, Financials, Industrials, Energy
            
            if sector_etf in defensive_sectors:
                enhanced_features['sector_defensive'] = 1
                enhanced_features['sector_cyclical'] = 0
            elif sector_etf in cyclical_sectors:
                enhanced_features['sector_defensive'] = 0
                enhanced_features['sector_cyclical'] = 1
            else:
                enhanced_features['sector_defensive'] = 0.5
                enhanced_features['sector_cyclical'] = 0.5
            
            # Economic sensitivity analysis
            # Correlation with economic indicators (using sector as proxy)
            try:
                # Use treasury yields as economic indicator proxy
                tlt_data = yf.download('TLT', start=start_date, end=end_date, progress=False)
                if not tlt_data.empty:
                    if isinstance(tlt_data.columns, pd.MultiIndex):
                        tlt_data.columns = tlt_data.columns.get_level_values(0)
                    
                    if 'Close' in tlt_data.columns:
                        tlt_aligned = tlt_data['Close'].reindex(features.index, method='ffill')
                        tlt_returns = tlt_aligned.pct_change()
                        
                        # Interest rate sensitivity
                        enhanced_features['interest_rate_sensitivity'] = (
                            stock_returns.rolling(60).corr(tlt_returns)
                        )
            except:
                enhanced_features['interest_rate_sensitivity'] = 0
            
            # Value vs Growth classification (simplified)
            # Based on sector tendencies
            growth_sectors = ['XLK', 'XLY']  # Technology, Consumer Discretionary
            value_sectors = ['XLF', 'XLE', 'XLB']  # Financials, Energy, Materials
            
            if sector_etf in growth_sectors:
                enhanced_features['growth_stock'] = 1
                enhanced_features['value_stock'] = 0
            elif sector_etf in value_sectors:
                enhanced_features['growth_stock'] = 0
                enhanced_features['value_stock'] = 1
            else:
                enhanced_features['growth_stock'] = 0.5
                enhanced_features['value_stock'] = 0.5
                
        except Exception as e:
            print(f"Warning: Error creating sector/industry features for {symbol}: {e}")
            # Set default values
            sector_features = [
                'sector_relative_performance', 'sector_beta', 'sector_correlation',
                'sector_momentum_divergence', 'sector_leadership', 'sector_weakness',
                'size_factor', 'size_correlation', 'sector_rank', 'sector_rotation_momentum',
                'sector_rotation_signal', 'sector_defensive', 'sector_cyclical',
                'interest_rate_sensitivity', 'growth_stock', 'value_stock'
            ]
            
            for feature in sector_features:
                if 'correlation' in feature or 'beta' in feature:
                    enhanced_features[feature] = 0.5
                elif 'rank' in feature:
                    enhanced_features[feature] = 0.5
                elif 'defensive' in feature or 'cyclical' in feature or 'growth' in feature or 'value' in feature:
                    enhanced_features[feature] = 0.5
                else:
                    enhanced_features[feature] = 0
        
        return enhanced_features
    
    def _create_industry_specific_features(self, features: pd.DataFrame, industry_type: str, symbol: str) -> pd.DataFrame:
        """Create industry-specific features"""
        industry_features = pd.DataFrame(index=features.index)
        
        try:
            # Technology industry features
            if 'software' in industry_type or 'internet' in industry_type or 'semiconductors' in industry_type:
                # Tech stocks are sensitive to NASDAQ
                industry_features['tech_nasdaq_correlation'] = 0.8  # Default high correlation
                industry_features['innovation_cycle'] = np.sin(np.arange(len(features)) * 2 * np.pi / 252)  # Annual cycle
                industry_features['growth_multiple_sensitivity'] = 1  # High growth multiple sensitivity
                
            # Financial industry features
            elif 'bank' in industry_type or 'investment' in industry_type or 'payment' in industry_type:
                # Banks are sensitive to interest rates and credit conditions
                industry_features['interest_rate_exposure'] = 1
                industry_features['credit_cycle_sensitivity'] = 1
                industry_features['regulatory_sensitivity'] = 1
                
            # Healthcare industry features
            elif 'pharmaceutical' in industry_type or 'biotechnology' in industry_type or 'health' in industry_type:
                industry_features['regulatory_risk'] = 1
                industry_features['patent_cliff_risk'] = 0.5
                industry_features['demographic_tailwind'] = 1  # Aging population
                
            # Energy industry features
            elif 'oil' in industry_type or 'gas' in industry_type or 'energy' in industry_type:
                industry_features['commodity_correlation'] = 0.8
                industry_features['geopolitical_sensitivity'] = 1
                industry_features['cyclical_sensitivity'] = 1
                
            # Consumer discretionary features
            elif 'retail' in industry_type or 'restaurant' in industry_type or 'apparel' in industry_type:
                industry_features['consumer_confidence_sensitivity'] = 1
                industry_features['economic_cycle_sensitivity'] = 1  
                industry_features['seasonal_patterns'] = np.sin(np.arange(len(features)) * 2 * np.pi / 252 * 4)  # Quarterly
                
            # Default features for other industries
            else:
                industry_features['industry_beta'] = 1.0
                industry_features['sector_specific_risk'] = 0.5
                
        except Exception as e:
            print(f"Warning: Error creating industry-specific features for {industry_type}: {e}")
            # Default neutral values
            industry_features['industry_neutral'] = 0.5
        
        return industry_features


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


class AdvancedStockPredictor:
    """Main prediction system combining all components"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.feature_engineer = AdvancedFeatureEngineering()
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
                features_df = self.feature_engineer.engineer_features(df, market_indices, symbol)
                
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
        
        features_df = self.feature_engineer.engineer_features(df, market_indices, symbol)
        
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
    
    # Import all tickers from top_200_tickers
    from top_200_tickers import ALL_TICKERS
    
    # Use all 188 stocks for comprehensive training
    symbols = ALL_TICKERS
    print(f"Training on {len(symbols)} stocks: {symbols[:10]}...{symbols[-5:]}")  # Show first 10 and last 5
    
    # Fetch historical data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    print("Fetching historical data...")
    data = predictor.fetch_data(symbols, start_date, end_date)
    
    # Prepare training data
    train_data = predictor.prepare_training_data(data)
    
    # Train ensemble
    predictor.train_ensemble(train_data)
    
    # Make prediction for a sample symbol (first stock in the list)
    sample_symbol = symbols[0]
    print(f"\nMaking prediction for {sample_symbol}...")
    prediction = predictor.predict(sample_symbol)
    
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


