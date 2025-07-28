# Advanced Stock Predictor 🚀

A comprehensive machine learning system for stock prediction that combines technical analysis, sentiment analysis, and multiple neural network architectures to predict weekly stock price movements.

## 🌟 Features

- **Multi-Modal Prediction**: Combines price data, technical indicators, and sentiment analysis
- **Advanced Neural Networks**: Transformer, LSTM, and Graph Neural Network architectures
- **Real-Time Sentiment Analysis**: Scrapes news and social media for sentiment data
- **Comprehensive Technical Analysis**: 80+ technical indicators and features
- **Backtesting System**: Historical performance testing with detailed metrics
- **Real-Time Monitoring**: Live prediction system with alerts
- **200+ Stock Coverage**: Supports analysis of major US stocks and ETFs

## 🏗️ Architecture

### Core Components

1. **AdvancedFeatureEngineering**: Creates 80+ technical and market regime features
2. **TransformerPredictor**: Attention-based model for sequence prediction
3. **MultiModalFusion**: Combines price and sentiment data with gated fusion
4. **GraphNeuralNetwork**: Captures inter-stock relationships
5. **SentimentAnalyzer**: Multi-source sentiment analysis
6. **RealTimePredictor**: Live monitoring and alert system

### Prediction Classes
- **0**: Down (< -2% weekly return)
- **1**: Neutral (-2% to +2% weekly return)  
- **2**: Up (> +2% weekly return)

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, for faster training)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Marketman2
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data** (automatic on first run)
```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

## 🚀 Quick Start

### Basic Prediction

```python
from advanced_stock_predictor import AdvancedStockPredictor

# Initialize predictor
predictor = AdvancedStockPredictor()

# Get prediction for a stock
result = predictor.predict('AAPL')
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Training Custom Model

```python
# Define stocks to analyze
symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']

# Fetch historical data
data = predictor.fetch_data(symbols, '2022-01-01', '2024-01-01')

# Prepare training data
train_data = predictor.prepare_training_data(data)

# Train ensemble
predictor.train_ensemble(train_data)

# Save model
predictor.save_model('my_model.pth')
```

### Run Complete Analysis

```bash
# Train on all 200 stocks with sentiment analysis
python run_advanced_predictor.py

# Quick mode (faster, less comprehensive)
python run_advanced_predictor.py --quick

# Train on specific sector
python run_advanced_predictor.py --tickers tech
```

## 📊 Usage Examples

### 1. Single Stock Analysis

```python
from advanced_stock_predictor import AdvancedStockPredictor

predictor = AdvancedStockPredictor()
result = predictor.predict('AAPL')

print(f"""
Stock: {result['symbol']}
Prediction: {result['prediction']}
Confidence: {result['confidence']:.2%}
Action: {result['recommendation']['action']}
Current Price: ${result['technical_indicators']['current_price']:.2f}
""")
```

### 2. Backtesting Strategy

```python
# Historical performance testing
backtest_results = predictor.backtest(
    test_data, 
    start_date='2023-01-01', 
    end_date='2024-01-01'
)

print(f"Total Return: {backtest_results['metrics']['total_return']:.2%}")
print(f"Sharpe Ratio: {backtest_results['metrics']['sharpe_ratio']:.2f}")
print(f"Win Rate: {backtest_results['metrics']['win_rate']:.2%}")
```

### 3. Real-Time Monitoring

```python
from advanced_stock_predictor import RealTimePredictor

# Start monitoring
monitor = RealTimePredictor('trained_model.pth')
monitor.monitor_symbols(['AAPL', 'TSLA', 'NVDA'], interval=300)
```

## 🔧 Configuration

### Model Parameters

- **Sequence Length**: 60 days for LSTM/Transformer models
- **Training Epochs**: 30-50 depending on model
- **Batch Size**: 32-64 samples
- **Learning Rate**: 1e-4 with AdamW optimizer

### Feature Engineering

- **Price Features**: Returns, volatility, price positions (20+ features)
- **Technical Indicators**: RSI, MACD, Bollinger Bands, etc. (30+ features)
- **Market Regime**: Trend strength, volatility regimes (10+ features)
- **Sentiment Features**: News sentiment, social media sentiment (10+ features)

### Data Sources

- **Price Data**: Yahoo Finance API
- **News Sentiment**: Multiple financial news sources
- **Market Data**: S&P 500, VIX, sector indices

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:

- **Return Metrics**: Total return, Sharpe ratio, maximum drawdown
- **Accuracy Metrics**: Prediction accuracy, precision, recall
- **Risk Metrics**: Volatility, beta, correlation
- **Trading Metrics**: Win rate, profit factor, trade frequency

## 🗂️ File Structure

```
Marketman2/
├── advanced_stock_predictor.py    # Main prediction system
├── run_advanced_predictor.py      # Enhanced runner with sentiment
├── sentiment_analysis_implementation.py  # Sentiment analysis
├── top_200_tickers.py             # Stock ticker lists
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── data/                          # Data storage (auto-created)
│   ├── models/                    # Saved models
│   ├── sentiment/                 # Sentiment cache
│   └── *.csv                      # Historical price data
└── venv/                          # Virtual environment (gitignored)
```

## 🔍 Technical Details

### Neural Network Architectures

1. **Transformer Model**
   - Multi-head attention mechanism
   - Positional encoding for time series
   - 4 layers, 8 attention heads

2. **LSTM Model**
   - 3-layer bidirectional LSTM
   - Dropout regularization
   - Dense output layers

3. **Graph Neural Network**
   - Captures stock correlations
   - Graph attention mechanism
   - Inter-stock relationship modeling

### Sentiment Analysis

- **Multi-Source**: News articles, financial reports, social media
- **Real-Time**: Fresh sentiment data for each prediction
- **Cached**: Intelligent caching to avoid API limits
- **Multiple Models**: VADER, FinBERT, custom scoring

### Risk Management

- **Position Sizing**: Automated based on volatility
- **Stop Loss**: Dynamic based on ATR
- **Take Profit**: Risk-adjusted targets
- **Portfolio Limits**: Maximum exposure controls

## ⚡ Performance Tips

1. **GPU Acceleration**: Use CUDA-enabled GPU for 10x faster training
2. **Batch Processing**: Process multiple stocks simultaneously
3. **Data Caching**: Historical data is cached locally
4. **Sentiment Caching**: Sentiment data cached daily
5. **Quick Mode**: Use `--quick` flag for faster iteration

## 🚨 Limitations & Risks

- **No Financial Advice**: This is for educational/research purposes only
- **Market Risk**: Past performance doesn't guarantee future results
- **Data Dependencies**: Requires reliable internet for real-time data
- **Computational Cost**: Full training requires significant resources
- **Regulatory Compliance**: Ensure compliance with local trading regulations

## 🛠️ Troubleshooting

### Common Issues

1. **"Truth value of Series is ambiguous"**
   - Fixed in latest version with proper pandas handling

2. **CUDA out of memory**
   - Reduce batch size or use CPU mode

3. **Network timeouts**
   - Check internet connection
   - Sentiment analysis has built-in retries

4. **Missing data**
   - Some stocks may have insufficient historical data
   - System automatically skips problematic tickers

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📝 Requirements

See `requirements.txt` for complete list. Key dependencies:
- torch >= 1.9.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- yfinance >= 0.1.70
- scikit-learn >= 1.0.0
- transformers >= 4.10.0
- ta >= 0.8.0

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not financial advice. Trading stocks involves risk of loss. Always do your own research and consult with financial professionals before making investment decisions.

## 🙏 Acknowledgments

- Yahoo Finance for market data
- Hugging Face for transformer models
- TA-Lib community for technical analysis
- OpenAI for inspiration and methodologies

---

**Happy Trading! 📈🤖**