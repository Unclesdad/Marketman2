#!/usr/bin/env python3
"""
Stock Market Recommendations Script
Load your trained model and get top buy/short recommendations
"""

from advanced_stock_predictor import AdvancedStockPredictor
import os

def main():
    # Find the latest model
    model_dir = 'data/models'
    if not os.path.exists(model_dir):
        print("❌ No models directory found. Please train a model first by running:")
        print("   python run_advanced_predictor.py")
        return
    
    models = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not models:
        print("❌ No trained models found. Please train a model first by running:")
        print("   python run_advanced_predictor.py")
        return
    
    # Use the latest model
    latest_model = sorted(models)[-1]
    print(f"📚 Loading model: {latest_model}")
    
    # Initialize predictor and load model
    predictor = AdvancedStockPredictor()
    predictor.load_model(latest_model)
    print("✅ Model loaded successfully!")
    
    # Analyze all stocks and get recommendations
    print("\n🚀 Starting comprehensive market analysis...")
    results = predictor.predict_all_stocks(min_confidence=0.6)
    
    # Display formatted results
    predictor.display_recommendations(results)
    
    # Optional: Save results to file
    import json
    results_file = f"market_analysis_{results['timestamp'].replace(':', '-').replace(' ', '_')}.json"
    
    # Convert results to JSON-serializable format
    json_results = {
        'timestamp': results['timestamp'],
        'summary': results['summary'],
        'top_buys': [
            {
                'symbol': stock['symbol'],
                'confidence': stock['confidence'],
                'prediction': stock['prediction'],
                'current_price': stock['technical_indicators']['current_price'],
                'action': stock['recommendation']['action'],
                'reason': stock['recommendation']['reason']
            }
            for stock in results['top_buys']
        ],
        'top_shorts': [
            {
                'symbol': stock['symbol'],
                'confidence': stock['confidence'],
                'prediction': stock['prediction'],
                'current_price': stock['technical_indicators']['current_price'],
                'action': stock['recommendation']['action'],
                'reason': stock['recommendation']['reason']
            }
            for stock in results['top_shorts']
        ],
        'failed_symbols': results['failed_symbols']
    }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Quick summary for copy-paste
    print(f"\n📋 QUICK SUMMARY:")
    if results['top_buys']:
        buy_symbols = [stock['symbol'] for stock in results['top_buys'][:5]]
        print(f"   🟢 Top 5 BUYS: {', '.join(buy_symbols)}")
    
    if results['top_shorts']:
        short_symbols = [stock['symbol'] for stock in results['top_shorts'][:5]]
        print(f"   🔴 Top 5 SHORTS: {', '.join(short_symbols)}")

if __name__ == "__main__":
    main()