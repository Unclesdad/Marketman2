"""
Real Sentiment Analysis Implementation
Multiple approaches from free to premium
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import time
import json
from typing import Dict, List
import feedparser
import re

# Option 1: Reddit Sentiment (Free, Real-time)
class RedditSentimentAnalyzer:
    """
    Analyzes sentiment from Reddit (WallStreetBets, stocks, investing)
    Free but requires careful rate limiting
    """
    
    def __init__(self):
        self.headers = {'User-Agent': 'StockPredictor/1.0'}
        
    def get_reddit_sentiment(self, ticker: str) -> Dict:
        """Get sentiment from Reddit discussions"""
        
        # Use Reddit API (pushshift is deprecated, use alternative approach)
        base_url = "https://www.reddit.com/r/wallstreetbets/search.json"
        
        sentiments = []
        
        # Search multiple subreddits using direct Reddit JSON API
        subreddits = ['wallstreetbets', 'stocks', 'investing', 'StockMarket']
        
        for subreddit in subreddits:
            try:
                # Reddit JSON API endpoint
                url = f"https://www.reddit.com/r/{subreddit}/search.json"
                params = {
                    'q': f'${ticker} OR {ticker}',
                    'restrict_sr': 'true',
                    'sort': 'new',
                    'limit': 25,
                    't': 'week'  # Last week
                }
                
                response = requests.get(url, params=params, headers=self.headers)
                if response.status_code == 200:
                    data = response.json()
                    
                    for post in data.get('data', {}).get('children', []):
                        post_data = post.get('data', {})
                        title = post_data.get('title', '').lower()
                        score = post_data.get('score', 0)
                        comments = post_data.get('num_comments', 0)
                        selftext = post_data.get('selftext', '').lower()
                        
                        # Combined text for analysis
                        text = f"{title} {selftext}"
                        
                        # Enhanced sentiment rules
                        sentiment_score = 0
                        bullish_words = ['calls', 'buy', 'long', 'bull', 'moon', 'squeeze', 'rocket', 'diamond', 'hodl', 'to the moon']
                        bearish_words = ['puts', 'sell', 'short', 'bear', 'crash', 'dump', 'tank', 'red', 'down']
                        
                        bullish_count = sum(1 for word in bullish_words if word in text)
                        bearish_count = sum(1 for word in bearish_words if word in text)
                        
                        if bullish_count > bearish_count:
                            sentiment_score = min(1.0, bullish_count / 3)
                        elif bearish_count > bullish_count:
                            sentiment_score = -min(1.0, bearish_count / 3)
                        
                        # Weight by engagement and recency
                        if sentiment_score != 0 and score > 5:
                            engagement_weight = min(3.0, 1 + (score + comments) / 100)
                            sentiments.append(sentiment_score * engagement_weight)
                
                time.sleep(1)  # Rate limit to be respectful
                
            except Exception as e:
                print(f"Reddit API error for {subreddit}/{ticker}: {e}")
        
        if sentiments:
            return {
                'reddit_sentiment': np.mean(sentiments),
                'reddit_posts': len(sentiments),
                'reddit_bullish_ratio': sum(1 for s in sentiments if s > 0) / len(sentiments)
            }
        else:
            return {'reddit_sentiment': 0, 'reddit_posts': 0, 'reddit_bullish_ratio': 0.5}

# Option 2: Financial News RSS Feeds (Free)
class RSSNewsSentimentAnalyzer:
    """
    Analyzes sentiment from free RSS feeds
    More reliable than scraping
    """
    
    def __init__(self):
        # Enhanced financial RSS feeds with better coverage
        self.feeds = {
            'yahoo_finance': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/topstories',
            'seeking_alpha': 'https://seekingalpha.com/feed.xml',
            'benzinga': 'https://www.benzinga.com/feed',
            'forbes_investing': 'https://www.forbes.com/investing/feed/',
            'cnbc_markets': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'zacks': 'https://www.zacks.com/commentary/rss'
        }
        
        # Enhanced sentiment keywords
        self.positive_words = [
            'surge', 'gain', 'rise', 'jump', 'advance', 'rally', 'bullish', 'soar',
            'upgrade', 'beat', 'strong', 'record', 'growth', 'profit', 'buy', 'outperform',
            'upside', 'momentum', 'breakthrough', 'boost', 'optimistic', 'exceed', 
            'success', 'winning', 'leader', 'innovate', 'expansion', 'positive'
        ]
        
        self.negative_words = [
            'fall', 'drop', 'decline', 'plunge', 'crash', 'bear', 'loss', 'slump',
            'downgrade', 'miss', 'weak', 'concern', 'risk', 'sell', 'warning', 'underperform',
            'downside', 'pressure', 'challenge', 'struggle', 'pessimistic', 'disappoint',
            'failure', 'losing', 'lag', 'cut', 'reduce', 'negative', 'volatile'
        ]
        
        # Company-specific indicators
        self.strength_indicators = [
            'earnings beat', 'revenue growth', 'market share', 'new product',
            'partnership', 'acquisition', 'expansion', 'innovation'
        ]
        
        self.weakness_indicators = [
            'earnings miss', 'revenue decline', 'market loss', 'lawsuit',
            'investigation', 'layoffs', 'bankruptcy', 'debt'
        ]
    
    def analyze_text_sentiment(self, text: str) -> float:
        """Enhanced keyword-based sentiment analysis"""
        text_lower = text.lower()
        
        # Count basic sentiment words
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        # Count strength/weakness indicators (higher weight)
        strength_count = sum(2 for indicator in self.strength_indicators if indicator in text_lower)
        weakness_count = sum(2 for indicator in self.weakness_indicators if indicator in text_lower)
        
        # Combine counts
        total_positive = positive_count + strength_count
        total_negative = negative_count + weakness_count
        
        if total_positive + total_negative == 0:
            return 0
        
        # Normalized sentiment score
        sentiment = (total_positive - total_negative) / (total_positive + total_negative)
        
        # Apply intensity modifiers
        if 'very' in text_lower or 'extremely' in text_lower or 'significantly' in text_lower:
            sentiment *= 1.5
        elif 'slightly' in text_lower or 'somewhat' in text_lower:
            sentiment *= 0.7
            
        return max(-1.0, min(1.0, sentiment))
    
    def get_news_sentiment(self, ticker: str) -> Dict:
        """Get sentiment from RSS feeds"""
        
        all_sentiments = []
        relevant_articles = 0
        
        for source, feed_url in self.feeds.items():
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:50]:  # Last 50 articles
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    full_text = f"{title} {summary}"
                    
                    # Enhanced ticker matching
                    ticker_patterns = [
                        ticker.upper(),
                        f'${ticker.upper()}',
                        f'{ticker.lower()}',
                        f' {ticker.upper()} ',
                        f'({ticker.upper()})',
                        f'{ticker.upper()}:',
                        f'{ticker.upper()}.'
                    ]
                    
                    if any(pattern in full_text.upper() for pattern in ticker_patterns):
                        sentiment = self.analyze_text_sentiment(full_text)
                        all_sentiments.append(sentiment)
                        relevant_articles += 1
                
                time.sleep(0.2)  # Be nice to servers
                
            except Exception as e:
                print(f"RSS feed error for {source}: {e}")
        
        if all_sentiments:
            return {
                'news_sentiment': np.mean(all_sentiments),
                'news_articles': relevant_articles,
                'positive_news_ratio': sum(1 for s in all_sentiments if s > 0) / len(all_sentiments)
            }
        else:
            return {'news_sentiment': 0, 'news_articles': 0, 'positive_news_ratio': 0.5}

# Option 3: Twitter/X Sentiment (Requires API key but has free tier)
class TwitterSentimentAnalyzer:
    """
    Analyzes sentiment from Twitter/X
    Requires API key but has free tier (1500 tweets/month)
    """
    
    def __init__(self, bearer_token=None):
        self.bearer_token = bearer_token
        self.headers = {'Authorization': f'Bearer {bearer_token}'} if bearer_token else None
        
    def get_twitter_sentiment(self, ticker: str) -> Dict:
        """Get sentiment from Twitter"""
        
        if not self.bearer_token:
            return {'twitter_sentiment': 0, 'tweet_count': 0}
        
        # Twitter API v2 endpoint
        url = "https://api.twitter.com/2/tweets/search/recent"
        
        # Cashtag search (e.g., $AAPL)
        query = f"${ticker} -is:retweet lang:en"
        
        params = {
            'query': query,
            'max_results': 100,
            'tweet.fields': 'public_metrics,created_at'
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                tweets = data.get('data', [])
                
                sentiments = []
                total_engagement = 0
                
                for tweet in tweets:
                    text = tweet.get('text', '').lower()
                    metrics = tweet.get('public_metrics', {})
                    engagement = metrics.get('like_count', 0) + metrics.get('retweet_count', 0)
                    
                    # Simple sentiment
                    sentiment = 0
                    if any(word in text for word in ['bull', 'calls', 'buy', 'long', 'moon']):
                        sentiment = 1
                    elif any(word in text for word in ['bear', 'puts', 'sell', 'short', 'crash']):
                        sentiment = -1
                    
                    if sentiment != 0:
                        # Weight by engagement
                        weight = 1 + min(engagement / 100, 5)
                        sentiments.append(sentiment * weight)
                        total_engagement += engagement
                
                if sentiments:
                    return {
                        'twitter_sentiment': np.mean(sentiments),
                        'tweet_count': len(tweets),
                        'twitter_engagement': total_engagement
                    }
            
        except Exception as e:
            print(f"Twitter API error: {e}")
        
        return {'twitter_sentiment': 0, 'tweet_count': 0, 'twitter_engagement': 0}

# Option 4: Free News API with limited calls
class FreeNewsAPI:
    """
    Uses free news APIs with rate limits
    """
    
    def __init__(self):
        # NewsAPI.org gives 100 requests/day free
        self.newsapi_key = "YOUR_FREE_API_KEY"  # Sign up at newsapi.org
        
    def get_newsapi_sentiment(self, ticker: str) -> Dict:
        """Get sentiment from NewsAPI.org"""
        
        if self.newsapi_key == "YOUR_FREE_API_KEY":
            return {'newsapi_sentiment': 0, 'article_count': 0}
        
        url = "https://newsapi.org/v2/everything"
        
        params = {
            'q': ticker,
            'apiKey': self.newsapi_key,
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': 20,
            'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        }
        
        try:
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                sentiments = []
                for article in articles:
                    title = article.get('title', '')
                    description = article.get('description', '')
                    
                    # Use simple sentiment analysis
                    analyzer = RSSNewsSentimentAnalyzer()
                    sentiment = analyzer.analyze_text_sentiment(f"{title} {description}")
                    sentiments.append(sentiment)
                
                if sentiments:
                    return {
                        'newsapi_sentiment': np.mean(sentiments),
                        'article_count': len(articles)
                    }
            
        except Exception as e:
            print(f"NewsAPI error: {e}")
        
        return {'newsapi_sentiment': 0, 'article_count': 0}

# Combine all sources
class ComprehensiveSentimentAnalyzer:
    """
    Combines multiple free sentiment sources
    """
    
    def __init__(self, twitter_token=None, newsapi_key=None):
        self.reddit = RedditSentimentAnalyzer()
        self.rss = RSSNewsSentimentAnalyzer()
        self.twitter = TwitterSentimentAnalyzer(twitter_token) if twitter_token else None
        self.newsapi = FreeNewsAPI()
        if newsapi_key:
            self.newsapi.newsapi_key = newsapi_key
        
        # Cache to avoid repeated API calls
        self.cache = {}
        self.cache_duration = timedelta(hours=6)
    
    def get_combined_sentiment(self, ticker: str) -> Dict:
        """Get sentiment from all available sources"""
        
        # Check cache
        cache_key = f"{ticker}_{datetime.now().date()}"
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                return cached_data
        
        print(f"Fetching sentiment for {ticker}...")
        
        combined_sentiment = {}
        
        # Get Reddit sentiment (free)
        try:
            reddit_data = self.reddit.get_reddit_sentiment(ticker)
            combined_sentiment.update(reddit_data)
        except:
            combined_sentiment.update({'reddit_sentiment': 0, 'reddit_posts': 0})
        
        # Get RSS news sentiment (free)
        try:
            rss_data = self.rss.get_news_sentiment(ticker)
            combined_sentiment.update(rss_data)
        except:
            combined_sentiment.update({'news_sentiment': 0, 'news_articles': 0})
        
        # Get Twitter sentiment (if API key provided)
        if self.twitter:
            try:
                twitter_data = self.twitter.get_twitter_sentiment(ticker)
                combined_sentiment.update(twitter_data)
            except:
                combined_sentiment.update({'twitter_sentiment': 0, 'tweet_count': 0})
        
        # Calculate overall sentiment
        sentiments = []
        weights = []
        
        if combined_sentiment.get('reddit_posts', 0) > 0:
            sentiments.append(combined_sentiment['reddit_sentiment'])
            weights.append(min(combined_sentiment['reddit_posts'] / 10, 1))
        
        if combined_sentiment.get('news_articles', 0) > 0:
            sentiments.append(combined_sentiment['news_sentiment'])
            weights.append(min(combined_sentiment['news_articles'] / 5, 1))
        
        if combined_sentiment.get('tweet_count', 0) > 0:
            sentiments.append(combined_sentiment['twitter_sentiment'])
            weights.append(min(combined_sentiment['tweet_count'] / 50, 1))
        
        if sentiments:
            combined_sentiment['overall_sentiment'] = np.average(sentiments, weights=weights)
        else:
            combined_sentiment['overall_sentiment'] = 0
        
        # Cache the result
        self.cache[cache_key] = (datetime.now(), combined_sentiment)
        
        return combined_sentiment

# Integration with stock predictor
def add_sentiment_to_features(features_df: pd.DataFrame, ticker: str, 
                            sentiment_analyzer: ComprehensiveSentimentAnalyzer) -> pd.DataFrame:
    """Add sentiment features to the dataframe"""
    
    # Get sentiment data
    sentiment_data = sentiment_analyzer.get_combined_sentiment(ticker)
    
    # Add as features (constant for historical data - simplified)
    for key, value in sentiment_data.items():
        features_df[f'sentiment_{key}'] = value
    
    # For more sophisticated approach, you'd want rolling sentiment
    # This would require storing historical sentiment data
    
    return features_df

# Example usage
if __name__ == "__main__":
    # Initialize sentiment analyzer
    analyzer = ComprehensiveSentimentAnalyzer(
        twitter_token=None,  # Add your Twitter bearer token if you have one
        newsapi_key=None     # Add your NewsAPI key if you have one
    )
    
    # Test on a few tickers
    test_tickers = ['AAPL', 'TSLA', 'GME']
    
    for ticker in test_tickers:
        sentiment = analyzer.get_combined_sentiment(ticker)
        
        print(f"\n{ticker} Sentiment Analysis:")
        print(f"  Reddit: {sentiment.get('reddit_sentiment', 0):.2f} ({sentiment.get('reddit_posts', 0)} posts)")
        print(f"  News: {sentiment.get('news_sentiment', 0):.2f} ({sentiment.get('news_articles', 0)} articles)")
        print(f"  Overall: {sentiment.get('overall_sentiment', 0):.2f}")
        
        time.sleep(2)  # Rate limiting