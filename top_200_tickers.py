"""
Top 200 Stock Tickers for Prediction
Organized by category and selected for liquidity, volatility, and prediction potential
"""

TOP_200_TICKERS = {
    # Mega Cap Tech (Most Liquid)
    "mega_cap_tech": [
        "AAPL",   # Apple
        "MSFT",   # Microsoft
        "GOOGL",  # Alphabet Class A
        "GOOG",   # Alphabet Class C
        "AMZN",   # Amazon
        "NVDA",   # NVIDIA
        "META",   # Meta Platforms
        "TSM",    # Taiwan Semiconductor
        "AVGO",   # Broadcom
        "ORCL",   # Oracle
    ],
    
    # Large Cap Tech
    "large_cap_tech": [
        "CSCO",   # Cisco
        "ADBE",   # Adobe
        "CRM",    # Salesforce
        "INTC",   # Intel
        "AMD",    # Advanced Micro Devices
        "QCOM",   # Qualcomm
        "TXN",    # Texas Instruments
        "IBM",    # IBM
        "AMAT",   # Applied Materials
        "NOW",    # ServiceNow
        "INTU",   # Intuit
        "UBER",   # Uber
        "SHOP",   # Shopify
        "SNOW",   # Snowflake
        "PLTR",   # Palantir
        "MU",     # Micron Technology
        "DELL",   # Dell Technologies
        "HPQ",    # HP Inc
        "MRVL",   # Marvell Technology
        "PANW",   # Palo Alto Networks
    ],
    
    # Software & Cloud
    "software_cloud": [
        "MSCI",   # MSCI Inc
        "FTNT",   # Fortinet
        "CDNS",   # Cadence Design
        "SNPS",   # Synopsys
        "CRWD",   # CrowdStrike
        "DDOG",   # Datadog
        "TEAM",   # Atlassian
        "HUBS",   # HubSpot
        "DOCN",   # DigitalOcean
        "NET",    # Cloudflare
        "ZM",     # Zoom
        "DOCU",   # DocuSign
        "OKTA",   # Okta
        "VEEV",   # Veeva Systems
        "WDAY",   # Workday
        "ZS",     # Zscaler
        "MDB",    # MongoDB
        "BILL",   # Bill.com
        "S",      # SentinelOne
        "PATH",   # UiPath
    ],
    
    # Electric Vehicles & Clean Energy
    "ev_clean_energy": [
        "TSLA",   # Tesla
        "RIVN",   # Rivian
        "LCID",   # Lucid Motors
        "NIO",    # NIO
        "XPEV",   # XPeng
        "LI",     # Li Auto
        "CHPT",   # ChargePoint
        "ENPH",   # Enphase Energy
        "SEDG",   # SolarEdge
        "PLUG",   # Plug Power
        "FCEL",   # FuelCell Energy
        "BE",     # Bloom Energy
        "RUN",    # Sunrun
    ],
    
    # Financial Services
    "financial": [
        "BRK-B",  # Berkshire Hathaway (corrected format)
        "JPM",    # JPMorgan Chase
        "V",      # Visa
        "MA",     # Mastercard
        "BAC",    # Bank of America
        "WFC",    # Wells Fargo
        "GS",     # Goldman Sachs
        "MS",     # Morgan Stanley
        "AXP",    # American Express
        "C",      # Citigroup
        "SCHW",   # Charles Schwab
        "BLK",    # BlackRock
        "SPGI",   # S&P Global
        "CB",     # Chubb
        "PGR",    # Progressive
        "COF",    # Capital One
        "USB",    # U.S. Bancorp
        "PNC",    # PNC Financial
        "TFC",    # Truist Financial
        "AIG",    # AIG
    ],
    
    # Healthcare & Pharma
    "healthcare": [
        "UNH",    # UnitedHealth
        "JNJ",    # Johnson & Johnson
        "LLY",    # Eli Lilly
        "PFE",    # Pfizer
        "ABBV",   # AbbVie
        "MRK",    # Merck
        "TMO",    # Thermo Fisher
        "ABT",    # Abbott
        "DHR",    # Danaher
        "BMY",    # Bristol-Myers Squibb
        "AMGN",   # Amgen
        "CVS",    # CVS Health
        "GILD",   # Gilead Sciences
        "ISRG",   # Intuitive Surgical
        "VRTX",   # Vertex Pharma
        "REGN",   # Regeneron
        "MRNA",   # Moderna
        "HCA",    # HCA Healthcare
        "CI",     # Cigna
        "ELV",    # Elevance Health
    ],
    
    # Consumer & Retail
    "consumer_retail": [
        "WMT",    # Walmart
        "HD",     # Home Depot
        "PG",     # Procter & Gamble
        "KO",     # Coca-Cola
        "PEP",    # PepsiCo
        "COST",   # Costco
        "NKE",    # Nike
        "MCD",    # McDonald's
        "SBUX",   # Starbucks
        "LOW",    # Lowe's
        "TGT",    # Target
        "TJX",    # TJX Companies
        "DIS",    # Disney
        "NFLX",   # Netflix
        "CMCSA",  # Comcast
        "ABNB",   # Airbnb
        "BKNG",   # Booking Holdings
        "MAR",    # Marriott
        "HLT",    # Hilton
        "CCL",    # Carnival
    ],
    
    # Industrial & Energy
    "industrial_energy": [
        "XOM",    # Exxon Mobil
        "CVX",    # Chevron
        "COP",    # ConocoPhillips
        "SLB",    # Schlumberger
        "EOG",    # EOG Resources
        "OXY",    # Occidental Petroleum
        "CAT",    # Caterpillar
        "BA",     # Boeing
        "GE",     # General Electric
        "RTX",    # Raytheon
        "LMT",    # Lockheed Martin
        "HON",    # Honeywell
        "UPS",    # UPS
        "UNP",    # Union Pacific
        "DE",     # John Deere
        "MMM",    # 3M
        "ADP",    # ADP
        "CSX",    # CSX Corp
        "NSC",    # Norfolk Southern
        "FDX",    # FedEx
    ],
    
    # Semiconductors & Hardware
    "semiconductors": [
        "LRCX",   # Lam Research
        "KLAC",   # KLA Corp
        "ASML",   # ASML Holding
        "NXPI",   # NXP Semiconductors
        "ADI",    # Analog Devices
        "ON",     # ON Semiconductor
        "MCHP",   # Microchip Technology
        "MPWR",   # Monolithic Power
        "SWKS",   # Skyworks Solutions
        "STX",    # Seagate
        "WDC",    # Western Digital
        "QRVO",   # Qorvo
        "ENTG",   # Entegris
        "ANET",   # Arista Networks
        "KEYS",   # Keysight Technologies
    ],
    
    # High Volatility Growth
    "high_volatility": [
        "COIN",   # Coinbase
        "HOOD",   # Robinhood
        "SOFI",   # SoFi
        "UPST",   # Upstart
        "AFRM",   # Affirm
        "RKLB",   # Rocket Lab
        "ROKU",   # Roku
        "RBLX",   # Roblox
        "U",      # Unity Software
        "APP",    # AppLovin
        "DASH",   # DoorDash
        "LYFT",   # Lyft
        "PINS",   # Pinterest
        "SNAP",   # Snapchat
        "TWLO",   # Twilio
        "DBX",    # Dropbox
        "BOX",    # Box Inc
        "FVRR",   # Fiverr
        "ETSY",   # Etsy
        "W",      # Wayfair
    ],
    
    # ETFs for Market Sentiment
    "etfs": [
        "SPY",    # S&P 500 ETF
        "QQQ",    # Nasdaq 100 ETF
        "IWM",    # Russell 2000 ETF
        "DIA",    # Dow Jones ETF
        "VTI",    # Total Market ETF
        "VOO",    # Vanguard S&P 500
        "XLF",    # Financial Sector ETF
        "XLK",    # Technology Sector ETF
        "VNQ",    # Real Estate ETF
        "GLD",    # Gold ETF
    ]
}

# Flatten to simple list
ALL_TICKERS = []
for category, tickers in TOP_200_TICKERS.items():
    ALL_TICKERS.extend(tickers)

# Function to get tickers by category
def get_tickers_by_category(categories=None):
    """
    Get tickers filtered by category
    
    Args:
        categories: List of category names, or None for all
        
    Returns:
        List of ticker symbols
    """
    if categories is None:
        return ALL_TICKERS
    
    tickers = []
    for category in categories:
        if category in TOP_200_TICKERS:
            tickers.extend(TOP_200_TICKERS[category])
    
    return tickers

# Function to get high-volume tickers for day trading
def get_high_volume_tickers():
    """Get tickers with highest average volume (best for prediction)"""
    return [
        "SPY", "QQQ", "AAPL", "TSLA", "NVDA", "AMD", "META", "AMZN", 
        "MSFT", "GOOGL", "SOFI", "PLTR", "NIO", "F", "BAC", "INTC",
        "T", "WFC", "C", "AAL", "CCL", "COIN", "RIVN", "LCID"
    ]

# Function to get tickers for different strategies
def get_strategy_tickers(strategy):
    """
    Get tickers optimized for specific strategies
    
    Args:
        strategy: 'momentum', 'volatility', 'value', 'growth'
    """
    strategies = {
        'momentum': ["NVDA", "TSLA", "AMD", "PLTR", "COIN", "SOFI", "UPST", "RKLB", "IONQ", "AI"],
        'volatility': ["TSLA", "RIVN", "LCID", "COIN", "HOOD", "SOFI", "UPST", "RKLB", "SPCE", "CLOV"],
        'value': ["BRK-B", "JPM", "XOM", "CVX", "WMT", "JNJ", "PG", "HD", "BAC", "WFC"],
        'growth': ["NVDA", "META", "AMZN", "GOOGL", "MSFT", "CRM", "NOW", "PANW", "CRWD", "SNOW"]
    }
    
    return strategies.get(strategy, [])

# Weekly options tickers (high liquidity for prediction)
WEEKLY_OPTIONS_TICKERS = [
    "SPY", "QQQ", "IWM", "AAPL", "TSLA", "AMZN", "GOOGL", "META", 
    "MSFT", "NVDA", "AMD", "NFLX", "BA", "GS", "JPM", "BAC"
]

# Print summary
if __name__ == "__main__":
    print(f"Total tickers: {len(ALL_TICKERS)}")
    print("\nTickers by category:")
    for category, tickers in TOP_200_TICKERS.items():
        print(f"  {category}: {len(tickers)} tickers")
    
    print(f"\nHigh volume tickers: {len(get_high_volume_tickers())}")
    print(f"Weekly options tickers: {len(WEEKLY_OPTIONS_TICKERS)}")
    
    # Example usage with the predictor
    print("\nExample usage:")
    print("predictor.train(ALL_TICKERS[:50], start_date, end_date)")
    print("predictor.train(get_tickers_by_category(['mega_cap_tech', 'semiconductors']), start_date, end_date)")
    print("predictor.train(get_high_volume_tickers(), start_date, end_date)")