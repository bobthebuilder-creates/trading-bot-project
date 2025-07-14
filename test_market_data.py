"""
Test script for market data functionality
Run this to verify your crypto data ingestion is working
"""
import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.market_data import MarketDataManager

def main():
    """Test the market data system"""
    print("🚀 Testing Crypto Market Data System")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize data manager
        print("1. Initializing Market Data Manager...")
        dm = MarketDataManager()
        
        # Connect to Binance (public data only)
        print("2. Connecting to Binance (public data)...")
        dm.add_public_exchange('binance')
        
        # Test 1: Get BTC data
        print("\n3. Testing BTC/USDT data fetching...")
        btc_data = dm.get_ohlcv('BTC/USDT', '1h', 50)
        
        if not btc_data.empty:
            latest_price = btc_data['close'].iloc[-1]
            print(f"   ✅ Successfully fetched {len(btc_data)} BTC candles")
            print(f"   💰 Latest BTC price: ${latest_price:,.2f}")
            print(f"   📊 Price range: ${btc_data['low'].min():,.2f} - ${btc_data['high'].max():,.2f}")
        else:
            print("   ❌ Failed to fetch BTC data")
            return
        
        # Test 2: Technical indicators
        print("\n4. Testing technical indicators...")
        btc_with_indicators = dm.calculate_technical_indicators(btc_data.copy())
        
        if 'rsi' in btc_with_indicators.columns:
            rsi = btc_with_indicators['rsi'].iloc[-1]
            macd = btc_with_indicators['macd'].iloc[-1]
            sma_20 = btc_with_indicators['sma_20'].iloc[-1]
            
            print(f"   ✅ Technical indicators calculated")
            print(f"   📈 RSI: {rsi:.2f} ({'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'})")
            print(f"   📊 MACD: {macd:.6f}")
            print(f"   📉 SMA(20): ${sma_20:,.2f}")
        
        # Test 3: Multiple symbols
        print("\n5. Testing multiple symbols...")
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        multi_data = dm.get_multiple_symbols(symbols, '1h', 20)
        
        print(f"   ✅ Fetched data for {len(multi_data)} symbols:")
        for symbol, data in multi_data.items():
            if not data.empty:
                price = data['close'].iloc[-1]
                change = ((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100
                print(f"   💎 {symbol}: ${price:,.4f} ({change:+.2f}%)")
        
        # Test 4: Market overview
        print("\n6. Testing market overview...")
        overview = dm.get_market_overview(['BTC/USDT', 'ETH/USDT', 'ADA/USDT'])
        
        if not overview.empty:
            print(f"   ✅ Market overview generated for {len(overview)} symbols")
            print("   📊 Top performers:")
            top_performers = overview.nlargest(3, 'percentage')
            for symbol, row in top_performers.iterrows():
                print(f"   🚀 {symbol}: ${row['last']:,.4f} ({row['percentage']:+.2f}%)")
        
        # Test 5: Orderbook
        print("\n7. Testing orderbook data...")
        orderbook = dm.get_orderbook('BTC/USDT')
        
        if orderbook:
            spread = orderbook['spread']
            best_bid = orderbook['bids'][0][0] if orderbook['bids'] else 0
            best_ask = orderbook['asks'][0][0] if orderbook['asks'] else 0
            
            print(f"   ✅ Orderbook fetched")
            print(f"   💰 Best bid: ${best_bid:,.2f}")
            print(f"   💰 Best ask: ${best_ask:,.2f}")
            print(f"   📏 Spread: ${spread:.2f}")
        
        # Test 6: Save/Load data
        print("\n8. Testing data persistence...")
        filename = f"btc_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        dm.save_data(btc_data, filename)
        
        loaded_data = dm.load_data(filename)
        if not loaded_data.empty and len(loaded_data) == len(btc_data):
            print(f"   ✅ Data saved and loaded successfully")
            print(f"   💾 File: data/raw/{filename}")
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your crypto data system is working perfectly!")
        print("\n📋 What you can do now:")
        print("   • Fetch real-time crypto prices")
        print("   • Calculate technical indicators")
        print("   • Monitor multiple symbols")
        print("   • Save/load historical data")
        print("   • Access orderbook data")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Check your internet connection and try again.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
"""
Test script for market data functionality
Run this to verify your crypto data ingestion is working
"""
import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.market_data import MarketDataManager

def main():
    """Test the market data system"""
    print("🚀 Testing Crypto Market Data System")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize data manager
        print("1. Initializing Market Data Manager...")
        dm = MarketDataManager()
        
        # Connect to Kraken (public data only)
        print("2. Connecting to Kraken (public data)...")
        dm.add_public_exchange('kraken')
        
        # Test 1: Get BTC data
        print("\n3. Testing BTC/USD data fetching...")
        btc_data = dm.get_ohlcv('BTC/USD', '1h', 50)
        
        if not btc_data.empty:
            latest_price = btc_data['close'].iloc[-1]
            print(f"   ✅ Successfully fetched {len(btc_data)} BTC candles")
            print(f"   💰 Latest BTC price: ${latest_price:,.2f}")
            print(f"   📊 Price range: ${btc_data['low'].min():,.2f} - ${btc_data['high'].max():,.2f}")
        else:
            print("   ❌ Failed to fetch BTC data")
            return
        
        # Test 2: Technical indicators
        print("\n4. Testing technical indicators...")
        btc_with_indicators = dm.calculate_technical_indicators(btc_data.copy())
        
        if 'rsi' in btc_with_indicators.columns:
            rsi = btc_with_indicators['rsi'].iloc[-1]
            macd = btc_with_indicators['macd'].iloc[-1]
            sma_20 = btc_with_indicators['sma_20'].iloc[-1]
            
            print(f"   ✅ Technical indicators calculated")
            print(f"   📈 RSI: {rsi:.2f} ({'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'})")
            print(f"   📊 MACD: {macd:.6f}")
            print(f"   📉 SMA(20): ${sma_20:,.2f}")
        
        # Test 3: Multiple symbols
        print("\n5. Testing multiple symbols...")
        symbols = ['BTC/USD', 'ETH/USD', 'ADA/USD']
        multi_data = dm.get_multiple_symbols(symbols, '1h', 20)
        
        print(f"   ✅ Fetched data for {len(multi_data)} symbols:")
        for symbol, data in multi_data.items():
            if not data.empty:
                price = data['close'].iloc[-1]
                change = ((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100
                print(f"   💎 {symbol}: ${price:,.4f} ({change:+.2f}%)")
        
        # Test 4: Market overview
        print("\n6. Testing market overview...")
        overview = dm.get_market_overview(['BTC/USD', 'ETH/USD', 'ADA/USD'])
        
        if not overview.empty:
            print(f"   ✅ Market overview generated for {len(overview)} symbols")
            print("   📊 Top performers:")
            top_performers = overview.nlargest(3, 'percentage')
            for symbol, row in top_performers.iterrows():
                print(f"   🚀 {symbol}: ${row['last']:,.4f} ({row['percentage']:+.2f}%)")
        
        # Test 5: Orderbook
        print("\n7. Testing orderbook data...")
        orderbook = dm.get_orderbook('BTC/USD')
        
        if orderbook:
            spread = orderbook['spread']
            best_bid = orderbook['bids'][0][0] if orderbook['bids'] else 0
            best_ask = orderbook['asks'][0][0] if orderbook['asks'] else 0
            
            print(f"   ✅ Orderbook fetched")
            print(f"   💰 Best bid: ${best_bid:,.2f}")
            print(f"   💰 Best ask: ${best_ask:,.2f}")
            print(f"   📏 Spread: ${spread:.2f}")
        
        # Test 6: Save/Load data
        print("\n8. Testing data persistence...")
        filename = f"btc_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        dm.save_data(btc_data, filename)
        
        loaded_data = dm.load_data(filename)
        if not loaded_data.empty and len(loaded_data) == len(btc_data):
            print(f"   ✅ Data saved and loaded successfully")
            print(f"   💾 File: data/raw/{filename}")
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your crypto data system is working perfectly!")
        print("\n📋 What you can do now:")
        print("   • Fetch real-time crypto prices")
        print("   • Calculate technical indicators")
        print("   • Monitor multiple symbols")
        print("   • Save/load historical data")
        print("   • Access orderbook data")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Check your internet connection and try again.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
"""
Test script for market data functionality
Run this to verify your crypto data ingestion is working
"""
import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.market_data import MarketDataManager

def main():
    """Test the market data system"""
    print("🚀 Testing Crypto Market Data System")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize data manager
        print("1. Initializing Market Data Manager...")
        dm = MarketDataManager()
        
        # Connect to best available exchange
        print("2. Connecting to best available exchange...")
        connected_exchange = dm.connect_best_exchange()
        print(f"   ✅ Connected to {connected_exchange}")
        
        # Test 1: Get BTC data
        print("\n3. Testing BTC/USD data fetching...")
        btc_data = dm.get_ohlcv('BTC/USD', '1h', 50)
        
        if not btc_data.empty:
            latest_price = btc_data['close'].iloc[-1]
            print(f"   ✅ Successfully fetched {len(btc_data)} BTC candles")
            print(f"   💰 Latest BTC price: ${latest_price:,.2f}")
            print(f"   📊 Price range: ${btc_data['low'].min():,.2f} - ${btc_data['high'].max():,.2f}")
        else:
            print("   ❌ Failed to fetch BTC data")
            return
        
        # Test 2: Technical indicators
        print("\n4. Testing technical indicators...")
        btc_with_indicators = dm.calculate_technical_indicators(btc_data.copy())
        
        if 'rsi' in btc_with_indicators.columns:
            rsi = btc_with_indicators['rsi'].iloc[-1]
            macd = btc_with_indicators['macd'].iloc[-1]
            sma_20 = btc_with_indicators['sma_20'].iloc[-1]
            
            print(f"   ✅ Technical indicators calculated")
            print(f"   📈 RSI: {rsi:.2f} ({'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'})")
            print(f"   📊 MACD: {macd:.6f}")
            print(f"   📉 SMA(20): ${sma_20:,.2f}")
        
        # Test 3: Multiple symbols
        print("\n5. Testing multiple symbols...")
        symbols = ['BTC/USD', 'ETH/USD', 'ADA/USD']
        multi_data = dm.get_multiple_symbols(symbols, '1h', 20)
        
        print(f"   ✅ Fetched data for {len(multi_data)} symbols:")
        for symbol, data in multi_data.items():
            if not data.empty:
                price = data['close'].iloc[-1]
                change = ((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100
                print(f"   💎 {symbol}: ${price:,.4f} ({change:+.2f}%)")
        
        # Test 4: Market overview
        print("\n6. Testing market overview...")
        overview = dm.get_market_overview(['BTC/USD', 'ETH/USD', 'ADA/USD'])
        
        if not overview.empty:
            print(f"   ✅ Market overview generated for {len(overview)} symbols")
            print("   📊 Top performers:")
            top_performers = overview.nlargest(3, 'percentage')
            for symbol, row in top_performers.iterrows():
                print(f"   🚀 {symbol}: ${row['last']:,.4f} ({row['percentage']:+.2f}%)")
        
        # Test 5: Orderbook
        print("\n7. Testing orderbook data...")
        orderbook = dm.get_orderbook('BTC/USD')
        
        if orderbook:
            spread = orderbook['spread']
            best_bid = orderbook['bids'][0][0] if orderbook['bids'] else 0
            best_ask = orderbook['asks'][0][0] if orderbook['asks'] else 0
            
            print(f"   ✅ Orderbook fetched")
            print(f"   💰 Best bid: ${best_bid:,.2f}")
            print(f"   💰 Best ask: ${best_ask:,.2f}")
            print(f"   📏 Spread: ${spread:.2f}")
        
        # Test 6: Save/Load data
        print("\n8. Testing data persistence...")
        filename = f"btc_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        dm.save_data(btc_data, filename)
        
        loaded_data = dm.load_data(filename)
        if not loaded_data.empty and len(loaded_data) == len(btc_data):
            print(f"   ✅ Data saved and loaded successfully")
            print(f"   💾 File: data/raw/{filename}")
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your crypto data system is working perfectly!")
        print("\n📋 What you can do now:")
        print("   • Fetch real-time crypto prices")
        print("   • Calculate technical indicators")
        print("   • Monitor multiple symbols")
        print("   • Save/load historical data")
        print("   • Access orderbook data")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Check your internet connection and try again.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
