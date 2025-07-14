"""
Market data ingestion from various sources
"""
import ccxt
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class MarketDataManager:
    """Handles market data from multiple exchanges"""
    
    def __init__(self):
        self.exchanges = {}
        self.supported_exchanges = ['binance', 'coinbase', 'kraken']
    
    def add_exchange(self, exchange_name: str, config: Dict):
        """Add an exchange connection"""
        if exchange_name not in self.supported_exchanges:
            raise ValueError(f"Exchange {exchange_name} not supported")
        
        # TODO: Initialize exchange with config
        logger.info(f"Adding exchange: {exchange_name}")
    
    def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Get OHLCV data for a symbol"""
        # TODO: Implement data fetching
        logger.info(f"Fetching {symbol} data with {timeframe} timeframe")
        return pd.DataFrame()
    
    def get_orderbook(self, symbol: str) -> Dict:
        """Get current orderbook for a symbol"""
        # TODO: Implement orderbook fetching
        return {}
"""
Market data ingestion from various crypto exchanges
"""
import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import time
from datetime import datetime, timedelta
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class MarketDataManager:
    """Handles market data from multiple exchanges"""
    
    def __init__(self, config: Dict = None):
        self.exchanges = {}
        self.supported_exchanges = ['binance', 'coinbase', 'kraken']
        self.data_cache = {}
        self.config = config or {}
        
        # Default symbols to track
        self.default_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 
            'SOL/USDT', 'XRP/USDT', 'DOT/USDT', 'AVAX/USDT'
        ]
        
        logger.info("MarketDataManager initialized")
    
    def add_exchange(self, exchange_name: str, config: Dict = None):
        """Add an exchange connection"""
        if exchange_name not in self.supported_exchanges:
            raise ValueError(f"Exchange {exchange_name} not supported")
        
        try:
            if exchange_name == 'binance':
                exchange = ccxt.binance({
                    'apiKey': config.get('api_key', ''),
                    'secret': config.get('api_secret', ''),
                    'sandbox': config.get('sandbox', True),  # Start with sandbox
                    'enableRateLimit': True,
                })
            elif exchange_name == 'coinbase':
                exchange = ccxt.coinbasepro({
                    'apiKey': config.get('api_key', ''),
                    'secret': config.get('api_secret', ''),
                    'passphrase': config.get('passphrase', ''),
                    'sandbox': config.get('sandbox', True),
                    'enableRateLimit': True,
                })
            elif exchange_name == 'kraken':
                exchange = ccxt.kraken({
                    'apiKey': config.get('api_key', ''),
                    'secret': config.get('api_secret', ''),
                    'enableRateLimit': True,
                })
            
            # Test connection
            exchange.load_markets()
            self.exchanges[exchange_name] = exchange
            logger.info(f"Successfully connected to {exchange_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to {exchange_name}: {e}")
            raise
    
    def add_public_exchange(self, exchange_name: str):
        """Add exchange for public data only (no API keys needed)"""
        try:
            if exchange_name == 'binance':
                exchange = ccxt.binance({'enableRateLimit': True})
            elif exchange_name == 'coinbase':
                exchange = ccxt.coinbasepro({'enableRateLimit': True})
            elif exchange_name == 'kraken':
                exchange = ccxt.kraken({'enableRateLimit': True})
            else:
                raise ValueError(f"Exchange {exchange_name} not supported")
            
            exchange.load_markets()
            self.exchanges[exchange_name] = exchange
            logger.info(f"Connected to {exchange_name} (public data only)")
            
        except Exception as e:
            logger.error(f"Failed to connect to {exchange_name}: {e}")
            raise
    
    def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100, 
                  exchange_name: str = None) -> pd.DataFrame:
        """Get OHLCV data for a symbol"""
        if not self.exchanges:
            raise ValueError("No exchanges configured")
        
        # Use first available exchange if none specified
        if exchange_name is None:
            exchange_name = list(self.exchanges.keys())[0]
        
        if exchange_name not in self.exchanges:
            raise ValueError(f"Exchange {exchange_name} not configured")
        
        exchange = self.exchanges[exchange_name]
        
        try:
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add metadata
            df.attrs['symbol'] = symbol
            df.attrs['timeframe'] = timeframe
            df.attrs['exchange'] = exchange_name
            
            logger.info(f"Fetched {len(df)} candles for {symbol} from {exchange_name}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_multiple_symbols(self, symbols: List[str], timeframe: str = '1h', 
                           limit: int = 100, exchange_name: str = None) -> Dict[str, pd.DataFrame]:
        """Get OHLCV data for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            try:
                df = self.get_ohlcv(symbol, timeframe, limit, exchange_name)
                if not df.empty:
                    results[symbol] = df
                    time.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                continue
        
        logger.info(f"Successfully fetched data for {len(results)} symbols")
        return results
    
    def get_orderbook(self, symbol: str, exchange_name: str = None) -> Dict:
        """Get current orderbook for a symbol"""
        if exchange_name is None:
            exchange_name = list(self.exchanges.keys())[0]
        
        if exchange_name not in self.exchanges:
            raise ValueError(f"Exchange {exchange_name} not configured")
        
        exchange = self.exchanges[exchange_name]
        
        try:
            orderbook = exchange.fetch_order_book(symbol)
            return {
                'symbol': symbol,
                'exchange': exchange_name,
                'timestamp': datetime.now(),
                'bids': orderbook['bids'][:10],  # Top 10 bids
                'asks': orderbook['asks'][:10],  # Top 10 asks
                'spread': orderbook['asks'][0][0] - orderbook['bids'][0][0] if orderbook['asks'] and orderbook['bids'] else 0
            }
        except Exception as e:
            logger.error(f"Failed to fetch orderbook for {symbol}: {e}")
            return {}
    
    def get_ticker(self, symbol: str, exchange_name: str = None) -> Dict:
        """Get current ticker information"""
        if exchange_name is None:
            exchange_name = list(self.exchanges.keys())[0]
        
        exchange = self.exchanges[exchange_name]
        
        try:
            ticker = exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'exchange': exchange_name,
                'last': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'change': ticker['change'],
                'percentage': ticker['percentage'],
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            return {}
    
    def get_market_overview(self, symbols: List[str] = None, exchange_name: str = None) -> pd.DataFrame:
        """Get market overview for multiple symbols"""
        if symbols is None:
            symbols = self.default_symbols
        
        tickers = []
        for symbol in symbols:
            ticker = self.get_ticker(symbol, exchange_name)
            if ticker:
                tickers.append(ticker)
                time.sleep(0.1)  # Rate limiting
        
        if tickers:
            df = pd.DataFrame(tickers)
            df.set_index('symbol', inplace=True)
            return df
        else:
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic technical indicators"""
        if df.empty:
            return df
        
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        logger.info(f"Added technical indicators to {df.attrs.get('symbol', 'unknown')} data")
        return df
    
    def save_data(self, data: pd.DataFrame, filename: str, directory: str = "data/raw"):
        """Save market data to file"""
        # Create directory if it doesn't exist
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        filepath = os.path.join(directory, filename)
        data.to_csv(filepath)
        logger.info(f"Saved data to {filepath}")
    
    def load_data(self, filename: str, directory: str = "data/raw") -> pd.DataFrame:
        """Load market data from file"""
        filepath = os.path.join(directory, filename)
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            logger.info(f"Loaded data from {filepath}")
            return df
        else:
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()
    
    def get_available_symbols(self, exchange_name: str = None) -> List[str]:
        """Get list of available trading symbols"""
        if exchange_name is None:
            exchange_name = list(self.exchanges.keys())[0]
        
        exchange = self.exchanges[exchange_name]
        markets = exchange.load_markets()
        
        # Filter for USDT pairs (most liquid)
        usdt_pairs = [symbol for symbol in markets.keys() if '/USDT' in symbol]
        
        logger.info(f"Found {len(usdt_pairs)} USDT pairs on {exchange_name}")
        return sorted(usdt_pairs)


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize data manager
    dm = MarketDataManager()
    
    try:
        # Add public exchange (no API keys needed)
        dm.add_public_exchange('binance')
        
        # Test basic functionality
        print("Testing market data fetching...")
        
        # Get BTC data
        btc_data = dm.get_ohlcv('BTC/USDT', '1h', 100)
        if not btc_data.empty:
            print(f"BTC/USDT data: {len(btc_data)} candles")
            print(f"Latest price: ${btc_data['close'].iloc[-1]:,.2f}")
            
            # Add technical indicators
            btc_with_indicators = dm.calculate_technical_indicators(btc_data.copy())
            print(f"RSI: {btc_with_indicators['rsi'].iloc[-1]:.2f}")
            print(f"MACD: {btc_with_indicators['macd'].iloc[-1]:.6f}")
        
        # Get market overview
        overview = dm.get_market_overview()
        if not overview.empty:
            print(f"\nMarket Overview ({len(overview)} symbols):")
            print(overview[['last', 'change', 'percentage']].head())
        
        # Get orderbook
        orderbook = dm.get_orderbook('BTC/USDT')
        if orderbook:
            print(f"\nBTC/USDT Spread: ${orderbook['spread']:.2f}")
        
        print("\n✅ Market data system working correctly!")
        
    except Exception as e:
        print(f"❌ Error testing market data: {e}")
"""
Market data ingestion from various crypto exchanges
"""
import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import time
from datetime import datetime, timedelta
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class MarketDataManager:
    """Handles market data from multiple exchanges"""
    
    def __init__(self, config: Dict = None):
        self.exchanges = {}
        self.supported_exchanges = ['binance', 'coinbase', 'kraken']
        self.data_cache = {}
        self.config = config or {}
        
        # Default symbols to track (Kraken format)
        self.default_symbols = [
            'BTC/USD', 'ETH/USD', 'ADA/USD', 'SOL/USD', 
            'XRP/USD', 'DOT/USD', 'AVAX/USD', 'LINK/USD'
        ]
        
        logger.info("MarketDataManager initialized")
    
    def add_exchange(self, exchange_name: str, config: Dict = None):
        """Add an exchange connection"""
        if exchange_name not in self.supported_exchanges:
            raise ValueError(f"Exchange {exchange_name} not supported")
        
        try:
            if exchange_name == 'binance':
                exchange = ccxt.binance({
                    'apiKey': config.get('api_key', ''),
                    'secret': config.get('api_secret', ''),
                    'sandbox': config.get('sandbox', True),  # Start with sandbox
                    'enableRateLimit': True,
                })
            elif exchange_name == 'coinbase':
                exchange = ccxt.coinbasepro({
                    'apiKey': config.get('api_key', ''),
                    'secret': config.get('api_secret', ''),
                    'passphrase': config.get('passphrase', ''),
                    'sandbox': config.get('sandbox', True),
                    'enableRateLimit': True,
                })
            elif exchange_name == 'kraken':
                exchange = ccxt.kraken({
                    'apiKey': config.get('api_key', ''),
                    'secret': config.get('api_secret', ''),
                    'enableRateLimit': True,
                })
            
            # Test connection
            exchange.load_markets()
            self.exchanges[exchange_name] = exchange
            logger.info(f"Successfully connected to {exchange_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to {exchange_name}: {e}")
            raise
    
    def add_public_exchange(self, exchange_name: str):
        """Add exchange for public data only (no API keys needed)"""
        try:
            if exchange_name == 'binance':
                exchange = ccxt.binance({'enableRateLimit': True})
            elif exchange_name == 'coinbase':
                exchange = ccxt.coinbasepro({'enableRateLimit': True})
            elif exchange_name == 'kraken':
                exchange = ccxt.kraken({
                    'enableRateLimit': True,
                    'rateLimit': 1000,  # Be extra careful with rate limits
                })
            else:
                raise ValueError(f"Exchange {exchange_name} not supported")
            
            exchange.load_markets()
            self.exchanges[exchange_name] = exchange
            logger.info(f"Connected to {exchange_name} (public data only)")
            
        except Exception as e:
            logger.error(f"Failed to connect to {exchange_name}: {e}")
            # Try alternative exchanges
            if exchange_name == 'binance':
                logger.info("Binance failed, trying Kraken...")
                self.add_public_exchange('kraken')
            elif exchange_name == 'kraken':
                logger.info("Kraken failed, trying Coinbase...")
                self.add_public_exchange('coinbase')
            else:
                raise
    
    def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100, 
                  exchange_name: str = None) -> pd.DataFrame:
        """Get OHLCV data for a symbol"""
        if not self.exchanges:
            raise ValueError("No exchanges configured")
        
        # Use first available exchange if none specified
        if exchange_name is None:
            exchange_name = list(self.exchanges.keys())[0]
        
        if exchange_name not in self.exchanges:
            raise ValueError(f"Exchange {exchange_name} not configured")
        
        exchange = self.exchanges[exchange_name]
        
        try:
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add metadata
            df.attrs['symbol'] = symbol
            df.attrs['timeframe'] = timeframe
            df.attrs['exchange'] = exchange_name
            
            logger.info(f"Fetched {len(df)} candles for {symbol} from {exchange_name}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_multiple_symbols(self, symbols: List[str], timeframe: str = '1h', 
                           limit: int = 100, exchange_name: str = None) -> Dict[str, pd.DataFrame]:
        """Get OHLCV data for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            try:
                df = self.get_ohlcv(symbol, timeframe, limit, exchange_name)
                if not df.empty:
                    results[symbol] = df
                    time.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                continue
        
        logger.info(f"Successfully fetched data for {len(results)} symbols")
        return results
    
    def get_orderbook(self, symbol: str, exchange_name: str = None) -> Dict:
        """Get current orderbook for a symbol"""
        if exchange_name is None:
            exchange_name = list(self.exchanges.keys())[0]
        
        if exchange_name not in self.exchanges:
            raise ValueError(f"Exchange {exchange_name} not configured")
        
        exchange = self.exchanges[exchange_name]
        
        try:
            orderbook = exchange.fetch_order_book(symbol)
            return {
                'symbol': symbol,
                'exchange': exchange_name,
                'timestamp': datetime.now(),
                'bids': orderbook['bids'][:10],  # Top 10 bids
                'asks': orderbook['asks'][:10],  # Top 10 asks
                'spread': orderbook['asks'][0][0] - orderbook['bids'][0][0] if orderbook['asks'] and orderbook['bids'] else 0
            }
        except Exception as e:
            logger.error(f"Failed to fetch orderbook for {symbol}: {e}")
            return {}
    
    def get_ticker(self, symbol: str, exchange_name: str = None) -> Dict:
        """Get current ticker information"""
        if exchange_name is None:
            exchange_name = list(self.exchanges.keys())[0]
        
        exchange = self.exchanges[exchange_name]
        
        try:
            ticker = exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'exchange': exchange_name,
                'last': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'change': ticker['change'],
                'percentage': ticker['percentage'],
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            return {}
    
    def get_market_overview(self, symbols: List[str] = None, exchange_name: str = None) -> pd.DataFrame:
        """Get market overview for multiple symbols"""
        if symbols is None:
            symbols = self.default_symbols
        
        tickers = []
        for symbol in symbols:
            ticker = self.get_ticker(symbol, exchange_name)
            if ticker:
                tickers.append(ticker)
                time.sleep(0.1)  # Rate limiting
        
        if tickers:
            df = pd.DataFrame(tickers)
            df.set_index('symbol', inplace=True)
            return df
        else:
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic technical indicators"""
        if df.empty:
            return df
        
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        logger.info(f"Added technical indicators to {df.attrs.get('symbol', 'unknown')} data")
        return df
    
    def save_data(self, data: pd.DataFrame, filename: str, directory: str = "data/raw"):
        """Save market data to file"""
        # Create directory if it doesn't exist
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        filepath = os.path.join(directory, filename)
        data.to_csv(filepath)
        logger.info(f"Saved data to {filepath}")
    
    def load_data(self, filename: str, directory: str = "data/raw") -> pd.DataFrame:
        """Load market data from file"""
        filepath = os.path.join(directory, filename)
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            logger.info(f"Loaded data from {filepath}")
            return df
        else:
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()
    
    def get_available_symbols(self, exchange_name: str = None) -> List[str]:
        """Get list of available trading symbols"""
        if exchange_name is None:
            exchange_name = list(self.exchanges.keys())[0]
        
        exchange = self.exchanges[exchange_name]
        markets = exchange.load_markets()
        
        # Filter for USDT pairs (most liquid)
        usdt_pairs = [symbol for symbol in markets.keys() if '/USDT' in symbol]
        
        logger.info(f"Found {len(usdt_pairs)} USDT pairs on {exchange_name}")
        return sorted(usdt_pairs)


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize data manager
    dm = MarketDataManager()
    
    try:
        # Add public exchange (no API keys needed)
        dm.add_public_exchange('binance')
        
        # Test basic functionality
        print("Testing market data fetching...")
        
        # Get BTC data
        btc_data = dm.get_ohlcv('BTC/USDT', '1h', 100)
        if not btc_data.empty:
            print(f"BTC/USDT data: {len(btc_data)} candles")
            print(f"Latest price: ${btc_data['close'].iloc[-1]:,.2f}")
            
            # Add technical indicators
            btc_with_indicators = dm.calculate_technical_indicators(btc_data.copy())
            print(f"RSI: {btc_with_indicators['rsi'].iloc[-1]:.2f}")
            print(f"MACD: {btc_with_indicators['macd'].iloc[-1]:.6f}")
        
        # Get market overview
        overview = dm.get_market_overview()
        if not overview.empty:
            print(f"\nMarket Overview ({len(overview)} symbols):")
            print(overview[['last', 'change', 'percentage']].head())
        
        # Get orderbook
        orderbook = dm.get_orderbook('BTC/USDT')
        if orderbook:
            print(f"\nBTC/USDT Spread: ${orderbook['spread']:.2f}")
        
        print("\n✅ Market data system working correctly!")
        
    except Exception as e:
        print(f"❌ Error testing market data: {e}")
"""
Market data ingestion from various crypto exchanges
"""
import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import time
from datetime import datetime, timedelta
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class MarketDataManager:
    """Handles market data from multiple exchanges"""
    
    def __init__(self, config: Dict = None):
        self.exchanges = {}
        self.supported_exchanges = ['binance', 'coinbase', 'kraken']
        self.data_cache = {}
        self.config = config or {}
        
        # Default symbols to track (Kraken format)
        self.default_symbols = [
            'BTC/USD', 'ETH/USD', 'ADA/USD', 'SOL/USD', 
            'XRP/USD', 'DOT/USD', 'AVAX/USD', 'LINK/USD'
        ]
        
        logger.info("MarketDataManager initialized")
    
    def add_exchange(self, exchange_name: str, config: Dict = None):
        """Add an exchange connection"""
        if exchange_name not in self.supported_exchanges:
            raise ValueError(f"Exchange {exchange_name} not supported")
        
        try:
            if exchange_name == 'binance':
                exchange = ccxt.binance({
                    'apiKey': config.get('api_key', ''),
                    'secret': config.get('api_secret', ''),
                    'sandbox': config.get('sandbox', True),  # Start with sandbox
                    'enableRateLimit': True,
                })
            elif exchange_name == 'coinbase':
                exchange = ccxt.coinbasepro({
                    'apiKey': config.get('api_key', ''),
                    'secret': config.get('api_secret', ''),
                    'passphrase': config.get('passphrase', ''),
                    'sandbox': config.get('sandbox', True),
                    'enableRateLimit': True,
                })
            elif exchange_name == 'kraken':
                exchange = ccxt.kraken({
                    'apiKey': config.get('api_key', ''),
                    'secret': config.get('api_secret', ''),
                    'enableRateLimit': True,
                })
            
            # Test connection
            exchange.load_markets()
            self.exchanges[exchange_name] = exchange
            logger.info(f"Successfully connected to {exchange_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to {exchange_name}: {e}")
            raise
    
    def add_public_exchange(self, exchange_name: str):
        """Add exchange for public data only (no API keys needed)"""
        try:
            if exchange_name == 'binance':
                exchange = ccxt.binance({'enableRateLimit': True})
            elif exchange_name == 'coinbase':
                exchange = ccxt.coinbasepro({'enableRateLimit': True})
            elif exchange_name == 'kraken':
                exchange = ccxt.kraken({
                    'enableRateLimit': True,
                    'rateLimit': 1000,  # Be extra careful with rate limits
                })
            else:
                raise ValueError(f"Exchange {exchange_name} not supported")
            
            exchange.load_markets()
            self.exchanges[exchange_name] = exchange
            logger.info(f"Connected to {exchange_name} (public data only)")
            
        except Exception as e:
            logger.error(f"Failed to connect to {exchange_name}: {e}")
            # Try alternative exchanges
            if exchange_name == 'binance':
                logger.info("Binance failed, trying Kraken...")
                self.add_public_exchange('kraken')
            elif exchange_name == 'kraken':
                logger.info("Kraken failed, trying Coinbase...")
                self.add_public_exchange('coinbase')
            else:
                raise
    
    def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100, 
                  exchange_name: str = None) -> pd.DataFrame:
        """Get OHLCV data for a symbol"""
        if not self.exchanges:
            raise ValueError("No exchanges configured")
        
        # Use first available exchange if none specified
        if exchange_name is None:
            exchange_name = list(self.exchanges.keys())[0]
        
        if exchange_name not in self.exchanges:
            raise ValueError(f"Exchange {exchange_name} not configured")
        
        exchange = self.exchanges[exchange_name]
        
        try:
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add metadata
            df.attrs['symbol'] = symbol
            df.attrs['timeframe'] = timeframe
            df.attrs['exchange'] = exchange_name
            
            logger.info(f"Fetched {len(df)} candles for {symbol} from {exchange_name}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_multiple_symbols(self, symbols: List[str], timeframe: str = '1h', 
                           limit: int = 100, exchange_name: str = None) -> Dict[str, pd.DataFrame]:
        """Get OHLCV data for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            try:
                df = self.get_ohlcv(symbol, timeframe, limit, exchange_name)
                if not df.empty:
                    results[symbol] = df
                    time.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                continue
        
        logger.info(f"Successfully fetched data for {len(results)} symbols")
        return results
    
    def get_orderbook(self, symbol: str, exchange_name: str = None) -> Dict:
        """Get current orderbook for a symbol"""
        if exchange_name is None:
            exchange_name = list(self.exchanges.keys())[0]
        
        if exchange_name not in self.exchanges:
            raise ValueError(f"Exchange {exchange_name} not configured")
        
        exchange = self.exchanges[exchange_name]
        
        try:
            orderbook = exchange.fetch_order_book(symbol)
            return {
                'symbol': symbol,
                'exchange': exchange_name,
                'timestamp': datetime.now(),
                'bids': orderbook['bids'][:10],  # Top 10 bids
                'asks': orderbook['asks'][:10],  # Top 10 asks
                'spread': orderbook['asks'][0][0] - orderbook['bids'][0][0] if orderbook['asks'] and orderbook['bids'] else 0
            }
        except Exception as e:
            logger.error(f"Failed to fetch orderbook for {symbol}: {e}")
            return {}
    
    def get_ticker(self, symbol: str, exchange_name: str = None) -> Dict:
        """Get current ticker information"""
        if exchange_name is None:
            exchange_name = list(self.exchanges.keys())[0]
        
        exchange = self.exchanges[exchange_name]
        
        try:
            ticker = exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'exchange': exchange_name,
                'last': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume'],
                'change': ticker['change'],
                'percentage': ticker['percentage'],
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            return {}
    
    def get_market_overview(self, symbols: List[str] = None, exchange_name: str = None) -> pd.DataFrame:
        """Get market overview for multiple symbols"""
        if symbols is None:
            symbols = self.default_symbols
        
        tickers = []
        for symbol in symbols:
            ticker = self.get_ticker(symbol, exchange_name)
            if ticker:
                tickers.append(ticker)
                time.sleep(0.1)  # Rate limiting
        
        if tickers:
            df = pd.DataFrame(tickers)
            df.set_index('symbol', inplace=True)
            return df
        else:
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic technical indicators"""
        if df.empty:
            return df
        
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        logger.info(f"Added technical indicators to {df.attrs.get('symbol', 'unknown')} data")
        return df
    
    def save_data(self, data: pd.DataFrame, filename: str, directory: str = "data/raw"):
        """Save market data to file"""
        # Create directory if it doesn't exist
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        filepath = os.path.join(directory, filename)
        data.to_csv(filepath)
        logger.info(f"Saved data to {filepath}")
    
    def load_data(self, filename: str, directory: str = "data/raw") -> pd.DataFrame:
        """Load market data from file"""
        filepath = os.path.join(directory, filename)
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            logger.info(f"Loaded data from {filepath}")
            return df
        else:
            logger.warning(f"File not found: {filepath}")
            return pd.DataFrame()
    
    def connect_best_exchange(self):
        """Connect to the best available exchange automatically"""
        exchanges_to_try = ['kraken', 'coinbase', 'binance']
        
        for exchange_name in exchanges_to_try:
            try:
                logger.info(f"Trying to connect to {exchange_name}...")
                self.add_public_exchange(exchange_name)
                logger.info(f"✅ Successfully connected to {exchange_name}")
                return exchange_name
            except Exception as e:
                logger.warning(f"❌ {exchange_name} failed: {e}")
                continue
        
        raise Exception("Could not connect to any exchange")
    
    def get_available_symbols(self, exchange_name: str = None) -> List[str]:
        """Get list of available trading symbols"""
        if exchange_name is None:
            exchange_name = list(self.exchanges.keys())[0]
        
        exchange = self.exchanges[exchange_name]
        markets = exchange.load_markets()
        
        # Filter for USD pairs (most common on Kraken)
        usd_pairs = [symbol for symbol in markets.keys() if '/USD' in symbol]
        
        logger.info(f"Found {len(usd_pairs)} USD pairs on {exchange_name}")
        return sorted(usd_pairs)


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize data manager
    dm = MarketDataManager()
    
    try:
        # Add Kraken exchange (works globally)
        dm.connect_best_exchange()
        
        # Test basic functionality
        print("Testing market data fetching...")
        
        # Get BTC data
        btc_data = dm.get_ohlcv('BTC/USD', '1h', 100)
        if not btc_data.empty:
            print(f"BTC/USDT data: {len(btc_data)} candles")
            print(f"Latest price: ${btc_data['close'].iloc[-1]:,.2f}")
            
            # Add technical indicators
            btc_with_indicators = dm.calculate_technical_indicators(btc_data.copy())
            print(f"RSI: {btc_with_indicators['rsi'].iloc[-1]:.2f}")
            print(f"MACD: {btc_with_indicators['macd'].iloc[-1]:.6f}")
        
        # Get market overview
        overview = dm.get_market_overview()
        if not overview.empty:
            print(f"\nMarket Overview ({len(overview)} symbols):")
            print(overview[['last', 'change', 'percentage']].head())
        
        # Get orderbook
        orderbook = dm.get_orderbook('BTC/USD')
        if orderbook:
            print(f"\nBTC/USD Spread: ${orderbook['spread']:.2f}")
        
        print("\n✅ Market data system working correctly!")
        
    except Exception as e:
        print(f"❌ Error testing market data: {e}")
