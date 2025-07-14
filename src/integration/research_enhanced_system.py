"""
Research Enhanced Trading System
Integrates advanced ML research findings with practical trading implementation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import existing modules if available
try:
    from data.market_data import MarketDataManager
    from models.base_model import BaseModel
    from models.linear_model import LinearModel
    EXISTING_MODULES_AVAILABLE = True
except ImportError:
    EXISTING_MODULES_AVAILABLE = False
    print("Note: Existing modules not found. Using standalone implementation.")

# Import our new components
try:
    from models.transformer_model import FinancialTransformer
    from strategies.ensemble_strategy import EnsembleIntegrationHelper
    from features.sentiment_features import AdvancedSentimentAnalyzer
    NEW_COMPONENTS_AVAILABLE = True
except ImportError as e:
    NEW_COMPONENTS_AVAILABLE = False
    print(f"Warning: New components not available: {e}")

class ResearchEnhancedTradingSystem:
    """
    Main trading system that integrates research findings with practical implementation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the research enhanced trading system
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()
        
        # Initialize components
        self.ensemble_helper = None
        self.transformer_model = None
        self.sentiment_analyzer = None
        self.market_data_manager = None
        
        # Performance tracking
        self.performance_history = []
        self.current_positions = {}
        self.portfolio_value = self.config.get('initial_capital', 10000)
        
        # Initialize all available components
        self._initialize_components()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'initial_capital': 10000,
            'risk_per_trade': 0.02,
            'max_positions': 3,
            'models': {
                'use_ensemble': True,
                'use_transformer': True,
                'use_sentiment': True
            },
            'trading': {
                'min_confidence': 0.6,
                'stop_loss': 0.05,
                'take_profit': 0.10
            },
            'data': {
                'symbols': ['BTC/USD', 'ETH/USD'],
                'timeframe': '1h',
                'lookback_days': 30
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('ResearchEnhancedSystem')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_components(self):
        """Initialize all available components"""
        try:
            # Initialize ensemble helper
            if NEW_COMPONENTS_AVAILABLE and self.config['models']['use_ensemble']:
                self.ensemble_helper = EnsembleIntegrationHelper(self.config)
                self.logger.info("✅ Ensemble helper initialized")
            
            # Initialize transformer model
            if NEW_COMPONENTS_AVAILABLE and self.config['models']['use_transformer']:
                self.transformer_model = FinancialTransformer(
                    input_dim=20,  # Will adjust based on actual features
                    model_dim=128,
                    num_heads=8,
                    num_layers=4
                )
                self.logger.info("✅ Transformer model initialized")
            
            # Initialize sentiment analyzer
            if NEW_COMPONENTS_AVAILABLE and self.config['models']['use_sentiment']:
                self.sentiment_analyzer = AdvancedSentimentAnalyzer()
                self.logger.info("✅ Sentiment analyzer initialized")
            
            # Initialize market data manager if available
            if EXISTING_MODULES_AVAILABLE:
                self.market_data_manager = MarketDataManager()
                self.logger.info("✅ Market data manager initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
    
    def get_market_data(self, symbol: str, timeframe: str = '1h', 
                       limit: int = 1000) -> pd.DataFrame:
        """
        Get market data for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USD')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if self.market_data_manager:
                # Use existing market data manager
                return self.market_data_manager.get_data(symbol, timeframe, limit)
            else:
                # Generate sample data for testing
                return self._generate_sample_data(symbol, limit)
                
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return self._generate_sample_data(symbol, limit)
    
    def _generate_sample_data(self, symbol: str, limit: int) -> pd.DataFrame:
        """Generate sample market data for testing"""
        dates = pd.date_range(
            start=datetime.now() - timedelta(hours=limit),
            periods=limit,
            freq='H'
        )
        
        # Generate realistic price data
        np.random.seed(42)
        price = 50000 if 'BTC' in symbol else 3000
        returns = np.random.normal(0, 0.02, limit)
        prices = [price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, limit)
        })
        
        return data.set_index('timestamp')
    
    def analyze_market_sentiment(self, symbol: str) -> Dict[str, float]:
        """
        Analyze market sentiment for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with sentiment metrics
        """
        try:
            if self.sentiment_analyzer:
                # Generate sample news text for testing
                sample_news = [
                    f"{symbol} showing strong bullish momentum with increased volume",
                    f"Technical analysis suggests {symbol} may continue upward trend",
                    f"Market sentiment for {symbol} remains positive despite volatility"
                ]
                
                sentiment_scores = []
                for news in sample_news:
                    sentiment = self.sentiment_analyzer.analyze_sentiment(news)
                    sentiment_scores.append(sentiment)
                
                # Aggregate sentiment
                avg_sentiment = np.mean(sentiment_scores)
                sentiment_strength = abs(avg_sentiment)
                
                return {
                    'sentiment_score': avg_sentiment,
                    'sentiment_strength': sentiment_strength,
                    'bullish_probability': (avg_sentiment + 1) / 2
                }
            else:
                # Return neutral sentiment if analyzer not available
                return {
                    'sentiment_score': 0.0,
                    'sentiment_strength': 0.0,
                    'bullish_probability': 0.5
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return {
                'sentiment_score': 0.0,
                'sentiment_strength': 0.0,
                'bullish_probability': 0.5
            }
    
    def generate_trading_signal(self, symbol: str) -> Dict[str, Any]:
        """
        Generate comprehensive trading signal for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with trading signal and metadata
        """
        try:
            # Get market data
            market_data = self.get_market_data(symbol)
            
            if len(market_data) < 50:
                return self._no_signal_response("Insufficient data")
            
            # Get sentiment analysis
            sentiment = self.analyze_market_sentiment(symbol)
            
            # Prepare features for ensemble
            if self.ensemble_helper:
                features = self.ensemble_helper.prepare_features(market_data)
                ensemble_prediction = self.ensemble_helper.predict_ensemble(features.tail(1))
            else:
                ensemble_prediction = {'ensemble_signal': 0, 'confidence': 0.5}
            
            # Transformer prediction (placeholder for now)
            transformer_signal = 0
            if self.transformer_model:
                # Would implement transformer prediction here
                transformer_signal = 0  # Placeholder
            
            # Combine all signals
            ensemble_weight = 0.5
            sentiment_weight = 0.3
            transformer_weight = 0.2
            
            combined_signal = (
                ensemble_prediction['ensemble_signal'] * ensemble_weight +
                sentiment['bullish_probability'] * sentiment_weight +
                transformer_signal * transformer_weight
            )
            
            # Generate final recommendation
            if combined_signal > 0.7 and ensemble_prediction['confidence'] > self.config['trading']['min_confidence']:
                action = 'BUY'
                confidence = ensemble_prediction['confidence']
            elif combined_signal < 0.3:
                action = 'SELL'
                confidence = 1 - ensemble_prediction['confidence']
            else:
                action = 'HOLD'
                confidence = 0.5
            
            return {
                'symbol': symbol,
                'action': action,
                'confidence': confidence,
                'combined_signal': combined_signal,
                'components': {
                    'ensemble': ensemble_prediction,
                    'sentiment': sentiment,
                    'transformer': transformer_signal
                },
                'current_price': market_data['close'].iloc[-1],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating trading signal: {e}")
            return self._no_signal_response(f"Error: {e}")
    
    def _no_signal_response(self, reason: str) -> Dict[str, Any]:
        """Generate a no-signal response"""
        return {
            'symbol': '',
            'action': 'HOLD',
            'confidence': 0.0,
            'combined_signal': 0.5,
            'reason': reason,
            'timestamp': datetime.now()
        }
    
    def execute_strategy(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Execute the complete trading strategy
        
        Args:
            symbols: List of symbols to trade (uses config default if None)
            
        Returns:
            Dictionary with execution results
        """
        if symbols is None:
            symbols = self.config['data']['symbols']
        
        results = {
            'timestamp': datetime.now(),
            'signals': {},
            'portfolio_value': self.portfolio_value,
            'positions': self.current_positions.copy(),
            'actions_taken': []
        }
        
        try:
            for symbol in symbols:
                # Generate signal
                signal = self.generate_trading_signal(symbol)
                results['signals'][symbol] = signal
                
                # Execute trade if signal is strong enough
                if signal['confidence'] > self.config['trading']['min_confidence']:
                    action_result = self._execute_trade(signal)
                    if action_result:
                        results['actions_taken'].append(action_result)
            
            # Update portfolio value
            results['portfolio_value'] = self._calculate_portfolio_value()
            
            # Store performance
            self.performance_history.append({
                'timestamp': datetime.now(),
                'portfolio_value': results['portfolio_value'],
                'num_positions': len(self.current_positions),
                'signals_generated': len(results['signals'])
            })
            
            self.logger.info(f"Strategy executed: {len(results['actions_taken'])} actions taken")
            
        except Exception as e:
            self.logger.error(f"Error executing strategy: {e}")
            results['error'] = str(e)
        
        return results
    
    def _execute_trade(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a trade based on the signal"""
        try:
            symbol = signal['symbol']
            action = signal['action']
            current_price = signal['current_price']
            
            if action == 'BUY' and symbol not in self.current_positions:
                if len(self.current_positions) < self.config['max_positions']:
                    # Calculate position size based on risk
                    risk_amount = self.portfolio_value * self.config['risk_per_trade']
                    position_size = risk_amount / current_price
                    
                    self.current_positions[symbol] = {
                        'size': position_size,
                        'entry_price': current_price,
                        'entry_time': datetime.now(),
                        'stop_loss': current_price * (1 - self.config['trading']['stop_loss']),
                        'take_profit': current_price * (1 + self.config['trading']['take_profit'])
                    }
                    
                    return {
                        'action': 'BUY',
                        'symbol': symbol,
                        'size': position_size,
                        'price': current_price,
                        'timestamp': datetime.now()
                    }
            
            elif action == 'SELL' and symbol in self.current_positions:
                position = self.current_positions.pop(symbol)
                pnl = (current_price - position['entry_price']) * position['size']
                
                return {
                    'action': 'SELL',
                    'symbol': symbol,
                    'size': position['size'],
                    'price': current_price,
                    'pnl': pnl,
                    'timestamp': datetime.now()
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return None
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        try:
            total_value = self.portfolio_value
            
            for symbol, position in self.current_positions.items():
                # Get current price (simplified)
                current_data = self.get_market_data(symbol, limit=1)
                if len(current_data) > 0:
                    current_price = current_data['close'].iloc[-1]
                    position_value = position['size'] * current_price
                    entry_value = position['size'] * position['entry_price']
                    total_value += (position_value - entry_value)
            
            return total_value
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {e}")
            return self.portfolio_value
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        try:
            if not self.performance_history:
                return {'error': 'No performance history available'}
            
            values = [p['portfolio_value'] for p in self.performance_history]
            initial_value = values[0] if values else self.config['initial_capital']
            current_value = values[-1] if values else initial_value
            
            total_return = (current_value - initial_value) / initial_value
            
            # Calculate other metrics
            returns = np.diff(values) / values[:-1] if len(values) > 1 else [0]
            
            metrics = {
                'total_return': total_return,
                'current_value': current_value,
                'initial_value': initial_value,
                'num_trades': len([p for p in self.performance_history if p.get('signals_generated', 0) > 0]),
                'current_positions': len(self.current_positions),
                'max_drawdown': self._calculate_max_drawdown(values),
                'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                'win_rate': 0.0  # Would need trade history to calculate
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown"""
        try:
            if len(values) < 2:
                return 0.0
            
            peak = values[0]
            max_drawdown = 0.0
            
            for value in values[1:]:
                if value > peak:
                    peak = value
                else:
                    drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown
            
        except Exception as e:
            return 0.0
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        try:
            if len(returns) < 2:
                return 0.0
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            return mean_return / std_return
            
        except Exception as e:
            return 0.0
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'timestamp': datetime.now(),
            'components': {
                'ensemble_helper': self.ensemble_helper is not None,
                'transformer_model': self.transformer_model is not None,
                'sentiment_analyzer': self.sentiment_analyzer is not None,
                'market_data_manager': self.market_data_manager is not None
            },
            'portfolio': {
                'value': self.portfolio_value,
                'positions': len(self.current_positions),
                'available_capital': self.portfolio_value - sum(
                    pos['size'] * pos['entry_price'] 
                    for pos in self.current_positions.values()
                )
            },
            'performance_history_length': len(self.performance_history),
            'config': self.config
        }

# Convenience function for easy system creation
def create_trading_system(config: Optional[Dict] = None) -> ResearchEnhancedTradingSystem:
    """
    Create and initialize a research enhanced trading system
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized trading system
    """
    return ResearchEnhancedTradingSystem(config)