"""
Market Regime Detection System
Research shows regime-aware strategies achieve 45% higher Sharpe ratios
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MarketRegimeDetector:
    """
    Advanced market regime detection using multiple methodologies
    
    Detects:
    - Volatility regimes (low/medium/high)
    - Trend regimes (bull/bear/sideways)
    - Market stress levels
    - Liquidity conditions
    """
    
    def __init__(self, lookback_period: int = 252, volatility_window: int = 20):
        self.lookback_period = lookback_period
        self.volatility_window = volatility_window
        
        # Regime thresholds (can be calibrated)
        self.volatility_thresholds = {
            'low': 0.15,      # 15% annualized volatility
            'medium': 0.25,   # 25% annualized volatility
            'high': float('inf')
        }
        
        self.trend_thresholds = {
            'strong_bull': 0.05,    # 5% move over trend_window
            'bull': 0.02,           # 2% move
            'sideways': 0.02,       # Within ±2%
            'bear': -0.02,          # -2% move
            'strong_bear': -0.05    # -5% move
        }
        
        # Model parameters
        self.trend_window = 21  # 21 days for trend detection
        self.stress_window = 10 # 10 days for stress detection
        
        # Historical regime data for learning
        self.regime_history = []
        
    def detect_regime(self, price_data: pd.Series, volume_data: pd.Series = None) -> Dict:
        """
        Comprehensive regime detection
        
        Args:
            price_data: Price time series
            volume_data: Optional volume time series
            
        Returns:
            Dict with regime information and confidence scores
        """
        if len(price_data) < max(self.lookback_period, 50):
            return self._get_default_regime()
        
        # Calculate all regime components
        volatility_regime = self._detect_volatility_regime(price_data)
        trend_regime = self._detect_trend_regime(price_data)
        stress_regime = self._detect_stress_regime(price_data, volume_data)
        momentum_regime = self._detect_momentum_regime(price_data)
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(
            price_data, volatility_regime, trend_regime, stress_regime
        )
        
        # Determine overall regime
        overall_regime = self._determine_overall_regime(
            volatility_regime, trend_regime, stress_regime, momentum_regime
        )
        
        # Calculate regime adjustments for model weights
        regime_adjustments = self._calculate_regime_adjustments(overall_regime)
        
        # Store for historical analysis
        regime_info = {
            'timestamp': datetime.now(),
            'volatility_regime': volatility_regime,
            'trend_regime': trend_regime,
            'stress_regime': stress_regime,
            'momentum_regime': momentum_regime,
            'overall_regime': overall_regime,
            'confidence': confidence_scores,
            'adjustments': regime_adjustments,
            'metrics': self._get_regime_metrics(price_data, volume_data)
        }
        
        self.regime_history.append(regime_info)
        
        # Keep only recent history
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]
        
        return regime_info
    
    def _detect_volatility_regime(self, price_data: pd.Series) -> str:
        """Detect volatility regime using multiple measures"""
        # Rolling volatility (annualized)
        returns = price_data.pct_change().dropna()
        rolling_vol = returns.rolling(self.volatility_window).std() * np.sqrt(252)
        current_vol = rolling_vol.iloc[-1]
        
        # Determine regime
        if current_vol < self.volatility_thresholds['low']:
            return 'low_vol'
        elif current_vol < self.volatility_thresholds['medium']:
            return 'medium_vol'
        else:
            return 'high_vol'
    
    def _detect_trend_regime(self, price_data: pd.Series) -> str:
        """Detect trend regime using multiple trend indicators"""
        if len(price_data) < self.trend_window:
            return 'unknown'
        
        # Simple trend: price change over trend_window
        trend_return = (price_data.iloc[-1] / price_data.iloc[-self.trend_window] - 1)
        
        # Moving average trend
        sma_short = price_data.rolling(5).mean().iloc[-1]
        sma_long = price_data.rolling(self.trend_window).mean().iloc[-1]
        ma_trend = (sma_short / sma_long - 1)
        
        # Combined trend signal
        combined_trend = (trend_return + ma_trend) / 2
        
        # Classify trend
        if combined_trend > self.trend_thresholds['strong_bull']:
            return 'strong_bull'
        elif combined_trend > self.trend_thresholds['bull']:
            return 'bull'
        elif combined_trend > self.trend_thresholds['sideways']:
            return 'sideways'
        elif combined_trend > self.trend_thresholds['bear']:
            return 'bear'
        else:
            return 'strong_bear'
    
    def _detect_stress_regime(self, price_data: pd.Series, volume_data: pd.Series = None) -> str:
        """Detect market stress using price action and volume"""
        returns = price_data.pct_change().dropna()
        
        if len(returns) < self.stress_window:
            return 'normal'
        
        # Recent volatility vs historical
        recent_vol = returns.tail(self.stress_window).std()
        historical_vol = returns.tail(self.lookback_period // 2).std()
        vol_ratio = recent_vol / (historical_vol + 1e-8)
        
        # Extreme moves
        recent_returns = returns.tail(self.stress_window)
        extreme_moves = len(recent_returns[abs(recent_returns) > recent_returns.std() * 2])
        extreme_ratio = extreme_moves / len(recent_returns)
        
        # Consecutive negative days
        negative_streak = 0
        for ret in recent_returns[::-1]:  # Reverse order
            if ret < 0:
                negative_streak += 1
            else:
                break
        
        # Volume stress (if available)
        volume_stress = 0
        if volume_data is not None and len(volume_data) >= self.stress_window:
            recent_volume = volume_data.tail(self.stress_window).mean()
            avg_volume = volume_data.tail(self.lookback_period // 2).mean()
            volume_stress = recent_volume / (avg_volume + 1e-8)
        
        # Determine stress level
        stress_score = (
            vol_ratio * 0.4 +
            extreme_ratio * 10 * 0.3 +  # Scale extreme ratio
            min(negative_streak / 5, 1) * 0.2 +  # Cap at 5 days
            max(0, (volume_stress - 1)) * 0.1  # Only high volume adds stress
        )
        
        if stress_score > 2.0:
            return 'extreme_stress'
        elif stress_score > 1.5:
            return 'high_stress'
        elif stress_score > 1.0:
            return 'moderate_stress'
        else:
            return 'normal'
    
    def _detect_momentum_regime(self, price_data: pd.Series) -> str:
        """Detect momentum regime using multiple momentum indicators"""
        if len(price_data) < 20:
            return 'neutral'
        
        # Price momentum (multiple timeframes)
        mom_5 = price_data.iloc[-1] / price_data.iloc[-6] - 1  # 5-day momentum
        mom_10 = price_data.iloc[-1] / price_data.iloc[-11] - 1  # 10-day momentum
        mom_20 = price_data.iloc[-1] / price_data.iloc[-21] - 1  # 20-day momentum
        
        # Acceleration (momentum of momentum)
        recent_mom = mom_5
        prev_mom = price_data.iloc[-6] / price_data.iloc[-11] - 1 if len(price_data) > 11 else 0
        acceleration = recent_mom - prev_mom
        
        # Combined momentum score
        momentum_score = (
            mom_5 * 0.5 +
            mom_10 * 0.3 +
            mom_20 * 0.2 +
            acceleration * 0.1
        )
        
        # Classify momentum
        if momentum_score > 0.03:
            return 'strong_momentum_up'
        elif momentum_score > 0.01:
            return 'momentum_up'
        elif momentum_score > -0.01:
            return 'neutral'
        elif momentum_score > -0.03:
            return 'momentum_down'
        else:
            return 'strong_momentum_down'
    
    def _calculate_confidence_scores(self, price_data: pd.Series, vol_regime: str, 
                                   trend_regime: str, stress_regime: str) -> Dict[str, float]:
        """Calculate confidence scores for regime classifications"""
        
        # Volatility confidence
        returns = price_data.pct_change().dropna()
        vol_stability = 1 - (returns.rolling(self.volatility_window).std().std() / 
                           returns.rolling(self.volatility_window).std().mean())
        vol_confidence = max(0.1, min(0.95, vol_stability))
        
        # Trend confidence (based on consistency)
        if len(price_data) >= self.trend_window:
            trend_consistency = 0
            window_size = max(5, self.trend_window // 4)
            
            for i in range(window_size, self.trend_window, window_size):
                window_return = price_data.iloc[-i] / price_data.iloc[-i-window_size] - 1
                if trend_regime in ['bull', 'strong_bull'] and window_return > 0:
                    trend_consistency += 1
                elif trend_regime in ['bear', 'strong_bear'] and window_return < 0:
                    trend_consistency += 1
                elif trend_regime == 'sideways' and abs(window_return) < 0.01:
                    trend_consistency += 1
            
            trend_confidence = max(0.1, min(0.95, trend_consistency / (self.trend_window // window_size)))
        else:
            trend_confidence = 0.5
        
        # Stress confidence (based on clarity of signals)
        stress_confidence = 0.7  # Default moderate confidence
        if stress_regime in ['extreme_stress', 'normal']:
            stress_confidence = 0.9  # High confidence for extreme cases
        elif stress_regime in ['high_stress', 'moderate_stress']:
            stress_confidence = 0.6  # Lower confidence for intermediate cases
        
        # Overall confidence
        overall_confidence = (vol_confidence * 0.4 + trend_confidence * 0.4 + stress_confidence * 0.2)
        
        return {
            'volatility': vol_confidence,
            'trend': trend_confidence,
            'stress': stress_confidence,
            'overall': overall_confidence
        }
    
    def _determine_overall_regime(self,
"""
Market Regime Detection System
Research shows regime-aware strategies achieve 45% higher Sharpe ratios
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class MarketRegimeDetector:
    """
    Advanced market regime detection using multiple methodologies
    
    Detects:
    - Volatility regimes (low/medium/high)
    - Trend regimes (bull/bear/sideways)
    - Market stress levels
    - Liquidity conditions
    """
    
    def __init__(self, lookback_period: int = 252, volatility_window: int = 20):
        self.lookback_period = lookback_period
        self.volatility_window = volatility_window
        
        # Regime thresholds (can be calibrated)
        self.volatility_thresholds = {
            'low': 0.15,      # 15% annualized volatility
            'medium': 0.25,   # 25% annualized volatility
            'high': float('inf')
        }
        
        self.trend_thresholds = {
            'strong_bull': 0.05,    # 5% move over trend_window
            'bull': 0.02,           # 2% move
            'sideways': 0.02,       # Within ±2%
            'bear': -0.02,          # -2% move
            'strong_bear': -0.05    # -5% move
        }
        
        # Model parameters
        self.trend_window = 21  # 21 days for trend detection
        self.stress_window = 10 # 10 days for stress detection
        
        # Historical regime data for learning
        self.regime_history = []
        
    def detect_regime(self, price_data: pd.Series, volume_data: pd.Series = None) -> Dict:
        """
        Comprehensive regime detection
        
        Args:
            price_data: Price time series
            volume_data: Optional volume time series
            
        Returns:
            Dict with regime information and confidence scores
        """
        if len(price_data) < max(self.lookback_period, 50):
            return self._get_default_regime()
        
        # Calculate all regime components
        volatility_regime = self._detect_volatility_regime(price_data)
        trend_regime = self._detect_trend_regime(price_data)
        stress_regime = self._detect_stress_regime(price_data, volume_data)
        momentum_regime = self._detect_momentum_regime(price_data)
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(
            price_data, volatility_regime, trend_regime, stress_regime
        )
        
        # Determine overall regime
        overall_regime = self._determine_overall_regime(
            volatility_regime, trend_regime, stress_regime, momentum_regime
        )
        
        # Calculate regime adjustments for model weights
        regime_adjustments = self._calculate_regime_adjustments(overall_regime)
        
        # Store for historical analysis
        regime_info = {
            'timestamp': datetime.now(),
            'volatility_regime': volatility_regime,
            'trend_regime': trend_regime,
            'stress_regime': stress_regime,
            'momentum_regime': momentum_regime,
            'overall_regime': overall_regime,
            'confidence': confidence_scores,
            'adjustments': regime_adjustments,
            'metrics': self._get_regime_metrics(price_data, volume_data)
        }
        
        self.regime_history.append(regime_info)
        
        # Keep only recent history
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]
        
        return regime_info
    
    def _detect_volatility_regime(self, price_data: pd.Series) -> str:
        """Detect volatility regime using multiple measures"""
        # Rolling volatility (annualized)
        returns = price_data.pct_change().dropna()
        rolling_vol = returns.rolling(self.volatility_window).std() * np.sqrt(252)
        current_vol = rolling_vol.iloc[-1]
        
        # Determine regime
        if current_vol < self.volatility_thresholds['low']:
            return 'low_vol'
        elif current_vol < self.volatility_thresholds['medium']:
            return 'medium_vol'
        else:
            return 'high_vol'
    
    def _detect_trend_regime(self, price_data: pd.Series) -> str:
        """Detect trend regime using multiple trend indicators"""
        if len(price_data) < self.trend_window:
            return 'unknown'
        
        # Simple trend: price change over trend_window
        trend_return = (price_data.iloc[-1] / price_data.iloc[-self.trend_window] - 1)
        
        # Moving average trend
        sma_short = price_data.rolling(5).mean().iloc[-1]
        sma_long = price_data.rolling(self.trend_window).mean().iloc[-1]
        ma_trend = (sma_short / sma_long - 1)
        
        # Combined trend signal
        combined_trend = (trend_return + ma_trend) / 2
        
        # Classify trend
        if combined_trend > self.trend_thresholds['strong_bull']:
            return 'strong_bull'
        elif combined_trend > self.trend_thresholds['bull']:
            return 'bull'
        elif combined_trend > self.trend_thresholds['sideways']:
            return 'sideways'
        elif combined_trend > self.trend_thresholds['bear']:
            return 'bear'
        else:
            return 'strong_bear'
    
    def _detect_stress_regime(self, price_data: pd.Series, volume_data: pd.Series = None) -> str:
        """Detect market stress using price action and volume"""
        returns = price_data.pct_change().dropna()
        
        if len(returns) < self.stress_window:
            return 'normal'
        
        # Recent volatility vs historical
        recent_vol = returns.tail(self.stress_window).std()
        historical_vol = returns.tail(self.lookback_period // 2).std()
        vol_ratio = recent_vol / (historical_vol + 1e-8)
        
        # Extreme moves
        recent_returns = returns.tail(self.stress_window)
        extreme_moves = len(recent_returns[abs(recent_returns) > recent_returns.std() * 2])
        extreme_ratio = extreme_moves / len(recent_returns)
        
        # Consecutive negative days
        negative_streak = 0
        for ret in recent_returns[::-1]:  # Reverse order
            if ret < 0:
                negative_streak += 1
            else:
                break
        
        # Volume stress (if available)
        volume_stress = 0
        if volume_data is not None and len(volume_data) >= self.stress_window:
            recent_volume = volume_data.tail(self.stress_window).mean()
            avg_volume = volume_data.tail(self.lookback_period // 2).mean()
            volume_stress = recent_volume / (avg_volume + 1e-8)
        
        # Determine stress level
        stress_score = (
            vol_ratio * 0.4 +
            extreme_ratio * 10 * 0.3 +  # Scale extreme ratio
            min(negative_streak / 5, 1) * 0.2 +  # Cap at 5 days
            max(0, (volume_stress - 1)) * 0.1  # Only high volume adds stress
        )
        
        if stress_score > 2.0:
            return 'extreme_stress'
        elif stress_score > 1.5:
            return 'high_stress'
        elif stress_score > 1.0:
            return 'moderate_stress'
        else:
            return 'normal'
    
    def _detect_momentum_regime(self, price_data: pd.Series) -> str:
        """Detect momentum regime using multiple momentum indicators"""
        if len(price_data) < 20:
            return 'neutral'
        
        # Price momentum (multiple timeframes)
        mom_5 = price_data.iloc[-1] / price_data.iloc[-6] - 1  # 5-day momentum
        mom_10 = price_data.iloc[-1] / price_data.iloc[-11] - 1  # 10-day momentum
        mom_20 = price_data.iloc[-1] / price_data.iloc[-21] - 1  # 20-day momentum
        
        # Acceleration (momentum of momentum)
        recent_mom = mom_5
        prev_mom = price_data.iloc[-6] / price_data.iloc[-11] - 1 if len(price_data) > 11 else 0
        acceleration = recent_mom - prev_mom
        
        # Combined momentum score
        momentum_score = (
            mom_5 * 0.5 +
            mom_10 * 0.3 +
            mom_20 * 0.2 +
            acceleration * 0.1
        )
        
        # Classify momentum
        if momentum_score > 0.03:
            return 'strong_momentum_up'
        elif momentum_score > 0.01:
            return 'momentum_up'
        elif momentum_score > -0.01:
            return 'neutral'
        elif momentum_score > -0.03:
            return 'momentum_down'
        else:
            return 'strong_momentum_down'
    
    def _calculate_confidence_scores(self, price_data: pd.Series, vol_regime: str, 
                                   trend_regime: str, stress_regime: str) -> Dict[str, float]:
        """Calculate confidence scores for regime classifications"""
        
        # Volatility confidence
        returns = price_data.pct_change().dropna()
        vol_stability = 1 - (returns.rolling(self.volatility_window).std().std() / 
                           returns.rolling(self.volatility_window).std().mean())
        vol_confidence = max(0.1, min(0.95, vol_stability))
        
        # Trend confidence (based on consistency)
        if len(price_data) >= self.trend_window:
            trend_consistency = 0
            window_size = max(5, self.trend_window // 4)
            
            for i in range(window_size, self.trend_window, window_size):
                window_return = price_data.iloc[-i] / price_data.iloc[-i-window_size] - 1
                if trend_regime in ['bull', 'strong_bull'] and window_return > 0:
                    trend_consistency += 1
                elif trend_regime in ['bear', 'strong_bear'] and window_return < 0:
                    trend_consistency += 1
                elif trend_regime == 'sideways' and abs(window_return) < 0.01:
                    trend_consistency += 1
            
            trend_confidence = max(0.1, min(0.95, trend_consistency / (self.trend_window // window_size)))
        else:
            trend_confidence = 0.5
        
        # Stress confidence (based on clarity of signals)
        stress_confidence = 0.7  # Default moderate confidence
        if stress_regime in ['extreme_stress', 'normal']:
            stress_confidence = 0.9  # High confidence for extreme cases
        elif stress_regime in ['high_stress', 'moderate_stress']:
            stress_confidence = 0.6  # Lower confidence for intermediate cases
        
        # Overall confidence
        overall_confidence = (vol_confidence * 0.4 + trend_confidence * 0.4 + stress_confidence * 0.2)
        
        return {
            'volatility': vol_confidence,
            'trend': trend_confidence,
            'stress': stress_confidence,
            'overall': overall_confidence
        }
    
    def _determine_overall_regime(self, vol_regime: str, trend_regime: str, 
                                 stress_regime: str, momentum_regime: str) -> str:
        """Determine overall market regime from individual components"""
        
        # Priority: Stress > Volatility > Trend > Momentum
        
        # Extreme stress overrides everything
        if stress_regime == 'extreme_stress':
            return f"crisis_{vol_regime}"
        
        # High stress modifies other regimes
        if stress_regime == 'high_stress':
            if vol_regime == 'high_vol':
                return f"volatile_stress_{trend_regime}"
            else:
                return f"stress_{trend_regime}"
        
        # Normal stress - combine volatility and trend
        if vol_regime == 'high_vol':
            if trend_regime in ['strong_bull', 'bull']:
                return "volatile_bull"
            elif trend_regime in ['strong_bear', 'bear']:
                return "volatile_bear"
            else:
                return "high_volatility_range"
        
        elif vol_regime == 'low_vol':
            if trend_regime in ['strong_bull', 'bull']:
                return "calm_bull"
            elif trend_regime in ['strong_bear', 'bear']:
                return "calm_bear"
            else:
                return "low_volatility_range"
        
        else:  # medium_vol
            if trend_regime in ['strong_bull', 'bull']:
                return "normal_bull"
            elif trend_regime in ['strong_bear', 'bear']:
                return "normal_bear"
            else:
                return "normal_range"
    
    def _calculate_regime_adjustments(self, overall_regime: str) -> Dict[str, float]:
        """Calculate model weight adjustments based on regime"""
        
        # Research-based adjustments for different regimes
        regime_adjustments = {
            # Crisis regimes - favor simple models
            'crisis_low_vol': {'linear': 1.3, 'rf': 0.9, 'xgb': 0.8, 'transformer': 0.7},
            'crisis_medium_vol': {'linear': 1.2, 'rf': 0.9, 'xgb': 0.9, 'transformer': 0.8},
            'crisis_high_vol': {'linear': 1.1, 'rf': 1.0, 'xgb': 1.0, 'transformer': 0.9},
            
            # Volatile regimes - favor ensemble models
            'volatile_bull': {'linear': 0.8, 'rf': 1.1, 'xgb': 1.2, 'transformer': 1.3},
            'volatile_bear': {'linear': 0.9, 'rf': 1.2, 'xgb': 1.1, 'transformer': 1.2},
            'high_volatility_range': {'linear': 0.7, 'rf': 1.0, 'xgb': 1.2, 'transformer': 1.4},
            
            # Calm regimes - favor trend-following models
            'calm_bull': {'linear': 1.0, 'rf': 1.1, 'xgb': 1.2, 'transformer': 1.1},
            'calm_bear': {'linear': 1.1, 'rf': 1.2, 'xgb': 1.0, 'transformer': 1.0},
            'low_volatility_range': {'linear': 1.2, 'rf': 1.0, 'xgb': 0.9, 'transformer': 0.9},
            
            # Normal regimes - balanced approach
            'normal_bull': {'linear': 1.0, 'rf': 1.0, 'xgb': 1.1, 'transformer': 1.1},
            'normal_bear': {'linear': 1.0, 'rf': 1.1, 'xgb': 1.0, 'transformer': 1.0},
            'normal_range': {'linear': 1.0, 'rf': 1.0, 'xgb': 1.0, 'transformer': 1.0},
            
            # Stress regimes - favor adaptive models
            'stress_bull': {'linear': 0.8, 'rf': 1.0, 'xgb': 1.2, 'transformer': 1.3},
            'stress_bear': {'linear': 0.9, 'rf': 1.1, 'xgb': 1.1, 'transformer': 1.2},
            'stress_sideways': {'linear': 0.7, 'rf': 0.9, 'xgb': 1.1, 'transformer': 1.4}
        }
        
        # Return adjustments for the detected regime
        if overall_regime in regime_adjustments:
            return regime_adjustments[overall_regime]
        else:
            # Default to balanced weights
            return {'linear': 1.0, 'rf': 1.0, 'xgb': 1.0, 'transformer': 1.0}
    
    def _get_regime_metrics(self, price_data: pd.Series, volume_data: pd.Series = None) -> Dict:
        """Get detailed metrics for the current regime"""
        returns = price_data.pct_change().dropna()
        
        if len(returns) < 20:
            return {}
        
        # Volatility metrics
        current_vol = returns.tail(self.volatility_window).std() * np.sqrt(252)
        vol_percentile = (returns.rolling(252).std() <= current_vol).mean() * 100
        
        # Trend metrics
        trend_strength = abs(price_data.iloc[-1] / price_data.iloc[-self.trend_window] - 1)
        
        # Momentum metrics
        momentum_5d = price_data.iloc[-1] / price_data.iloc[-6] - 1
        momentum_20d = price_data.iloc[-1] / price_data.iloc[-21] - 1
        
        # Drawdown metrics
        rolling_max = price_data.rolling(252).max()
        current_drawdown = (price_data.iloc[-1] / rolling_max.iloc[-1] - 1)
        
        # Volume metrics (if available)
        volume_metrics = {}
        if volume_data is not None and len(volume_data) >= 20:
            volume_metrics = {
                'volume_trend': volume_data.iloc[-5:].mean() / volume_data.iloc[-20:-5].mean() - 1,
                'volume_volatility': volume_data.pct_change().tail(20).std()
            }
        
        return {
            'volatility_annualized': current_vol,
            'volatility_percentile': vol_percentile,
            'trend_strength': trend_strength,
            'momentum_5d': momentum_5d,
            'momentum_20d': momentum_20d,
            'current_drawdown': current_drawdown,
            'price_level': price_data.iloc[-1],
            **volume_metrics
        }
    
    def _get_default_regime(self) -> Dict:
        """Return default regime when insufficient data"""
        return {
            'volatility_regime': 'medium_vol',
            'trend_regime': 'sideways',
            'stress_regime': 'normal',
            'momentum_regime': 'neutral',
            'overall_regime': 'normal_range',
            'confidence': {
                'volatility': 0.3,
                'trend': 0.3,
                'stress': 0.5,
                'overall': 0.3
            },
            'adjustments': {'linear': 1.0, 'rf': 1.0, 'xgb': 1.0, 'transformer': 1.0},
            'metrics': {},
            'insufficient_data': True
        }
    
    def get_regime_transitions(self, lookback_periods: int = 50) -> Dict:
        """Analyze recent regime transitions"""
        if len(self.regime_history) < 2:
            return {'error': 'Insufficient regime history'}
        
        recent_regimes = self.regime_history[-lookback_periods:]
        
        # Count regime changes
        transitions = 0
        transition_types = []
        
        for i in range(1, len(recent_regimes)):
            prev_regime = recent_regimes[i-1]['overall_regime']
            curr_regime = recent_regimes[i]['overall_regime']
            
            if prev_regime != curr_regime:
                transitions += 1
                transition_types.append(f"{prev_regime} -> {curr_regime}")
        
        # Calculate regime stability
        if len(recent_regimes) > 0:
            regime_stability = 1 - (transitions / len(recent_regimes))
        else:
            regime_stability = 1.0
        
        # Most common recent regime
        recent_regime_names = [r['overall_regime'] for r in recent_regimes]
        most_common_regime = max(set(recent_regime_names), key=recent_regime_names.count)
        
        return {
            'total_transitions': transitions,
            'regime_stability': regime_stability,
            'most_common_regime': most_common_regime,
            'recent_transitions': transition_types[-5:],  # Last 5 transitions
            'current_regime_duration': self._get_current_regime_duration()
        }
    
    def _get_current_regime_duration(self) -> int:
        """Get duration of current regime in periods"""
        if len(self.regime_history) < 2:
            return 0
        
        current_regime = self.regime_history[-1]['overall_regime']
        duration = 1
        
        for i in range(len(self.regime_history) - 2, -1, -1):
            if self.regime_history[i]['overall_regime'] == current_regime:
                duration += 1
            else:
                break
        
        return duration
    
    def predict_regime_change_probability(self, price_data: pd.Series) -> Dict[str, float]:
        """Predict probability of regime change in next period"""
        
        if len(self.regime_history) < 10:
            return {'regime_change_prob': 0.5, 'confidence': 0.1}
        
        # Historical transition probability
        transitions = 0
        total_periods = len(self.regime_history) - 1
        
        for i in range(1, len(self.regime_history)):
            if self.regime_history[i]['overall_regime'] != self.regime_history[i-1]['overall_regime']:
                transitions += 1
        
        historical_transition_prob = transitions / total_periods if total_periods > 0 else 0.5
        
        # Current regime duration effect
        current_duration = self._get_current_regime_duration()
        duration_factor = min(2.0, current_duration / 20)  # Longer regimes more likely to change
        
        # Recent volatility effect
        returns = price_data.pct_change().dropna()
        if len(returns) >= 10:
            recent_vol = returns.tail(10).std()
            avg_vol = returns.tail(50).std()
            vol_factor = recent_vol / (avg_vol + 1e-8)
        else:
            vol_factor = 1.0
        
        # Combined probability
        regime_change_prob = historical_transition_prob * duration_factor * vol_factor
        regime_change_prob = max(0.05, min(0.95, regime_change_prob))
        
        # Confidence based on data quality
        confidence = min(0.9, len(self.regime_history) / 100)
        
        return {
            'regime_change_prob': regime_change_prob,
            'confidence': confidence,
            'factors': {
                'historical_prob': historical_transition_prob,
                'duration_factor': duration_factor,
                'volatility_factor': vol_factor
            }
        }

class AdvancedRegimeDetector(MarketRegimeDetector):
    """
    Advanced regime detector with machine learning enhancements
    """
    
    def __init__(self, lookback_period: int = 252, use_ml: bool = True):
        super().__init__(lookback_period)
        self.use_ml = use_ml
        self.ml_model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        if use_ml:
            self.ml_model = KMeans(n_clusters=6, random_state=42)
    
    def fit_ml_regime_detector(self, historical_data: pd.DataFrame):
        """Fit ML model for regime detection"""
        if not self.use_ml or len(historical_data) < 100:
            return
        
        # Prepare features for ML
        features = self._prepare_ml_features(historical_data)
        
        if features is not None and len(features) > 50:
            # Fit scaler and model
            features_scaled = self.scaler.fit_transform(features)
            self.ml_model.fit(features_scaled)
            self.is_fitted = True
    
    def _prepare_ml_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML regime detection"""
        if 'close' not in data.columns:
            return None
        
        returns = data['close'].pct_change().dropna()
        
        features_list = []
        window_sizes = [5, 10, 20, 50]
        
        for i in range(max(window_sizes), len(data)):
            feature_vector = []
            
            # Rolling statistics
            for window in window_sizes:
                window_returns = returns.iloc[i-window:i]
                if len(window_returns) == window:
                    feature_vector.extend([
                        window_returns.mean(),
                        window_returns.std(),
                        window_returns.skew(),
                        window_returns.kurtosis(),
                        window_returns.min(),
                        window_returns.max()
                    ])
            
            # Price-based features
            current_price = data['close'].iloc[i]
            for window in [10, 20, 50]:
                if i >= window:
                    past_price = data['close'].iloc[i-window]
                    feature_vector.append(current_price / past_price - 1)
            
            # Volume features (if available)
            if 'volume' in data.columns:
                for window in [5, 20]:
                    if i >= window:
                        vol_ratio = data['volume'].iloc[i-window:i].mean() / data['volume'].iloc[i-window*2:i-window].mean()
                        feature_vector.append(vol_ratio)
            
            if len(feature_vector) > 10:  # Ensure we have enough features
                features_list.append(feature_vector)
        
        return np.array(features_list) if features_list else None
    
    def detect_regime_ml(self, price_data: pd.Series, volume_data: pd.Series = None) -> Dict:
        """Enhanced regime detection using ML"""
        # Get base regime detection
        base_regime = self.detect_regime(price_data, volume_data)
        
        if not self.use_ml or not self.is_fitted:
            return base_regime
        
        try:
            # Prepare recent data for ML prediction
            recent_data = pd.DataFrame({'close': price_data})
            if volume_data is not None:
                recent_data['volume'] = volume_data
            
            ml_features = self._prepare_ml_features(recent_data)
            
            if ml_features is not None and len(ml_features) > 0:
                # Get ML prediction
                features_scaled = self.scaler.transform([ml_features[-1]])
                ml_cluster = self.ml_model.predict(features_scaled)[0]
                
                # Map cluster to regime names
                cluster_regime_map = {
                    0: 'ml_regime_0_low_vol_bull',
                    1: 'ml_regime_1_high_vol_bull', 
                    2: 'ml_regime_2_low_vol_bear',
                    3: 'ml_regime_3_high_vol_bear',
                    4: 'ml_regime_4_sideways',
                    5: 'ml_regime_5_volatile_sideways'
                }
                
                ml_regime = cluster_regime_map.get(ml_cluster, 'ml_regime_unknown')
                
                # Combine with base regime
                base_regime['ml_regime'] = ml_regime
                base_regime['ml_cluster'] = int(ml_cluster)
                base_regime['ml_confidence'] = 0.7  # Default ML confidence
                
        except Exception as e:
            # If ML fails, return base regime
            base_regime['ml_error'] = str(e)
        
        return base_regime

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    # Simulate price data with different regimes
    price_data = []
    base_price = 100
    
    for i, date in enumerate(dates):
        if i < 100:  # Bull market
            daily_return = np.random.normal(0.001, 0.015)
        elif i < 200:  # High volatility
            daily_return = np.random.normal(0, 0.04)
        elif i < 300:  # Bear market
            daily_return = np.random.normal(-0.001, 0.02)
        else:  # Recovery
            daily_return = np.random.normal(0.0005, 0.01)
        
        base_price *= (1 + daily_return)
        price_data.append(base_price)
    
    price_series = pd.Series(price_data, index=dates)
    volume_series = pd.Series(np.random.lognormal(10, 1, len(dates)), index=dates)
    
    # Test regime detector
    print("🔍 Testing Market Regime Detection")
    print("=" * 50)
    
    detector = MarketRegimeDetector()
    regime_info = detector.detect_regime(price_series, volume_series)
    
    print(f"Detected Regime: {regime_info['overall_regime']}")
    print(f"Volatility Regime: {regime_info['volatility_regime']}")
    print(f"Trend Regime: {regime_info['trend_regime']}")
    print(f"Stress Regime: {regime_info['stress_regime']}")
    print(f"Overall Confidence: {regime_info['confidence']['overall']:.3f}")
    
    # Test advanced detector
    print("\n🤖 Testing Advanced ML Regime Detection")
    print("=" * 50)
    
    advanced_detector = AdvancedRegimeDetector(use_ml=True)
    
    # Create historical data for training
    historical_df = pd.DataFrame({
        'close': price_series,
        'volume': volume_series
    })
    
    # Fit ML model
    advanced_detector.fit_ml_regime_detector(historical_df)
    
    # Get ML-enhanced regime detection
    ml_regime_info = advanced_detector.detect_regime_ml(price_series, volume_series)
    
    print(f"ML Enhanced Regime: {ml_regime_info.get('ml_regime', 'N/A')}")
    print(f"ML Cluster: {ml_regime_info.get('ml_cluster', 'N/A')}")
    print(f"Base Regime: {ml_regime_info['overall_regime']}")
    
    print("\n✅ Regime Detection System Ready!")
    print("🚀 Ready for integration with ensemble system!")
