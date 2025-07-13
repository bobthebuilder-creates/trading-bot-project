"""
Base strategy class for all trading strategies
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str, risk_params: Dict):
        self.name = name
        self.risk_params = risk_params
        self.positions = {}
        self.performance = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0
        }
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals"""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: float, current_price: float) -> float:
        """Calculate position size based on signal and risk management"""
        pass
    
    def update_performance(self, trade_result: Dict) -> None:
        """Update strategy performance metrics"""
        # TODO: Implement performance tracking
        logger.info(f"Updating performance for {self.name}")
