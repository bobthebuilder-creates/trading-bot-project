"""
Base model class for all trading models
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class BaseTradingModel(ABC):
    """Abstract base class for all trading models"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
        self.model = None
        self.performance_metrics = {}
    
    @abstractmethod
    def train(self, data: pd.DataFrame) -> None:
        """Train the model on historical data"""
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        pass
    
    @abstractmethod
    def evaluate(self, data: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        pass
    
    def save_model(self, path: str) -> None:
        """Save the trained model"""
        # TODO: Implement model saving
        logger.info(f"Saving model {self.name} to {path}")
    
    def load_model(self, path: str) -> None:
        """Load a trained model"""
        # TODO: Implement model loading
        logger.info(f"Loading model {self.name} from {path}")
