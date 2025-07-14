"""
Linear Regression Model for Trading Signal Generation
First model in the progressive learning system
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
import joblib
from typing import Dict, Tuple, Optional
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.base_model import BaseTradingModel

logger = logging.getLogger(__name__)

class LinearTradingModel(BaseTradingModel):
    """Linear model for predicting price direction using technical indicators"""
    
    def __init__(self, name: str = "LinearModel"):
        super().__init__(name)
        self.model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = []
        self.min_periods = 60  # Minimum data points needed
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for the model"""
        df = data.copy()
        
        # Calculate technical indicators if not already present
        if 'rsi' not in df.columns:
            from data.market_data import MarketDataManager
            dm = MarketDataManager()
            df = dm.calculate_technical_indicators(df)
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['price_change_2'] = df['close'].pct_change(2)
        df['price_change_5'] = df['close'].pct_change(5)
        
        # Volatility features
        df['volatility_5'] = df['close'].rolling(5).std()
        df['volatility_20'] = df['close'].rolling(20).std()
        
        # Volume features
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_change'] = df['volume'].pct_change()
        
        # Trend features
        df['sma_ratio'] = df['close'] / df['sma_20']
        df['ema_ratio'] = df['ema_12'] / df['ema_26']
        
        # Bollinger Band position
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI features
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        # MACD features
        df['macd_signal_cross'] = (df['macd'] > df['macd_signal']).astype(int)
        
        # Select features for the model
        feature_cols = [
            'rsi', 'macd', 'macd_histogram',
            'price_change', 'price_change_2', 'price_change_5',
            'volatility_5', 'volatility_20',
            'volume_ratio', 'volume_change',
            'sma_ratio', 'ema_ratio', 'bb_position',
            'rsi_oversold', 'rsi_overbought', 'macd_signal_cross'
        ]
        
        # Keep only available features
        available_features = [col for col in feature_cols if col in df.columns]
        self.feature_names = available_features
        
        features_df = df[available_features].copy()
        
        # Fill NaN values
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        logger.info(f"Prepared {len(available_features)} features for training")
        return features_df
    
    def create_targets(self, data: pd.DataFrame, lookahead: int = 1) -> pd.Series:
        """Create binary targets (1 = price goes up, 0 = price goes down)"""
        # Future price change
        future_return = data['close'].shift(-lookahead) / data['close'] - 1
        
        # Binary classification: 1 if price goes up, 0 if down
        targets = (future_return > 0).astype(int)
        
        # Remove last lookahead rows (no future data)
        targets = targets[:-lookahead]
        
        logger.info(f"Created targets with {targets.sum()} positive samples out of {len(targets)}")
        return targets
    
    def train(self, data: pd.DataFrame, test_size: float = 0.2) -> Dict[str, float]:
        """Train the linear model"""
        if len(data) < self.min_periods:
            raise ValueError(f"Need at least {self.min_periods} data points, got {len(data)}")
        
        logger.info(f"Training linear model with {len(data)} data points")
        
        # Prepare features and targets
        features = self.prepare_features(data)
        targets = self.create_targets(data)
        
        # Align features and targets
        min_len = min(len(features), len(targets))
        X = features.iloc[:min_len]
        y = targets.iloc[:min_len]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        self.performance_metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_count': len(self.feature_names)
        }
        
        self.is_trained = True
        
        logger.info(f"Model trained - Test Accuracy: {test_accuracy:.3f}, CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, abs(self.model.coef_[0])))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("Top 5 most important features:")
        for feature, importance in sorted_features[:5]:
            logger.info(f"  {feature}: {importance:.4f}")
        
        return self.performance_metrics
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        features = self.prepare_features(data)
        
        # Use only the features the model was trained on
        features_subset = features[self.feature_names]
        features_scaled = self.scaler.transform(features_subset)
        
        # Get predictions and probabilities
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        # Return probabilities for the positive class (price goes up)
        return probabilities[:, 1]
    
    def predict_latest(self, data: pd.DataFrame) -> Dict:
        """Get prediction for the latest data point"""
        probabilities = self.predict(data)
        latest_prob = probabilities[-1]
        
        prediction = {
            'signal': 1 if latest_prob > 0.5 else 0,
            'confidence': latest_prob,
            'direction': 'BUY' if latest_prob > 0.6 else 'SELL' if latest_prob < 0.4 else 'HOLD',
            'strength': 'STRONG' if abs(latest_prob - 0.5) > 0.2 else 'WEAK'
        }
        
        return prediction
    
    def evaluate(self, data: pd.DataFrame, targets: pd.Series) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(data)
        binary_predictions = (predictions > 0.5).astype(int)
        
        accuracy = accuracy_score(targets, binary_predictions)
        
        # Calculate profit simulation (simplified)
        returns = []
        for i in range(len(binary_predictions) - 1):
            if binary_predictions[i] == 1:  # Predicted UP
                actual_return = data['close'].iloc[i+1] / data['close'].iloc[i] - 1
                returns.append(actual_return)
            elif binary_predictions[i] == 0:  # Predicted DOWN (short)
                actual_return = data['close'].iloc[i] / data['close'].iloc[i+1] - 1
                returns.append(actual_return)
        
        if returns:
            total_return = sum(returns)
            win_rate = sum(1 for r in returns if r > 0) / len(returns)
        else:
            total_return = 0
            win_rate = 0
        
        metrics = {
            'accuracy': accuracy,
            'total_return': total_return,
            'win_rate': win_rate,
            'num_trades': len(returns)
        }
        
        return metrics
    
    def save_model(self, path: str) -> None:
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'performance_metrics': self.performance_metrics,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load a trained model"""
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.performance_metrics = model_data['performance_metrics']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {path}")


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Import data manager
    sys.path.append('..')
    from data.market_data import MarketDataManager
    
    print("🤖 Testing Linear Trading Model")
    print("=" * 40)
    
    try:
        # Get market data
        dm = MarketDataManager()
        dm.connect_best_exchange()
        
        # Get BTC data with technical indicators
        print("📊 Fetching BTC data...")
        btc_data = dm.get_ohlcv('BTC/USD', '1h', 500)  # Get more data for training
        btc_with_indicators = dm.calculate_technical_indicators(btc_data)
        
        if len(btc_with_indicators) < 100:
            print("❌ Not enough data for training")
            exit()
        
        # Create and train model
        print("🔧 Training linear model...")
        model = LinearTradingModel()
        
        # Train the model
        performance = model.train(btc_with_indicators)
        
        print(f"✅ Model trained successfully!")
        print(f"📈 Test Accuracy: {performance['test_accuracy']:.3f}")
        print(f"📊 Cross-validation: {performance['cv_mean']:.3f} ± {performance['cv_std']:.3f}")
        
        # Get latest prediction
        print("\n🔮 Latest prediction:")
        latest_prediction = model.predict_latest(btc_with_indicators)
        print(f"Signal: {latest_prediction['direction']}")
        print(f"Confidence: {latest_prediction['confidence']:.3f}")
        print(f"Strength: {latest_prediction['strength']}")
        
        # Save model
        model.save_model('linear_model.pkl')
        print("💾 Model saved!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
