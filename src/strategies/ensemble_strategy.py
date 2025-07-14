"""
Enhanced Ensemble Strategy for Trading Bot
Integrates multiple ML models with research-based techniques
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    import xgboost as xgb
    import lightgbm as lgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: Some ML libraries not available. Install with: pip install scikit-learn xgboost lightgbm")

class EnsembleIntegrationHelper:
    """
    Helper class for integrating ensemble strategies with existing trading bot
    Provides research-enhanced features and model management
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the ensemble integration helper
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config or self._get_default_config()
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        self.logger = self._setup_logging()
        
        # Initialize models if ML libraries are available
        if ML_AVAILABLE:
            self._initialize_models()
        else:
            self.logger.warning("ML libraries not available. Limited functionality.")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for ensemble models"""
        return {
            'models': {
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42
                },
                'xgboost': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                },
                'lightgbm': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42
                },
                'logistic': {
                    'random_state': 42,
                    'max_iter': 1000
                }
            },
            'ensemble': {
                'voting': 'soft',
                'weights': None
            },
            'features': {
                'lookback_periods': [5, 10, 20, 50],
                'technical_indicators': ['sma', 'ema', 'rsi', 'macd', 'bollinger'],
                'sentiment_weight': 0.2
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the ensemble system"""
        logger = logging.getLogger('EnsembleStrategy')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_models(self):
        """Initialize all ensemble models"""
        if not ML_AVAILABLE:
            return
        
        try:
            # Random Forest
            self.models['random_forest'] = RandomForestClassifier(
                **self.config['models']['random_forest']
            )
            
            # XGBoost
            self.models['xgboost'] = xgb.XGBClassifier(
                **self.config['models']['xgboost']
            )
            
            # LightGBM
            self.models['lightgbm'] = lgb.LGBMClassifier(
                **self.config['models']['lightgbm']
            )
            
            # Logistic Regression
            self.models['logistic'] = LogisticRegression(
                **self.config['models']['logistic']
            )
            
            # Scalers for each model
            for model_name in self.models.keys():
                self.scalers[model_name] = StandardScaler()
            
            self.logger.info(f"Initialized {len(self.models)} ensemble models")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
    
    def prepare_features(self, data: pd.DataFrame, 
                        sentiment_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare features for ensemble models
        
        Args:
            data: Price data with OHLCV columns
            sentiment_data: Optional sentiment analysis data
            
        Returns:
            DataFrame with engineered features
        """
        try:
            features_df = data.copy()
            
            # Technical indicators
            features_df = self._add_technical_indicators(features_df)
            
            # Price-based features
            features_df = self._add_price_features(features_df)
            
            # Volatility features
            features_df = self._add_volatility_features(features_df)
            
            # Sentiment features if available
            if sentiment_data is not None:
                features_df = self._add_sentiment_features(features_df, sentiment_data)
            
            # Remove NaN values
            features_df = features_df.dropna()
            
            self.logger.info(f"Prepared {len(features_df.columns)} features for {len(features_df)} samples")
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return data
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        try:
            # Simple Moving Averages
            for period in self.config['features']['lookback_periods']:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        try:
            # Returns
            df['return_1d'] = df['close'].pct_change()
            df['return_5d'] = df['close'].pct_change(5)
            df['return_10d'] = df['close'].pct_change(10)
            
            # Price position features
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Volume features
            if 'volume' in df.columns:
                df['volume_sma'] = df['volume'].rolling(window=20).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding price features: {e}")
            return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        try:
            # Historical volatility
            df['volatility_10d'] = df['return_1d'].rolling(window=10).std()
            df['volatility_20d'] = df['return_1d'].rolling(window=20).std()
            
            # True Range and ATR
            df['tr1'] = df['high'] - df['low']
            df['tr2'] = abs(df['high'] - df['close'].shift())
            df['tr3'] = abs(df['low'] - df['close'].shift())
            df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr'] = df['true_range'].rolling(window=14).mean()
            
            # Clean up temporary columns
            df = df.drop(['tr1', 'tr2', 'tr3'], axis=1)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding volatility features: {e}")
            return df
    
    def _add_sentiment_features(self, df: pd.DataFrame, 
                               sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment-based features"""
        try:
            # Merge sentiment data with price data
            if 'timestamp' in sentiment_data.columns:
                sentiment_features = sentiment_data.set_index('timestamp')
                df = df.join(sentiment_features, how='left')
            
            # Fill missing sentiment values
            sentiment_cols = [col for col in df.columns if 'sentiment' in col.lower()]
            for col in sentiment_cols:
                df[col] = df[col].fillna(0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding sentiment features: {e}")
            return df
    
    def train_ensemble(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """
        Train the ensemble of models
        
        Args:
            features: Feature matrix
            targets: Target labels (1 for buy, 0 for hold/sell)
            
        Returns:
            Dictionary with training results
        """
        if not ML_AVAILABLE:
            self.logger.error("ML libraries not available for training")
            return {}
        
        try:
            results = {}
            
            # Select numeric features only
            numeric_features = features.select_dtypes(include=[np.number])
            
            for model_name, model in self.models.items():
                self.logger.info(f"Training {model_name}...")
                
                # Scale features
                X_scaled = self.scalers[model_name].fit_transform(numeric_features)
                
                # Train model
                model.fit(X_scaled, targets)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_scaled, targets, cv=5, scoring='accuracy')
                
                # Store results
                results[model_name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'feature_count': X_scaled.shape[1]
                }
                
                # Store feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[model_name] = dict(
                        zip(numeric_features.columns, model.feature_importances_)
                    )
                
                self.logger.info(f"{model_name} CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            self.performance_metrics = results
            return results
            
        except Exception as e:
            self.logger.error(f"Error training ensemble: {e}")
            return {}
    
    def predict_ensemble(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate ensemble predictions
        
        Args:
            features: Feature matrix for prediction
            
        Returns:
            Dictionary with individual and ensemble predictions
        """
        if not ML_AVAILABLE:
            return {'ensemble_signal': 0, 'confidence': 0.0}
        
        try:
            predictions = {}
            probabilities = {}
            
            # Select numeric features only
            numeric_features = features.select_dtypes(include=[np.number])
            
            for model_name, model in self.models.items():
                if model_name in self.scalers:
                    # Scale features
                    X_scaled = self.scalers[model_name].transform(numeric_features)
                    
                    # Get prediction and probability
                    pred = model.predict(X_scaled)
                    pred_proba = model.predict_proba(X_scaled)
                    
                    predictions[model_name] = pred[0] if len(pred) > 0 else 0
                    probabilities[model_name] = pred_proba[0] if len(pred_proba) > 0 else [0.5, 0.5]
            
            # Ensemble prediction (simple voting)
            if predictions:
                ensemble_pred = np.mean(list(predictions.values()))
                ensemble_signal = 1 if ensemble_pred >= 0.5 else 0
                
                # Calculate confidence as average of max probabilities
                confidences = [max(prob) for prob in probabilities.values()]
                ensemble_confidence = np.mean(confidences) if confidences else 0.5
            else:
                ensemble_signal = 0
                ensemble_confidence = 0.0
            
            return {
                'individual_predictions': predictions,
                'individual_probabilities': probabilities,
                'ensemble_signal': ensemble_signal,
                'confidence': ensemble_confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error generating ensemble predictions: {e}")
            return {'ensemble_signal': 0, 'confidence': 0.0}
    
    def get_feature_importance(self, top_n: int = 10) -> Dict[str, Dict]:
        """
        Get feature importance from trained models
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary with feature importance for each model
        """
        importance_summary = {}
        
        for model_name, importance_dict in self.feature_importance.items():
            if importance_dict:
                # Sort by importance
                sorted_features = sorted(
                    importance_dict.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                importance_summary[model_name] = dict(sorted_features[:top_n])
        
        return importance_summary
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all models"""
        return self.performance_metrics
    
    def save_models(self, filepath: str):
        """Save trained models to file"""
        try:
            import joblib
            
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_importance': self.feature_importance,
                'performance_metrics': self.performance_metrics,
                'config': self.config
            }
            
            joblib.dump(model_data, filepath)
            self.logger.info(f"Models saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def load_models(self, filepath: str):
        """Load trained models from file"""
        try:
            import joblib
            
            model_data = joblib.load(filepath)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_importance = model_data['feature_importance']
            self.performance_metrics = model_data['performance_metrics']
            self.config = model_data['config']
            
            self.logger.info(f"Models loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")

# Additional utility functions for integration

def create_target_labels(data: pd.DataFrame, 
                        future_periods: int = 1, 
                        threshold: float = 0.01) -> pd.Series:
    """
    Create target labels for training
    
    Args:
        data: Price data with 'close' column
        future_periods: Number of periods to look ahead
        threshold: Minimum return threshold for buy signal
        
    Returns:
        Series with binary labels (1=buy, 0=hold/sell)
    """
    future_returns = data['close'].shift(-future_periods) / data['close'] - 1
    labels = (future_returns > threshold).astype(int)
    return labels

def backtest_ensemble_strategy(ensemble_helper: EnsembleIntegrationHelper,
                              data: pd.DataFrame,
                              initial_capital: float = 10000) -> Dict[str, Any]:
    """
    Simple backtest for ensemble strategy
    
    Args:
        ensemble_helper: Trained ensemble helper
        data: Historical price data
        initial_capital: Starting capital
        
    Returns:
        Dictionary with backtest results
    """
    try:
        results = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'trades': []
        }
        
        capital = initial_capital
        position = 0
        entry_price = 0
        
        for i in range(len(data)):
            if i < 50:  # Skip initial period for feature calculation
                continue
            
            # Get features for current timestamp
            current_data = data.iloc[:i+1]
            features = ensemble_helper.prepare_features(current_data)
            
            if len(features) == 0:
                continue
            
            # Get prediction
            prediction = ensemble_helper.predict_ensemble(features.tail(1))
            signal = prediction.get('ensemble_signal', 0)
            confidence = prediction.get('confidence', 0.0)
            
            current_price = data.iloc[i]['close']
            
            # Trading logic
            if signal == 1 and position == 0 and confidence > 0.6:
                # Buy signal
                position = capital / current_price
                entry_price = current_price
                capital = 0
                
            elif signal == 0 and position > 0:
                # Sell signal
                capital = position * current_price
                trade_return = (current_price - entry_price) / entry_price
                
                results['trades'].append({
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'return': trade_return,
                    'timestamp': data.index[i]
                })
                
                position = 0
                entry_price = 0
        
        # Final position value
        if position > 0:
            final_value = position * data.iloc[-1]['close']
        else:
            final_value = capital
        
        # Calculate metrics
        results['total_return'] = (final_value - initial_capital) / initial_capital
        
        if results['trades']:
            trade_returns = [trade['return'] for trade in results['trades']]
            results['win_rate'] = sum(1 for r in trade_returns if r > 0) / len(trade_returns)
            
            if len(trade_returns) > 1:
                returns_std = np.std(trade_returns)
                if returns_std > 0:
                    results['sharpe_ratio'] = np.mean(trade_returns) / returns_std
        
        return results
        
    except Exception as e:
        print(f"Error in backtest: {e}")
        return results
