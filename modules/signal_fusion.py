# modules/signal_fusion.py - UPGRADED WITH ULTRA ALPHA INTEGRATION
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import datetime
import sys
import os

from core.redis_manager import redis_manager

# ULTRA ALPHA INTEGRATION
try:
    from core.alpha_integration import integration_engine, IntegratedAlphaSignal
    ULTRA_ALPHA_AVAILABLE = True
except ImportError:
    ULTRA_ALPHA_AVAILABLE = False
    logging.warning("Ultra Alpha Integration not available - using legacy signal fusion only")

logger = logging.getLogger(__name__)

class SignalFusion:
    def __init__(self):
        self.technical_model = None
        self.sentiment_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_trained = False
        
    def prepare_features(self, technical_signals: Dict, sentiment_score: float, 
                        price_features: Dict, news_confidence: float = 0.0) -> np.ndarray:
        """Prepare feature vector for ML model"""
        try:
            features = []
            
            # Technical indicators (normalized to 0-1 range)
            features.append(technical_signals.get('rsi_oversold', 0))
            features.append(technical_signals.get('rsi_overbought', 0))
            features.append(technical_signals.get('macd_bullish_cross', 0))
            features.append(technical_signals.get('macd_bearish_cross', 0))
            features.append(technical_signals.get('sma_golden_cross', 0))
            features.append(technical_signals.get('sma_death_cross', 0))
            
            # Price-based features
            features.append(price_features.get('returns', 0.0))
            features.append(price_features.get('volatility', 0.0))
            features.append(price_features.get('volume_ratio', 1.0))
            features.append(price_features.get('atr_normalized', 0.0))
            
            # Sentiment features
            features.append(max(-1, min(1, sentiment_score)))  # Clamp to [-1, 1]
            features.append(max(0, min(1, news_confidence)))   # Clamp to [0, 1]
            
            # Time-based features
            now = datetime.datetime.now()
            features.append(now.hour / 24.0)  # Hour of day (0-1)
            features.append(now.weekday() / 6.0)  # Day of week (0-1)
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return np.zeros((1, 14))  # Return zero features if failed
    
    def create_training_data(self, historical_data: List[Dict]) -> tuple:
        """Create training dataset from historical signals and outcomes
        
        FIXED: Look-ahead bias eliminated by using only available information at time t
        to predict outcome at time t+1
        """
        try:
            features_list = []
            targets = []
            
            # CRITICAL FIX: Process data chronologically and use only past information
            for i, data_point in enumerate(historical_data[:-1]):  # Exclude last point (no future data)
                # Extract features from current time point (t)
                technical = data_point.get('technical_signals', {})
                sentiment = data_point.get('sentiment_score', 0.0)
                price_features = data_point.get('price_features', {})
                news_conf = data_point.get('news_confidence', 0.0)
                
                features = self.prepare_features(technical, sentiment, price_features, news_conf)
                features_list.append(features.flatten())
                
                # Extract target from NEXT time point (t+1) - NO LOOK-AHEAD BIAS
                next_data_point = historical_data[i + 1]
                current_price = data_point.get('price', 0.0)
                next_price = next_data_point.get('price', 0.0)
                
                if current_price > 0:
                    next_return = (next_price - current_price) / current_price
                    
                    # Convert to classification target using REALISTIC thresholds
                    if next_return > 0.0005:  # 5 bps profit threshold (realistic)
                        target = 1  # Buy signal
                    elif next_return < -0.0005:  # 5 bps loss threshold
                        target = -1  # Sell signal
                    else:
                        target = 0  # Hold signal
                else:
                    target = 0  # Default to hold if price data unavailable
                
                targets.append(target)
            
            X = np.array(features_list)
            y = np.array(targets)
            
            logger.info(f"Training data created: {len(X)} samples, Look-ahead bias eliminated")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Training data creation failed: {e}")
            return np.array([]), np.array([])
    
    def train_models(self, historical_data: List[Dict]):
        """Train ML models for signal fusion"""
        try:
            X, y = self.create_training_data(historical_data)
            
            if len(X) < 50:  # Need minimum data for training
                logger.warning("Insufficient data for model training")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Random Forest for multi-class classification
            self.technical_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            self.technical_model.fit(X_train_scaled, y_train)
            
            # Train Logistic Regression for probability estimates
            self.sentiment_model = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                multi_class='ovr'
            )
            self.sentiment_model.fit(X_train_scaled, y_train)
            
            # Evaluate models
            rf_score = self.technical_model.score(X_test_scaled, y_test)
            lr_score = self.sentiment_model.score(X_test_scaled, y_test)
            
            logger.info(f"Model training completed - RF Score: {rf_score:.3f}, LR Score: {lr_score:.3f}")
            
            self.model_trained = True
            
            # Save models to Redis cache
            self.save_models_to_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
    
    def predict_signal(self, technical_signals: Dict, sentiment_score: float, 
                      price_features: Dict, news_confidence: float = 0.0) -> Dict[str, Any]:
        """Generate fused trading signal with Ultra Alpha Integration"""
        try:
            # ULTRA ALPHA INTEGRATION - Use advanced system if available
            if ULTRA_ALPHA_AVAILABLE:
                return self.generate_ultra_alpha_signal(technical_signals, sentiment_score, 
                                                       price_features, news_confidence)
            
            # LEGACY SYSTEM - Fallback to original ML approach
            if not self.model_trained:
                # Try to load from cache
                if not self.load_models_from_cache():
                    return self.fallback_signal_fusion(technical_signals, sentiment_score)
            
            # Prepare features
            features = self.prepare_features(technical_signals, sentiment_score, price_features, news_confidence)
            features_scaled = self.scaler.transform(features)
            
            # Get predictions from both models
            rf_prediction = self.technical_model.predict(features_scaled)[0]
            rf_probabilities = self.technical_model.predict_proba(features_scaled)[0]
            
            lr_prediction = self.sentiment_model.predict(features_scaled)[0]
            lr_probabilities = self.sentiment_model.predict_proba(features_scaled)[0]
            
            # Ensemble prediction (weighted average)
            technical_weight = 0.6
            sentiment_weight = 0.4
            
            ensemble_probs = (technical_weight * rf_probabilities + 
                            sentiment_weight * lr_probabilities)
            
            ensemble_prediction = np.argmax(ensemble_probs) - 1  # Convert to -1, 0, 1
            
            # Calculate confidence
            confidence = float(np.max(ensemble_probs))
            
            # Generate final signal
            signal_result = {
                'signal': int(ensemble_prediction),
                'confidence': confidence,
                'technical_signal': int(rf_prediction),
                'sentiment_signal': int(lr_prediction),
                'technical_confidence': float(np.max(rf_probabilities)),
                'sentiment_confidence': float(np.max(lr_probabilities)),
                'feature_importance': self.get_feature_importance(),
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            # Cache result
            redis_manager.set_cache('latest_ml_signal', signal_result, ttl=300)
            
            return signal_result
            
        except Exception as e:
            logger.error(f"ML signal prediction failed: {e}")
            return self.fallback_signal_fusion(technical_signals, sentiment_score)
    
    def fallback_signal_fusion(self, technical_signals: Dict, sentiment_score: float) -> Dict[str, Any]:
        """Simple rule-based signal fusion when ML models are not available"""
        try:
            # Count technical signals
            buy_signals = sum([
                technical_signals.get('rsi_oversold', 0),
                technical_signals.get('macd_bullish_cross', 0),
                technical_signals.get('sma_golden_cross', 0)
            ])
            
            sell_signals = sum([
                technical_signals.get('rsi_overbought', 0),
                technical_signals.get('macd_bearish_cross', 0),
                technical_signals.get('sma_death_cross', 0)
            ])
            
            # Sentiment adjustment
            sentiment_weight = 0.3
            adjusted_buy = buy_signals + max(0, sentiment_score) * sentiment_weight
            adjusted_sell = sell_signals + max(0, -sentiment_score) * sentiment_weight
            
            # Generate signal
            if adjusted_buy > adjusted_sell and adjusted_buy >= 1.5:
                signal = 1
                confidence = min(0.8, adjusted_buy / 3.0)
            elif adjusted_sell > adjusted_buy and adjusted_sell >= 1.5:
                signal = -1
                confidence = min(0.8, adjusted_sell / 3.0)
            else:
                signal = 0
                confidence = 0.5
            
            return {
                'signal': signal,
                'confidence': confidence,
                'technical_signal': 1 if buy_signals > sell_signals else (-1 if sell_signals > buy_signals else 0),
                'sentiment_signal': 1 if sentiment_score > 0.1 else (-1 if sentiment_score < -0.1 else 0),
                'method': 'rule_based',
                'timestamp': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Fallback signal fusion failed: {e}")
            return {'signal': 0, 'confidence': 0.0}
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained models"""
        if not self.model_trained or self.technical_model is None:
            return {}
        
        try:
            feature_names = [
                'rsi_oversold', 'rsi_overbought', 'macd_bullish', 'macd_bearish',
                'sma_golden', 'sma_death', 'returns', 'volatility', 'volume_ratio',
                'atr', 'sentiment', 'news_confidence', 'hour', 'weekday'
            ]
            
            importance = self.technical_model.feature_importances_
            return dict(zip(feature_names, importance))
            
        except Exception as e:
            logger.error(f"Feature importance extraction failed: {e}")
            return {}
    
    def save_models_to_cache(self):
        """Save trained models to Redis cache"""
        try:
            if self.model_trained:
                model_data = {
                    'technical_model': pickle.dumps(self.technical_model),
                    'sentiment_model': pickle.dumps(self.sentiment_model),
                    'scaler': pickle.dumps(self.scaler),
                    'trained_at': datetime.datetime.now().isoformat()
                }
                
                redis_manager.set_cache('ml_models', model_data, ttl=86400)  # 24 hours
                logger.info("ML models saved to cache")
                
        except Exception as e:
            logger.error(f"Model saving failed: {e}")
    
    def load_models_from_cache(self) -> bool:
        """Load trained models from Redis cache"""
        try:
            model_data = redis_manager.get_cache('ml_models')
            if not model_data:
                return False
            
            self.technical_model = pickle.loads(model_data['technical_model'])
            self.sentiment_model = pickle.loads(model_data['sentiment_model'])
            self.scaler = pickle.loads(model_data['scaler'])
            self.model_trained = True
            
            logger.info("ML models loaded from cache")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False
    
    def update_performance_feedback(self, signal_id: str, actual_return: float):
        """Update model performance with actual trading results"""
        try:
            feedback_data = {
                'signal_id': signal_id,
                'actual_return': actual_return,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            redis_manager.add_to_stream('ml_feedback', feedback_data)
            logger.debug(f"Added performance feedback for signal {signal_id}")
            
        except Exception as e:
            logger.error(f"Performance feedback update failed: {e}")
    
    def generate_ultra_alpha_signal(self, technical_signals: Dict, sentiment_score: float,
                                   price_features: Dict, news_confidence: float) -> Dict[str, Any]:
        """Generate signal using Ultra Alpha Integration Engine"""
        try:
            # Convert inputs to market_data format for integration engine
            market_data = self.convert_to_market_data_format(
                technical_signals, sentiment_score, price_features, news_confidence
            )
            
            # Generate integrated alpha signal
            current_price = price_features.get('current_price', 0.0)
            integrated_signal = integration_engine.generate_integrated_alpha_signal(
                market_data, current_price
            )
            
            if integrated_signal is None:
                # Fallback to legacy system
                logger.warning("Ultra Alpha signal generation failed - using legacy system")
                return self.fallback_signal_fusion(technical_signals, sentiment_score)
            
            # Convert IntegratedAlphaSignal to legacy format for compatibility
            legacy_format_signal = {
                'signal': int(np.sign(integrated_signal.signal_strength)),
                'confidence': integrated_signal.confidence,
                'signal_strength': integrated_signal.signal_strength,
                
                # Enhanced Ultra Alpha data
                'expected_edge_bps': integrated_signal.expected_edge_bps,
                'optimal_position_size': integrated_signal.optimal_position_size,
                'risk_adjusted_return': integrated_signal.risk_adjusted_return,
                
                # Execution guidance
                'entry_price': integrated_signal.entry_price,
                'stop_loss_price': integrated_signal.stop_loss_price,
                'take_profit_price': integrated_signal.take_profit_price,
                'holding_period_minutes': integrated_signal.holding_period_minutes,
                
                # Strategy breakdown
                'timeframe_components': integrated_signal.timeframe_components,
                'strategy_components': integrated_signal.strategy_components,
                
                # Risk metrics
                'sharpe_estimate': integrated_signal.sharpe_estimate,
                'max_drawdown_estimate': integrated_signal.max_drawdown_estimate,
                
                # Meta data
                'market_regime': integrated_signal.market_regime,
                'method': 'ultra_alpha_integration',
                'timestamp': integrated_signal.timestamp.isoformat(),
                'signal_expiry': integrated_signal.expiry.isoformat()
            }
            
            # Cache the enhanced signal
            redis_manager.set_cache('latest_ultra_alpha_signal', legacy_format_signal, ttl=600)
            
            logger.info(f"Ultra Alpha signal generated: {integrated_signal.signal_strength:.3f} "
                       f"confidence: {integrated_signal.confidence:.3f}, "
                       f"edge: {integrated_signal.expected_edge_bps:.1f}bps")
            
            return legacy_format_signal
            
        except Exception as e:
            logger.error(f"Ultra Alpha signal generation failed: {e}")
            return self.fallback_signal_fusion(technical_signals, sentiment_score)
    
    def convert_to_market_data_format(self, technical_signals: Dict, sentiment_score: float,
                                    price_features: Dict, news_confidence: float) -> Dict:
        """Convert legacy format to market_data format for integration engine"""
        
        # Extract price series from price_features or use defaults
        current_price = price_features.get('current_price', 100.0)
        volatility = price_features.get('volatility', 0.01)
        volume_ratio = price_features.get('volume_ratio', 1.0)
        
        # Generate synthetic price series (in real implementation, use actual historical data)
        price_series = self.generate_synthetic_price_series(current_price, volatility)
        volume_series = [volume_ratio * 1000] * len(price_series)  # Synthetic volume
        
        # Create high/low series from price series
        high_series = [p * (1 + volatility * 0.5) for p in price_series]
        low_series = [p * (1 - volatility * 0.5) for p in price_series]
        
        market_data = {
            'price_series': price_series,
            'volume_series': volume_series,
            'high_series': high_series,
            'low_series': low_series,
            'sentiment_score': sentiment_score,
            'news_confidence': news_confidence,
            'technical_signals': technical_signals
        }
        
        return market_data
    
    def generate_synthetic_price_series(self, current_price: float, volatility: float, length: int = 100) -> List[float]:
        """Generate synthetic price series for testing (replace with real data in production)"""
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(0, volatility, length)
        prices = [current_price]
        
        for r in returns:
            next_price = prices[-1] * (1 + r)
            prices.append(next_price)
        
        return prices
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics and performance metrics"""
        try:
            stats = {
                'model_trained': self.model_trained,
                'models_available': self.technical_model is not None,
                'ultra_alpha_available': ULTRA_ALPHA_AVAILABLE,
                'last_signal': redis_manager.get_cache('latest_ml_signal'),
                'last_ultra_signal': redis_manager.get_cache('latest_ultra_alpha_signal'),
                'feature_importance': self.get_feature_importance()
            }
            
            # Add Ultra Alpha performance if available
            if ULTRA_ALPHA_AVAILABLE:
                ultra_performance = integration_engine.get_performance_summary()
                stats['ultra_alpha_performance'] = ultra_performance
            
            # Get recent feedback
            feedback_stream = redis_manager.read_stream('ml_feedback', count=100)
            if feedback_stream:
                returns = [float(msg['data'].get('actual_return', 0)) for msg in feedback_stream]
                stats['recent_performance'] = {
                    'total_trades': len(returns),
                    'avg_return': np.mean(returns) if returns else 0,
                    'win_rate': sum(1 for r in returns if r > 0) / len(returns) if returns else 0,
                    'sharpe_ratio': np.mean(returns) / np.std(returns) if returns and np.std(returns) > 0 else 0
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Model stats calculation failed: {e}")
            return {'model_trained': False, 'ultra_alpha_available': False}

# Global signal fusion instance
signal_fusion = SignalFusion()