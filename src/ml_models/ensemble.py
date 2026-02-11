"""
ML Ensemble Orchestrator

Combines all ML layers (Forecaster, Classifier, Explainer) into a unified pipeline.
Manages model loading, inference, and result aggregation.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base import (
    ClassifierOutput,
    ExplainerOutput,
    ForecastOutput,
    ModelConfig,
    ModelRegistry,
    ModelStatus,
    ModelType,
)
from .classifier import (
    CatBoostClassifier,
    LightGBMClassifier,
    SignalClassifier,
    TrainingConfig,
    create_classifier,
)
from .explainer import (
    AnalysisContext,
    AnalysisExplainer,
    QwenExplainer,
    RuleBasedExplainer,
    create_explainer,
)
from .features import FeatureEngineer, FeatureSet
from .forecaster import (
    ChronosForecaster,
    LagLlamaForecaster,
    TimeSeriesForecaster,
    create_forecaster,
)
from .model_manager import ModelManager

logger = logging.getLogger(__name__)


@dataclass
class MLPrediction:
    """Combined prediction from all ML layers."""
    
    symbol: str
    success: bool
    
    # Layer outputs
    forecast: Optional[ForecastOutput] = None
    classification: Optional[ClassifierOutput] = None
    explanation: Optional[ExplainerOutput] = None
    
    # Aggregated results
    ml_signal: str = "HOLD"
    ml_confidence: float = 0.0
    ml_score: float = 50.0  # 0-100 scale to integrate with existing system
    
    # Price predictions
    price_trend: str = "neutral"
    price_prediction_5d: Optional[float] = None
    price_prediction_30d: Optional[float] = None
    
    # Explanation
    summary: str = ""
    key_insights: List[str] = field(default_factory=list)

    # Feature set (for explainability)
    feature_set: Optional[FeatureSet] = None
    
    # Performance metrics
    total_inference_time_ms: float = 0.0
    layers_used: List[str] = field(default_factory=list)
    
    # Errors
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "success": self.success,
            "ml_signal": self.ml_signal,
            "ml_confidence": self.ml_confidence,
            "ml_score": self.ml_score,
            "price_trend": self.price_trend,
            "price_prediction_5d": self.price_prediction_5d,
            "price_prediction_30d": self.price_prediction_30d,
            "summary": self.summary,
            "key_insights": self.key_insights,
            "total_inference_time_ms": self.total_inference_time_ms,
            "layers_used": self.layers_used,
            "errors": self.errors,
        }


@dataclass
class EnsembleConfig:
    """Configuration for the ML ensemble."""
    
    # Model selections
    forecaster_model: Optional[ModelType] = None
    classifier_model: Optional[ModelType] = None
    explainer_model: Optional[ModelType] = None
    
    # Paths
    models_dir: Path = Path("models")
    
    # Device settings
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    
    # Layer weights for combining with rule-based scores
    forecast_weight: float = 0.3  # How much forecast influences final score
    classifier_weight: float = 0.5  # How much classifier influences final score
    
    # Feature engineering
    include_technical_features: bool = True
    
    # Inference settings
    forecast_horizon: int = 30
    num_forecast_samples: int = 100
    
    # Fallback behavior
    use_rule_based_fallback: bool = True  # Use rule-based if ML fails


class MLEnsemble:
    """
    Orchestrates all ML layers for stock analysis.
    
    Usage:
        config = EnsembleConfig(
            forecaster_model=ModelType.CHRONOS_T5_BASE,
            classifier_model=ModelType.LIGHTGBM,
            explainer_model=ModelType.QWEN_3B,
        )
        ensemble = MLEnsemble(config)
        ensemble.initialize()
        
        result = ensemble.predict(
            symbol="RELIANCE",
            prices=price_series,
            governance_result=gov,
            financial_result=fin,
            valuation_result=val,
            market_result=mkt,
        )
    """
    
    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()
        self.model_manager = ModelManager(str(self.config.models_dir))
        
        # Layer instances
        self._forecaster: Optional[TimeSeriesForecaster] = None
        self._classifier: Optional[SignalClassifier] = None
        self._explainer: Optional[AnalysisExplainer] = None
        
        # Feature engineer
        self._feature_engineer = FeatureEngineer(
            include_technical=self.config.include_technical_features,
            include_forecast=self.config.forecaster_model is not None
        )
        
        self._initialized = False
        self._logger = logging.getLogger(self.__class__.__name__)
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all layers."""
        return {
            "initialized": self._initialized,
            "forecaster": {
                "model": self.config.forecaster_model.value if self.config.forecaster_model else None,
                "ready": self._forecaster.is_ready if self._forecaster else False,
            },
            "classifier": {
                "model": self.config.classifier_model.value if self.config.classifier_model else None,
                "ready": self._classifier.is_ready if self._classifier else False,
            },
            "explainer": {
                "model": self.config.explainer_model.value if self.config.explainer_model else None,
                "ready": self._explainer.is_ready if self._explainer else False,
            },
            "device": self.config.device,
        }

    def get_classifier_artifacts(self) -> Dict[str, Any]:
        """Expose classifier model and feature names for explainability."""
        if not self._classifier or not self._classifier.is_ready:
            return {}
        return {
            "model": getattr(self._classifier, "_model", None),
            "feature_names": getattr(self._classifier, "_feature_names", []),
        }
    
    def initialize(self, load_models: bool = True) -> Tuple[bool, List[str]]:
        """
        Initialize the ensemble by setting up all layers.
        
        Args:
            load_models: Whether to load models into memory immediately
            
        Returns:
            (success, list of warnings/errors)
        """
        messages = []
        success = True
        
        # Initialize forecaster
        if self.config.forecaster_model:
            try:
                model_path = self.model_manager.get_model_path(self.config.forecaster_model)
                self._forecaster = create_forecaster(
                    self.config.forecaster_model,
                    model_path,
                    self.config.device
                )
                
                if load_models:
                    if self._forecaster.load_model():
                        messages.append(f"Forecaster loaded: {self.config.forecaster_model.value}")
                    else:
                        messages.append(f"Warning: Could not load forecaster model")
            except Exception as e:
                messages.append(f"Error initializing forecaster: {e}")
                self._logger.error(f"Forecaster init failed: {e}")
        
        # Initialize classifier
        if self.config.classifier_model:
            try:
                model_path = self.model_manager.get_model_path(self.config.classifier_model)
                self._classifier = create_classifier(
                    self.config.classifier_model,
                    model_path
                )
                
                if load_models:
                    if self._classifier.load_model():
                        messages.append(f"Classifier loaded: {self.config.classifier_model.value}")
                    else:
                        messages.append(f"Note: Classifier needs training - no pre-trained model found")
            except Exception as e:
                messages.append(f"Error initializing classifier: {e}")
                self._logger.error(f"Classifier init failed: {e}")
        
        # Initialize explainer
        if self.config.explainer_model:
            try:
                model_path = self.model_manager.get_model_path(self.config.explainer_model)
                self._explainer = create_explainer(
                    self.config.explainer_model,
                    model_path,
                    self.config.device
                )
                
                if load_models:
                    if self._explainer.load_model():
                        messages.append(f"Explainer loaded: {self.config.explainer_model.value}")
                    else:
                        messages.append(f"Warning: Could not load explainer model")
            except Exception as e:
                messages.append(f"Error initializing explainer: {e}")
                self._logger.error(f"Explainer init failed: {e}")
        
        # Setup rule-based fallback explainer if no LLM
        if not self._explainer and self.config.use_rule_based_fallback:
            self._explainer = RuleBasedExplainer(ModelConfig(
                model_type=ModelType.QWEN_3B,  # Placeholder
                model_path=Path(""),
                device="cpu"
            ))
            self._explainer.load_model()
            messages.append("Using rule-based explainer (no LLM model)")
        
        self._initialized = True
        return success, messages
    
    def predict(
        self,
        symbol: str,
        prices: Union[pd.Series, pd.DataFrame],
        governance_result: Optional[Any] = None,
        financial_result: Optional[Any] = None,
        valuation_result: Optional[Any] = None,
        market_result: Optional[Any] = None,
        macro_data: Optional[Dict[str, Any]] = None,
        company_name: str = "",
        current_signal: str = "HOLD",
        current_confidence: float = 0.5,
        red_flags: Optional[List[str]] = None,
    ) -> MLPrediction:
        """
        Run full ML prediction pipeline.
        
        Args:
            symbol: Stock symbol
            prices: Historical closing prices (Series or DataFrame with 'close' column)
            governance_result: Output from GovernanceAnalyzer
            financial_result: Output from FinancialAnalyzer
            valuation_result: Output from ValuationAnalyzer
            market_result: Output from MarketBehaviourAnalyzer
            company_name: Company name for explanations
            current_signal: Current signal from rule-based system
            current_confidence: Current confidence from rule-based system
            red_flags: List of red flags
        """
        start_time = time.time()
        prediction = MLPrediction(symbol=symbol, success=False)
        
        # Convert DataFrame to Series if needed
        if isinstance(prices, pd.DataFrame):
            if 'close' in prices.columns:
                price_series = prices['close']
            elif 'Close' in prices.columns:
                price_series = prices['Close']
            else:
                # Assume first numeric column
                price_series = prices.select_dtypes(include=[np.number]).iloc[:, 0]
        else:
            price_series = prices
        
        if not self._initialized:
            success, messages = self.initialize()
            if not success:
                prediction.errors.extend(messages)
                return prediction
        
        forecast_result = None
        classifier_result = None
        explainer_result = None
        
        # Step 1: Run forecaster
        if self._forecaster:
            try:
                forecast_result = self._forecaster.forecast(
                    price_series,
                    horizon=self.config.forecast_horizon,
                    num_samples=self.config.num_forecast_samples
                )
                
                if forecast_result.success:
                    prediction.forecast = forecast_result
                    prediction.price_trend = forecast_result.trend_direction
                    prediction.layers_used.append("forecaster")
                    
                    # Extract price predictions
                    if forecast_result.predictions is not None:
                        current_price = float(price_series.iloc[-1])
                        if len(forecast_result.predictions) >= 5:
                            prediction.price_prediction_5d = float(forecast_result.predictions[4])
                        if len(forecast_result.predictions) >= 30:
                            prediction.price_prediction_30d = float(forecast_result.predictions[29])
                else:
                    prediction.errors.append(f"Forecast failed: {forecast_result.error}")
                    
            except Exception as e:
                prediction.errors.append(f"Forecaster error: {e}")
                self._logger.error(f"Forecaster failed: {e}")
        
        # Step 2: Extract features (only if needed by classifier/explainer)
        feature_set = None
        if self._classifier or self._explainer:
            feature_set = self._feature_engineer.extract_features(
                symbol=symbol,
                prices=price_series,
                governance_result=governance_result,
                financial_result=financial_result,
                valuation_result=valuation_result,
                market_result=market_result,
                forecast_result=forecast_result,
                macro_data=macro_data,
            )
            prediction.feature_set = feature_set
        
        # Step 3: Run classifier
        if self._classifier and self._classifier.is_ready and feature_set:
            try:
                classifier_result = self._classifier.classify(feature_set.features)
                
                if classifier_result.success:
                    prediction.classification = classifier_result
                    prediction.ml_signal = classifier_result.signal
                    prediction.ml_confidence = classifier_result.confidence
                    prediction.layers_used.append("classifier")
                    
                    # Convert confidence to score (0-100)
                    # Weight by signal type
                    signal_base_scores = {"BUY": 80, "HOLD": 55, "AVOID": 35, "SELL": 20}
                    base_score = signal_base_scores.get(classifier_result.signal, 50)
                    confidence_adjustment = (classifier_result.confidence - 0.5) * 20
                    prediction.ml_score = np.clip(base_score + confidence_adjustment, 0, 100)
                else:
                    prediction.errors.append(f"Classifier failed: {classifier_result.error}")
                    # Use rule-based signal as fallback
                    prediction.ml_signal = current_signal
                    prediction.ml_confidence = current_confidence
                    
            except Exception as e:
                prediction.errors.append(f"Classifier error: {e}")
                self._logger.error(f"Classifier failed: {e}")
                prediction.ml_signal = current_signal
                prediction.ml_confidence = current_confidence
        else:
            # No classifier - use rule-based
            prediction.ml_signal = current_signal
            # Normalize confidence if it's on 0-100 scale
            if current_confidence > 1.0:
                prediction.ml_confidence = current_confidence / 100.0
            else:
                prediction.ml_confidence = current_confidence
            prediction.ml_score = prediction.ml_confidence * 100
        
        # Step 4: Generate explanation
        if self._explainer and feature_set:
            try:
                # Build context
                context = AnalysisContext(
                    symbol=symbol,
                    company_name=company_name or symbol,
                    signal=prediction.ml_signal,
                    confidence=prediction.ml_confidence,
                    governance_score=getattr(governance_result, 'overall_score', 50) if governance_result else 50,
                    financial_score=getattr(financial_result, 'overall_score', 50) if financial_result else 50,
                    valuation_score=getattr(valuation_result, 'overall_score', 50) if valuation_result else 50,
                    market_score=getattr(market_result, 'overall_score', 50) if market_result else 50,
                    composite_score=prediction.ml_score,
                    metrics=feature_set.features,
                    forecast_trend=prediction.price_trend,
                    forecast_strength=forecast_result.trend_strength if forecast_result and forecast_result.success else 0.0,
                    price_prediction_30d=prediction.price_prediction_30d,
                    red_flags=red_flags or [],
                    signal_probabilities=classifier_result.signal_probabilities if classifier_result and classifier_result.success else None,
                )
                
                explainer_result = self._explainer.explain(context)
                
                if explainer_result.success:
                    prediction.explanation = explainer_result
                    prediction.summary = explainer_result.summary
                    prediction.key_insights = (
                        explainer_result.key_points[:3] +
                        explainer_result.risk_factors[:2]
                    )
                    prediction.layers_used.append("explainer")
                else:
                    prediction.errors.append(f"Explainer failed: {explainer_result.error}")
                    
            except Exception as e:
                prediction.errors.append(f"Explainer error: {e}")
                self._logger.error(f"Explainer failed: {e}")
        
        # Calculate total inference time
        prediction.total_inference_time_ms = (time.time() - start_time) * 1000
        
        # Set success if at least one layer worked
        prediction.success = len(prediction.layers_used) > 0 or self.config.use_rule_based_fallback
        
        return prediction
    
    def train_classifier(
        self,
        training_data: List[Dict[str, Any]],
        labels: pd.Series,
        training_config: Optional[TrainingConfig] = None
    ) -> Tuple[bool, str]:
        """
        Train the classifier on historical data.
        
        Args:
            training_data: List of dicts with stock data and engine results
            labels: Series mapping symbol to signal label
            training_config: Optional training configuration
            
        Returns:
            (success, message)
        """
        if not self._classifier:
            return False, "No classifier configured"
        
        try:
            # Create training dataset
            X, y = self._feature_engineer.create_training_dataset(training_data, labels)
            
            if len(X) < 50:
                return False, f"Insufficient training data: {len(X)} samples (need at least 50)"
            
            # Train
            result = self._classifier.train(X, y)
            
            if result.success:
                return True, (
                    f"Training complete. Accuracy: {result.metrics.get('accuracy', 0):.2%}, "
                    f"F1: {result.metrics.get('f1_weighted', 0):.2%}"
                )
            else:
                return False, f"Training failed: {result.error}"
                
        except Exception as e:
            return False, f"Training error: {e}"
    
    def unload_all(self) -> None:
        """Unload all models to free memory."""
        if self._forecaster:
            self._forecaster.unload_model()
        if self._classifier:
            self._classifier.unload_model()
        if self._explainer:
            self._explainer.unload_model()
        
        self._initialized = False
        self._logger.info("All models unloaded")
    
    def get_download_commands(self) -> str:
        """Get shell commands to download all configured models."""
        commands = ["# ML Model Download Commands", ""]
        
        if self.config.forecaster_model:
            commands.append(f"# Layer 1: Forecaster - {self.config.forecaster_model.value}")
            commands.append(self.model_manager.get_download_command(self.config.forecaster_model))
            commands.append("")
        
        if self.config.classifier_model:
            commands.append(f"# Layer 2: Classifier - {self.config.classifier_model.value}")
            commands.append("# Trained locally - no download needed")
            commands.append("")
        
        if self.config.explainer_model:
            commands.append(f"# Layer 3: Explainer - {self.config.explainer_model.value}")
            commands.append(self.model_manager.get_download_command(self.config.explainer_model))
            commands.append("")
        
        return "\n".join(commands)


class FundamentalAwareEnsemble:
    """
    Two-layer ensemble that combines technical + fundamental signals.
    
    Layer 1: Uses technical/price features only (momentum, RSI, MACD, etc.)
    Layer 2: Meta-model that combines Layer 1 predictions with fundamental features
    
    This architecture ensures that:
    1. Technical signals are captured without fundamental bias
    2. Fundamentals can override technical signals when appropriate
    3. The model doesn't over-rely on any single feature type
    """
    
    # Technical features (price-based)
    TECHNICAL_FEATURES = [
        "return_1d", "return_5d", "return_20d", "return_60d", "return_252d",
        "volatility_20d", "volatility_60d",
        "momentum_10d", "momentum_30d",
        "sma_ratio_20", "sma_ratio_50", "sma_ratio_200",
        "high_52w_pct", "low_52w_pct",
        "rsi_14", "rsi_28",
        "macd", "macd_signal", "macd_histogram",
        "bb_position", "atr_14", "adx_14",
    ]
    
    # Fundamental features
    FUNDAMENTAL_FEATURES = [
        "years_listed", "promoter_holding", "pledge_ratio",
        "dividend_consistency", "auditor_stability_score", "governance_score",
        "revenue_cagr_3y", "revenue_cagr_5y",
        "pat_cagr_3y", "pat_cagr_5y",
        "roce", "fcf_yield", "debt_to_equity",
        "operating_margin", "earnings_quality", "financial_score",
        "pe_ratio", "pe_percentile",
        "pb_ratio", "pb_percentile",
        "ev_ebitda", "peg_ratio", "valuation_score",
        "max_drawdown", "avg_recovery_months",
        "volatility_1y", "volatility_3y",
        "beta", "sharpe_ratio",
        "relative_performance_1y", "market_score",
    ]
    
    def __init__(self, n_classes: int = 4):
        """
        Initialize two-layer ensemble.
        
        Args:
            n_classes: Number of output classes (BUY, HOLD, AVOID, SELL = 4)
        """
        self.n_classes = n_classes
        self.layer1_model = None  # Technical features only
        self.layer2_model = None  # Meta-model: L1 probs + fundamentals
        self._is_trained = False
        self._feature_names_l1 = []
        self._feature_names_l2 = []
        self._logger = logging.getLogger(self.__class__.__name__)
    
    @property
    def is_ready(self) -> bool:
        return self._is_trained and self.layer1_model is not None and self.layer2_model is not None
    
    def _get_technical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract technical features from full feature set."""
        available = [c for c in X.columns if c in self.TECHNICAL_FEATURES]
        return X[available]
    
    def _get_fundamental_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract fundamental features from full feature set."""
        available = [c for c in X.columns if c in self.FUNDAMENTAL_FEATURES]
        return X[available]
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train two-layer ensemble.
        
        Args:
            X: Full feature DataFrame
            y: Labels (encoded as integers)
            
        Returns:
            Training metrics dictionary
        """
        try:
            import lightgbm as lgb
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            lgb = None
        
        # Split features
        X_technical = self._get_technical_features(X)
        X_fundamental = self._get_fundamental_features(X)
        
        self._feature_names_l1 = list(X_technical.columns)
        
        # Train Layer 1: Technical features only
        self._logger.info(f"Training Layer 1 with {len(X_technical.columns)} technical features...")
        
        if lgb:
            self.layer1_model = lgb.LGBMClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=31,
                min_child_samples=20,
                class_weight='balanced',
                random_state=42,
                verbose=-1,
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            self.layer1_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42,
            )
        
        self.layer1_model.fit(X_technical.fillna(0), y)
        
        # Get Layer 1 predictions as features
        l1_probs = self.layer1_model.predict_proba(X_technical.fillna(0))
        l1_prob_df = pd.DataFrame(
            l1_probs, 
            columns=[f'l1_prob_{i}' for i in range(l1_probs.shape[1])],
            index=X.index
        )
        
        # Create Layer 2 features: L1 predictions + fundamental features
        X_layer2 = pd.concat([l1_prob_df.reset_index(drop=True), 
                              X_fundamental.reset_index(drop=True)], axis=1)
        
        self._feature_names_l2 = list(X_layer2.columns)
        
        # Train Layer 2: Meta-model
        self._logger.info(f"Training Layer 2 with {len(X_layer2.columns)} meta-features...")
        
        if lgb:
            self.layer2_model = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=6,
                num_leaves=31,
                min_child_samples=15,
                class_weight='balanced',
                random_state=42,
                verbose=-1,
            )
        else:
            self.layer2_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
            )
        
        self.layer2_model.fit(X_layer2.fillna(0), y.reset_index(drop=True))
        
        self._is_trained = True
        
        # Calculate metrics
        y_pred = self.predict(X)
        from sklearn.metrics import accuracy_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'f1_weighted': f1_score(y, y_pred, average='weighted'),
            'layer1_features': len(X_technical.columns),
            'layer2_features': len(X_layer2.columns),
            'training_samples': len(X),
        }
        
        self._logger.info(f"Training complete. Accuracy: {metrics['accuracy']:.2%}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Two-stage prediction.
        
        Args:
            X: Full feature DataFrame
            
        Returns:
            Predicted class labels
        """
        if not self.is_ready:
            raise ValueError("Model not trained. Call train() first.")
        
        X_technical = self._get_technical_features(X)
        X_fundamental = self._get_fundamental_features(X)
        
        # Layer 1 predictions
        l1_probs = self.layer1_model.predict_proba(X_technical.fillna(0))
        l1_prob_df = pd.DataFrame(
            l1_probs,
            columns=[f'l1_prob_{i}' for i in range(l1_probs.shape[1])],
            index=X.index
        )
        
        # Create Layer 2 input
        X_layer2 = pd.concat([l1_prob_df.reset_index(drop=True),
                              X_fundamental.reset_index(drop=True)], axis=1)
        
        # Layer 2 prediction
        return self.layer2_model.predict(X_layer2.fillna(0))
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions from Layer 2."""
        if not self.is_ready:
            raise ValueError("Model not trained. Call train() first.")
        
        X_technical = self._get_technical_features(X)
        X_fundamental = self._get_fundamental_features(X)
        
        l1_probs = self.layer1_model.predict_proba(X_technical.fillna(0))
        l1_prob_df = pd.DataFrame(
            l1_probs,
            columns=[f'l1_prob_{i}' for i in range(l1_probs.shape[1])],
            index=X.index
        )
        
        X_layer2 = pd.concat([l1_prob_df.reset_index(drop=True),
                              X_fundamental.reset_index(drop=True)], axis=1)
        
        return self.layer2_model.predict_proba(X_layer2.fillna(0))
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """
        Get feature importance from both layers.
        
        Returns:
            Dictionary with 'layer1' and 'layer2' importance dictionaries
        """
        if not self.is_ready:
            return {'layer1': {}, 'layer2': {}}
        
        result = {}
        
        # Layer 1 importance
        try:
            l1_importance = dict(zip(
                self._feature_names_l1,
                self.layer1_model.feature_importances_
            ))
            result['layer1'] = dict(sorted(
                l1_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
        except:
            result['layer1'] = {}
        
        # Layer 2 importance
        try:
            l2_importance = dict(zip(
                self._feature_names_l2,
                self.layer2_model.feature_importances_
            ))
            result['layer2'] = dict(sorted(
                l2_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
        except:
            result['layer2'] = {}
        
        return result
    
    def save(self, path: Path) -> bool:
        """Save both layer models."""
        import pickle
        
        try:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'layer1_model': self.layer1_model,
                'layer2_model': self.layer2_model,
                'feature_names_l1': self._feature_names_l1,
                'feature_names_l2': self._feature_names_l2,
                'n_classes': self.n_classes,
            }
            
            with open(path / 'two_layer_ensemble.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            self._logger.info(f"Model saved to {path}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to save model: {e}")
            return False
    
    def load(self, path: Path) -> bool:
        """Load both layer models."""
        import pickle
        
        try:
            model_file = Path(path) / 'two_layer_ensemble.pkl'
            
            if not model_file.exists():
                return False
            
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.layer1_model = model_data['layer1_model']
            self.layer2_model = model_data['layer2_model']
            self._feature_names_l1 = model_data['feature_names_l1']
            self._feature_names_l2 = model_data['feature_names_l2']
            self.n_classes = model_data.get('n_classes', 4)
            self._is_trained = True
            
            self._logger.info(f"Model loaded from {path}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to load model: {e}")
            return False


def create_ensemble_from_config(config_dict: Dict[str, Any]) -> MLEnsemble:
    """
    Create ensemble from configuration dictionary.
    
    Expected format:
    {
        "forecaster": "chronos-t5-base",  # or "lag-llama", null
        "classifier": "lightgbm",          # or "catboost", null
        "explainer": "qwen2.5-3b",         # or "qwen2.5-7b", null
        "device": "auto",
        "models_dir": "models",
    }
    """
    # Map string names to ModelType
    forecaster_map = {
        "chronos-t5-tiny": ModelType.CHRONOS_T5_TINY,
        "chronos-t5-small": ModelType.CHRONOS_T5_SMALL,
        "chronos-t5-base": ModelType.CHRONOS_T5_BASE,
        "lag-llama": ModelType.LAG_LLAMA,
    }
    
    classifier_map = {
        "lightgbm": ModelType.LIGHTGBM,
        "catboost": ModelType.CATBOOST,
        "xgboost": ModelType.XGBOOST,
    }
    
    explainer_map = {
        "qwen2.5-3b": ModelType.QWEN_3B,
        "qwen2.5-7b": ModelType.QWEN_7B,
        "phi-3-mini": ModelType.PHI3_MINI,
    }
    
    forecaster = config_dict.get("forecaster")
    classifier = config_dict.get("classifier")
    explainer = config_dict.get("explainer")
    
    config = EnsembleConfig(
        forecaster_model=forecaster_map.get(forecaster) if forecaster else None,
        classifier_model=classifier_map.get(classifier) if classifier else None,
        explainer_model=explainer_map.get(explainer) if explainer else None,
        device=config_dict.get("device", "auto"),
        models_dir=Path(config_dict.get("models_dir", "models")),
    )
    
    return MLEnsemble(config)
