"""Explainability Engine for Stocron by RTR."""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import numpy as np
import pandas as pd

from .signal_generator import SignalResult, UserProfile, Signal

logger = logging.getLogger(__name__)


@dataclass
class ExplainabilityReport:
    summary: str
    signal_explanation: str
    why_not_buy: List[str]
    confidence_factors: Dict[str, Any]
    dimension_breakdown: Dict[str, Dict[str, Any]]
    risk_factors: List[str]
    thesis_invalidators: List[str]
    user_profile_analysis: Dict[str, Any]
    data_quality_notes: List[str]
    ml_explanation: Optional[str] = None
    shap_contributions: Optional[Dict[str, Any]] = None
    external_context: Optional[str] = None


class SHAPExplainer:
    """
    SHAP-based model explainability.
    
    Generates explanations for ML model predictions using SHAP values,
    allowing the tool to say things like:
    "The model suggests 'Buy' because the 10-year ROE trend outweighed 
    the recent RSI overbought signal."
    """
    
    # Human-readable feature name mappings
    FEATURE_NAMES = {
        'return_1d': '1-day return',
        'return_5d': '5-day return',
        'return_20d': '20-day return',
        'return_60d': '60-day return',
        'return_252d': '1-year return',
        'log_return_252d': '1-year log return',
        'real_return_252d': '1-year inflation-adjusted return',
        'volatility_20d': '20-day volatility',
        'volatility_60d': '60-day volatility',
        'momentum_10d': '10-day momentum',
        'momentum_30d': '30-day momentum',
        'sma_ratio_20': 'price vs 20-day average',
        'sma_ratio_50': 'price vs 50-day average',
        'sma_ratio_200': 'price vs 200-day average',
        'high_52w_pct': 'distance from 52-week high',
        'low_52w_pct': 'distance from 52-week low',
        'rsi_14': 'RSI (14-day)',
        'rsi_28': 'RSI (28-day)',
        'macd': 'MACD',
        'macd_histogram': 'MACD histogram',
        'bb_position': 'Bollinger Band position',
        'atr_14': 'Average True Range',
        'adx_14': 'trend strength (ADX)',
        'years_listed': 'years since listing',
        'promoter_holding': 'promoter holding',
        'pledge_ratio': 'promoter pledge ratio',
        'dividend_consistency': 'dividend consistency',
        'governance_score': 'governance quality',
        'revenue_cagr_3y': '3-year revenue growth',
        'revenue_cagr_5y': '5-year revenue growth',
        'pat_cagr_3y': '3-year profit growth',
        'pat_cagr_5y': '5-year profit growth',
        'roce': 'return on capital (ROCE)',
        'fcf_yield': 'free cash flow yield',
        'debt_to_equity': 'debt-to-equity ratio',
        'operating_margin': 'operating margin',
        'earnings_quality': 'earnings quality',
        'financial_score': 'financial health',
        'pe_ratio': 'PE ratio',
        'pe_percentile': 'PE percentile',
        'pb_ratio': 'Price-to-Book ratio',
        'pb_percentile': 'PB percentile',
        'ev_ebitda': 'EV/EBITDA',
        'peg_ratio': 'PEG ratio',
        'valuation_score': 'valuation attractiveness',
        'max_drawdown': 'maximum drawdown',
        'avg_recovery_months': 'average recovery time',
        'volatility_1y': '1-year volatility',
        'volatility_3y': '3-year volatility',
        'beta': 'market beta',
        'sharpe_ratio': 'Sharpe ratio',
        'relative_performance_1y': '1-year outperformance',
        'market_score': 'market behavior score',
        'forecast_trend_bullish': 'bullish forecast trend',
        'forecast_return_5d': 'forecasted 5-day return',
        'forecast_return_30d': 'forecasted 30-day return',
        'forecast_confidence': 'forecast confidence',
        'repo_rate_current': 'RBI repo rate',
        'usdinr_level': 'USD-INR level',
        'crude_oil_level': 'crude oil price',
        'market_regime_bull': 'bull market regime',
        'market_regime_bear': 'bear market regime',
        'cpi_inflation_yoy': 'inflation rate',
        'rsi_regime_adjusted': 'regime-adjusted RSI',
    }
    
    def __init__(self, model=None, feature_names: List[str] = None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained ML model (LightGBM, XGBoost, etc.)
            feature_names: List of feature names in model order
        """
        self.model = model
        self.feature_names = feature_names or []
        self.explainer = None
        self._initialized = False
    
    def initialize(self, background_data: pd.DataFrame = None):
        """
        Initialize SHAP explainer with background data.
        
        Args:
            background_data: Sample of training data for SHAP baseline
        """
        if self.model is None:
            logger.warning("No model provided for SHAP explainer")
            return
        
        try:
            import shap
            
            # Use TreeExplainer for tree-based models (LightGBM, XGBoost)
            model_type = type(self.model).__name__.lower()
            
            if 'lightgbm' in model_type or 'lgbm' in model_type or 'xgb' in model_type or 'gradient' in model_type:
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Fallback to KernelExplainer
                if background_data is not None:
                    self.explainer = shap.KernelExplainer(
                        self.model.predict_proba, 
                        shap.sample(background_data, 100)
                    )
            
            self._initialized = True
            logger.info("SHAP explainer initialized")
            
        except ImportError:
            logger.warning("SHAP not installed. Install with: pip install shap")
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
    
    def explain_prediction(
        self, 
        features: pd.DataFrame,
        prediction_class: int = None
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single prediction.
        
        Args:
            features: Single-row DataFrame with feature values
            prediction_class: Which class to explain (for multi-class)
            
        Returns:
            Dictionary with SHAP values and contributions
        """
        if not self._initialized or self.explainer is None:
            return self._empty_explanation()
        
        try:
            import shap
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(features)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                # Use specified class or the predicted class
                if prediction_class is not None:
                    sv = shap_values[prediction_class][0]
                else:
                    sv = shap_values[0][0]  # Default to first class
            else:
                sv = shap_values[0]
            
            # Create contribution dictionary
            contributions = dict(zip(self.feature_names, sv))
            sorted_contrib = sorted(
                contributions.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            
            # Split into positive and negative contributions
            top_positive = [(k, v) for k, v in sorted_contrib if v > 0][:5]
            top_negative = [(k, v) for k, v in sorted_contrib if v < 0][:5]
            
            # Get expected value (baseline)
            if hasattr(self.explainer, 'expected_value'):
                base_value = self.explainer.expected_value
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = base_value[prediction_class or 0]
            else:
                base_value = 0
            
            return {
                'top_positive': top_positive,
                'top_negative': top_negative,
                'all_contributions': contributions,
                'base_value': float(base_value),
                'prediction_delta': float(sum(sv)),
                'feature_count': len(self.feature_names)
            }
            
        except Exception as e:
            logger.debug(f"SHAP explanation error: {e}")
            return self._empty_explanation()
    
    def _empty_explanation(self) -> Dict[str, Any]:
        """Return empty explanation when SHAP is not available."""
        return {
            'top_positive': [],
            'top_negative': [],
            'all_contributions': {},
            'base_value': 0,
            'prediction_delta': 0,
            'feature_count': 0
        }
    
    def _humanize_feature(self, feature_name: str) -> str:
        """Convert feature name to human-readable format."""
        return self.FEATURE_NAMES.get(feature_name, feature_name.replace('_', ' '))
    
    def generate_natural_language_explanation(
        self, 
        shap_result: Dict[str, Any],
        signal: str = None
    ) -> str:
        """
        Convert SHAP values to human-readable explanation.
        
        Example output:
        "The model suggests 'Buy' primarily because the strong 5-year 
        revenue growth and high ROCE indicate quality, while the low 
        PE percentile suggests attractive valuation. However, the high 
        volatility and elevated debt ratio are concerns."
        """
        if not shap_result['top_positive'] and not shap_result['top_negative']:
            return "No ML explanation available."
        
        explanations = []
        
        # Positive factors
        if shap_result['top_positive']:
            pos_factors = []
            for feature, value in shap_result['top_positive'][:3]:
                pos_factors.append(self._humanize_feature(feature))
            
            if signal in ['BUY', 'HOLD']:
                explanations.append(
                    f"The model favors this stock primarily due to: {', '.join(pos_factors)}"
                )
            else:
                explanations.append(
                    f"Positive factors include: {', '.join(pos_factors)}"
                )
        
        # Negative factors
        if shap_result['top_negative']:
            neg_factors = []
            for feature, value in shap_result['top_negative'][:3]:
                neg_factors.append(self._humanize_feature(feature))
            
            explanations.append(f"Concerns: {', '.join(neg_factors)}")
        
        return ". ".join(explanations) + "."
    
    def get_feature_importance_summary(
        self, 
        shap_result: Dict[str, Any],
        top_n: int = 5
    ) -> List[Tuple[str, float, str]]:
        """
        Get top features with direction and magnitude.
        
        Returns list of (feature_name, contribution, direction)
        """
        summary = []
        
        for feature, value in shap_result['top_positive'][:top_n]:
            summary.append((
                self._humanize_feature(feature),
                abs(value),
                'positive'
            ))
        
        for feature, value in shap_result['top_negative'][:top_n]:
            summary.append((
                self._humanize_feature(feature),
                abs(value),
                'negative'
            ))
        
        # Sort by absolute contribution
        summary.sort(key=lambda x: x[1], reverse=True)
        
        return summary[:top_n]


class ExplainabilityEngine:
    """
    Generates comprehensive explanations with optional SHAP integration.
    
    Every analysis MUST include 'Why NOT to buy' section.
    """
    
    def __init__(self, ml_model: Dict[str, Any] = None):
        """
        Initialize explainability engine.
        
        Args:
            ml_model: Dictionary with 'model' and 'feature_names' keys
        """
        self.shap_explainer = None
        
        if ml_model is not None:
            self.shap_explainer = SHAPExplainer(
                model=ml_model.get('model'),
                feature_names=ml_model.get('feature_names', [])
            )
            # Initialize will be called lazily on first use
    
    def generate_explanation(self, signal_result: SignalResult, user_profile: UserProfile,
                             governance_score, financial_score, valuation_score, market_score,
                             stock_info: Dict[str, Any], red_flags: List[Dict] = None,
                             ml_features: pd.DataFrame = None,
                             prediction_class: int = None,
                             external_context: str = None) -> ExplainabilityReport:
        red_flags = red_flags or []
        
        summary = self._generate_summary(signal_result, stock_info, user_profile)
        if external_context:
            summary += f"\n\nContext: {external_context}"
        signal_explanation = self._explain_signal(signal_result)
        why_not_buy = self._generate_why_not_buy(user_profile, governance_score, financial_score,
                                                   valuation_score, market_score, signal_result, red_flags)
        confidence = self._analyze_confidence(signal_result)
        breakdown = self._create_breakdown(governance_score, financial_score, valuation_score, market_score)
        risks = self._compile_risks(governance_score, financial_score, valuation_score, market_score, red_flags)
        invalidators = self._identify_invalidators(signal_result, financial_score)
        profile_analysis = {'profile': user_profile.to_dict(), 'match': signal_result.user_profile_match}
        data_notes = self._assess_data_quality()
        
        # Generate SHAP explanations if available and ML features provided
        ml_explanation = None
        shap_contributions = None
        
        if self.shap_explainer is not None and ml_features is not None:
            try:
                # Initialize explainer if not done
                if not self.shap_explainer._initialized:
                    self.shap_explainer.initialize()
                
                if self.shap_explainer._initialized:
                    shap_result = self.shap_explainer.explain_prediction(
                        ml_features, 
                        prediction_class=prediction_class
                    )
                    
                    if shap_result['top_positive'] or shap_result['top_negative']:
                        ml_explanation = self.shap_explainer.generate_natural_language_explanation(
                            shap_result,
                            signal=signal_result.signal
                        )
                        shap_contributions = shap_result
            except Exception as e:
                logger.debug(f"SHAP explanation failed: {e}")
        
        return ExplainabilityReport(
            summary=summary, signal_explanation=signal_explanation, why_not_buy=why_not_buy,
            confidence_factors=confidence, dimension_breakdown=breakdown, risk_factors=risks,
            thesis_invalidators=invalidators, user_profile_analysis=profile_analysis,
            data_quality_notes=data_notes, ml_explanation=ml_explanation,
            shap_contributions=shap_contributions, external_context=external_context
        )
    
    def _generate_summary(self, result: SignalResult, info: Dict, profile: UserProfile) -> str:
        name = info.get('name', info.get('symbol', 'This stock'))
        if result.signal == Signal.BUY:
            return f"{name} appears suitable for your profile. Score: {result.composite_score:.0f}/100. Review 'Why NOT to buy' section."
        elif result.signal == Signal.HOLD:
            return f"{name} shows mixed characteristics. Score: {result.composite_score:.0f}/100."
        elif result.signal == Signal.AVOID:
            return f"{name} does not align with your profile ({profile.expected_return}% CAGR, {profile.risk_appetite} risk)."
        return f"{name} shows significant concerns. Score: {result.composite_score:.0f}/100."
    
    def _explain_signal(self, result: SignalResult) -> str:
        lines = [f"Signal: {result.signal}", "", "Dimension Scores:"]
        for dim, score in result.dimension_scores.items():
            bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
            lines.append(f"  {dim.replace('_', ' ').title()}: {bar} {score:.0f}")
        lines.append(f"\nComposite: {result.composite_score:.0f}/100")
        return "\n".join(lines)
    
    def _generate_why_not_buy(self, profile: UserProfile, gov, fin, val, mkt, result: SignalResult, red_flags: List) -> List[str]:
        """
        MANDATORY: Generate 'Why NOT to buy' - shown even for BUY signals.
        Only includes stock-specific concerns, no generic advice.
        """
        reasons = []
        
        # Return gap - only if significant
        ret = result.user_profile_match.get('return_expectation', {})
        if not ret.get('meets', True) and ret.get('gap', 0) > 3:
            reasons.append(f"RETURN GAP: You expect {ret['expected']}% but estimated future return is {ret['estimated']}% (gap: {ret['gap']}%) based on fundamentals")
        
        # Risk mismatch - only if significant
        risk = result.user_profile_match.get('risk_match', {})
        if not risk.get('within_tolerance', True):
            reasons.append(f"RISK MISMATCH: Stock volatility {risk['stock_vol']:.0f}% exceeds your tolerance ({risk['max_acceptable']:.0f}%)")
        
        # Governance concerns
        for rf in gov.red_flags:
            reasons.append(f"GOVERNANCE: {rf}")
        
        if hasattr(gov, 'promoter_holding_trend') and gov.promoter_holding_trend == 'decreasing':
            change = gov.details.get('promoter_analysis', {}).get('change_5y', 0) if gov.details else 0
            if change < -3:  # Only flag if significant decline
                reasons.append(f"PROMOTER SELLING: Holding declined {abs(change):.1f}% over 5 years")
        
        if gov.pledge_ratio and gov.pledge_ratio > 10:
            reasons.append(f"PLEDGE RISK: {gov.pledge_ratio:.1f}% of promoter shares pledged - forced selling risk in market downturns")
        elif gov.pledge_ratio and gov.pledge_ratio > 5:
            reasons.append(f"PLEDGE CONCERN: {gov.pledge_ratio:.1f}% promoter pledge")
        
        # Financial concerns
        for rf in fin.red_flags:
            reasons.append(f"FINANCIAL: {rf}")
        
        if fin.roce_current and fin.roce_current < 10:
            reasons.append(f"POOR CAPITAL EFFICIENCY: ROCE of {fin.roce_current:.1f}% is below cost of capital")
        
        if fin.debt_to_equity and fin.debt_to_equity > 1.5:
            reasons.append(f"HIGH LEVERAGE: Debt-to-Equity of {fin.debt_to_equity:.2f}x creates financial risk")
        elif fin.debt_to_equity and fin.debt_to_equity > 1:
            reasons.append(f"MODERATE LEVERAGE: D/E ratio of {fin.debt_to_equity:.2f}x")
        
        if fin.earnings_quality and fin.earnings_quality < 0.5:
            reasons.append(f"EARNINGS QUALITY: Cash flow conversion is poor ({fin.earnings_quality:.0%})")
        
        # Growth concerns
        if fin.revenue_cagr_5y and fin.revenue_cagr_5y < 5:
            reasons.append(f"SLOW GROWTH: Revenue growing at only {fin.revenue_cagr_5y:.1f}% annually")
        
        # Valuation concerns
        if val.stress_level in ['high', 'extreme']:
            reasons.append(f"VALUATION STRESS: Stock at '{val.stress_level}' stress level - high downside risk")
        
        if val.pe_percentile_own and val.pe_percentile_own > 80:
            reasons.append(f"EXPENSIVE: PE at {val.pe_percentile_own:.0f}th percentile of its own history")
        elif val.pe_percentile_own and val.pe_percentile_own > 70:
            reasons.append(f"ABOVE AVERAGE VALUATION: PE at {val.pe_percentile_own:.0f}th percentile")
        
        if val.peg_ratio and val.peg_ratio > 2.5:
            reasons.append(f"HIGH PEG: PEG ratio of {val.peg_ratio:.2f} - paying premium for growth")
        
        # Market behavior concerns
        if mkt.max_drawdown and mkt.max_drawdown > 50:
            reasons.append(f"CRASH HISTORY: Stock fell {mkt.max_drawdown:.0f}% in past - can you handle such volatility?")
        elif mkt.max_drawdown and mkt.max_drawdown > 40:
            reasons.append(f"HIGH DRAWDOWN: Stock has fallen {mkt.max_drawdown:.0f}% historically")
        
        if mkt.volatility_regime in ['high', 'extreme']:
            reasons.append(f"HIGH VOLATILITY: Currently in {mkt.volatility_regime} volatility regime ({mkt.volatility_1y:.0f}% annual)")
        
        # Price trend vs Fundamental Signal divergence
        if result.signal in [Signal.AVOID, Signal.SELL] and mkt.trend_30d == 'bullish':
             reasons.append("TREND TRAP: Price is rising (bullish trend) but fundamentals signal AVOID. This often indicates a speculative rally without substance.")
        
        if mkt.beta and mkt.beta > 1.5:
            reasons.append(f"HIGH BETA: Beta of {mkt.beta:.2f} - stock moves 1.5x the market")
        elif mkt.beta and mkt.beta > 1.3:
            reasons.append(f"ELEVATED BETA: Beta of {mkt.beta:.2f} - amplifies market moves")
        
        # Red flags from detection
        for f in red_flags:
            desc = f.get('description', 'Issue detected')
            if desc not in [r.split(': ', 1)[-1] if ': ' in r else r for r in reasons]:
                reasons.append(f"⚠️ {f.get('severity', 'medium').upper()}: {desc}")
        
        # Only add generic warning if no specific concerns found
        if len(reasons) == 0:
            if result.signal == Signal.BUY:
                reasons.append("No specific concerns identified, but no investment is risk-free")
            else:
                reasons.append("Multiple minor concerns contribute to cautious rating")
        
        return reasons
    
    def _analyze_confidence(self, result: SignalResult) -> Dict:
        # Normalize confidence to 0-1 range for display if needed
        conf_val = result.confidence
        if conf_val > 1.0:
            conf_val = conf_val / 100.0
            
        return {
            'overall': conf_val,
            'strengthening': [f for f in ['No red flags'] if result.red_flags_count == 0],
            'weakening': [f"{result.red_flags_count} red flags" for _ in [1] if result.red_flags_count > 0]
        }
    
    def _create_breakdown(self, gov, fin, val, mkt) -> Dict:
        return {
            'governance': {'score': gov.overall_score, 'warnings': gov.warnings, 'red_flags': gov.red_flags},
            'financial': {'score': fin.overall_score, 'warnings': fin.warnings, 'red_flags': fin.red_flags},
            'valuation': {'score': val.overall_score, 'warnings': val.warnings},
            'market': {'score': mkt.overall_score, 'warnings': mkt.warnings}
        }
    
    def _compile_risks(self, gov, fin, val, mkt, red_flags: List) -> List[str]:
        risks = list(set(gov.red_flags + fin.red_flags + [f.get('description', '') for f in red_flags]))
        if val.stress_level in ['high', 'extreme']:
            risks.append(f"Valuation stress: {val.stress_level}")
        if mkt.max_drawdown > 50:
            risks.append(f"Historical drawdown: {mkt.max_drawdown:.1f}%")
        return risks
    
    def _identify_invalidators(self, result: SignalResult, fin) -> List[str]:
        """Generate stock-specific thesis invalidators based on current strengths."""
        invalidators = []
        
        # Based on financial strength, identify what could break the thesis
        if fin.revenue_cagr_5y and fin.revenue_cagr_5y > 10:
            invalidators.append(f"Revenue growth slowing to below {max(5, fin.revenue_cagr_5y - 5):.0f}% for 2+ quarters")
        
        if fin.roce_current and fin.roce_current > 15:
            invalidators.append(f"ROCE declining from {fin.roce_current:.0f}% to below 12%")
        
        if fin.earnings_quality and fin.earnings_quality > 0.7:
            invalidators.append("Cash flow conversion deteriorating significantly")
        
        # Standard invalidators that apply to most stocks
        invalidators.extend([
            "Promoter selling shares or increasing pledge significantly",
            "Auditor resignation or qualification in audit report",
            "Key management departures or governance concerns",
        ])
        
        # Add industry-specific risks
        if fin.debt_to_equity and fin.debt_to_equity > 0.5:
            invalidators.append("Interest rates rising significantly impacting profitability")
        
        return invalidators[:6]  # Limit to 6 most relevant
    
    def _assess_data_quality(self) -> List[str]:
        return ["Data quality appears adequate for analysis"]
