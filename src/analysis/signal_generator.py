"""Signal Generation for Stocron by RTR."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass
import logging

from ..utils.config import get_config, get_signal_weights, get_risk_profile
from ..utils.sectors import get_sector_thresholds

logger = logging.getLogger(__name__)


class Signal:
    BUY = "BUY"
    HOLD = "HOLD"
    AVOID = "AVOID BUYING"
    SELL = "SELL / EXIT"


@dataclass
class UserProfile:
    expected_return: float
    risk_appetite: str
    holding_tenure: int
    
    def to_dict(self) -> Dict:
        return {'expected_return': self.expected_return, 'risk_appetite': self.risk_appetite,
                'holding_tenure': self.holding_tenure}


@dataclass
class SignalResult:
    signal: str
    confidence: float
    composite_score: float
    dimension_scores: Dict[str, float]
    signal_reasoning: List[str]
    key_positives: List[str]
    key_negatives: List[str]
    user_profile_match: Dict[str, Any]
    red_flags_count: int
    warnings_count: int


class SignalGenerator:
    """
    Generates investment signals based on multi-dimensional analysis.
    
    Signal Logic (Equity Market Principles):
    - BUY: Quality business + Fair/Cheap valuation + Acceptable risk + Meets user's return expectation
    - HOLD: Good business with some concerns OR expensive valuation for quality
    - AVOID: Poor fundamentals OR very expensive OR high risk for user's profile
    - SELL: Serious governance/financial issues OR critical red flags
    """
    
    def __init__(self):
        self.weights = get_signal_weights()
        # More balanced thresholds
        self.buy_threshold = get_config('signals.buy_threshold', 65)  # Lowered from 70
        self.hold_threshold = get_config('signals.hold_threshold', 45)  # Lowered from 50
        self.avoid_threshold = get_config('signals.avoid_threshold', 30)
    
    def generate_signal(self, user_profile: UserProfile, governance_score, financial_score,
                        valuation_score, market_score, ml_context=None, red_flags: List[Dict] = None,
                        sector: str = None) -> SignalResult:
        red_flags = red_flags or []
        
        dimension_scores = {
            'governance': governance_score.overall_score,
            'financial': financial_score.overall_score,
            'valuation': valuation_score.overall_score,
            'market_behaviour': market_score.overall_score
        }
        
        # Calculate composite score with quality-weighted approach
        composite = self._calculate_composite(dimension_scores, financial_score, governance_score, sector)
        
        # Apply user profile adjustments
        adjusted, profile_match = self._apply_profile(composite, user_profile, financial_score, 
                                                       valuation_score, market_score)
        
        # Augment red flags with composite interactions
        red_flags = self._augment_red_flags(red_flags, financial_score, governance_score)
        
        # Apply red flag penalties (but don't over-penalize)
        penalty = self._calculate_penalty(red_flags)
        final_score = max(0, adjusted - penalty)
        
        # Determine signal with quality considerations
        signal, confidence = self._determine_signal(
            final_score, red_flags, governance_score, financial_score, 
            valuation_score, profile_match, dimension_scores, sector
        )
        
        reasoning = self._generate_reasoning(signal, user_profile, dimension_scores, profile_match, red_flags)
        positives = self._extract_positives(governance_score, financial_score, valuation_score, market_score)
        negatives = self._extract_negatives(governance_score, financial_score, valuation_score, market_score, red_flags)
        
        return SignalResult(
            signal=signal, confidence=confidence, composite_score=round(final_score, 1),
            dimension_scores=dimension_scores, signal_reasoning=reasoning,
            key_positives=positives, key_negatives=negatives,
            user_profile_match=profile_match, red_flags_count=len(red_flags),
            warnings_count=self._count_warnings(governance_score, financial_score, valuation_score, market_score)
        )
    
    def _calculate_composite(self, scores: Dict[str, float], fin, gov, sector: str = None) -> float:
        """
        Calculate composite score with quality-weighted approach.
        Quality companies (high ROCE, good governance) get bonus points.
        """
        base_score = sum(scores[d] * self.weights.get(d, 0.25) for d in scores)
        
        # Quality bonus for high-quality businesses
        quality_bonus = 0
        
        # Get sector thresholds
        thresholds = get_sector_thresholds(sector)
        metric = thresholds.get('roce_metric', 'roce')
        min_val = thresholds.get('min_roce', 15)
        high_val = thresholds.get('high_roce', 20)
        
        # Check metric (ROCE or ROA)
        if metric == 'roa':
            val = fin.roa_current
        else:
            val = fin.roce_current
            
        if val:
            if val > high_val:
                quality_bonus += 8
            elif val > min_val:
                quality_bonus += 5
        
        # Strong governance
        if gov.overall_score >= 70:
            quality_bonus += 5
        
        # Consistent earnings
        if fin.earnings_quality and fin.earnings_quality > 0.8:
            quality_bonus += 3
        
        return min(100, base_score + quality_bonus)
    
    def _apply_profile(self, score: float, profile: UserProfile, fin, val, mkt) -> Tuple[float, Dict]:
        """
        Apply user profile adjustments.
        
        Return Expectation Logic:
        - Estimate total return = Earnings growth + Dividend yield + Valuation expansion/contraction
        - For quality businesses, historical returns are typically 15-20% CAGR
        - If stock can meet user's expected return, it's a positive
        """
        adjusted = score
        match = {'return_expectation': {}, 'risk_match': {}, 'tenure_fit': {}}
        
        # Calculate estimated return potential
        # Components: Earnings growth + Dividend yield + PE expansion/contraction potential
        earnings_growth = fin.pat_cagr_5y or fin.revenue_cagr_5y or 10
        
        # Estimate dividend yield (from details if available)
        div_yield = fin.details.get('dividend_yield', 1.5) if fin.details else 1.5
        
        # PE expansion/contraction based on current valuation
        pe_pct = val.pe_percentile_own or 50
        if pe_pct < 30:  # Cheap - potential for PE expansion
            pe_adjustment = 5
        elif pe_pct > 70:  # Expensive - risk of PE contraction
            pe_adjustment = -5
        else:
            pe_adjustment = 0
        
        # Total estimated return potential
        estimated_return = earnings_growth + div_yield + pe_adjustment
        
        # For quality businesses, floor the estimate at reasonable levels
        roce = fin.roce_current
        if roce and roce > 15:  # Quality business
            estimated_return = max(estimated_return, 12)  # Quality businesses typically return 12%+
        
        gap = profile.expected_return - estimated_return
        meets_expectation = estimated_return >= profile.expected_return
        
        match['return_expectation'] = {
            'expected': profile.expected_return, 
            'estimated': round(estimated_return, 1),
            'gap': round(gap, 1), 
            'meets': meets_expectation,
            'components': {
                'earnings_growth': round(earnings_growth, 1),
                'dividend_yield': round(div_yield, 1),
                'valuation_adjustment': pe_adjustment
            }
        }
        
        # Adjust score based on return expectation
        if meets_expectation:
            adjusted += 5  # Bonus for meeting expectation
        elif gap > 10:
            adjusted -= 10  # Only penalize if gap is large
        elif gap > 5:
            adjusted -= 5
        
        # Risk assessment
        vol = mkt.volatility_1y or 30
        risk_cfg = get_risk_profile(profile.risk_appetite)
        max_vol = risk_cfg.get('max_volatility', 35)
        
        match['risk_match'] = {
            'user_risk': profile.risk_appetite, 
            'stock_vol': vol,
            'max_acceptable': max_vol, 
            'within_tolerance': vol <= max_vol * 1.2  # 20% buffer
        }
        
        # Less aggressive volatility penalty
        if vol > max_vol * 1.5:
            adjusted -= 15
        elif vol > max_vol * 1.2:
            adjusted -= 8
        elif vol < max_vol * 0.7:
            adjusted += 3  # Low volatility bonus
        
        match['tenure_fit'] = {'tenure': profile.holding_tenure, 'suitable': True, 'reason': 'Suitable'}
        
        return adjusted, match
    
    def _augment_red_flags(self, red_flags: List[Dict], fin, gov) -> List[Dict]:
        """Add composite red flags based on interactions."""
        augmented = red_flags.copy()
        
        # Check for Rising Debt + Declining CFO/Margins
        rising_debt = any("increasing" in w.lower() and "debt" in w.lower() for w in fin.warnings) or \
                      (fin.debt_to_equity and fin.debt_to_equity > 1.0)
        
        declining_cfo = any("poor earnings quality" in f.lower() for f in fin.red_flags) or \
                        (fin.earnings_quality and fin.earnings_quality < 0.7)
                        
        if rising_debt and declining_cfo:
            augmented.append({
                'type': 'COMPOSITE',
                'description': 'Rising Debt with Weak Cash Flows',
                'severity': 'high'
            })
            
        # Governance + Financial Weakness
        weak_gov = gov.overall_score < 50
        weak_fin = fin.overall_score < 50
        
        if weak_gov and weak_fin:
             augmented.append({
                'type': 'COMPOSITE',
                'description': 'Double Trouble: Weak Governance & Financials',
                'severity': 'critical'
            })
            
        return augmented

    def _calculate_penalty(self, red_flags: List[Dict]) -> float:
        """Calculate penalty for red flags - more nuanced approach."""
        if not red_flags:
            return 0
        
        total_penalty = 0
        for f in red_flags:
            severity = f.get('severity', 'medium')
            if severity == 'critical':
                total_penalty += 20
            elif severity == 'high':
                total_penalty += 10
            else:
                total_penalty += 5
        
        # Cap penalty to avoid over-penalization
        return min(40, total_penalty)
    
    def _determine_signal(self, score: float, red_flags: List, gov, fin, val, 
                          profile_match: Dict, dimension_scores: Dict, sector: str = None) -> Tuple[str, float]:
        """
        Determine signal based on comprehensive analysis.
        
        Logic:
        - Critical red flags -> AVOID (governance/fraud concerns trump everything)
        - High quality + Fair value + Meets return -> BUY
        - Good quality but expensive OR some concerns -> HOLD
        - Poor quality OR doesn't meet user needs -> AVOID
        - Serious issues -> SELL
        """
        
        # Critical red flags are deal-breakers
        if any(f.get('severity') == 'critical' for f in red_flags):
            return Signal.AVOID, 85.0
        
        # Serious governance + financial issues -> SELL
        if gov.overall_score < 30 and fin.overall_score < 30:
            return Signal.SELL, 80.0
        
        # Quality indicators (Sector aware)
        thresholds = get_sector_thresholds(sector)
        metric = thresholds.get('roce_metric', 'roce')
        min_val = thresholds.get('min_roce', 15)
        
        if metric == 'roa':
            val = fin.roa_current
        else:
            val = fin.roce_current
            
        is_quality = val and val > min_val and gov.overall_score >= 55
        
        meets_return = profile_match.get('return_expectation', {}).get('meets', False)
        within_risk = profile_match.get('risk_match', {}).get('within_tolerance', True)
        
        # Average dimension score
        avg_score = sum(dimension_scores.values()) / len(dimension_scores)
        
        # Decision logic
        if score >= self.buy_threshold:
            # High score - likely BUY, but check quality
            if is_quality or avg_score >= 60:
                return Signal.BUY, min(95, 70 + (score - self.buy_threshold))
            else:
                return Signal.HOLD, 65.0  # High score but not quality
        
        elif score >= self.hold_threshold:
            # Medium score - HOLD or BUY based on quality
            if is_quality and meets_return and within_risk:
                # Quality business at fair value that meets needs -> BUY
                return Signal.BUY, 70.0
            elif is_quality:
                # Quality business but doesn't fully meet needs -> HOLD
                return Signal.HOLD, 60 + (score - self.hold_threshold) * 0.5
            else:
                return Signal.HOLD, 55.0
        
        elif score >= self.avoid_threshold:
            # Low-medium score
            if is_quality:
                # Quality business with temporary issues -> HOLD
                return Signal.HOLD, 50.0
            else:
                return Signal.AVOID, 60 + (self.hold_threshold - score)
        
        else:
            # Low score
            return Signal.SELL, 70 + (self.avoid_threshold - score)
    
    def _generate_reasoning(self, signal: str, profile: UserProfile, scores: Dict, 
                            match: Dict, red_flags: List) -> List[str]:
        reasons = []
        if signal == Signal.BUY:
            reasons.append(f"Stock aligns with your profile ({profile.expected_return}% CAGR, {profile.risk_appetite} risk).")
        elif signal == Signal.HOLD:
            reasons.append("Mixed characteristics. Consider monitoring if already holding.")
        elif signal == Signal.AVOID:
            reasons.append("Stock does not align well with your investment profile.")
        else:
            reasons.append("Significant concerns detected. Review any existing position.")
        
        strongest = max(scores, key=scores.get)
        weakest = min(scores, key=scores.get)
        reasons.append(f"Strongest: {strongest.replace('_', ' ').title()} ({scores[strongest]:.0f}/100)")
        reasons.append(f"Weakest: {weakest.replace('_', ' ').title()} ({scores[weakest]:.0f}/100)")
        
        if not match['return_expectation'].get('meets', True):
            reasons.append(f"Return gap: Expected {match['return_expectation']['expected']}%, estimated {match['return_expectation']['estimated']}%")
        if not match['risk_match'].get('within_tolerance', True):
            reasons.append(f"Risk mismatch: Volatility {match['risk_match']['stock_vol']:.0f}% exceeds {match['risk_match']['max_acceptable']:.0f}%")
        if red_flags:
            reasons.append(f"Red flags detected: {len(red_flags)}")
        
        return reasons
    
    def _extract_positives(self, gov, fin, val, mkt) -> List[str]:
        pos = []
        if gov.promoter_holding > 50:
            pos.append(f"Strong promoter holding: {gov.promoter_holding:.1f}%")
        if gov.pledge_ratio == 0:
            pos.append("No promoter pledge")
        if fin.roce_current and fin.roce_current > 18:
            pos.append(f"High ROCE: {fin.roce_current:.1f}%")
        if fin.revenue_cagr_5y and fin.revenue_cagr_5y > 15:
            pos.append(f"Strong growth: {fin.revenue_cagr_5y:.1f}% revenue CAGR")
        if val.valuation_zone == 'undervalued':
            pos.append("Trading at attractive valuations")
        if mkt.volatility_regime == 'low':
            pos.append("Low volatility stock")
        return pos[:6]
    
    def _extract_negatives(self, gov, fin, val, mkt, red_flags: List) -> List[str]:
        neg = []
        neg.extend(gov.warnings[:2])
        neg.extend(gov.red_flags)
        neg.extend(fin.warnings[:2])
        neg.extend(fin.red_flags)
        neg.extend(val.warnings[:2])
        neg.extend(mkt.warnings[:2])
        for f in red_flags[:3]:
            neg.append(f.get('description', 'Red flag'))
        return neg[:8]
    
    def _count_warnings(self, gov, fin, val, mkt) -> int:
        return len(gov.warnings) + len(fin.warnings) + len(val.warnings) + len(mkt.warnings)
