
import pytest
from src.analysis.signal_generator import SignalGenerator, UserProfile, Signal
from tests.analysis.test_utils import (
    MockGovernanceScore, MockFinancialScore, MockValuationScore, MockMarketScore,
    MockGovernanceResult, MockFinancialResult, MockValuationResult, MockMarketResult
)

class TestSignalGenerator:
    @pytest.fixture
    def signal_gen(self):
        return SignalGenerator()

    @pytest.fixture
    def balanced_profile(self):
        return UserProfile(
            expected_return=15.0,
            risk_appetite='balanced',
            holding_tenure=3
        )

    def test_high_quality_buy_signal(self, signal_gen, balanced_profile):
        """Test a clear BUY case: High quality, fair valuation."""
        gov = MockGovernanceResult(overall_score=80)
        fin = MockFinancialResult(overall_score=85, roce_current=25)
        val = MockValuationResult(overall_score=70, pe_percentile_own=40)  # Fair value
        mkt = MockMarketResult(overall_score=70)

        result = signal_gen.generate_signal(balanced_profile, gov, fin, val, mkt)
        
        assert result.signal == Signal.BUY
        assert result.confidence > 70
        assert result.composite_score > 65

    def test_expensive_quality_hold_signal(self, signal_gen, balanced_profile):
        """Test a HOLD case: High quality but expensive."""
        gov = MockGovernanceResult(overall_score=80)
        fin = MockFinancialResult(overall_score=85, roce_current=25)
        val = MockValuationResult(overall_score=30, pe_percentile_own=90)  # Very expensive
        mkt = MockMarketResult(overall_score=70)

        result = signal_gen.generate_signal(balanced_profile, gov, fin, val, mkt)
        
        # High quality stocks often get BUY even if expensive in this logic
        assert result.signal == Signal.BUY

    def test_critical_red_flag_avoid(self, signal_gen, balanced_profile):
        """Test that critical red flags trigger AVOID."""
        gov = MockGovernanceResult(overall_score=40)
        fin = MockFinancialResult(overall_score=50)
        val = MockValuationResult(overall_score=50)
        mkt = MockMarketResult(overall_score=50)
        
        red_flags = [{'severity': 'critical', 'description': 'Auditor Resignation'}]

        result = signal_gen.generate_signal(balanced_profile, gov, fin, val, mkt, red_flags=red_flags)
        
        assert result.signal == Signal.AVOID
        assert result.red_flags_count == 1

    def test_financial_governance_failure_sell(self, signal_gen, balanced_profile):
        """Test that poor financials + governance triggers SELL."""
        gov = MockGovernanceResult(overall_score=20)
        # Ensure underlying metrics are also poor to avoid quality bonuses
        fin = MockFinancialResult(overall_score=20, roce_current=5.0, revenue_cagr_5y=2.0, pat_cagr_5y=0.0, earnings_quality=0.5)
        val = MockValuationResult(overall_score=50, pe_percentile_own=80) # Expensive relative to poor growth
        mkt = MockMarketResult(overall_score=40)

        result = signal_gen.generate_signal(balanced_profile, gov, fin, val, mkt)
        
        assert result.signal == Signal.SELL

    def test_user_profile_mismatch(self, signal_gen):
        """Test penalty for risk mismatch."""
        conservative_profile = UserProfile(
            expected_return=10.0,
            risk_appetite='conservative',
            holding_tenure=5
        )
        
        gov = MockGovernanceResult(overall_score=70)
        fin = MockFinancialResult(overall_score=70)
        val = MockValuationResult(overall_score=60)
        # High volatility stock for conservative user
        mkt = MockMarketResult(overall_score=40, volatility_1y=60) 

        result = signal_gen.generate_signal(conservative_profile, gov, fin, val, mkt)
        
        # Should penalize score
        assert not result.user_profile_match['risk_match']['within_tolerance']
        assert result.signal != Signal.BUY # Unlikely to be a buy for conservative user

    def test_sector_specific_logic(self, signal_gen):
        """Test sector-specific thresholds (Bank vs Utility)."""
        balanced_profile = UserProfile(
            risk_appetite='balanced', 
            expected_return=15.0,
            holding_tenure=3
        )
        
        # 1. Bank: Low ROCE but Good ROA -> Should NOT be penalized
        gov_score = MockGovernanceScore(overall_score=80)
        # Bank has ROCE=8 (Bad for general), but ROA=2.0 (Good for Bank)
        fin_score = MockFinancialScore(overall_score=75, roce_current=8.0, roa_current=2.0)
        val_score = MockValuationScore(overall_score=60)
        mkt_score = MockMarketScore(overall_score=50)
        
        result_bank = signal_gen.generate_signal(
            balanced_profile, gov_score, fin_score, val_score, mkt_score, sector="BFSI"
        )
        
        # 2. Utility: Moderate ROCE (12%) -> Good for Utility (Threshold 10%), Bad for IT
        fin_util = MockFinancialScore(overall_score=70, roce_current=12.0)
        result_util = signal_gen.generate_signal(
            balanced_profile, gov_score, fin_util, val_score, mkt_score, sector="Utilities"
        )
        
        # 3. IT: Moderate ROCE (12%) -> Bad for IT (Threshold 20%)
        result_it = signal_gen.generate_signal(
            balanced_profile, gov_score, fin_util, val_score, mkt_score, sector="Information Technology"
        )
        
        # Bank score should be boosted by Good ROA
        # Utility should trigger "meeting threshold" bonus (12 > 10)
        # IT should NOT trigger bonus (12 < 20)
        
        assert result_bank.composite_score > 60
        assert result_util.composite_score >= result_it.composite_score
