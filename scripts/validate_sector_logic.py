
import sys
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Ensure project root is in path
sys.path.append(os.getcwd())

try:
    from src.analysis.signal_generator import SignalGenerator, UserProfile, Signal
    from src.engine.financial import FinancialScore
    from src.engine.governance import GovernanceScore
    from src.engine.valuation import ValuationScore
    from src.engine.market import MarketBehaviourScore
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please run this script from the project root (e.g., inside the Docker container at /app)")
    sys.exit(1)

# Mock Classes for Validation
@dataclass
class MockGovernanceScore:
    overall_score: float = 80
    warnings: List[str] = None
    red_flags: List[str] = None
    def __post_init__(self): self.warnings, self.red_flags = [], []

@dataclass
class MockFinancialScore:
    overall_score: float
    roce_current: float
    roa_current: float = 2.0
    revenue_cagr_5y: float = 10.0
    pat_cagr_5y: float = 10.0
    debt_to_equity: float = 0.5
    earnings_quality: float = 0.9
    warnings: List[str] = None
    red_flags: List[str] = None
    details: Dict = None
    def __post_init__(self):
        self.warnings, self.red_flags = [], []
        self.details = {'margins': {'opm_current': 20}}

@dataclass
class MockValuationScore:
    overall_score: float = 60
    pe_percentile_own: float = 40
    warnings: List[str] = None
    def __post_init__(self): self.warnings = []

@dataclass
class MockMarketScore:
    overall_score: float = 50
    warnings: List[str] = None
    def __post_init__(self): self.warnings = []

def run_validation():
    print("="*60)
    print("Stocron Sector Logic Validation")
    print("="*60)
    
    signal_gen = SignalGenerator()
    profile = UserProfile(expected_return=15.0, risk_appetite='balanced', holding_tenure=3)
    
    # Scene 1: Banking Scenario
    # Banks often have low ROCE but we care about ROA.
    # Rules: BFSI min_roce is lower (8%), and high_roa (>1.5%) gives bonus.
    print(f"\n[Scenario 1] Banking Stock (BFSI)")
    print(f"Metrics: ROCE=8% (Low), ROA=2.0% (Excellent)")
    
    gov = MockGovernanceScore()
    fin_bank = MockFinancialScore(overall_score=75, roce_current=8.0, roa_current=2.0)
    val = MockValuationScore()
    mkt = MockMarketScore()
    
    result_bank = signal_gen.generate_signal(profile, gov, fin_bank, val, mkt, sector="BFSI")
    print(f"Signal: {result_bank.signal}")
    print(f"Composite Score: {result_bank.composite_score}")
    print(f"Reasoning: {result_bank.signal_reasoning}")
    
    # Scene 2: IT Scenario
    # IT Companies must have high ROCE.
    # Rules: IT min_roce is high (20%). 
    # Using same metrics as bank (ROCE=8%) should be penalized or at least not bonused.
    print(f"\n[Scenario 2] IT Stock (Information Technology)")
    print(f"Metrics: ROCE=8% (Very Low for IT)")
    
    fin_it = MockFinancialScore(overall_score=70, roce_current=8.0, roa_current=2.0) # Lower overall score context
    
    result_it = signal_gen.generate_signal(profile, gov, fin_it, val, mkt, sector="Information Technology")
    print(f"Signal: {result_it.signal}")
    print(f"Composite Score: {result_it.composite_score}")
    print(f"Reasoning: {result_it.signal_reasoning}")
    
    print("-" * 30)
    if result_bank.composite_score > result_it.composite_score:
        print("SUCCESS: Bank score is higher than IT score for same low ROCE, confirming sector rules applied.")
    else:
        print("FAILURE: Sector specific logic did not differentiate enough.")

if __name__ == "__main__":
    run_validation()
