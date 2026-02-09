
import pytest
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

# Mock result classes to mimic engine outputs
@dataclass
class MockGovernanceResult:
    overall_score: float = 75.0
    promoter_holding: float = 60.0
    pledge_ratio: float = 0.0
    years_listed: int = 10
    dividend_consistency_years: int = 5
    auditor_stability: int = 10
    warnings: List[str] = None
    red_flags: List[str] = None

    def __post_init__(self):
        if self.warnings is None: self.warnings = []
        if self.red_flags is None: self.red_flags = []

@dataclass
class MockFinancialResult:
    overall_score: float = 75.0
    roce_current: float = 25.0
    revenue_cagr_5y: float = 15.0
    pat_cagr_5y: float = 18.0
    earnings_quality: float = 0.9
    fcf_yield: float = 3.0
    debt_to_equity: float = 0.2
    roa_current: Optional[float] = None  # Added for BFSI tests
    warnings: List[str] = None
    red_flags: List[str] = None
    details: Dict = None

    def __post_init__(self):
        if self.warnings is None: self.warnings = []
        if self.red_flags is None: self.red_flags = []
        if self.details is None: self.details = {'dividend_yield': 1.5, 'margins': {'opm_current': 20.0}}

@dataclass
class MockValuationResult:
    overall_score: float = 60.0
    pe_percentile_own: float = 40.0
    current_pe: float = 25.0
    current_pb: float = 4.0
    ev_ebitda: float = 15.0
    peg_ratio: float = 1.2
    valuation_zone: str = 'fair'
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None: self.warnings = []

@dataclass
class MockMarketResult:
    overall_score: float = 65.0
    volatility_1y: float = 25.0
    beta: float = 0.9
    sharpe_ratio: float = 1.2
    vs_nifty_1y: float = 10.0
    volatility_regime: str = 'normal'
    max_drawdown: float = -15.0
    avg_recovery_months: int = 4
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None: self.warnings = []

# Aliases for compatibility with tests expecting 'Score' naming
MockGovernanceScore = MockGovernanceResult
MockFinancialScore = MockFinancialResult
MockValuationScore = MockValuationResult
MockMarketScore = MockMarketResult
