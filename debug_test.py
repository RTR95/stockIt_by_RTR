
import sys
import os
from pathlib import Path

# Add project root to python path
root = Path(__file__).parent
sys.path.append(str(root))

try:
    from src.analysis.signal_generator import SignalGenerator, UserProfile, Signal
    from tests.analysis.test_utils import MockGovernanceResult, MockFinancialResult, MockValuationResult, MockMarketResult
    print("Imports successful")
except Exception as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def run_test():
    try:
        signal_gen = SignalGenerator()
        profile = UserProfile(expected_return=15.0, risk_appetite='balanced', holding_tenure=3)
        
        gov = MockGovernanceResult(overall_score=80)
        fin = MockFinancialResult(overall_score=85, roce_current=25)
        val = MockValuationResult(overall_score=70, pe_percentile_own=40)
        mkt = MockMarketResult(overall_score=70)

        result = signal_gen.generate_signal(profile, gov, fin, val, mkt)
        print(f"Signal Result: {result.signal}, Score: {result.composite_score}")
        
    except Exception as e:
        print(f"Test Execution Error: {e}")
        # Print traceback
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
