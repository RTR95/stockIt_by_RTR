
import pandas as pd
from src.analysis.buffett import BuffettAnalyzer

def run_debug():
    try:
        analyzer = BuffettAnalyzer()
        
        # Create minimal mock data
        dates = pd.date_range(start='2014-01-01', periods=10, freq='Y')
        income_stmt = pd.DataFrame({
            'Net Income': [100]*10, 'Total Revenue': [1000]*10, 'EPS': [10]*10
        }, index=dates)
        balance_sheet = pd.DataFrame({
            'Total Assets': [5000]*10, 'Total Equity': [2000]*10, 'Total Debt': [500]*10, 
            'Cash And Cash Equivalents': [200]*10
        }, index=dates)
        cash_flow = pd.DataFrame({
            'Free Cash Flow': [50]*10, 'Operating Cash Flow': [100]*10, 'Capital Expenditure': [-50]*10
        }, index=dates)
        
        financials = {
            'income_statement': income_stmt, 'balance_sheet': balance_sheet,
            'cash_flow': cash_flow, 'dividends': pd.Series([1]*10, index=dates)
        }
        
        stock_info = {'marketCap': 1e12, 'sector': 'Technology'}
        
        print("Running analyze...")
        result = analyzer.analyze(stock_info, financials)
        print(f"Analysis complete. Score: {result.score}. Approved: {result.approved}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_debug()
