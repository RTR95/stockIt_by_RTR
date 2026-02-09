
import pandas as pd
from src.analysis.buffett import BuffettAnalyzer

def run_debug():
    try:
        analyzer = BuffettAnalyzer()
        
        dates = pd.date_range(start='2014-01-01', periods=10, freq='Y')
        
        # Exact data from test_buffett.py
        income_stmt = pd.DataFrame({
            'Net Income': [500, 550, 600, 650, 700, 750, 800, 850, 900, 950],
            'Total Revenue': [2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800],
            'EPS': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
            'Gross Profit': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900], 
            'EBIT': [700, 770, 840, 910, 980, 1050, 1120, 1190, 1260, 1330],
            'Interest Expense': [50] * 10 # Added Interest Expense!
        }, index=dates)
        
        balance_sheet = pd.DataFrame({
            'Total Assets': [5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500],
            'Total Equity': [2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800],
            'Total Debt': [500] * 10,
            'Cash And Cash Equivalents': [200] * 10,
            'Total Current Assets': [2000] * 10, 
            'Total Current Liabilities': [1000] * 10 
        }, index=dates)
        
        cash_flow = pd.DataFrame({
            'Free Cash Flow': [200, 220, 240, 260, 280, 300, 320, 340, 360, 380],
            'Operating Cash Flow': [400] * 10,
            'Capital Expenditure': [-100] * 10
        }, index=dates)
        
        financials = {
            'income_statement': income_stmt, 'balance_sheet': balance_sheet,
            'cash_flow': cash_flow, 'dividends': pd.Series([1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9], index=dates)
        }
        
        stock_info = {'marketCap': 1e12, 'sector': 'Technology'}
        
        print("Running analyze...")
        result = analyzer.analyze(stock_info, financials)
        print(f"Score: {result.score}. Approved: {result.approved}")
        for r in result.rules:
            print(f"Rule: {r.name} | Passed: {r.passed} | Value: {r.value_display}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_debug()
