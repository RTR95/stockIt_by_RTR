
import pytest
import pandas as pd
import numpy as np
from src.analysis.buffett import BuffettAnalyzer

class TestBuffettAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return BuffettAnalyzer()

    @pytest.fixture
    def mock_financials(self):
        # Create dummy dataframes
        dates = pd.date_range(start='2014-01-01', periods=10, freq='Y')
        
        # Stronger financials to pass Buffett rules
        # Stronger financials to pass Buffett rules
        income_stmt = pd.DataFrame({
            'Net Income': [500, 550, 600, 650, 700, 750, 800, 850, 900, 950],
            'Total Revenue': [2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800],
            'EPS': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
            'Gross Profit': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900], # 50% Margin
            'EBIT': [700, 770, 840, 910, 980, 1050, 1120, 1190, 1260, 1330],
            'Interest Expense': [50] * 10 # 50 Interest. EBIT 700. Coverage 14x. (>5x)
        }, index=dates)
        
        balance_sheet = pd.DataFrame({
            'Total Assets': [5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500],
            'Total Equity': [2000, 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800], # Growing Book Value
            'Total Debt': [500] * 10,
            'Cash And Cash Equivalents': [200] * 10,
            'Total Current Assets': [2000] * 10, 
            'Total Current Liabilities': [1000] * 10 
        }, index=dates)
        
        cash_flow = pd.DataFrame({
            'Free Cash Flow': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
            'Operating Cash Flow': [100] * 10,
            'Capital Expenditure': [-50] * 10
        }, index=dates)
        
        return {
            'income_statement': income_stmt,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow,
            'dividends': pd.Series([1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9], index=dates)
        }

    def test_strong_company_approval(self, analyzer, mock_financials):
        """Test that a strong company passes Buffett criteria."""
        stock_info = {'marketCap': 1e12, 'sector': 'Technology'} # Large cap
        
        result = analyzer.analyze(stock_info, mock_financials)
        
        assert result.approved is True
        assert result.score >= 8
        # Check specific rules
        debt_rule = next(r for r in result.rules if 'Debt' in r.name)
        assert debt_rule.passed is True

    def test_high_debt_disapproval(self, analyzer, mock_financials):
        """Test that high debt fails the hard filter."""
        # Modify to high debt
        mock_financials['balance_sheet']['Total Debt'] = 5000 # Debt/Equity > 2.5
        
        stock_info = {'marketCap': 1e12}
        result = analyzer.analyze(stock_info, mock_financials)
        
        debt_rule = next(r for r in result.rules if 'Debt' in r.name)
        assert debt_rule.passed is False
        # Might still be approved if other rules pass, but check score impact
        
    def test_inconsistent_earnings(self, analyzer, mock_financials):
        """Test penalty for inconsistent earnings."""
        # Make earnings volatile: Only 4 years of growth
        # 10->5 (D), 5->12 (U), 12->-2 (D), -2->-5 (D), -5->15 (U), 15->10 (D), 10->17 (U), 17->12 (D), 12->19 (U)
        mock_financials['income_statement']['EPS'] = [10, 5, 12, -2, -5, 15, 10, 17, 12, 19]
        
        stock_info = {'marketCap': 1e12}
        result = analyzer.analyze(stock_info, mock_financials)
        
        eps_rule = next(r for r in result.rules if 'EPS' in r.name)
        assert eps_rule.passed is False
