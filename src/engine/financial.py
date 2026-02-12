"""Financial Trajectory Analysis for Stocron by RTR."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
import logging

from ..utils.config import get_threshold
from ..utils.helpers import calculate_cagr, safe_divide, calculate_consistency_score
from typing import Tuple

logger = logging.getLogger(__name__)


@dataclass
class FinancialScore:
    """Container for financial analysis results."""
    overall_score: float
    revenue_cagr_3y: Optional[float]
    revenue_cagr_5y: Optional[float]
    revenue_cagr_10y: Optional[float]
    pat_cagr_3y: Optional[float]
    pat_cagr_5y: Optional[float]
    roce_current: Optional[float]
    roa_current: Optional[float]
    roce_avg_5y: Optional[float]
    roce_consistency: float
    fcf_yield: Optional[float]
    earnings_quality: float
    debt_to_equity: Optional[float]
    margin_trend: str
    details: Dict[str, Any]
    warnings: List[str]
    red_flags: List[str]
    investable: bool
    investable_reasons: List[str]


class FinancialAnalyzer:
    """Analyzes financial trajectory and health."""
    
    def __init__(self, db_manager=None):
        self.db = db_manager
        self.min_revenue_cagr_3y = get_threshold('financial', 'min_revenue_cagr_3y', 10.0)
        self.min_roce = get_threshold('financial', 'min_roce', 12.0)
        self.max_debt_equity = get_threshold('financial', 'max_debt_to_equity', 1.5)
    
    def analyze(self, symbol: str, financials: Dict[str, pd.DataFrame], stock_info: Dict[str, Any],
                cutoff_date: Optional[datetime] = None) -> FinancialScore:
        cutoff_date = cutoff_date or datetime.now()
        warnings, red_flags, details = [], [], {}
        
        income_stmt = financials.get('income_statement', pd.DataFrame())
        balance_sheet = financials.get('balance_sheet', pd.DataFrame())
        cash_flow = financials.get('cash_flow', pd.DataFrame())
        
        # Revenue Growth
        revenue = self._analyze_revenue(income_stmt)
        details['revenue'] = revenue
        if revenue['cagr_3y'] is not None and revenue['cagr_3y'] < self.min_revenue_cagr_3y:
            warnings.append(f"Slow revenue growth: {revenue['cagr_3y']:.1f}% (3Y)")
        
        # Profitability
        profit = self._analyze_profitability(income_stmt)
        details['profitability'] = profit
        
        # ROCE
        roce = self._analyze_roce(income_stmt, balance_sheet)
        details['roce'] = roce
        if roce['current'] and roce['current'] < self.min_roce:
            warnings.append(f"Low ROCE: {roce['current']:.1f}%")
            
        # ROA (for Banks/NBFCs context)
        roa = self._analyze_return_on_assets(income_stmt, balance_sheet)
        details['roa'] = roa
        
        # Cash Flow
        cashflow = self._analyze_cashflow(cash_flow, income_stmt, stock_info)
        details['cash_flow'] = cashflow
        if cashflow['earnings_quality'] < 0.5:
            red_flags.append(f"Poor earnings quality: CFO/PAT = {cashflow['earnings_quality']:.2f}")
        if cashflow['negative_fcf_years'] >= 3:
            red_flags.append(f"Persistent negative FCF: {cashflow['negative_fcf_years']} years")
        
        # Leverage
        leverage = self._analyze_leverage(balance_sheet)
        details['leverage'] = leverage
        if leverage['debt_to_equity'] and leverage['debt_to_equity'] > self.max_debt_equity:
            warnings.append(f"High leverage: D/E = {leverage['debt_to_equity']:.2f}")
        
        # Margins
        margins = self._analyze_margins(income_stmt)
        details['margins'] = margins

        # Investable universe filter (e.g., positive cash flow for 5+ years)
        investable_reasons = []
        min_positive_fcf_years = get_threshold('financial', 'min_positive_fcf_years', 5)
        min_earnings_quality = get_threshold('financial', 'min_earnings_quality', 0.6)
        if cashflow.get('has_cashflow'):
            if cashflow.get('positive_fcf_years', 0) < min_positive_fcf_years:
                investable_reasons.append(
                    f"Positive FCF only {cashflow.get('positive_fcf_years', 0)} of last {min_positive_fcf_years} years"
                )
            if cashflow.get('earnings_quality', 0) < min_earnings_quality:
                investable_reasons.append(
                    f"Earnings quality below {min_earnings_quality:.0%}"
                )
        investable = len(investable_reasons) == 0
        
        overall_score = self._calculate_score(revenue['score'], profit['score'], roce['score'],
                                               cashflow['score'], leverage['score'], margins['score'])
        
        return FinancialScore(
            overall_score=overall_score, revenue_cagr_3y=revenue['cagr_3y'],
            revenue_cagr_5y=revenue['cagr_5y'], revenue_cagr_10y=revenue['cagr_10y'],
            pat_cagr_3y=profit['pat_cagr_3y'], pat_cagr_5y=profit['pat_cagr_5y'],
            roce_current=roce['current'], roa_current=roa['current'], roce_avg_5y=roce['avg_5y'],
            roce_consistency=roce['consistency'], fcf_yield=cashflow['fcf_yield'],
            earnings_quality=cashflow['earnings_quality'],
            debt_to_equity=leverage['debt_to_equity'], margin_trend=margins['trend'],
            details=details, warnings=warnings, red_flags=red_flags,
            investable=investable, investable_reasons=investable_reasons
        )
    
    def _analyze_revenue(self, income_stmt: pd.DataFrame) -> Dict:
        if income_stmt.empty:
            return {'cagr_3y': None, 'cagr_5y': None, 'cagr_10y': None, 'score': 50}
        
        rev_cols = ['Total Revenue', 'Revenue', 'Net Sales']
        revenue = None
        for col in rev_cols:
            if col in income_stmt.columns:
                revenue = income_stmt[col].dropna().sort_index()
                break
        
        if revenue is None or len(revenue) < 2:
            return {'cagr_3y': None, 'cagr_5y': None, 'cagr_10y': None, 'score': 50}
        
        cagr_3y = self._calc_cagr(revenue, 3)
        cagr_5y = self._calc_cagr(revenue, 5)
        cagr_10y = self._calc_cagr(revenue, 10)
        
        primary = cagr_5y or cagr_3y
        score = 50 if primary is None else 95 if primary >= 20 else 85 if primary >= 15 else 75 if primary >= 10 else 60 if primary >= 5 else 45 if primary >= 0 else 25
        
        return {'cagr_3y': cagr_3y, 'cagr_5y': cagr_5y, 'cagr_10y': cagr_10y, 'score': score}
    
    def _analyze_profitability(self, income_stmt: pd.DataFrame) -> Dict:
        if income_stmt.empty:
            return {'pat_cagr_3y': None, 'pat_cagr_5y': None, 'score': 50}
        
        pat_cols = ['Net Income', 'Profit After Tax', 'PAT']
        pat = None
        for col in pat_cols:
            if col in income_stmt.columns:
                pat = income_stmt[col].dropna().sort_index()
                pat = pat[pat > 0]
                break
        
        if pat is None or len(pat) < 2:
            return {'pat_cagr_3y': None, 'pat_cagr_5y': None, 'score': 50}
        
        cagr_3y = self._calc_cagr(pat, 3)
        cagr_5y = self._calc_cagr(pat, 5)
        primary = cagr_5y or cagr_3y
        score = 50 if primary is None else 95 if primary >= 25 else 85 if primary >= 18 else 75 if primary >= 12 else 60 if primary >= 5 else 45 if primary >= 0 else 25
        
        return {'pat_cagr_3y': cagr_3y, 'pat_cagr_5y': cagr_5y, 'score': score}
    
    def _analyze_return_on_assets(self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame) -> Dict:
        """
        Calculate Return on Assets (ROA).
        ROA = Net Income / Total Assets
        """
        if income_stmt.empty or balance_sheet.empty:
            return {'current': None, 'avg_5y': None, 'score': 50}
        
        try:
            # Get Net Income
            pat_cols = ['Net Income', 'Profit After Tax', 'PAT', 'Net Profit']
            net_income = None
            for col in pat_cols:
                if col in income_stmt.columns:
                    net_income = income_stmt[col].dropna()
                    break
            
            if net_income is None:
                return {'current': None, 'avg_5y': None, 'score': 50}

            # Get Total Assets
            total_assets_cols = ['Total Assets', 'Total Asset']
            total_assets = None
            for col in total_assets_cols:
                if col in balance_sheet.columns:
                    total_assets = balance_sheet[col].dropna()
                    break
            
            if total_assets is None:
                return {'current': None, 'avg_5y': None, 'score': 50}
            
            common_idx = net_income.index.intersection(total_assets.index)
            if len(common_idx) == 0:
                return {'current': None, 'avg_5y': None, 'score': 50}
            
            roa_series = (net_income.loc[common_idx] / total_assets.loc[common_idx].replace(0, np.nan)) * 100
            roa_series = roa_series.dropna().sort_index()
            
            if roa_series.empty:
                return {'current': None, 'avg_5y': None, 'score': 50}
            
            current_roa = float(roa_series.iloc[-1])
            avg_5y = float(roa_series.tail(5).mean()) if len(roa_series) >= 2 else current_roa
            
            return {
                'current': round(current_roa, 2),
                'avg_5y': round(avg_5y, 2),
                'score': 50 # Placeholder, not used for scoring directly yet
            }
            
        except Exception as e:
            logger.debug(f"ROA calculation error: {e}")
            return {'current': None, 'avg_5y': None, 'score': 50}

    def _analyze_roce(self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame) -> Dict:
        """
        Calculate Return on Capital Employed (ROCE).
        
        ROCE = EBIT / Capital Employed
        Capital Employed = Total Assets - Current Liabilities
        """
        if income_stmt.empty or balance_sheet.empty:
            return {'current': None, 'avg_5y': None, 'consistency': 0, 'score': 50}
        
        try:
            ebit = self._get_ebit(income_stmt)
            if ebit is None or ebit.empty:
                return {'current': None, 'avg_5y': None, 'consistency': 0, 'score': 50}
            
            capital_employed = self._get_capital_employed(balance_sheet)
            if capital_employed is None or capital_employed.empty:
                return {'current': None, 'avg_5y': None, 'consistency': 0, 'score': 50}
            
            common_idx = ebit.index.intersection(capital_employed.index)
            if len(common_idx) == 0:
                return {'current': None, 'avg_5y': None, 'consistency': 0, 'score': 50}
            
            ebit = ebit.loc[common_idx]
            capital_employed = capital_employed.loc[common_idx]
            
            roce_series = (ebit / capital_employed.replace(0, np.nan)) * 100
            roce_series = roce_series.dropna().sort_index()
            
            if roce_series.empty:
                return {'current': None, 'avg_5y': None, 'consistency': 0, 'score': 50}
            
            current_roce = float(roce_series.iloc[-1])
            avg_5y = float(roce_series.tail(5).mean()) if len(roce_series) >= 2 else current_roce
            
            if len(roce_series) >= 3:
                recent = roce_series.iloc[-1]
                older = roce_series.iloc[-3] if len(roce_series) >= 3 else roce_series.iloc[0]
                trend = 'improving' if recent > older else 'declining' if recent < older * 0.9 else 'stable'
            else:
                trend = 'insufficient_data'
            
            consistency = calculate_consistency_score(roce_series, threshold=12.0)
            score = self._score_roce(current_roce, avg_5y, consistency)
            
            return {
                'current': round(current_roce, 2),
                'avg_5y': round(avg_5y, 2),
                'trend': trend,
                'consistency': consistency,
                'score': score,
            }
            
        except Exception as e:
            logger.debug(f"ROCE calculation error: {e}")
            return {'current': None, 'avg_5y': None, 'consistency': 0, 'score': 50}
    
    def _get_ebit(self, income_stmt: pd.DataFrame) -> Optional[pd.Series]:
        """Extract EBIT from income statement."""
        ebit_cols = ['EBIT', 'Operating Income', 'Operating Profit', 'Earnings Before Interest And Taxes']
        
        for col in ebit_cols:
            if col in income_stmt.columns:
                return income_stmt[col].dropna()
        
        net_income_cols = ['Net Income', 'Profit After Tax', 'PAT', 'Net Profit']
        interest_cols = ['Interest Expense', 'Interest', 'Finance Costs']
        tax_cols = ['Tax Provision', 'Income Tax Expense', 'Tax']
        
        net_income = None
        for col in net_income_cols:
            if col in income_stmt.columns:
                net_income = income_stmt[col]
                break
        
        if net_income is not None:
            interest = pd.Series(0, index=income_stmt.index)
            for col in interest_cols:
                if col in income_stmt.columns:
                    interest = income_stmt[col].fillna(0)
                    break
            
            tax = pd.Series(0, index=income_stmt.index)
            for col in tax_cols:
                if col in income_stmt.columns:
                    tax = income_stmt[col].fillna(0)
                    break
            
            return (net_income + interest.abs() + tax.abs()).dropna()
        
        return None
    
    def _get_capital_employed(self, balance_sheet: pd.DataFrame) -> Optional[pd.Series]:
        """Calculate Capital Employed = Total Assets - Current Liabilities."""
        total_assets_cols = ['Total Assets', 'Total Asset']
        current_liab_cols = ['Current Liabilities', 'Total Current Liabilities']
        
        total_assets = None
        for col in total_assets_cols:
            if col in balance_sheet.columns:
                total_assets = balance_sheet[col]
                break
        
        current_liab = None
        for col in current_liab_cols:
            if col in balance_sheet.columns:
                current_liab = balance_sheet[col]
                break
        
        if total_assets is not None and current_liab is not None:
            return (total_assets - current_liab).dropna()
        
        equity_cols = ['Total Stockholder Equity', 'Stockholders Equity', 'Total Equity']
        debt_cols = ['Long Term Debt', 'Long-term Debt']
        
        equity = None
        for col in equity_cols:
            if col in balance_sheet.columns:
                equity = balance_sheet[col]
                break
        
        long_term_debt = pd.Series(0, index=balance_sheet.index)
        for col in debt_cols:
            if col in balance_sheet.columns:
                long_term_debt = balance_sheet[col].fillna(0)
                break
        
        if equity is not None:
            return (equity + long_term_debt).dropna()
        
        return None
    
    def _score_roce(self, current: float, avg_5y: float, consistency: float) -> float:
        """Score ROCE based on Indian market context."""
        score = 50.0
        
        if current >= 25:
            score += 30
        elif current >= 20:
            score += 25
        elif current >= 15:
            score += 18
        elif current >= 12:
            score += 10
        elif current >= 8:
            score += 0
        else:
            score -= 15
        
        if consistency >= 80:
            score += 15
        elif consistency >= 60:
            score += 10
        elif consistency >= 40:
            score += 5
        
        if current > avg_5y * 1.1:
            score += 5
        elif current < avg_5y * 0.8:
            score -= 5
        
        return min(100, max(0, score))
    
    def _analyze_cashflow(self, cash_flow: pd.DataFrame, income_stmt: pd.DataFrame, stock_info: Dict) -> Dict:
        result = {
            'fcf_yield': None,
            'earnings_quality': 1.0,
            'negative_fcf_years': 0,
            'positive_fcf_years': 0,
            'has_cashflow': False,
            'score': 50
        }
        if cash_flow.empty:
            return result
        result['has_cashflow'] = True
        
        cfo_cols = ['Operating Cash Flow', 'Cash From Operating Activities']
        cfo = None
        for col in cfo_cols:
            if col in cash_flow.columns:
                cfo = cash_flow[col]
                break
        
        capex_cols = [
            'Capital Expenditure', 'Capital Expenditures', 'Purchase Of Property Plant Equipment',
            'CAPEX'
        ]
        capex = None
        for col in capex_cols:
            if col in cash_flow.columns:
                capex = cash_flow[col]
                break

        if cfo is not None and capex is not None:
            common_idx = cfo.index.intersection(capex.index)
            if len(common_idx) > 0:
                fcf_series = (cfo.loc[common_idx] - capex.loc[common_idx]).dropna().sort_index()
                if not fcf_series.empty:
                    result['negative_fcf_years'] = int((fcf_series < 0).sum())
                    recent = fcf_series.tail(5) if len(fcf_series) >= 5 else fcf_series
                    result['positive_fcf_years'] = int((recent > 0).sum())

                    market_cap = stock_info.get('market_cap') or 0
                    if market_cap > 0:
                        result['fcf_yield'] = round(float(fcf_series.iloc[-1]) / market_cap * 100, 2)

        # Earnings quality: CFO / PAT
        pat_cols = ['Net Income', 'Profit After Tax', 'PAT', 'Net Profit']
        pat = None
        for col in pat_cols:
            if col in income_stmt.columns:
                pat = income_stmt[col]
                break
        if cfo is not None and pat is not None:
            common_idx = cfo.index.intersection(pat.index)
            if len(common_idx) > 0:
                ratio = cfo.loc[common_idx] / pat.loc[common_idx].replace(0, np.nan)
                ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
                if not ratio.empty:
                    result['earnings_quality'] = float(round(ratio.tail(3).mean(), 2))

        score = 50.0
        if result['fcf_yield'] is not None:
            if result['fcf_yield'] >= get_threshold('financial', 'min_fcf_yield', 2.0) * 2:
                score += 20
            elif result['fcf_yield'] >= get_threshold('financial', 'min_fcf_yield', 2.0):
                score += 10
            elif result['fcf_yield'] < 0:
                score -= 15
        if result['earnings_quality'] >= 1.0:
            score += 10
        elif result['earnings_quality'] >= 0.8:
            score += 5
        elif result['earnings_quality'] < 0.5:
            score -= 15
        if result['negative_fcf_years'] >= 3:
            score -= 20

        result['score'] = min(100, max(0, score))
        return result
    
    def _analyze_leverage(self, balance_sheet: pd.DataFrame) -> Dict:
        """Analyze leverage metrics from balance sheet."""
        if balance_sheet.empty:
            return {'debt_to_equity': None, 'score': 50}
        
        try:
            debt_cols = ['Total Debt', 'Long Term Debt', 'Long-term Debt']
            total_debt = None
            for col in debt_cols:
                if col in balance_sheet.columns:
                    total_debt = balance_sheet[col].dropna()
                    break
            
            if total_debt is None:
                short_debt = pd.Series(0, index=balance_sheet.index)
                long_debt = pd.Series(0, index=balance_sheet.index)
                
                for col in ['Short Term Debt', 'Current Debt']:
                    if col in balance_sheet.columns:
                        short_debt = balance_sheet[col].fillna(0)
                        break
                
                for col in ['Long Term Debt', 'Long-term Debt']:
                    if col in balance_sheet.columns:
                        long_debt = balance_sheet[col].fillna(0)
                        break
                
                total_debt = (short_debt + long_debt).dropna()
            
            equity_cols = ['Total Stockholder Equity', 'Stockholders Equity', 'Total Equity']
            total_equity = None
            for col in equity_cols:
                if col in balance_sheet.columns:
                    total_equity = balance_sheet[col].dropna()
                    break
            
            if total_debt is None or total_equity is None:
                return {'debt_to_equity': None, 'score': 50}
            
            common_idx = total_debt.index.intersection(total_equity.index)
            if len(common_idx) == 0:
                return {'debt_to_equity': None, 'score': 50}
            
            de_ratio = total_debt.loc[common_idx] / total_equity.loc[common_idx].replace(0, np.nan)
            de_ratio = de_ratio.dropna().sort_index()
            
            if de_ratio.empty:
                return {'debt_to_equity': None, 'score': 50}
            
            current_de = float(de_ratio.iloc[-1])
            
            if len(de_ratio) >= 3:
                trend = 'increasing' if de_ratio.iloc[-1] > de_ratio.iloc[0] * 1.1 else 'decreasing' if de_ratio.iloc[-1] < de_ratio.iloc[0] * 0.9 else 'stable'
            else:
                trend = 'insufficient_data'
            
            score = self._score_leverage(current_de, trend)
            
            return {'debt_to_equity': round(current_de, 2), 'trend': trend, 'score': score}
            
        except Exception as e:
            logger.debug(f"Leverage calculation error: {e}")
            return {'debt_to_equity': None, 'score': 50}
    
    def _score_leverage(self, de_ratio: float, trend: str) -> float:
        """Score leverage based on D/E ratio."""
        score = 50.0
        
        if de_ratio <= 0.3:
            score += 35
        elif de_ratio <= 0.5:
            score += 25
        elif de_ratio <= 1.0:
            score += 15
        elif de_ratio <= 1.5:
            score += 0
        elif de_ratio <= 2.0:
            score -= 15
        else:
            score -= 25
        
        if trend == 'decreasing':
            score += 5
        elif trend == 'increasing':
            score -= 5
        
        return min(100, max(0, score))
    
    def _analyze_margins(self, income_stmt: pd.DataFrame) -> Dict:
        """Analyze profit margins from income statement."""
        if income_stmt.empty:
            return {'opm_current': None, 'npm_current': None, 'trend': 'unknown', 'score': 50}
        
        try:
            revenue = None
            for col in ['Total Revenue', 'Revenue', 'Net Sales']:
                if col in income_stmt.columns:
                    revenue = income_stmt[col].dropna()
                    break
            
            if revenue is None or revenue.empty:
                return {'opm_current': None, 'npm_current': None, 'trend': 'unknown', 'score': 50}
            
            operating_income = None
            for col in ['Operating Income', 'Operating Profit', 'EBIT']:
                if col in income_stmt.columns:
                    operating_income = income_stmt[col].dropna()
                    break
            
            net_income = None
            for col in ['Net Income', 'Profit After Tax', 'PAT']:
                if col in income_stmt.columns:
                    net_income = income_stmt[col].dropna()
                    break
            
            result = {'opm_current': None, 'npm_current': None, 'trend': 'unknown', 'score': 50}
            
            if operating_income is not None:
                common_idx = revenue.index.intersection(operating_income.index)
                if len(common_idx) > 0:
                    opm_series = (operating_income.loc[common_idx] / revenue.loc[common_idx]) * 100
                    if not opm_series.dropna().empty:
                        result['opm_current'] = round(float(opm_series.dropna().iloc[-1]), 2)
            
            if net_income is not None:
                common_idx = revenue.index.intersection(net_income.index)
                if len(common_idx) > 0:
                    npm_series = (net_income.loc[common_idx] / revenue.loc[common_idx]) * 100
                    npm_series = npm_series.dropna().sort_index()
                    
                    if not npm_series.empty:
                        result['npm_current'] = round(float(npm_series.iloc[-1]), 2)
                        
                        if len(npm_series) >= 3:
                            recent = npm_series.iloc[-1]
                            older = npm_series.iloc[-3]
                            result['trend'] = 'improving' if recent > older * 1.1 else 'declining' if recent < older * 0.9 else 'stable'
            
            result['score'] = self._score_margins(result['opm_current'], result['npm_current'], result['trend'])
            
            return result
            
        except Exception as e:
            logger.debug(f"Margin calculation error: {e}")
            return {'opm_current': None, 'npm_current': None, 'trend': 'unknown', 'score': 50}
    
    def _score_margins(self, opm: Optional[float], npm: Optional[float], trend: str) -> float:
        """Score margins based on profitability."""
        score = 50.0
        
        if opm is not None:
            if opm >= 25:
                score += 20
            elif opm >= 20:
                score += 15
            elif opm >= 15:
                score += 10
            elif opm >= 10:
                score += 5
            elif opm < 5:
                score -= 10
        
        if npm is not None:
            if npm >= 20:
                score += 15
            elif npm >= 15:
                score += 10
            elif npm >= 10:
                score += 5
            elif npm < 0:
                score -= 15
        
        if trend == 'improving':
            score += 5
        elif trend == 'declining':
            score -= 5
        
        return min(100, max(0, score))
    
    def _calc_cagr(self, series: pd.Series, years: int) -> Optional[float]:
        if len(series) < 2:
            return None
        actual_years = min(years, len(series) - 1)
        if actual_years <= 0:
            return None
        start = series.iloc[-(actual_years + 1)]
        end = series.iloc[-1]
        return calculate_cagr(start, end, actual_years)
    
    def _calculate_score(self, revenue: float, profit: float, roce: float, 
                         cashflow: float, leverage: float, margin: float) -> float:
        return round(revenue * 0.20 + profit * 0.20 + roce * 0.20 + 
                     cashflow * 0.20 + leverage * 0.10 + margin * 0.10, 1)
    
    @staticmethod
    def get_investable_universe_filter() -> Dict[str, Any]:
        """Get filter criteria for the investable universe."""
        return {
            'min_positive_cashflow_years': 5,
            'min_roce': 10,
            'max_debt_to_equity': 2.0,
            'min_years_listed': 3,
            'min_market_cap_cr': 100,
            'min_revenue_growth_3y': 0,
            'require_profitable': True,
        }
    
    def passes_fundamental_filter(
        self, 
        symbol: str, 
        financials: Dict[str, pd.DataFrame],
        stock_info: Dict[str, Any],
        filter_criteria: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """Check if a stock passes the fundamental quality filter."""
        criteria = filter_criteria or self.get_investable_universe_filter()
        failed_reasons = []
        
        try:
            result = self.analyze(symbol, financials, stock_info)
        except Exception as e:
            return False, [f"Could not analyze: {e}"]
        
        market_cap = stock_info.get('market_cap', 0)
        min_market_cap = criteria.get('min_market_cap_cr', 100) * 1e7
        if market_cap < min_market_cap:
            failed_reasons.append(f"Market cap too small")
        
        min_roce = criteria.get('min_roce', 10)
        if result.roce_current is not None and result.roce_current < min_roce:
            failed_reasons.append(f"ROCE {result.roce_current:.1f}% below minimum")
        
        max_de = criteria.get('max_debt_to_equity', 2.0)
        if result.debt_to_equity is not None and result.debt_to_equity > max_de:
            failed_reasons.append(f"D/E ratio {result.debt_to_equity:.2f} above maximum")
        
        if result.red_flags:
            for rf in result.red_flags:
                failed_reasons.append(f"Red flag: {rf}")
        
        passes = len(failed_reasons) == 0
        return passes, failed_reasons
