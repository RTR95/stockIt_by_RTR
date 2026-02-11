"""Buffett checklist scoring and analysis."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.helpers import calculate_cagr, safe_divide


@dataclass
class BuffettRuleResult:
    name: str
    category: str
    passed: Optional[bool]
    value_display: str
    threshold: str
    notes: str = ""
    counts_for_score: bool = True
    weight: int = 1

    @property
    def status(self) -> str:
        if self.passed is None:
            return "Insufficient data"
        return "Pass" if self.passed else "Fail"


@dataclass
class BuffettAnalysisResult:
    score: int
    total_points: int
    approved: bool
    rules: List[BuffettRuleResult]

    @property
    def coverage(self) -> float:
        if self.total_points == 0:
            return 0.0
        available = sum(
            1
            for r in self.rules
            if r.counts_for_score and r.passed is not None
        )
        return round(available / self.total_points * 100, 1)


class BuffettAnalyzer:
    """Compute Buffett-style checklist and score."""

    def __init__(self, market_cap_inr_threshold: float = 8e11):
        # ~$10B USD ≈ ₹8e11 (rough conversion)
        self.market_cap_inr_threshold = market_cap_inr_threshold

    def analyze(self, stock_info: Dict[str, Any], financials: Dict[str, pd.DataFrame]) -> BuffettAnalysisResult:
        income_stmt = financials.get("income_statement", pd.DataFrame())
        balance_sheet = financials.get("balance_sheet", pd.DataFrame())
        cash_flow = financials.get("cash_flow", pd.DataFrame())
        dividends = financials.get("dividends")

        rules: List[BuffettRuleResult] = []

        # Hard filters
        debt_to_equity = self._debt_to_equity(balance_sheet)
        rules.append(self._rule(
            name="Debt / Equity < 0.5",
            category="Hard Filter",
            passed=self._passes_ratio_lt(debt_to_equity, 0.5),
            value_display=self._fmt_ratio(debt_to_equity),
            threshold="< 0.5",
            weight=2,
        ))

        current_ratio = self._current_ratio(balance_sheet)
        rules.append(self._rule(
            name="Current Ratio between 1.5 and 2.5",
            category="Hard Filter",
            passed=self._passes_range(current_ratio, 1.5, 2.5),
            value_display=self._fmt_ratio(current_ratio),
            threshold="1.5 – 2.5",
        ))

        pb_ratio = self._price_to_book(stock_info)
        rules.append(self._rule(
            name="Price / Book < 1.5",
            category="Hard Filter",
            passed=self._passes_ratio_lt(pb_ratio, 1.5),
            value_display=self._fmt_ratio(pb_ratio),
            threshold="< 1.5",
        ))

        roa = self._return_on_assets(income_stmt, balance_sheet)
        rules.append(self._rule(
            name="Return on Assets > 6%",
            category="Hard Filter",
            passed=self._passes_ratio_gt(roa, 0.06),
            value_display=self._fmt_percent_ratio(roa),
            threshold="> 6%",
        ))

        icr = self._interest_coverage(income_stmt)
        rules.append(self._rule(
            name="Interest Coverage > 5x",
            category="Hard Filter",
            passed=self._passes_ratio_gt(icr, 5),
            value_display=self._fmt_ratio(icr, suffix="x"),
            threshold="> 5x",
            weight=2,
        ))

        # Trend filters
        roe_result = self._roe_consistency(income_stmt, balance_sheet)
        rules.append(self._rule(
            name="ROE Avg 10Y > 8% and no year < 5%",
            category="Trend Filter",
            passed=roe_result[0],
            value_display=roe_result[1],
            threshold="Avg > 8%, Min > 5%",
            notes=roe_result[2],
            weight=2,
        ))

        book_growth = self._book_value_growth(balance_sheet)
        rules.append(self._rule(
            name="Book Value growth (current > 5Y ago)",
            category="Trend Filter",
            passed=book_growth[0],
            value_display=book_growth[1],
            threshold="CAGR > 0%",
            notes=book_growth[2],
        ))

        eps_growth = self._eps_growth(income_stmt)
        rules.append(self._rule(
            name="EPS grew in 7 of last 10 years",
            category="Trend Filter",
            passed=eps_growth[0],
            value_display=eps_growth[1],
            threshold=">= 7 increases",
            notes=eps_growth[2],
        ))

        dividend_growth = self._dividend_growth(cash_flow, stock_info, dividends)
        rules.append(self._rule(
            name="Stable dividend growth (5Y)",
            category="Trend Filter",
            passed=dividend_growth[0],
            value_display=dividend_growth[1],
            threshold="Yield > 0 and non-decreasing",
            notes=dividend_growth[2],
        ))

        # Economic moat (composite scoreable rule)
        moat_passed, moat_display, moat_note, moat_proxies = self._economic_moat(
            income_stmt, balance_sheet, stock_info
        )
        rules.append(self._rule(
            name="Economic Moat (2 of 3 proxies)",
            category="Economic Moat",
            passed=moat_passed,
            value_display=moat_display,
            threshold=">= 2 proxies pass",
            notes=moat_note,
        ))

        # Include moat proxies in output for transparency (not scored)
        rules.extend(moat_proxies)

        score, total = self._score_rules(rules)
        approved = score >= (total * 0.8) # 80% pass rate required
        return BuffettAnalysisResult(score=score, total_points=total, approved=approved, rules=rules)

    def _rule(
        self,
        name: str,
        category: str,
        passed: Optional[bool],
        value_display: str,
        threshold: str,
        notes: str = "",
        counts_for_score: bool = True,
        weight: int = 1,
    ) -> BuffettRuleResult:
        return BuffettRuleResult(
            name=name,
            category=category,
            passed=passed,
            value_display=value_display,
            threshold=threshold,
            notes=notes,
            counts_for_score=counts_for_score,
            weight=weight,
        )

    def _score_rules(self, rules: List[BuffettRuleResult]) -> Tuple[int, int]:
        total = sum(r.weight for r in rules if r.counts_for_score)
        score = sum(r.weight for r in rules if r.counts_for_score and r.passed)
        return score, total

    def _get_series(self, df: pd.DataFrame, columns: List[str]) -> Optional[pd.Series]:
        if df is None or df.empty:
            return None
        for col in columns:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                if not series.empty:
                    return series.sort_index()
        return None

    def _debt_to_equity(self, balance_sheet: pd.DataFrame) -> Optional[float]:
        total_debt = self._get_series(
            balance_sheet,
            ["Total Debt", "Long Term Debt", "Long-term Debt", "Short Term Debt", "Current Debt"],
        )
        if total_debt is None:
            return None
        equity = self._get_series(
            balance_sheet,
            ["Total Stockholder Equity", "Stockholders Equity", "Total Equity"],
        )
        if equity is None:
            return None
        return self._latest_ratio(total_debt, equity)

    def _current_ratio(self, balance_sheet: pd.DataFrame) -> Optional[float]:
        current_assets = self._get_series(
            balance_sheet, ["Current Assets", "Total Current Assets"]
        )
        current_liab = self._get_series(
            balance_sheet, ["Current Liabilities", "Total Current Liabilities"]
        )
        if current_assets is None or current_liab is None:
            return None
        return self._latest_ratio(current_assets, current_liab)

    def _price_to_book(self, stock_info: Dict[str, Any]) -> Optional[float]:
        pb = stock_info.get("pb_ratio")
        if pb is not None and not pd.isna(pb):
            return float(pb)
        book_value = stock_info.get("book_value")
        price = stock_info.get("current_price")
        if book_value and price:
            return safe_divide(price, book_value, default=None)
        return None

    def _return_on_assets(self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame) -> Optional[float]:
        net_income = self._get_series(
            income_stmt, ["Net Income", "Profit After Tax", "PAT", "Net Profit"]
        )
        total_assets = self._get_series(balance_sheet, ["Total Assets", "Total Asset"])
        if net_income is None or total_assets is None:
            return None
        return self._latest_ratio(net_income, total_assets)

    def _interest_coverage(self, income_stmt: pd.DataFrame) -> Optional[float]:
        ebit = self._get_series(
            income_stmt,
            ["EBIT", "Operating Income", "Operating Profit", "Earnings Before Interest And Taxes"],
        )
        interest = self._get_series(
            income_stmt, ["Interest Expense", "Interest", "Finance Costs"]
        )
        if ebit is None or interest is None:
            return None
        interest = interest.abs().replace(0, np.nan)
        return self._latest_ratio(ebit, interest)

    def _roe_consistency(
        self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame
    ) -> Tuple[Optional[bool], str, str]:
        net_income = self._get_series(
            income_stmt, ["Net Income", "Profit After Tax", "PAT", "Net Profit"]
        )
        equity = self._get_series(
            balance_sheet, ["Total Stockholder Equity", "Stockholders Equity", "Total Equity"]
        )
        if net_income is None or equity is None:
            return None, "N/A", "Missing net income or equity history"
        roe = (net_income / equity.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
        if roe.empty:
            return None, "N/A", "ROE series unavailable"
        roe = roe.sort_index()
        recent = roe.tail(10)
        avg = recent.mean()
        min_floor = recent.min()
        passed = bool(avg > 0.08 and min_floor > 0.05)
        value = f"Avg {avg*100:.1f}%, Min {min_floor*100:.1f}%"
        note = f"{len(recent)} years available"
        return passed, value, note

    def _book_value_growth(self, balance_sheet: pd.DataFrame) -> Tuple[Optional[bool], str, str]:
        equity = self._get_series(
            balance_sheet, ["Total Stockholder Equity", "Stockholders Equity", "Total Equity"]
        )
        if equity is None or equity.empty:
            return None, "N/A", "Missing equity history"
        equity = equity.sort_index()
        if len(equity) < 2:
            return None, "N/A", "Not enough history"
        years = min(5, len(equity) - 1)
        start = float(equity.iloc[-(years + 1)])
        end = float(equity.iloc[-1])
        cagr = calculate_cagr(start, end, years)
        passed = cagr is not None and cagr > 0
        value = f"CAGR {cagr:.1f}%" if cagr is not None else "N/A"
        note = f"{years}Y window"
        return passed, value, note

    def _eps_growth(self, income_stmt: pd.DataFrame) -> Tuple[Optional[bool], str, str]:
        eps_series = self._get_series(
            income_stmt,
            [
                "Basic EPS",
                "Diluted EPS",
                "Basic EPS - Continuing Ops",
                "Diluted EPS - Continuing Ops",
                "EPS",
            ],
        )
        proxy_note = ""
        if eps_series is None or eps_series.empty:
            eps_series = self._get_series(
                income_stmt,
                ["Net Income", "Profit After Tax", "PAT", "Net Profit"],
            )
            proxy_note = "Using net income as EPS proxy"
        if eps_series is None or eps_series.empty:
            return None, "N/A", "Missing EPS or earnings history"
        eps_series = eps_series.sort_index()
        if len(eps_series) < 8:
            return None, "N/A", "Need at least 8 years for 7 increases"
        recent = eps_series.tail(10)
        increases = int((recent.diff() > 0).sum())
        passed = increases >= 7
        value = f"{increases} increases in {len(recent)-1} intervals"
        note = f"{len(recent)} years available"
        if proxy_note:
            note = f"{note} ({proxy_note})"
        return passed, value, note

    def _dividend_growth(
        self,
        cash_flow: pd.DataFrame,
        stock_info: Dict[str, Any],
        dividends: Optional[pd.Series] = None,
    ) -> Tuple[Optional[bool], str, str]:
        div_series = None
        if dividends is not None and isinstance(dividends, pd.Series) and not dividends.empty:
            div_series = dividends.copy()
        if div_series is None:
            div_series = self._get_series(
                cash_flow, ["Dividends Paid", "Cash Dividends Paid", "Dividends"]
            )
        div_yield = stock_info.get("dividend_yield")
        if div_series is None or div_series.empty:
            return None, "N/A", "Missing dividend history"
        if isinstance(div_series.index, pd.DatetimeIndex):
            yearly = div_series.abs().groupby(div_series.index.year).sum().sort_index()
        else:
            yearly = div_series.abs().sort_index()
        if len(yearly) < 5:
            return None, "N/A", "Need at least 5 years"
        recent = yearly.tail(5)
        non_decreasing = bool((recent.diff().fillna(0) >= 0).all())
        has_yield = div_yield is not None and div_yield > 0
        passed = non_decreasing and has_yield
        value = f"Yield {div_yield:.2f}%" if div_yield is not None else "Yield N/A"
        note = "Non-decreasing in last 5Y" if non_decreasing else "Dividend dips detected"
        return passed, value, note

    def _economic_moat(
        self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame, stock_info: Dict[str, Any]
    ) -> Tuple[Optional[bool], str, str, List[BuffettRuleResult]]:
        proxies: List[BuffettRuleResult] = []

        gross_margin = self._gross_margin_avg(income_stmt)
        gm_pass = self._passes_ratio_gt(gross_margin, 0.40)
        proxies.append(self._rule(
            name="Gross Margin Avg 5Y > 40%",
            category="Economic Moat (Proxy)",
            passed=gm_pass,
            value_display=self._fmt_percent_ratio(gross_margin),
            threshold="> 40%",
            counts_for_score=False,
        ))

        roic = self._roic(income_stmt, balance_sheet)
        roic_pass = self._passes_ratio_gt(roic, 0.15)
        proxies.append(self._rule(
            name="ROIC > 15%",
            category="Economic Moat (Proxy)",
            passed=roic_pass,
            value_display=self._fmt_percent_ratio(roic),
            threshold="> 15%",
            counts_for_score=False,
        ))

        market_cap = stock_info.get("market_cap")
        cap_pass = None
        cap_display = "N/A"
        if market_cap is not None and not pd.isna(market_cap):
            cap_pass = float(market_cap) >= self.market_cap_inr_threshold
            cap_display = f"₹{float(market_cap):,.0f}"
        proxies.append(self._rule(
            name="Large Cap (Market Cap > ~$10B)",
            category="Economic Moat (Proxy)",
            passed=cap_pass,
            value_display=cap_display,
            threshold=f"> ₹{self.market_cap_inr_threshold:,.0f}",
            counts_for_score=False,
        ))

        available = [p for p in proxies if p.passed is not None]
        passed_count = sum(1 for p in available if p.passed)
        if len(available) < 2:
            return None, "N/A", "Not enough moat proxy data", proxies
        moat_passed = passed_count >= 2
        value = f"{passed_count} of {len(available)} proxies"
        note = "Uses gross margin, ROIC, and market cap as proxies"
        return moat_passed, value, note, proxies

    def _gross_margin_avg(self, income_stmt: pd.DataFrame) -> Optional[float]:
        gross_profit = self._get_series(income_stmt, ["Gross Profit"])
        revenue = self._get_series(income_stmt, ["Total Revenue", "Revenue", "Net Sales"])
        if gross_profit is None or revenue is None:
            return None
        margin = (gross_profit / revenue.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
        if margin.empty:
            return None
        return float(margin.tail(5).mean())

    def _roic(self, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame) -> Optional[float]:
        ebit = self._get_series(
            income_stmt,
            ["EBIT", "Operating Income", "Operating Profit", "Earnings Before Interest And Taxes"],
        )
        if ebit is None:
            return None
        total_assets = self._get_series(balance_sheet, ["Total Assets", "Total Asset"])
        current_liab = self._get_series(
            balance_sheet, ["Current Liabilities", "Total Current Liabilities"]
        )
        if total_assets is None or current_liab is None:
            return None
        invested_capital = (total_assets - current_liab).replace(0, np.nan)
        return self._latest_ratio(ebit, invested_capital)

    def _latest_ratio(self, numerator: pd.Series, denominator: pd.Series) -> Optional[float]:
        common = numerator.index.intersection(denominator.index)
        if len(common) == 0:
            return None
        num = numerator.loc[common].iloc[-1]
        denom = denominator.loc[common].iloc[-1]
        if denom == 0 or pd.isna(denom):
            return None
        return float(num) / float(denom)

    def _passes_ratio_lt(self, value: Optional[float], threshold: float) -> Optional[bool]:
        if value is None or pd.isna(value):
            return None
        return value < threshold

    def _passes_ratio_gt(self, value: Optional[float], threshold: float) -> Optional[bool]:
        if value is None or pd.isna(value):
            return None
        return value > threshold

    def _passes_range(self, value: Optional[float], lower: float, upper: float) -> Optional[bool]:
        if value is None or pd.isna(value):
            return None
        return lower < value < upper

    def _fmt_ratio(self, value: Optional[float], suffix: str = "") -> str:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{value:.2f}{suffix}"

    def _fmt_percent_ratio(self, value: Optional[float]) -> str:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{value * 100:.1f}%"
