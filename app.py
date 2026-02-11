"""
Stocron by RTR - Main Application
Tagline: Indian Equity Intelligence

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import json
from typing import Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config import load_config, get_config
from src.utils.helpers import calculate_cagr, format_percentage
from src.data.sources import DataSourceManager
from src.data.macro import get_macro_provider
from src.data.database import DatabaseManager
from src.data.cache import CacheManager
from src.engine.governance import GovernanceAnalyzer
from src.engine.financial import FinancialAnalyzer
from src.engine.valuation import ValuationAnalyzer
from src.engine.market import MarketBehaviourAnalyzer
from src.engine.ml_context import MLContextAnalyzer
from src.analysis.signal_generator import SignalGenerator, UserProfile, Signal
from src.analysis.explainability import ExplainabilityEngine
from src.analysis.red_flags import RedFlagDetector
from src.analysis.buffett import BuffettAnalyzer
from src.utils.web_search import WebSearch
from src.features.time_travel import TimeTravelEngine
from src.features.scenario_simulator import ScenarioSimulator
from src.ui.components import render_user_profile, render_signal_badge, render_why_not_buy, render_red_flags, render_dimension_scores, render_footer
from src.ui.charts import create_price_chart, create_score_radar, create_drawdown_chart

# ML Models Integration
from src.ml_models.ensemble import MLEnsemble, EnsembleConfig, create_ensemble_from_config
from src.ml_models.base import ModelType

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Stocron by RTR", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

SEBI_DISCLAIMER_PATH = Path(".streamlit/sebi_disclaimer.json")

st.markdown("""
<style>
    .main { padding: 0 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    div[data-testid="metric-container"] { background-color: #262730; padding: 10px; border-radius: 5px; }
    /* Widen dialog overlays (Streamlit v1.30+) */
    div[role="dialog"] { width: 90vw !important; max-width: 90vw !important; }
    div[role="dialog"] > div { max-height: 90vh; overflow-y: auto; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _load_stock_info_index() -> pd.DataFrame:
    """Load stock info JSONs into a DataFrame for suggestions."""
    info_dir = Path("data/stock_info")
    records = []
    if not info_dir.exists():
        return pd.DataFrame()
    for f in info_dir.glob("*.json"):
        try:
            raw = json.loads(f.read_text())
            records.append({
                "symbol": f.stem.upper(),
                "name": raw.get("longName", raw.get("shortName", f.stem)),
                "sector": raw.get("sector"),
                "industry": raw.get("industry"),
                "market_cap": raw.get("marketCap"),
                "current_price": raw.get("currentPrice", raw.get("regularMarketPrice")),
                "pe_ratio": raw.get("trailingPE"),
                "dividend_yield": (raw.get("dividendYield") or 0) * 100 if raw.get("dividendYield") is not None else None,
                "roe": (raw.get("returnOnEquity") or 0) * 100 if raw.get("returnOnEquity") else None,
                "debt_to_equity": raw.get("debtToEquity"),
                "beta": raw.get("beta"),
            })
        except Exception:
            continue
    df = pd.DataFrame(records)
    # Coerce numeric fields to floats for scoring
    for col in [
        "market_cap", "current_price", "pe_ratio", "dividend_yield",
        "roe", "debt_to_equity", "beta"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _get_sebi_disclaimer_state() -> str:
    """Return the last acknowledged date (YYYY-MM-DD) or empty string."""
    try:
        if SEBI_DISCLAIMER_PATH.exists():
            payload = json.loads(SEBI_DISCLAIMER_PATH.read_text())
            return payload.get("ack_date", "")
    except Exception:
        return ""
    return ""


def _set_sebi_disclaimer_state(ack_date: str) -> None:
    """Persist acknowledgment date (YYYY-MM-DD) to disk."""
    try:
        SEBI_DISCLAIMER_PATH.parent.mkdir(parents=True, exist_ok=True)
        SEBI_DISCLAIMER_PATH.write_text(json.dumps({"ack_date": ack_date}))
    except Exception:
        return


def _calculate_price_cagr(prices: pd.DataFrame, years: int) -> float:
    """Calculate price CAGR over the requested window (in years)."""
    if prices is None or prices.empty or years <= 0:
        return None
    df = prices.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        cutoff = datetime.now() - timedelta(days=years * 365)
        df = df[df["date"] >= cutoff]
        if df.empty:
            return None
        start_date = df["date"].iloc[0]
        end_date = df["date"].iloc[-1]
    else:
        df = df.sort_index()
        start_date = pd.to_datetime(df.index[0])
        end_date = pd.to_datetime(df.index[-1])

    price_series = None
    if "close" in df.columns:
        price_series = df["close"]
    elif "Close" in df.columns:
        price_series = df["Close"]
    elif df.shape[1] >= 4:
        price_series = df.iloc[:, 3]
    elif df.shape[1] >= 1:
        price_series = df.iloc[:, -1]

    if price_series is None or price_series.empty:
        return None

    years_span = (end_date - start_date).days / 365.25
    if years_span <= 0:
        return None

    start_price = float(price_series.iloc[0])
    end_price = float(price_series.iloc[-1])
    return calculate_cagr(start_price, end_price, years_span)


def _filter_market_cap(df: pd.DataFrame, market_cap_filter: str) -> pd.DataFrame:
    if df.empty or not market_cap_filter or market_cap_filter == "Any":
        return df
    # ₹20,000 Cr = 2e11; ₹5,000 Cr = 5e10
    large_cap = 2e11
    mid_cap = 5e10
    if "Large Cap" in market_cap_filter:
        return df[df["market_cap"] >= large_cap]
    if "Mid Cap" in market_cap_filter:
        return df[(df["market_cap"] >= mid_cap) & (df["market_cap"] < large_cap)]
    if "Small Cap" in market_cap_filter:
        return df[df["market_cap"] < mid_cap]
    return df


def _score_candidates(df: pd.DataFrame, profile: Dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df
    risk = profile.get("risk_appetite", "medium")
    expected = profile.get("expected_return", 15)
    tenure = profile.get("holding_tenure", 5)

    def score_row(row: pd.Series) -> float:
        score = 50.0
        roe = row.get("roe") or 0
        div = row.get("dividend_yield") or 0
        de = row.get("debt_to_equity") or 0
        pe = row.get("pe_ratio") or 0
        mcap = row.get("market_cap") or 0
        beta = row.get("beta") or 1

        # Quality / returns
        score += min(roe, 30) / 30 * 30
        if expected >= 20 and roe < 12:
            score -= 8
        if expected <= 10 and roe >= 20:
            score += 4

        # Leverage penalty
        score -= min(de, 3) / 3 * 15

        # Dividends favor long-tenure/low risk
        div_weight = 15 if risk == "low" else 8 if risk == "medium" else 4
        if tenure >= 10:
            div_weight += 3
        score += min(div, 5) / 5 * div_weight

        # PE preference
        if pe:
            if pe < 8:
                score -= 3
            elif pe <= 25:
                score += 8
            elif pe <= 50:
                score += 2 if risk == "high" else -4
            else:
                score -= 8

        # Market cap preference
        if mcap:
            if risk == "low":
                score += 10 if mcap >= 2e11 else 4 if mcap >= 5e10 else -6
            elif risk == "high":
                score += 8 if 5e10 <= mcap < 2e11 else 4 if mcap >= 2e11 else 6
            else:
                score += 6 if mcap >= 5e10 else -2

        # Beta preference
        if beta:
            if risk == "low":
                score += 6 if beta <= 1 else -4 if beta > 1.5 else 0
            elif risk == "high":
                score += 3 if beta >= 1 else 0

        return score

    scored = df.copy()
    scored["score"] = scored.apply(score_row, axis=1)
    return scored.sort_values("score", ascending=False)


def _render_dialog(title: str, render_body):
    """Render a floating dialog if supported, else return False."""
    if hasattr(st, "dialog"):
        @st.dialog(title)
        def _dlg():
            render_body()
        _dlg()
        return True
    if hasattr(st, "experimental_dialog"):
        @st.experimental_dialog(title)
        def _dlg():
            render_body()
        _dlg()
        return True
    return False


def _render_wiki_modal():
    def _body():
        st.markdown("""
### Core Concepts
- **Signal (BUY/HOLD/AVOID/SELL)**: Composite outcome across governance, financials, valuation, market behavior, and ML context. It’s a suitability score, not a trade trigger.
- **Composite Score (0–100)**: Weighted blend of engine scores (+ ML if enabled).
- **Why NOT to Buy**: Mandatory counter‑thesis to reduce confirmation bias.

### Governance (Stewardship Quality)
- **Promoter Holding**: % held by promoters. Higher + stable is generally safer.
- **Pledge Ratio**: % of promoter shares pledged. Rising pledges imply financing stress.
- **Dividend Consistency**: Years of steady payouts.
- **Auditor Stability**: Frequent changes are a governance risk marker.

### Financials (Business Strength)
- **Revenue CAGR**: `((End / Start)^(1/Years) - 1)`.
- **PAT CAGR**: Profit growth over 3/5 years.
- **ROCE**: `EBIT / (Total Assets - Current Liabilities)`.
- **Earnings Quality**: `CFO / PAT` (< 0.5 is weak).
- **FCF Yield**: `Free Cash Flow / Market Cap`.
- **Debt‑to‑Equity**: Leverage risk; rising trend is a concern.
- **Investable Universe Filter**: Flags inconsistent cash flow quality.

### Valuation (What You Pay)
- **P/E**: `Price / EPS`.
- **P/B**: `Price / Book Value`.
- **EV/EBITDA**: Capital‑structure‑adjusted valuation.
- **PEG**: `P/E / Earnings Growth`.
- **Historical Percentile**: Where current valuation sits vs its own history.

### Market Behavior (Risk & Cyclicality)
- **Max Drawdown**: Worst peak‑to‑trough fall.
- **Recovery Time**: Months to regain highs after >20% drawdowns.
- **Volatility Regime**: Low/Medium/High/Extreme from 1–3Y vol.
- **Beta**: Sensitivity to Nifty.
- **Sharpe**: `(Return – Risk‑Free) / Volatility`.
- **Relative Performance**: vs Nifty (1Y/3Y/5Y).

### Regime‑Aware Technicals
- **Market Regime**: Bull/Bear/Sideways from trend structure.
- **Regime‑Adjusted RSI**: RSI thresholds adapt by regime.

### Macro Context
- **Repo Rate**: Higher rates compress valuation.
- **USD‑INR**: Currency swings impact importers/exporters.
- **Crude Oil**: Energy/transport sensitivity.
- **CPI Inflation**: Higher inflation can squeeze margins and multiples.

### ML Insights (Optional)
- **Chronos Forecaster**: 5–30 day trend direction.
- **LightGBM Classifier**: Feature‑driven signal probabilities.
- **Walk‑Forward Validation**: Past → future split.
- **Risk‑Adjusted Targets**: Sharpe/Sortino instead of raw returns.
- **SHAP**: Feature‑level explanation of model reasoning.

### Time Travel Mode
- Runs analysis as‑of a historical date using only data available then.
- Useful to test logic in real past regimes.

### Scenario Simulator (Stress Testing)
Preset scenarios and intent:
- **Revenue Stagnation**: growth drops to 0% (growth dependency risk).
- **Margin Compression**: -500 bps OPM (cost pressure risk).
- **Rising Interest Rates**: +200 bps, PE compression (valuation risk).
- **Demand Slowdown**: negative volume growth (cycle risk).
- **Cost Inflation**: raw material spike with partial pass‑through.
- **Currency Depreciation**: INR down 15% (import cost risk).

Interpreting outcomes:
- **Good**: robust/resilient rating, small return impact, limited target damage.
- **Bad**: fragile rating, large return impact, material target price drawdown.
""")
    if _render_dialog("Wiki", _body):
        return
    with st.expander("Wiki", expanded=True):
        _body()


def _render_howto_modal():
    def _body():
        st.markdown("""
### Quick Start Flow
1. Set your profile in the sidebar.
2. Use **Suggest** to shortlist or search a stock directly.
3. Review **Overview → Why NOT → Charts → Scenarios → Time Travel**.
4. Use **ML Insights** as a second opinion if enabled.

### 1) Set Your Profile (Sidebar)
- **Expected CAGR**: Your target annual return.
- **Risk Appetite**: Low/Medium/High.
- **Holding Period**: Your investment horizon.
- **Price Range + Market Cap**: Filters for suggestion accuracy.
**Tip**: If you’re unsure, start with Medium risk and a 5–10 year horizon.

### 2) Suggestions
- Click **Suggest** to see profile-based candidates.
- Click any stock to run analysis.
**Tip**: Suggestions are ranked to match your profile; widen filters if the list is short.

### 3) Stock Search
- Use the dropdown to pick a symbol or type directly.
- Click **Analyze** to run the full pipeline.
**Tip**: Start with a known large-cap to learn how the signal behaves.

### 4) Overview Tab
- See the signal, composite score, and dimension breakdown.
- Review key positives and concerns.
- Read the red flags if present.
**Tip**: A strong score with major red flags should still be treated cautiously.
**Interpretation**:
- Governance strong + Financial weak = quality business but execution risk.
- Valuation expensive + Market strong = momentum, not necessarily margin of safety.

### 5) ML Insights Tab
- Shows ML signal, confidence, and price trend (if models loaded).
- Review SHAP explanations to understand why the model leaned bullish/bearish.
**Tip**: Use ML as a second opinion, not the primary decision.
**Interpretation**:
- High confidence but low composite score → ML sees short‑term upside, fundamentals lag.
- Low confidence across signals → treat as exploratory only.

### 6) Why NOT Tab
- Always read this section before acting.
- It highlights mismatches with your profile and core risks.
**Tip**: If any “Why NOT” item would keep you up at night, reduce position size or skip.

### 7) Charts Tab
- Full price history with candlesticks and volume.
- Drawdown chart to understand worst-case behavior.
**Tip**: Look for shallow drawdowns and faster recoveries for long-term holdings.
**Interpretation**:
- Repeated deep drawdowns → higher stress tolerance required.
- Long recovery → opportunity cost risk for long‑term capital.

### 8) Time Travel Tab
- Select a historical cutoff year and re-run analysis.
- Use this to sanity-check against past regimes.
**Tip**: Test major market events (e.g., 2008, 2020) to see how resilient the thesis is.
**Interpretation**:
- If signal flips wildly across regimes, demand a larger margin of safety.

### 9) Scenarios Tab
- Stress-test macro shocks and compare resilience.
- Use the results to size positions and risk-manage.
**Tip**: If multiple scenarios show “fragile,” wait for better valuation or safer entry.
**Interpretation**:
- Robust/Resilient → position sizing can be higher.
- Moderate/Fragile → size down or require better valuation.

### 10) Under the Hood (Sidebar)
- Expand **Under the Hood** for data source and ML model status.
**Tip**: If ML isn’t ready, run the training script or check model downloads.

### Common Beginner Mistakes (Avoid These)
- Over‑trusting a BUY signal without reading “Why NOT”.
- Ignoring drawdowns because returns look good.
- Using ML confidence as a substitute for fundamentals.
- Skipping Time Travel when testing new ideas.
""")
    if _render_dialog("How to Use", _body):
        return
    with st.expander("How to Use", expanded=True):
        _body()


def _render_sebi_disclaimer_modal():
    def _body():
        st.markdown(
            """
            ### SEBI Risk Disclosure
            - **Investments in securities market are subject to market risks.**
            - **Read all the related documents carefully before investing.**
            - Past performance is not indicative of future results.

            ### Important Notice
            This app provides decision support and educational insights only. It is **not** investment advice or a recommendation to buy or sell any security.
            """
        )

    return _render_dialog("Important Disclaimer", _body)


def _render_all_stocks_modal(app):
    def _body():
        st.session_state.all_stocks_rendered = True
        col_title, col_close = st.columns([5, 1])
        st.markdown("### Sectors")
        stock_info_df = _load_stock_info_index()
        if stock_info_df.empty:
            st.warning("No local stock metadata found. Run `python 2-download_all_stocks.py` first.")
            return
        
        stock_info_df = stock_info_df.copy()
        stock_info_df["sector"] = stock_info_df["sector"].fillna("Unknown")
        stock_info_df["current_price"] = pd.to_numeric(stock_info_df["current_price"], errors="coerce")
        
        # Sector filter buttons
        sectors = sorted([s for s in stock_info_df["sector"].unique() if s])
        sector_key = "all_stocks_sector"
        if sector_key not in st.session_state:
            st.session_state[sector_key] = "All sectors"
        
        buttons = ["All sectors"] + sectors
        columns_per_row = 6
        for row_start in range(0, len(buttons), columns_per_row):
            row_buttons = buttons[row_start:row_start + columns_per_row]
            cols = st.columns(columns_per_row)
            for col, label in zip(cols, row_buttons):
                is_selected = st.session_state.get(sector_key, "All sectors") == label
                btn_type = "primary" if is_selected else "secondary"
                if col.button(
                    label,
                    key=f"sector_btn_{label}",
                    width="stretch",
                    type=btn_type,
                ):
                    st.session_state[sector_key] = label
                    st.rerun()
            if len(row_buttons) < columns_per_row:
                for col in cols[len(row_buttons):]:
                    col.empty()
        
        selected_sector = st.session_state.get(sector_key, "All sectors")
        if selected_sector != "All sectors":
            stock_info_df = stock_info_df[stock_info_df["sector"] == selected_sector]

        holding_years = int(app.config.get("ui", {}).get("default_holding_tenure", 5) or 5)
        stock_info_df["cagr"] = stock_info_df["symbol"].apply(
            lambda s: app.data_manager.get_stock_cagr(s, holding_years)
        )
        
        display_df = stock_info_df[["symbol", "name", "sector", "current_price", "cagr"]].copy()
        display_df = display_df.sort_values("symbol")
        display_df.rename(columns={
            "symbol": "Symbol",
            "name": "Name",
            "sector": "Sector",
            "current_price": "Current Price",
            "cagr": f"CAGR ({holding_years}Y)",
        }, inplace=True)
        display_df["Analyze"] = False
        
        edited = st.data_editor(
            display_df,
            hide_index=True,
            num_rows="fixed",
            width="stretch",
            column_config={
                "Analyze": st.column_config.CheckboxColumn(
                    "Analyze",
                    help="Select a stock to analyze",
                    default=False,
                ),
                f"CAGR ({holding_years}Y)": st.column_config.NumberColumn(
                    f"CAGR ({holding_years}Y)",
                    format="%.2f%%",
                    help="Pre-calculated CAGR from the local database",
                ),
            },
            disabled=["Symbol", "Name", "Sector", "Current Price", f"CAGR ({holding_years}Y)"],
            key="all_stocks_editor",
        )
        
        selected = edited[edited["Analyze"] == True]
        if not selected.empty:
            symbol = selected.iloc[0]["Symbol"]
            st.session_state.selected_stock = symbol
            st.session_state.analyze_stock = symbol
            st.session_state.show_all_stocks = False
            st.rerun()
    
    if _render_dialog("All Stocks", _body):
        return
    with st.expander("All Stocks", expanded=True):
        _body()


class IndianEquityIntelligence:
    def __init__(self):
        self.config = load_config()
        self._data_manager = None
        self._db_manager = None
        self.governance = GovernanceAnalyzer()
        self.financial = FinancialAnalyzer()
        self.valuation = ValuationAnalyzer()
        self.market = MarketBehaviourAnalyzer()
        self.ml_context = MLContextAnalyzer()
        self.signal_gen = SignalGenerator()
        self.explainer = ExplainabilityEngine()
        self.red_flag = RedFlagDetector()
        self.buffett = BuffettAnalyzer()
        self.web_search = WebSearch()
        self.time_travel = TimeTravelEngine()
        self.scenario_sim = ScenarioSimulator()
        self.macro_provider = get_macro_provider()
        
        # Initialize ML Ensemble (3-layer architecture)
        self._ml_ensemble = None
        self._ml_initialized = False
        self._ml_enabled = self.config.get('ml_config', {}).get('enabled', True)
    
    @property
    def data_manager(self):
        if self._data_manager is None:
            offline_mode = bool(self.config.get("app", {}).get("offline_mode", True))
            self._data_manager = DataSourceManager(offline_mode=offline_mode)
        return self._data_manager
    
    @property
    def ml_ensemble(self):
        """Lazy initialization of ML ensemble."""
        if self._ml_ensemble is None and self._ml_enabled:
            try:
                ml_config = self.config.get('ml_config', {})
                
                # Build ensemble config from settings
                ensemble_config_dict = {
                    "forecaster": ml_config.get('forecaster', {}).get('model'),
                    "classifier": ml_config.get('classifier', {}).get('model'),
                    "explainer": ml_config.get('explainer', {}).get('model'),
                    "device": ml_config.get('device', 'auto'),
                    "models_dir": ml_config.get('models_dir', 'models'),
                }
                
                self._ml_ensemble = create_ensemble_from_config(ensemble_config_dict)
                
            except Exception as e:
                logger.warning(f"Failed to create ML ensemble: {e}")
                self._ml_enabled = False
        
        return self._ml_ensemble
    
    def initialize_ml(self):
        """Initialize ML models (call explicitly when ready to load into memory)."""
        if self._ml_initialized or not self._ml_enabled:
            return self._ml_initialized, []
        
        if self.ml_ensemble:
            try:
                success, messages = self.ml_ensemble.initialize(load_models=True)
                self._ml_initialized = success
                if success:
                    artifacts = self.ml_ensemble.get_classifier_artifacts()
                    if artifacts:
                        self.explainer = ExplainabilityEngine(ml_model=artifacts)
                return success, messages
            except Exception as e:
                logger.error(f"ML initialization failed: {e}")
                return False, [str(e)]
        
        return False, ["ML ensemble not configured"]
    
    def get_ml_status(self) -> dict:
        """Get status of ML models."""
        if not self._ml_enabled:
            return {"enabled": False, "reason": "ML disabled in config"}
        
        if not self.ml_ensemble:
            return {"enabled": True, "initialized": False, "reason": "Not yet initialized"}
        
        return {
            "enabled": True,
            "initialized": self._ml_initialized,
            **self.ml_ensemble.get_status()
        }
    
    def fetch_data(self, symbol: str, show_spinner: bool = True, show_errors: bool = True):
        def _fetch():
            info = self.data_manager.get_stock_info(symbol)
            if not info:
                if show_errors:
                    st.error(f"Could not find: {symbol}")
                return None
            # Fetch up to 30 years of history (or all available)
            prices = self.data_manager.get_price_history(symbol, years=30)
            if prices is None or prices.empty:
                if show_errors:
                    st.error(f"No price history for {symbol}")
                return None
            financials = self.data_manager.get_financials(symbol) or {}
            nifty = self.data_manager.get_price_history("NIFTY", years=30)
            return {'stock_info': info, 'price_history': prices, 'financials': financials,
                    'nifty_history': nifty, 'shareholding': pd.DataFrame(), 'dividends': pd.DataFrame()}
        if show_spinner:
            with st.spinner(f"Fetching data for {symbol}..."):
                return _fetch()
        return _fetch()
    
    def run_analysis(self, symbol: str, profile: UserProfile, data: dict, show_spinner: bool = True, enable_ml: bool = True):
        def _run():
            info = data['stock_info']
            prices = data['price_history']
            fins = data['financials']
            nifty = data['nifty_history']
            share = data.get('shareholding', pd.DataFrame())
            divs = data.get('dividends', pd.DataFrame())
            
            # Run traditional analysis engines
            gov = self.governance.analyze(symbol, info, share, divs)
            fin = self.financial.analyze(symbol, fins, info)
            val = self.valuation.analyze(symbol, info, prices, fins, earnings_growth=fin.pat_cagr_5y)
            mkt = self.market.analyze(symbol, prices, nifty)
            red_flags = self.red_flag.detect_all_flags(symbol, info, share, fins)
            buffett = self.buffett.analyze(info, fins)
            
            # Generate initial signal from rule-based system
            signal = self.signal_gen.generate_signal(
                profile, gov, fin, val, mkt,
                red_flags=[{'severity': r.severity, 'description': r.description} for r in red_flags]
            )
            
            # Run ML-enhanced analysis if enabled and available
            ml_prediction = None
            ml_skip_reason = None
            
            # Fetch news context (optional, can be async or parallelized in future)
            news_context = None
            if self.config.get('features', {}).get('enable_news', True):
                try:
                    news_context = self.web_search.get_stock_news(symbol, info.get('name', ''))
                except Exception as e:
                    logger.debug(f"News fetch failed: {e}")

            if enable_ml and self._ml_enabled and self.ml_ensemble and self._ml_initialized:
                try:
                    macro_data = None
                    if self.config.get('macro_config', {}).get('enabled', True):
                        try:
                            macro_data = self.macro_provider.get_all_macro_data(years=10)
                        except Exception as e:
                            logger.debug(f"Macro data fetch failed: {e}")

                    # Respect investable universe filter before ML forecasting
                    respect_investable_filter = bool(
                        self.config.get('ml_config', {}).get('respect_investable_filter', False)
                    )
                    if respect_investable_filter and hasattr(fin, 'investable') and not fin.investable:
                        if getattr(fin, "investable_reasons", None):
                            ml_skip_reason = "Investable universe filter: " + "; ".join(fin.investable_reasons)
                        else:
                            ml_skip_reason = "Investable universe filter blocked ML."
                        logger.info("Skipping ML prediction due to investable filter")
                    else:
                        with st.spinner("Running ML-enhanced analysis..."):
                            ml_prediction = self.ml_ensemble.predict(
                                symbol=symbol,
                                prices=prices,
                                governance_result=gov,
                                financial_result=fin,
                                valuation_result=val,
                                market_result=mkt,
                                macro_data=macro_data,
                                company_name=info.get('name', symbol),
                                current_signal=signal.signal,
                                current_confidence=signal.confidence,
                                red_flags=[r.description for r in red_flags],
                            )
                        
                        # Blend ML score with rule-based score
                        if ml_prediction and ml_prediction.success:
                            ml_weight = self.config.get('ml_config', {}).get('ml_weight_in_composite', 0.15)
                            
                            # Update composite score with ML contribution
                            blended_score = (
                                signal.composite_score * (1 - ml_weight) + 
                                ml_prediction.ml_score * ml_weight
                            )
                            signal.composite_score = blended_score
                            
                            # If ML strongly disagrees, add to warnings
                            if ml_prediction.ml_signal != signal.signal:
                                if ml_prediction.ml_confidence > 0.7:
                                    signal.key_negatives.append(
                                        f"ML model suggests {ml_prediction.ml_signal} "
                                        f"({ml_prediction.ml_confidence:.0%} confidence)"
                                    )
                                    
                except Exception as e:
                    logger.warning(f"ML prediction failed: {e}")
                    ml_prediction = None
            
            # Generate explanations
            explain = self.explainer.generate_explanation(
                signal, profile, gov, fin, val, mkt, info,
                [{'severity': r.severity, 'description': r.description} for r in red_flags],
                ml_features=(
                    ml_prediction.feature_set.to_dataframe()
                    if ml_prediction and ml_prediction.feature_set
                    else None
                ),
                prediction_class=(
                    {"BUY": 0, "HOLD": 1, "AVOID": 2, "SELL": 3}.get(
                        ml_prediction.ml_signal, None
                    )
                    if ml_prediction and ml_prediction.success
                    else None
                ),
                external_context=news_context
            )
            
            return {
                'signal': signal, 
                'explain': explain, 
                'governance': gov, 
                'financial': fin,
                'valuation': val, 
                'market': mkt, 
                'red_flags': red_flags,
                'buffett': buffett,
                'ml_prediction': ml_prediction,  # NEW: ML results
                'ml_skip_reason': ml_skip_reason,
            }
        if show_spinner:
            with st.spinner("Running analysis..."):
                return _run()
        return _run()


def main():
    if 'app' not in st.session_state:
        st.session_state.app = IndianEquityIntelligence()
        st.session_state.ml_init_attempted = False
    if 'enable_ml' not in st.session_state:
        st.session_state.enable_ml = False
    if 'sebi_disclaimer_shown' not in st.session_state:
        today = datetime.now().date().isoformat()
        last_ack = _get_sebi_disclaimer_state()
        st.session_state.sebi_disclaimer_shown = (last_ack == today)
        st.session_state.sebi_disclaimer_ack_date = last_ack
    app = st.session_state.app
    
    # Auto-initialize ML models on first load if configured
    if not st.session_state.ml_init_attempted:
        ml_config = app.config.get('ml_config', {})
        attempted = False
        if (
            st.session_state.enable_ml
            and ml_config.get('enabled')
            and ml_config.get('auto_initialize', True)
        ):
            attempted = True
            with st.spinner("🚀 Initializing ML models... (first time only)"):
                success, messages = app.initialize_ml()
                if success:
                    st.toast("✓ ML models ready!", icon="🤖")
                else:
                    for msg in messages:
                        logger.info(f"ML init: {msg}")
        if attempted:
            st.session_state.ml_init_attempted = True
    
    should_show_disclaimer = not st.session_state.sebi_disclaimer_shown
    if should_show_disclaimer:
        today = datetime.now().date().isoformat()
        st.session_state.sebi_disclaimer_shown = True
        st.session_state.sebi_disclaimer_ack_date = today
        _set_sebi_disclaimer_state(today)

        shown = _render_sebi_disclaimer_modal()
        if not shown:
            st.warning(
                "SEBI Risk Disclosure: Investments in securities market are subject to market risks. "
                "Read all the related documents carefully before investing."
            )

    with st.sidebar:
        st.markdown(
            """
            <style>
              @import url('https://fonts.googleapis.com/css2?family=Bungee+Shade&family=Lexend+Peta:wght@300;400;500;600&display=swap');
            </style>
            <div style="line-height: 1.1;">
              <div style="font-family: 'Bungee Shade', cursive; font-size: 30px; font-weight: 400;">
                📊 Stocron
              </div>
              <div style="font-family: 'Lexend Peta', sans-serif; font-size: 12px; color: #6c757d;">
                &nbsp;by&nbsp;&nbsp;&nbsp;<a href="https://www.youtube.com/@RTR-Unfiltered" target="_blank" rel="noopener noreferrer">RTR Unfiltered</a>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.caption("Local-first • Explainable")
        st.markdown("---")
        profile_dict = render_user_profile()
        profile = UserProfile(expected_return=profile_dict['expected_return'],
                              risk_appetite=profile_dict['risk_appetite'],
                              holding_tenure=profile_dict['holding_tenure'])
        profile_cache_key = json.dumps(profile_dict, sort_keys=True)
        st.markdown("---")
        st.subheader("Analysis Options")
        st.session_state.enable_ml = st.checkbox(
            "Enable ML (slower)",
            value=st.session_state.enable_ml,
            help="Uses Chronos/Qwen models; can add minutes per analysis."
        )
        st.markdown("---")
        with st.expander("Under the Hood", expanded=False):
            st.subheader("Data Status")
            try:
                available = app.data_manager.get_available_sources()
                for s in available:
                    st.success(f"✓ {s.replace('_', ' ').title()}")
            except Exception:
                st.info("Data sources load on first search")
            
            # ML Models Status
            st.markdown("---")
            st.subheader("ML Models")
            ml_status = app.get_ml_status()
            
            if ml_status.get('enabled'):
                if not ml_status.get('initialized'):
                    if st.button("🚀 Initialize ML Models", width="stretch"):
                        with st.spinner("Loading ML models..."):
                            success, messages = app.initialize_ml()
                            for msg in messages:
                                st.info(msg)
                            if success:
                                st.success("ML models ready! Re-run analysis to see ML insights.")
                                # Clear previous results to force re-analysis
                                if 'results' in st.session_state:
                                    del st.session_state.results
                            st.rerun()
                else:
                    # Show status of each layer
                    for layer in ['forecaster', 'classifier', 'explainer']:
                        layer_info = ml_status.get(layer, {})
                        model_name = layer_info.get('model') or 'Not configured'
                        is_ready = layer_info.get('ready', False)
                        
                        if is_ready:
                            st.success(f"✓ {layer.title()}: {model_name}")
                        elif model_name != 'Not configured':
                            st.warning(f"⚠ {layer.title()}: {model_name}")
                        else:
                            st.info(f"○ {layer.title()}: Disabled")
                    
                    # Show re-analyze button if results exist but ML wasn't used
                    if hasattr(st.session_state, 'results') and st.session_state.results:
                        ml_pred = st.session_state.results.get('ml_prediction')
                        if not ml_pred or not ml_pred.success:
                            if st.button("🔄 Re-analyze with ML", width="stretch"):
                                symbol = st.session_state.get('symbol')
                                current_profile = st.session_state.get('profile', profile)
                                if symbol:
                                    data = app.fetch_data(symbol)
                                    if data:
                                        st.session_state.data = data
                                        st.session_state.enable_ml = True
                                        st.session_state.results = app.run_analysis(
                                            symbol,
                                            current_profile,
                                            data,
                                            enable_ml=True
                                        )
                                        st.rerun()
            else:
                st.info("ML disabled in config")
        
        st.markdown("---")
        col_wiki, col_howto = st.columns(2)
        with col_wiki:
            if st.button("Wiki", width="stretch"):
                st.session_state.show_wiki = True
                st.session_state.show_howto = False
                st.session_state.show_all_stocks = False
        with col_howto:
            if st.button("How-to?", width="stretch"):
                st.session_state.show_howto = True
                st.session_state.show_wiki = False
                st.session_state.show_all_stocks = False
        if st.button("View all Stocks", width="stretch"):
            st.session_state.show_all_stocks = True
            st.session_state.show_wiki = False
            st.session_state.show_howto = False

        render_footer()

    if st.session_state.get("show_wiki"):
        _render_wiki_modal()
        st.session_state.show_wiki = False
    elif st.session_state.get("show_howto"):
        _render_howto_modal()
        st.session_state.show_howto = False
    elif st.session_state.get("show_all_stocks"):
        st.session_state.all_stocks_rendered = False
        _render_all_stocks_modal(app)
        if not st.session_state.all_stocks_rendered:
            st.session_state.show_all_stocks = False
    
    st.title("Stock Analysis")
    
    # Initialize session state for suggestions
    if 'show_suggestions' not in st.session_state:
        st.session_state.show_suggestions = False
    
    # Load available stock list from data/prices directory (all stocks with data)
    stock_list = []
    try:
        prices_dir = Path("data/prices")
        if prices_dir.exists():
            # Get all stock symbols from parquet files
            stock_list = sorted([f.stem.upper() for f in prices_dir.glob("*.parquet")])
    except Exception:
        pass
    
    # Popular stocks for quick autocomplete (shown at top)
    popular_stocks = [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "HINDUNILVR", 
        "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "BAJFINANCE", "AXISBANK",
        "MARUTI", "TITAN", "SUNPHARMA", "HCLTECH", "WIPRO", "TECHM",
        "SBILIFE", "HDFCLIFE", "ICICIPRULI", "BAJAJFINSV", "MUTHOOTFIN",
        "PERSISTENT", "COFORGE", "LTIM", "MPHASIS", "TATAELXSI",
    ]
    
    # Stock input with autocomplete
    col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
    with col1:
        if stock_list:
            # Build list: popular first, then rest alphabetically
            all_stocks = [""] + [s for s in popular_stocks if s in stock_list]
            all_stocks += [s for s in stock_list if s not in popular_stocks]
            
            # Find default index
            default_idx = 0
            if 'selected_stock' in st.session_state and st.session_state.selected_stock:
                selected = st.session_state.selected_stock.upper()
                if selected in all_stocks:
                    default_idx = all_stocks.index(selected)
            
            symbol = st.selectbox(
                "Select or type NSE Stock Symbol",
                all_stocks,
                index=default_idx,
                format_func=lambda x: x if x else "Type to search...",
                key="stock_selector"
            )
            if symbol:
                symbol = symbol.upper()
        else:
            symbol = st.text_input("Enter NSE Stock Symbol", placeholder="e.g., RELIANCE, TCS, INFY").upper().strip()
    with col2:
        analyze = st.button("🔍 Analyze", type="primary", width="stretch")
    with col3:
        suggest = st.button("💡 Suggest", width="stretch")
    with col4:
        reset = st.button("🔄 Reset UI", width="stretch")
    with col5:
        refresh = st.button("🔁 Refresh a Stock", width="stretch")
    
    # Handle reset
    if reset:
        for key in ['results', 'data', 'symbol', 'show_suggestions', 'selected_stock', 'ml_analysis_cache']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    # Handle refresh
    if refresh:
        if not symbol:
            st.warning("Select a stock symbol to refresh.")
        else:
            with st.spinner(f"Refreshing {symbol}..."):
                success, message = app.data_manager.refresh_data(symbol)
            if success:
                ml_cache = st.session_state.get("ml_analysis_cache", {})
                if ml_cache:
                    for key in list(ml_cache.keys()):
                        if key[0] == symbol:
                            del ml_cache[key]
                    st.session_state["ml_analysis_cache"] = ml_cache
                st.success(message)
            else:
                st.error(message)
    
    # Toggle suggestions panel
    if suggest:
        st.session_state.show_suggestions = not st.session_state.show_suggestions
    
    # Stock suggestions panel - based on user profile
    if st.session_state.show_suggestions:
        risk_appetite = profile_dict.get('risk_appetite', 'medium')

        stock_info_df = _load_stock_info_index()
        if stock_info_df.empty:
            st.warning("No local stock metadata found. Run `python 2-download_all_stocks.py` first.")
        else:
            # Apply user filters
            min_price, max_price = profile_dict.get('price_range', (1, 100000))
            filtered = stock_info_df.copy()
            filtered = filtered[(filtered["current_price"].fillna(0) >= min_price) &
                                (filtered["current_price"].fillna(0) <= max_price)]
            filtered = _filter_market_cap(filtered, profile_dict.get('market_cap', 'Any'))

            # Score against profile
            ranked = _score_candidates(filtered, profile_dict)
            if not ranked.empty:
                expected_cagr = float(profile_dict.get('expected_return', 15))
                holding_period = int(profile_dict.get('holding_tenure', 5))
                cagr_tolerance = 2.0
                max_candidates_for_signal = 200
                candidate_symbols = ranked["symbol"].tolist()[:max_candidates_for_signal]
                filtered_symbols = []
                fetched_data = {}
                
                def _get_cached_data(symbol: str):
                    if symbol in fetched_data:
                        return fetched_data[symbol]
                    data = app.fetch_data(symbol, show_spinner=False, show_errors=False)
                    if data:
                        fetched_data[symbol] = data
                    return data
                
                with st.spinner("Filtering candidates by CAGR and signal..."):
                    for s in candidate_symbols:
                        stock_cagr = app.data_manager.get_stock_cagr(s, holding_period)
                        data = None
                        if stock_cagr is None:
                            data = _get_cached_data(s)
                            if not data:
                                continue
                            stock_cagr = _calculate_price_cagr(data.get("price_history"), holding_period)
                        if stock_cagr is None:
                            continue
                        if not (expected_cagr - cagr_tolerance <= stock_cagr <= expected_cagr + cagr_tolerance):
                            continue
                        if data is None:
                            data = _get_cached_data(s)
                            if not data:
                                continue
                        results = app.run_analysis(s, profile, data, show_spinner=False, enable_ml=False)
                        if results['signal'].signal not in {Signal.BUY, Signal.HOLD}:
                            continue
                        filtered_symbols.append(s)
                ranked = ranked[ranked["symbol"].isin(filtered_symbols)]
            if ranked.empty:
                st.warning("No stocks match your current filters after CAGR and signal checks. Try widening price, market cap, or expected CAGR.")
            else:
                title = {
                    "low": "📊 Conservative Picks (Low Risk)",
                    "medium": "⚖️ Balanced Picks (Medium Risk)",
                    "high": "🚀 Growth Picks (High Risk)"
                }.get(risk_appetite, "⚖️ Balanced Picks (Medium Risk)")
                st.info(f"**{title}** (Based on your profile filters) - Click any stock to analyze")

                # Build categories from ranked list
                top_overall = ranked.head(5)["symbol"].tolist()
                top_quality = ranked.sort_values("roe", ascending=False).head(5)["symbol"].tolist()
                top_income = ranked.sort_values("dividend_yield", ascending=False).head(5)["symbol"].tolist()

                categories = {
                    "Top Matches": top_overall,
                    "Quality (ROE)": top_quality,
                    "Income (Dividend)": top_income,
                }

                cols = st.columns(len(categories))
                for col, (category, stocks) in zip(cols, categories.items()):
                    with col:
                        st.markdown(f"**{category}**")
                        for s in stocks:
                            if st.button(s, key=f"sug_{category}_{s}", width="stretch"):
                                st.session_state.selected_stock = s
                                st.session_state.analyze_stock = s
                                st.session_state.show_suggestions = False
                                st.rerun()
    
    # Handle stock selection from suggestions
    if 'analyze_stock' in st.session_state and st.session_state.analyze_stock:
        symbol = st.session_state.analyze_stock
        del st.session_state.analyze_stock
        # Auto-analyze
        st.session_state.symbol = symbol
        st.session_state.profile = profile
        with st.spinner(f"Analyzing {symbol}..."):
            data = app.fetch_data(symbol)
            if data:
                st.session_state.data = data
                ml_cache = st.session_state.get("ml_analysis_cache", {})
                cache_key = (symbol, bool(st.session_state.enable_ml), profile_cache_key)
                cached = ml_cache.get(cache_key)
                if cached:
                    st.session_state.results = cached
                else:
                    st.session_state.results = app.run_analysis(
                        symbol,
                        profile,
                        data,
                        enable_ml=st.session_state.enable_ml
                    )
                    ml_cache[cache_key] = st.session_state.results
                    st.session_state["ml_analysis_cache"] = ml_cache
    
    if analyze and symbol:
        st.session_state.symbol = symbol
        st.session_state.profile = profile
        data = app.fetch_data(symbol)
        if data:
            st.session_state.data = data
            ml_cache = st.session_state.get("ml_analysis_cache", {})
            cache_key = (symbol, bool(st.session_state.enable_ml), profile_cache_key)
            cached = ml_cache.get(cache_key)
            if cached:
                st.session_state.results = cached
            else:
                st.session_state.results = app.run_analysis(
                    symbol,
                    profile,
                    data,
                    enable_ml=st.session_state.enable_ml
                )
                ml_cache[cache_key] = st.session_state.results
                st.session_state["ml_analysis_cache"] = ml_cache
    
    if hasattr(st.session_state, 'results') and st.session_state.results:
        results = st.session_state.results
        data = st.session_state.data
        symbol = st.session_state.symbol
        
        signal = results['signal']
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            render_signal_badge(signal.signal, signal.composite_score)
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
            ["📊 Overview", "🧾 Buffett's Analysis", "🤖 ML Insights", "⚠️ Why NOT", "📈 Charts", "⏰ Time Travel", "🎯 Scenarios"]
        )
        
        with tab1:
            st.markdown(f"### {data['stock_info'].get('name', symbol)}")
            st.markdown(f"*{results['explain'].summary}*")
            info = data.get('stock_info', {})
            st.subheader("Company Profile")
            col_info1, col_info2, col_info3, col_info4, col_info5, col_info6 = st.columns(6)
            profile_obj = st.session_state.get("profile")
            holding_years = int(getattr(profile_obj, "holding_tenure", 5) or 5)
            pre_cagr = app.data_manager.get_stock_cagr(symbol, holding_years)
            if pre_cagr is None:
                pre_cagr = _calculate_price_cagr(data.get("price_history"), holding_years)
            buffett_result = results.get("buffett")
            with col_info1:
                st.metric(
                    "Current Price",
                    f"₹{info.get('current_price', 0):,.2f}" if info.get('current_price') else "N/A"
                )
            with col_info2:
                st.markdown(f"**Sector**: {info.get('sector', 'Unknown')}")
            with col_info3:
                st.markdown(f"**Industry**: {info.get('industry', 'Unknown')}")
            with col_info4:
                st.markdown(f"**City**: {info.get('city', 'Unknown')}")
            with col_info5:
                st.metric(
                    f"CAGR ({holding_years}Y)",
                    format_percentage(pre_cagr) if pre_cagr is not None else "N/A"
                )
            with col_info6:
                if buffett_result:
                    delta = "Buffett Approved" if buffett_result.approved else "Not Approved"
                    st.metric(
                        "Buffett Score",
                        f"{buffett_result.score}/{buffett_result.total_points}",
                        delta,
                    )
                else:
                    st.metric("Buffett Score", "N/A")

            summary = info.get('business_summary')
            if summary:
                with st.expander("Business Summary", expanded=False):
                    st.write(summary)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_score_radar(signal.dimension_scores), width="stretch")
            with col2:
                render_dimension_scores(signal.dimension_scores)
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("✅ Key Positives")
                for p in signal.key_positives:
                    st.success(p)
            with col2:
                st.subheader("⚠️ Key Concerns")
                for n in signal.key_negatives[:5]:
                    st.warning(n)

            if hasattr(results['financial'], 'investable') and not results['financial'].investable:
                st.markdown("---")
                st.subheader("🚫 Investable Universe Filter")
                for reason in results['financial'].investable_reasons:
                    st.error(reason)
            
            if results['red_flags']:
                st.markdown("---")
                render_red_flags([{'severity': r.severity, 'description': r.description} for r in results['red_flags']])
        
        with tab2:
            buffett_result = results.get("buffett")
            st.markdown("### 🧾 Buffett's Checklist Analysis")
            if not buffett_result:
                st.info("No Buffett analysis available for this stock.")
            else:
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric(
                        "Buffett Score",
                        f"{buffett_result.score}/{buffett_result.total_points}",
                    )
                with col_b:
                    st.metric(
                        "Status",
                        "Buffett Approved" if buffett_result.approved else "Not Yet",
                    )
                with col_c:
                    st.metric("Data Coverage", f"{buffett_result.coverage:.0f}%")

                st.markdown("---")
                rows = []
                for rule in buffett_result.rules:
                    rows.append({
                        "Category": rule.category,
                        "Rule": rule.name,
                        "Status": rule.status,
                        "Value": rule.value_display,
                        "Target": rule.threshold,
                        "Notes": rule.notes,
                    })
                if rows:
                    st.dataframe(
                        pd.DataFrame(rows),
                        width="stretch",
                        hide_index=True,
                    )

                st.caption(
                    "Scoring uses 10 rules (5 hard, 4 trend, 1 moat composite). "
                    "Moat proxies are shown for transparency."
                )

        with tab3:
            # ML Insights Tab - NEW
            st.markdown("### 🤖 ML-Enhanced Analysis")
            if not st.session_state.get('enable_ml', False):
                st.info("ML is disabled for this run. Enable it in the sidebar to include ML insights.")
            
            ml_prediction = results.get('ml_prediction')
            
            if ml_prediction and ml_prediction.success:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "ML Signal",
                        ml_prediction.ml_signal,
                        f"{ml_prediction.ml_confidence:.0%} confidence"
                    )
                
                with col2:
                    st.metric(
                        "ML Score",
                        f"{ml_prediction.ml_score:.0f}/100"
                    )
                
                with col3:
                    st.metric(
                        "Price Trend",
                        ml_prediction.price_trend.capitalize(),
                        f"Strength: {ml_prediction.forecast.trend_strength:.0%}" if ml_prediction.forecast else ""
                    )
                
                st.markdown("---")
                
                # Price Predictions
                if ml_prediction.price_prediction_5d or ml_prediction.price_prediction_30d:
                    st.subheader("📈 Price Predictions")
                    col1, col2 = st.columns(2)
                    
                    # Get current price from the 'close' column
                    price_df = data['price_history']
                    if 'close' in price_df.columns:
                        current_price = float(price_df['close'].iloc[-1])
                    elif 'Close' in price_df.columns:
                        current_price = float(price_df['Close'].iloc[-1])
                    else:
                        current_price = float(price_df.iloc[-1, 3])  # Fallback to 4th column
                    
                    with col1:
                        if ml_prediction.price_prediction_5d and current_price > 0:
                            change_5d = ((ml_prediction.price_prediction_5d - current_price) / current_price) * 100
                            st.metric(
                                "5-Day Prediction",
                                f"₹{ml_prediction.price_prediction_5d:.2f}",
                                f"{change_5d:+.1f}%"
                            )
                    
                    with col2:
                        if ml_prediction.price_prediction_30d and current_price > 0:
                            change_30d = ((ml_prediction.price_prediction_30d - current_price) / current_price) * 100
                            st.metric(
                                "30-Day Prediction",
                                f"₹{ml_prediction.price_prediction_30d:.2f}",
                                f"{change_30d:+.1f}%"
                            )
                    
                    st.caption("⚠️ Price predictions are probabilistic estimates, not guarantees.")
                
                st.markdown("---")
                
                # ML Summary
                if ml_prediction.summary:
                    st.subheader("📝 AI Analysis Summary")
                    st.markdown(ml_prediction.summary)

                # SHAP / Model Explainability
                if results.get('explain') and results['explain'].ml_explanation:
                    st.markdown("---")
                    st.subheader("🧠 Model Explanation (SHAP)")
                    st.markdown(results['explain'].ml_explanation)
                    shap = results['explain'].shap_contributions or {}
                    top_positive = shap.get('top_positive', [])
                    top_negative = shap.get('top_negative', [])
                    if top_positive or top_negative:
                        st.caption("Top contributing features")
                        for feature, value in top_positive[:5]:
                            st.success(f"+ {feature}: {value:+.3f}")
                        for feature, value in top_negative[:5]:
                            st.warning(f"- {feature}: {value:+.3f}")
                
                # Key Insights
                if ml_prediction.key_insights:
                    st.subheader("💡 Key Insights")
                    for insight in ml_prediction.key_insights:
                        st.info(f"• {insight}")
                
                st.markdown("---")
                
                # Classifier Probabilities
                if ml_prediction.classification and ml_prediction.classification.signal_probabilities:
                    st.subheader("📊 Signal Probabilities")
                    probs = ml_prediction.classification.signal_probabilities
                    
                    for signal, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                        color = {
                            "BUY": "green",
                            "HOLD": "blue", 
                            "AVOID": "orange",
                            "SELL": "red"
                        }.get(signal, "gray")
                        
                        st.progress(prob, text=f"{signal}: {prob:.0%}")
                
                # Performance info
                st.markdown("---")
                st.caption(
                    f"Layers used: {', '.join(ml_prediction.layers_used)} | "
                    f"Inference time: {ml_prediction.total_inference_time_ms:.0f}ms"
                )
                
                if ml_prediction.errors:
                    with st.expander("⚠️ Warnings"):
                        for error in ml_prediction.errors:
                            st.warning(error)
            else:
                # Check ML status to give specific guidance
                ml_status = app.get_ml_status()
                ml_skip_reason = results.get("ml_skip_reason")
                
                if not ml_status.get('enabled'):
                    st.warning("**ML is disabled in configuration.**")
                    st.info("Enable ML in `config/settings.yaml` by setting `ml_config.enabled: true`")
                    
                elif not ml_status.get('initialized'):
                    st.warning("**ML models not initialized.**")
                    st.info("👈 Click **'Initialize ML Models'** in the sidebar, then re-analyze.")
                    
                else:
                    # ML is initialized but prediction failed
                    st.warning("**ML analysis ran but didn't produce results.**")
                    if ml_skip_reason:
                        st.info(ml_skip_reason)
                    
                    # Show which layers are ready
                    st.markdown("**Layer Status:**")
                    for layer in ['forecaster', 'classifier', 'explainer']:
                        layer_info = ml_status.get(layer, {})
                        is_ready = layer_info.get('ready', False)
                        model_name = layer_info.get('model', 'Not configured')
                        
                        if is_ready:
                            st.success(f"✓ {layer.title()}: {model_name}")
                        elif model_name:
                            st.warning(f"⚠ {layer.title()}: {model_name} (not ready)")
                        else:
                            st.info(f"○ {layer.title()}: Disabled")
                    
                    # Check if classifier needs training
                    classifier_info = ml_status.get('classifier', {})
                    if classifier_info.get('model') and not classifier_info.get('ready'):
                        st.info("""
                        💡 **Tip**: The classifier needs training data. It will improve as you analyze more stocks.
                        For now, forecaster and explainer results are still available.
                        """)
                    
                    if st.button("🔄 Re-run ML Analysis", width="stretch"):
                        symbol = st.session_state.get('symbol')
                        current_profile = st.session_state.get('profile')
                        if symbol and current_profile:
                            data = app.fetch_data(symbol)
                            if data:
                                st.session_state.data = data
                                st.session_state.results = app.run_analysis(
                                    symbol,
                                    current_profile,
                                    data,
                                    enable_ml=True
                                )
                                st.rerun()
                
                # Show download instructions
                with st.expander("📥 Model Download Instructions"):
                    st.markdown("""
                    ### Quick Start
                    
                    ```bash
                    # Install ML dependencies
                    pip install -r requirements.txt
                    
                    # Download Chronos-T5-Base (forecaster)
                    huggingface-cli download amazon/chronos-t5-base --local-dir models/forecaster/chronos-t5-base
                    
                    # Download Qwen2.5-3B (explainer) - GGUF format
                    huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF --local-dir models/explainer/qwen2.5-3b --include "*.gguf"
                    ```
                    
                    See ML_MODELS.md for complete instructions.
                    """)
        
        with tab4:
            #st.markdown("### This section is shown for ALL stocks, even BUY signals")
            #st.markdown("---")
            render_why_not_buy(results['explain'].why_not_buy)
            st.markdown("---")
            st.subheader("Risk Factors")
            for r in results['explain'].risk_factors:
                st.error(f"⚠️ {r}")
            st.markdown("---")
            st.subheader("What Could Invalidate The Thesis")
            for i in results['explain'].thesis_invalidators:
                st.info(f"📌 {i}")
        
        with tab5:
            st.subheader("Price History")
            st.plotly_chart(create_price_chart(data['price_history'], f"{symbol} Price"), width="stretch")
            st.subheader("Drawdown History")
            st.plotly_chart(create_drawdown_chart(data['price_history']), width="stretch")

            st.markdown("---")
            st.subheader("Index History")
            index_options = app.data_manager.get_all_indices()
            if index_options:
                default_index = "NIFTY 50" if "NIFTY 50" in index_options else index_options[0]
                index_symbol = st.selectbox("Select Index", index_options, index=index_options.index(default_index))
                index_history = app.data_manager.get_index_history(index_symbol, years=30)
                if index_history is None or index_history.empty:
                    st.warning("No index history available.")
                else:
                    st.plotly_chart(
                        create_price_chart(index_history, f"{index_symbol} Index", show_volume=False),
                        width="stretch"
                    )
            else:
                st.info("No index datasets detected. Run `python 2-download_all_stocks.py --build-db-only`.")
        
        with tab6:
            st.subheader("⏰ Time Travel Mode")
            st.info("Re-analyze using only data available at a historical point. No future leakage.")
            cutoffs = app.time_travel.get_available_cutoffs(data['price_history'])
            if cutoffs:
                cutoff = st.selectbox("Travel back to December of:", cutoffs)
                if st.button("🕰️ Run Time Travel", type="primary"):
                    with st.spinner(f"Traveling back to {cutoff}..."):
                        try:
                            tt_result = app.time_travel.analyze_at_cutoff(
                                symbol=symbol,
                                cutoff_year=cutoff,
                                user_profile=profile,
                                stock_info=data['stock_info'],
                                price_history=data['price_history'],
                                financials=data['financials'],
                                shareholding=data.get('shareholding', pd.DataFrame()),
                                dividends=data.get('dividends', pd.DataFrame()),
                                nifty_history=data.get('nifty_history'),
                                include_outcome=True
                            )
                            
                            st.success(f"Analysis complete for December {cutoff}")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"### Signal at {cutoff}")
                                render_signal_badge(tt_result.signal_at_cutoff.signal, tt_result.signal_at_cutoff.composite_score)
                            
                            with col2:
                                st.markdown("### Actual Outcome")
                                if tt_result.actual_outcome:
                                    for period, data_out in tt_result.actual_outcome.items():
                                        if isinstance(data_out, dict) and 'return' in data_out:
                                            ret = data_out['return']
                                            color = "green" if ret > 0 else "red"
                                            st.markdown(f"**{period.upper()}**: <span style='color:{color}'>{ret:+.1f}%</span>", unsafe_allow_html=True)
                                else:
                                    st.warning("Outcome data not available")
                            
                            with col3:
                                st.markdown("### Optimal Signal (Hindsight)")
                                if tt_result.hindsight_signal:
                                    # Color based on hindsight signal
                                    hs_colors = {
                                        "STRONG BUY": "#006400",
                                        "BUY": "#228B22", 
                                        "HOLD": "#FFA500",
                                        "WEAK HOLD": "#DAA520",
                                        "AVOID": "#FF6347",
                                        "SELL": "#DC143C"
                                    }
                                    hs_color = hs_colors.get(tt_result.hindsight_signal, "#808080")
                                    st.markdown(f"""
                                    <div style='background-color:{hs_color}; padding:15px; border-radius:10px; text-align:center;'>
                                        <span style='color:white; font-size:18px; font-weight:bold;'>{tt_result.hindsight_signal}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Show accuracy badge
                                    if tt_result.signal_accuracy:
                                        acc_colors = {"CORRECT": "#228B22", "PARTIAL": "#FFA500", "INCORRECT": "#DC143C"}
                                        acc_color = acc_colors.get(tt_result.signal_accuracy, "#808080")
                                        st.markdown(f"<p style='text-align:center; margin-top:8px;'><span style='background-color:{acc_color}; color:white; padding:3px 8px; border-radius:5px; font-size:12px;'>{tt_result.signal_accuracy}</span></p>", unsafe_allow_html=True)
                            
                            st.markdown("---")
                            st.markdown("### Hindsight Analysis")
                            
                            # Show different colors based on accuracy
                            if tt_result.signal_accuracy == "CORRECT":
                                st.success(tt_result.hindsight_analysis)
                            elif tt_result.signal_accuracy == "PARTIAL":
                                st.warning(tt_result.hindsight_analysis)
                            else:
                                st.error(tt_result.hindsight_analysis)
                            
                            # Show red flags at that time
                            if tt_result.red_flags_at_cutoff:
                                st.markdown("### Red Flags (at that time)")
                                for flag in tt_result.red_flags_at_cutoff:
                                    st.warning(f"⚠️ {flag.get('description', flag)}")
                                    
                        except Exception as e:
                            st.error(f"Time Travel failed: {e}")
            else:
                st.warning("Not enough historical data for time travel.")
        
        with tab7:
            st.subheader("🎯 Scenario Simulator")
            st.info("Stress-test your investment thesis under various market conditions")
            
            scenarios = app.scenario_sim.list_scenarios()
            selected = st.selectbox("Select Scenario:", [s['key'] for s in scenarios],
                                    format_func=lambda x: next(s['name'] for s in scenarios if s['key'] == x))
            
            for s in scenarios:
                if s['key'] == selected:
                    st.caption(s['description'])
            
            if st.button("🎯 Run Scenario", type="primary"):
                with st.spinner("Running scenario simulation..."):
                    try:
                        # Build financial metrics and valuation from results
                        # Get operating margin from details dict or use default
                        fin_details = results['financial'].details or {}
                        opm = fin_details.get('opm_current', fin_details.get('operating_margin', 15))
                        
                        fin_metrics = {
                            'revenue_cagr_5y': results['financial'].revenue_cagr_5y or 10,
                            'operating_margin': opm,
                        }
                        # Get current price from price history
                        price_df = data['price_history']
                        if len(price_df) > 0:
                            if 'close' in price_df.columns:
                                curr_price = float(price_df['close'].iloc[-1])
                            elif 'Close' in price_df.columns:
                                curr_price = float(price_df['Close'].iloc[-1])
                            else:
                                curr_price = 100
                        else:
                            curr_price = 100
                        
                        val_metrics = {
                            'pe_ratio': results['valuation'].current_pe or 20,
                            'current_price': curr_price,
                        }
                        user_profile_dict = {
                            'holding_tenure': profile.holding_tenure,
                            'expected_return': profile.expected_return,
                        }
                        
                        scenario_result = app.scenario_sim.simulate(
                            scenario_name=selected,
                            stock_info=data['stock_info'],
                            financial_metrics=fin_metrics,
                            valuation=val_metrics,
                            user_profile=user_profile_dict
                        )
                        
                        st.success(f"Scenario: {scenario_result.scenario.name}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("### Base Case")
                            st.metric("Expected Return", f"{scenario_result.base_case['expected_return']:.1f}%")
                            st.metric("Target Price", f"₹{scenario_result.base_case['target_price']:.0f}")
                        
                        with col2:
                            st.markdown("### Stressed Case")
                            stressed_ret = scenario_result.stressed_case['expected_return']
                            base_ret = scenario_result.base_case['expected_return']
                            st.metric("Expected Return", f"{stressed_ret:.1f}%", f"{stressed_ret - base_ret:.1f}%")
                            st.metric("Target Price", f"₹{scenario_result.stressed_case['target_price']:.0f}")
                        
                        st.markdown("---")
                        
                        # Resilience rating
                        resilience_colors = {'robust': 'green', 'resilient': 'blue', 'moderate': 'orange', 'fragile': 'red'}
                        color = resilience_colors.get(scenario_result.resilience_rating, 'gray')
                        st.markdown(f"### Resilience Rating: <span style='color:{color}'>{scenario_result.resilience_rating.upper()}</span>", unsafe_allow_html=True)
                        
                        if scenario_result.signal_change:
                            st.warning(f"⚠️ {scenario_result.signal_change}")
                        
                        st.markdown("### Key Findings")
                        for finding in scenario_result.key_findings:
                            st.info(f"• {finding}")
                            
                    except Exception as e:
                        st.error(f"Scenario simulation failed: {e}")
    
    else:
        st.markdown("""
            ## Welcome to Stocron by RTR
            *The Indian Equity Intelligence*
            
            A **local-first**, **explainable** tool for long-term stock analysis.
            
            ### How to Use
            1. Enter a stock symbol (e.g., RELIANCE, TCS, INFY)
            2. Set your investment profile in the sidebar
            3. Click **Analyze**
            
            ### Core Philosophy
            > *"Is this stock suitable for this investor, under these assumptions — 
            > and what could invalidate the thesis?"*
            
            ### Features
            - **Contextual Signals**: BUY, HOLD, AVOID, SELL based on YOUR profile
            - **"Why NOT to Buy"**: Shown even for BUY signals
            - **Red Flag Detection**: Governance and financial warnings
            - **Time Travel Mode**: Re-analyze with historical data
            - **Scenario Simulation**: Stress-test assumptions
            
            ### Privacy
            - 100% local processing after data download
            - No data sent to external servers
        """)


if __name__ == "__main__":
    main()
