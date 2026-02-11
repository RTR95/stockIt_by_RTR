"""Market Behaviour Analysis for Stocron by RTR."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from ..utils.config import get_threshold, get_macro_rsi_thresholds
from ..utils.helpers import calculate_max_drawdown, calculate_volatility

logger = logging.getLogger(__name__)


@dataclass
class MarketBehaviourScore:
    """Container for market behaviour analysis results."""
    overall_score: float
    max_drawdown: float
    avg_recovery_months: Optional[float]
    volatility_1y: Optional[float]
    volatility_3y: Optional[float]
    beta: Optional[float]
    sharpe_ratio: Optional[float]
    vs_nifty_1y: Optional[float]
    vs_nifty_3y: Optional[float]
    vs_nifty_5y: Optional[float]
    volatility_regime: str
    trend_30d: str  # Added field
    details: Dict[str, Any]
    warnings: List[str]


class MarketBehaviourAnalyzer:
    """Analyzes long-term market behaviour patterns."""
    
    def __init__(self, db_manager=None):
        self.db = db_manager
        self.max_dd_threshold = get_threshold('market', 'max_drawdown_threshold', 50.0)
        self.recovery_warning = get_threshold('market', 'recovery_months_warning', 24)
        self.vol_high = get_threshold('market', 'volatility_high_threshold', 40.0)
    
    def analyze(self, symbol: str, price_history: pd.DataFrame,
                nifty_history: Optional[pd.DataFrame] = None,
                cutoff_date: Optional[datetime] = None) -> MarketBehaviourScore:
        cutoff_date = cutoff_date or datetime.now()
        warnings, details = [], {}
        
        if price_history.empty or len(price_history) < 30:
            return self._empty_result()
        
        df = price_history[price_history['date'] <= cutoff_date].sort_values('date')
        prices = df.set_index('date')['close']
        
        # Drawdown Analysis
        dd = self._analyze_drawdown(prices)
        details['drawdowns'] = dd
        if dd['max_drawdown'] > self.max_dd_threshold:
            warnings.append(f"Large drawdown: {dd['max_drawdown']:.1f}%")
        
        # Recovery Analysis
        recovery = self._analyze_recovery(prices)
        details['recovery'] = recovery
        if recovery['avg_months'] and recovery['avg_months'] > self.recovery_warning:
            warnings.append(f"Slow recovery: {recovery['avg_months']:.0f} months average")
        
        # Volatility Analysis
        vol = self._analyze_volatility(df, cutoff_date)
        details['volatility'] = vol
        if vol['regime'] in ['high', 'extreme']:
            warnings.append(f"High volatility: {vol['annual_1y']:.1f}%")
        
        # Beta
        beta = self._calculate_beta(df, nifty_history, cutoff_date)
        details['beta'] = beta
        if beta['beta'] and beta['beta'] > 1.5:
            warnings.append(f"High beta: {beta['beta']:.2f}")
        
        # Relative Performance
        relative = self._analyze_relative(df, nifty_history, cutoff_date)
        details['relative'] = relative
        
        # Sharpe Ratio
        sharpe = self._calculate_sharpe(df, cutoff_date)
        
        # Trend Analysis
        trend_30d = 'neutral'
        if not df.empty and len(df) > 30:
             p_curr = float(df['close'].iloc[-1])
             p_30d_ago = float(df['close'].iloc[-30])
             trend_30d = 'bullish' if p_curr > p_30d_ago * 1.05 else 'bearish' if p_curr < p_30d_ago * 0.95 else 'neutral'

        overall = self._calculate_score(dd['score'], recovery['score'], vol['score'], 
                                         beta['score'], relative['score'])
        
        return MarketBehaviourScore(
            overall_score=overall, max_drawdown=dd['max_drawdown'],
            avg_recovery_months=recovery['avg_months'], volatility_1y=vol['annual_1y'],
            volatility_3y=vol['annual_3y'], beta=beta['beta'], sharpe_ratio=sharpe,
            vs_nifty_1y=relative['vs_nifty_1y'], vs_nifty_3y=relative['vs_nifty_3y'],
            vs_nifty_5y=relative['vs_nifty_5y'], volatility_regime=vol['regime'],
            trend_30d=trend_30d,
            details=details, warnings=warnings
        )
    
    def _empty_result(self) -> MarketBehaviourScore:
        return MarketBehaviourScore(
            overall_score=50, max_drawdown=0, avg_recovery_months=None,
            volatility_1y=None, volatility_3y=None, beta=None, sharpe_ratio=None,
            vs_nifty_1y=None, vs_nifty_3y=None, vs_nifty_5y=None,
            volatility_regime='unknown', trend_30d='neutral', details={}, warnings=['Insufficient data']
        )
    
    def _analyze_drawdown(self, prices: pd.Series) -> Dict:
        max_dd, _, _ = calculate_max_drawdown(prices)
        score = 90 if max_dd <= 20 else 75 if max_dd <= 35 else 55 if max_dd <= 50 else 35 if max_dd <= 65 else 20
        return {'max_drawdown': max_dd, 'score': score}
    
    def _analyze_recovery(self, prices: pd.Series) -> Dict:
        """Analyze recovery time from drawdowns."""
        try:
            if len(prices) < 252:
                return {'avg_months': None, 'score': 50}
            
            running_max = prices.expanding().max()
            drawdown = (prices - running_max) / running_max
            
            in_drawdown = False
            dd_start = None
            recovery_times = []
            
            for i, (date, dd) in enumerate(drawdown.items()):
                if not in_drawdown and dd < -0.20:
                    in_drawdown = True
                    dd_start = date
                elif in_drawdown and dd >= -0.05:
                    if dd_start is not None:
                        recovery_days = (date - dd_start).days
                        recovery_times.append(recovery_days / 30)
                    in_drawdown = False
                    dd_start = None
            
            if len(recovery_times) == 0:
                return {'avg_months': None, 'score': 70}
            
            avg_months = float(np.mean(recovery_times))
            score = 85 if avg_months < 6 else 70 if avg_months < 12 else 55 if avg_months < 18 else 40
            
            return {'avg_months': round(avg_months, 1), 'score': score}
            
        except Exception:
            return {'avg_months': None, 'score': 50}
    
    def _analyze_volatility(self, df: pd.DataFrame, cutoff_date: datetime) -> Dict:
        df = df.sort_values('date').copy()
        df['returns'] = self._calculate_log_returns(df['close'])
        
        one_year_ago = cutoff_date - timedelta(days=365)
        df_1y = df[df['date'] >= one_year_ago]
        vol_1y = calculate_volatility(df_1y['returns'].dropna())
        
        three_years_ago = cutoff_date - timedelta(days=3*365)
        df_3y = df[df['date'] >= three_years_ago]
        vol_3y = calculate_volatility(df_3y['returns'].dropna())
        
        ref_vol = vol_1y or vol_3y or 30
        regime = 'low' if ref_vol <= 20 else 'medium' if ref_vol <= 30 else 'high' if ref_vol <= 40 else 'extreme'
        score = 90 if ref_vol <= 20 else 75 if ref_vol <= 30 else 55 if ref_vol <= 40 else 35 if ref_vol <= 55 else 20
        
        return {'annual_1y': vol_1y, 'annual_3y': vol_3y, 'regime': regime, 'score': score}
    
    def _calculate_beta(self, df: pd.DataFrame, nifty: Optional[pd.DataFrame], cutoff_date: datetime) -> Dict:
        """Calculate stock beta relative to Nifty 50."""
        if nifty is None or nifty.empty:
            return {'beta': None, 'score': 50}
        
        try:
            stock_df = df.sort_values('date').copy()
            stock_df['returns'] = self._calculate_log_returns(stock_df['close'])
            
            nifty_df = nifty.sort_values('date').copy()
            if 'close' not in nifty_df.columns and 'Close' in nifty_df.columns:
                nifty_df['close'] = nifty_df['Close']
            nifty_df['market_returns'] = self._calculate_log_returns(nifty_df['close'])
            
            stock_df['date'] = pd.to_datetime(stock_df['date']).dt.date
            nifty_df['date'] = pd.to_datetime(nifty_df['date']).dt.date
            
            merged = pd.merge(stock_df[['date', 'returns']], nifty_df[['date', 'market_returns']], on='date')
            
            one_year_ago = (cutoff_date - timedelta(days=365)).date()
            merged = merged[merged['date'] >= one_year_ago].dropna()
            
            if len(merged) < 100:
                return {'beta': None, 'score': 50}
            
            covariance = merged['returns'].cov(merged['market_returns'])
            market_variance = merged['market_returns'].var()
            
            if market_variance == 0:
                return {'beta': None, 'score': 50}
            
            beta = covariance / market_variance
            
            if beta < 0.5:
                score = 75
            elif beta <= 1.2:
                score = 70
            elif beta <= 1.5:
                score = 55
            else:
                score = 35
            
            return {'beta': round(float(beta), 2), 'score': score}
            
        except Exception:
            return {'beta': None, 'score': 50}
    
    def _analyze_relative(self, df: pd.DataFrame, nifty: Optional[pd.DataFrame], cutoff_date: datetime) -> Dict:
        """Calculate relative performance vs Nifty 50."""
        if nifty is None or nifty.empty:
            return {'vs_nifty_1y': None, 'vs_nifty_3y': None, 'vs_nifty_5y': None, 'score': 50}
        
        try:
            stock_df = df.sort_values('date').copy()
            nifty_df = nifty.sort_values('date').copy()
            
            if 'close' not in nifty_df.columns and 'Close' in nifty_df.columns:
                nifty_df['close'] = nifty_df['Close']
            
            stock_df['date'] = pd.to_datetime(stock_df['date']).dt.date
            nifty_df['date'] = pd.to_datetime(nifty_df['date']).dt.date
            
            result = {'vs_nifty_1y': None, 'vs_nifty_3y': None, 'vs_nifty_5y': None, 'score': 50}
            
            for years, key in [(1, 'vs_nifty_1y'), (3, 'vs_nifty_3y'), (5, 'vs_nifty_5y')]:
                start_date = (cutoff_date - timedelta(days=years * 365)).date()
                
                stock_period = stock_df[stock_df['date'] >= start_date]
                nifty_period = nifty_df[nifty_df['date'] >= start_date]
                
                if len(stock_period) < 100 or len(nifty_period) < 100:
                    continue
                
                stock_return = (stock_period['close'].iloc[-1] / stock_period['close'].iloc[0] - 1) * 100
                nifty_return = (nifty_period['close'].iloc[-1] / nifty_period['close'].iloc[0] - 1) * 100
                
                result[key] = round(stock_return - nifty_return, 1)
            
            outperformance = result['vs_nifty_1y'] or result['vs_nifty_3y'] or 0
            
            if outperformance >= 20:
                result['score'] = 90
            elif outperformance >= 10:
                result['score'] = 75
            elif outperformance >= 0:
                result['score'] = 60
            elif outperformance >= -10:
                result['score'] = 45
            else:
                result['score'] = 30
            
            return result
            
        except Exception:
            return {'vs_nifty_1y': None, 'vs_nifty_3y': None, 'vs_nifty_5y': None, 'score': 50}
    
    def detect_market_regime(self, nifty_prices: pd.Series, lookback: int = 200) -> str:
        """Detect current market regime (Bull/Bear/Sideways)."""
        if len(nifty_prices) < lookback:
            return 'sideways'
        
        try:
            sma_50 = nifty_prices.rolling(50).mean()
            sma_200 = nifty_prices.rolling(200).mean()
            
            current_price = nifty_prices.iloc[-1]
            current_sma50 = sma_50.iloc[-1]
            current_sma200 = sma_200.iloc[-1]
            
            if pd.isna(current_sma200):
                return 'sideways'
            
            if current_price > current_sma50 > current_sma200:
                return 'bull'
            elif current_price < current_sma50 < current_sma200:
                return 'bear'
            return 'sideways'
            
        except Exception:
            return 'sideways'
    
    @staticmethod
    def get_regime_rsi_thresholds(regime: str) -> Dict[str, int]:
        """Get dynamic RSI thresholds based on market regime."""
        thresholds = get_macro_rsi_thresholds()
        return thresholds.get(regime, thresholds['sideways'])

    @staticmethod
    def _calculate_log_returns(prices: pd.Series) -> pd.Series:
        """Calculate log returns for long-horizon stability."""
        returns = np.log(prices / prices.shift(1))
        return returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    def _calculate_sharpe(self, df: pd.DataFrame, cutoff_date: datetime, risk_free: float = 6.0) -> Optional[float]:
        try:
            df = df.sort_values('date').copy()
            df['returns'] = self._calculate_log_returns(df['close'])
            one_year_ago = cutoff_date - timedelta(days=365)
            df_1y = df[df['date'] >= one_year_ago]
            if len(df_1y) < 100:
                return None
            returns = df_1y['returns'].dropna()
            annual_return = returns.mean() * 252 * 100
            annual_vol = returns.std() * np.sqrt(252) * 100
            if annual_vol > 0:
                return round((annual_return - risk_free) / annual_vol, 2)
        except Exception:
            pass
        return None
    
    def _calculate_score(self, dd: float, recovery: float, vol: float, beta: float, relative: float) -> float:
        return round(dd * 0.25 + recovery * 0.20 + vol * 0.25 + beta * 0.15 + relative * 0.15, 1)
