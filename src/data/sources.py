"""Multi-source data ingestion for Stocron by RTR.

Priority:
1. Local database/parquet files (for offline operation)
2. Yahoo Finance API (for online refresh)
3. Jugaad Data (backup)
4. NSE Tools (for stock info only)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import logging
import time
import json

from ..utils.config import get_config
from ..utils.cache_utils import LRUCache
from .historic import (
    list_historic_stock_symbols,
    list_historic_index_names,
    load_historic_stock_prices,
    load_historic_index_prices,
)
from .refresh import refresh_symbol_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
PRICE_DIR = DATA_DIR / 'prices'
INFO_DIR = DATA_DIR / 'stock_info'
FINANCIALS_DIR = DATA_DIR / 'financials'
DB_PATH = DATA_DIR / 'db' / 'equity_intelligence.db'


class LocalDataSource:
    """Local data source - reads from parquet files and database."""
    
    def __init__(self):
        self._db_conn = None
        cache_config = get_config('cache', {}) or {}
        memory_cache = cache_config.get('memory', {}) or {}
        info_limit = memory_cache.get('stock_info_max_entries', 2048)
        self._stock_info_cache = LRUCache(info_limit)
        survivorship = get_config('survivorship_bias', {}) or {}
        self._include_delisted = survivorship.get('include_delisted', False)
        delisted_dir = survivorship.get('delisted_data_dir', 'data/delisted')
        self._delisted_dir = (PROJECT_ROOT / delisted_dir).resolve()

    def _get_delisted_price_file(self, symbol: str) -> Optional[Path]:
        if not self._include_delisted or not self._delisted_dir.exists():
            return None
        for subdir in ['prices', 'price_history', '']:
            base = self._delisted_dir / subdir if subdir else self._delisted_dir
            for ext in ['.parquet', '.csv']:
                candidate = base / f"{symbol}{ext}"
                if candidate.exists():
                    return candidate
        return None

    def _get_delisted_info_file(self, symbol: str) -> Optional[Path]:
        if not self._include_delisted or not self._delisted_dir.exists():
            return None
        for subdir in ['info', 'stock_info', '']:
            base = self._delisted_dir / subdir if subdir else self._delisted_dir
            candidate = base / f"{symbol}.json"
            if candidate.exists():
                return candidate
        return None
    
    def _get_db(self):
        """Get database connection."""
        if self._db_conn is None and DB_PATH.exists():
            try:
                import duckdb
                self._db_conn = duckdb.connect(str(DB_PATH), read_only=True)
            except Exception as e:
                logger.debug(f"Could not connect to database: {e}")
        return self._db_conn
    
    def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get stock info from local storage."""
        # Check cache
        if symbol in self._stock_info_cache:
            return self._stock_info_cache.get(symbol)
        
        # Try JSON file first
        info_file = INFO_DIR / f"{symbol}.json"
        if info_file.exists():
            try:
                with open(info_file) as f:
                    info = json.load(f)
                    # Normalize field names
                    result = {
                        'symbol': symbol,
                        'name': info.get('longName', info.get('shortName', symbol)),
                        'business_summary': info.get('longBusinessSummary'),
                        'city': info.get('city'),
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown'),
                        'market_cap': info.get('marketCap', 0),
                        'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                        'pe_ratio': info.get('trailingPE'),
                        'pb_ratio': info.get('priceToBook'),
                        'dividend_yield': (info.get('dividendYield', 0) or 0) * 100,
                        'eps': info.get('trailingEps'),
                        'book_value': info.get('bookValue'),
                        'roe': (info.get('returnOnEquity') or 0) * 100 if info.get('returnOnEquity') else None,
                        'debt_to_equity': info.get('debtToEquity'),
                        'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                        'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                        'beta': info.get('beta'),
                        'source': 'local_file'
                    }
                    self._stock_info_cache.set(symbol, result)
                    return result
            except Exception as e:
                logger.debug(f"Error reading info file for {symbol}: {e}")

        # Try delisted info
        delisted_info_file = self._get_delisted_info_file(symbol)
        if delisted_info_file:
            try:
                with open(delisted_info_file) as f:
                    info = json.load(f)
                    result = {
                        'symbol': symbol,
                        'name': info.get('name', symbol),
                        'business_summary': info.get('business_summary'),
                        'city': info.get('city'),
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown'),
                        'market_cap': info.get('market_cap', 0),
                        'current_price': info.get('last_known_price', 0),
                        'pe_ratio': info.get('pe_ratio'),
                        'pb_ratio': info.get('pb_ratio'),
                        'dividend_yield': info.get('dividend_yield'),
                        'roe': info.get('roe'),
                        'debt_to_equity': info.get('debt_to_equity'),
                        'fifty_two_week_high': info.get('peak_price'),
                        'fifty_two_week_low': info.get('low_price'),
                        'delisted': True,
                        'source': 'delisted_file'
                    }
                    self._stock_info_cache.set(symbol, result)
                    return result
            except Exception as e:
                logger.debug(f"Error reading delisted info file for {symbol}: {e}")
        
        # Try database
        db = self._get_db()
        if db:
            try:
                row = db.execute(
                    "SELECT * FROM stocks WHERE symbol = ?", [symbol]
                ).fetchone()
                if row:
                    result = {
                        'symbol': row[0],
                        'name': row[1] or symbol,
                        'business_summary': None,
                        'city': None,
                        'sector': row[2] or 'Unknown',
                        'industry': row[3] or 'Unknown',
                        'market_cap': row[4] or 0,
                        'current_price': row[5] or 0,
                        'pe_ratio': row[6],
                        'pb_ratio': row[7],
                        'dividend_yield': row[8],
                        'fifty_two_week_high': row[9],
                        'fifty_two_week_low': row[10],
                        'source': 'local_db'
                    }
                    self._stock_info_cache.set(symbol, result)
                    return result
            except Exception as e:
                logger.debug(f"Error querying database for {symbol}: {e}")
        
        return None
    
    def get_price_history(self, symbol: str, years: int = 10) -> Optional[pd.DataFrame]:
        """Get price history from local storage."""
        cutoff = datetime.now() - timedelta(days=years * 365)
        cutoff_date = cutoff.date()
        # Try parquet file first
        price_file = PRICE_DIR / f"{symbol}.parquet"
        if price_file.exists():
            try:
                df = pd.read_parquet(price_file)
                df['date'] = pd.to_datetime(df['date'])
                
                # Filter to requested years
                df = df[df['date'] >= cutoff]
                
                if not df.empty:
                    logger.info(f"Got {len(df)} local records for {symbol}")
                    return df.sort_values('date')
            except Exception as e:
                logger.debug(f"Error reading parquet for {symbol}: {e}")
        
        # Try database
        db = self._get_db()
        if db:
            try:
                df = db.execute("""
                    SELECT symbol, date, open, high, low, close, volume, source
                    FROM price_history 
                    WHERE symbol = ? AND date >= ?
                    ORDER BY date
                """, [symbol, cutoff_date]).fetchdf()
                
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    logger.info(f"Got {len(df)} database records for {symbol}")
                    return df
            except Exception as e:
                logger.debug(f"Error querying price history for {symbol}: {e}")

        # Try delisted price history
        delisted_price_file = self._get_delisted_price_file(symbol)
        if delisted_price_file:
            try:
                if delisted_price_file.suffix == '.parquet':
                    df = pd.read_parquet(delisted_price_file)
                else:
                    df = pd.read_csv(delisted_price_file)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                else:
                    df['date'] = pd.to_datetime(df.iloc[:, 0])
                df = df[df['date'] >= cutoff]
                if not df.empty:
                    logger.info(f"Got {len(df)} delisted records for {symbol}")
                    return df.sort_values('date')
            except Exception as e:
                logger.debug(f"Error reading delisted price history for {symbol}: {e}")
        
        # Try bundled historic datasets
        historic_df = load_historic_stock_prices(symbol)
        if historic_df is None and symbol.upper().startswith("NIFTY"):
            historic_df = load_historic_index_prices(symbol)

        if historic_df is not None and not historic_df.empty:
            historic_df = historic_df[historic_df["date"] >= cutoff]
            if not historic_df.empty:
                logger.info(f"Got {len(historic_df)} historic records for {symbol}")
                return historic_df.sort_values("date")

        return None

    def get_financials(self, symbol: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Get financial statements from local storage (if available)."""
        fin_file = FINANCIALS_DIR / f"{symbol}.json"
        if not fin_file.exists():
            return None
        try:
            with open(fin_file) as f:
                payload = json.load(f)
            result = {"source": payload.get("source", "local_file")}
            for key in ["income_statement", "balance_sheet", "cash_flow"]:
                data = payload.get(key)
                if data and isinstance(data, dict) and data.get("data") is not None:
                    df = pd.DataFrame(data.get("data"), columns=data.get("columns"))
                    idx = data.get("index")
                    if idx is not None:
                        df.index = idx
                    df = self._normalize_financial_df(df)
                    result[key] = df
                else:
                    result[key] = pd.DataFrame()
            div_payload = payload.get("dividends")
            if div_payload and isinstance(div_payload, dict):
                div_index = div_payload.get("index") or []
                div_values = div_payload.get("values") or []
                if len(div_index) == len(div_values) and div_index:
                    div_series = pd.Series(div_values, index=pd.to_datetime(div_index))
                    result["dividends"] = div_series.sort_index()
                else:
                    result["dividends"] = pd.Series(dtype=float)
            else:
                result["dividends"] = pd.Series(dtype=float)
            return result
        except Exception as e:
            logger.debug(f"Error reading financials for {symbol}: {e}")
        return None

    @staticmethod
    def _normalize_financial_df(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure line items are columns (match online mode)."""
        if df is None or df.empty:
            return df
        line_items = {
            "Total Revenue",
            "Revenue",
            "Net Sales",
            "Net Income",
            "Profit After Tax",
            "PAT",
            "Net Profit",
            "EBIT",
            "Operating Income",
            "Operating Profit",
            "Interest Expense",
            "Finance Costs",
            "Total Assets",
            "Total Stockholder Equity",
            "Stockholders Equity",
            "Total Equity",
            "Current Assets",
            "Current Liabilities",
            "Cash From Operating Activities",
            "Operating Cash Flow",
            "Capital Expenditure",
            "Dividends Paid",
        }
        has_line_items_in_index = any(item in df.index for item in line_items)
        has_line_items_in_cols = any(item in df.columns for item in line_items)
        if has_line_items_in_index and not has_line_items_in_cols:
            return df.T
        return df

    def get_index_history(self, index_symbol: str, years: int = 30) -> Optional[pd.DataFrame]:
        """Get index history from local storage."""
        db = self._get_db()
        if db:
            try:
                cutoff = datetime.now() - timedelta(days=years * 365)
                df = db.execute("""
                    SELECT index_symbol, date, open, high, low, close, volume, source
                    FROM index_history
                    WHERE index_symbol = ? AND date >= ?
                    ORDER BY date
                """, [index_symbol, cutoff.date()]).fetchdf()
                if not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                    return df
            except Exception as e:
                logger.debug(f"Error querying index history for {index_symbol}: {e}")

        historic_df = load_historic_index_prices(index_symbol)
        if historic_df is not None and not historic_df.empty:
            cutoff = datetime.now() - timedelta(days=years * 365)
            historic_df = historic_df[historic_df["date"] >= cutoff]
            if not historic_df.empty:
                return historic_df.sort_values("date")

        return None
    
    def get_asset_cagr(self, symbol: str, window_years: int, asset_type: str = "stock") -> Optional[float]:
        """Get cached CAGR for a stock or index from the local DB."""
        db = self._get_db()
        if not db:
            return None
        try:
            row = db.execute("""
                SELECT cagr_pct
                FROM asset_cagr
                WHERE asset_type = ? AND symbol = ? AND window_years = ?
            """, [asset_type, symbol.upper(), int(window_years)]).fetchone()
            if row:
                return row[0]
        except Exception as e:
            logger.debug(f"Error querying CAGR for {symbol}: {e}")
        return None
    
    def get_all_symbols(self) -> List[str]:
        """Get all available symbols from local storage."""
        symbols = set()
        
        # From parquet files
        if PRICE_DIR.exists():
            for f in PRICE_DIR.glob("*.parquet"):
                symbols.add(f.stem)
        
        # From info files
        if INFO_DIR.exists():
            for f in INFO_DIR.glob("*.json"):
                symbols.add(f.stem)
        
        # From database
        db = self._get_db()
        if db:
            try:
                rows = db.execute("SELECT DISTINCT symbol FROM stocks").fetchall()
                symbols.update(r[0] for r in rows)
            except:
                pass

        # From delisted data
        if self._include_delisted and self._delisted_dir.exists():
            for ext in ['*.parquet', '*.csv']:
                for f in self._delisted_dir.rglob(ext):
                    symbols.add(f.stem)
            for f in self._delisted_dir.rglob("*.json"):
                symbols.add(f.stem)

        # From bundled historic dataset
        for symbol in list_historic_stock_symbols():
            symbols.add(symbol)
        
        return sorted(list(symbols))

    def get_all_indices(self) -> List[str]:
        """Get all available indices from local storage."""
        indices = set()
        db = self._get_db()
        if db:
            try:
                rows = db.execute("SELECT DISTINCT index_symbol FROM index_history").fetchall()
                indices.update(r[0] for r in rows)
            except Exception:
                pass

        for index_name in list_historic_index_names():
            indices.add(index_name)

        return sorted(indices)
    
    def has_data(self) -> bool:
        """Check if local data exists."""
        return (PRICE_DIR.exists() and any(PRICE_DIR.glob("*.parquet"))) or DB_PATH.exists()


class YahooFinanceSource:
    """Yahoo Finance data source (online only)."""
    
    def __init__(self):
        self._yf = None
    
    def _get_yf(self):
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError:
                logger.warning("yfinance not installed")
        return self._yf
    
    def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get stock info from Yahoo Finance."""
        yf = self._get_yf()
        if not yf:
            return None
        
        # Special handling for NIFTY index
        if symbol.upper() == 'NIFTY':
            try:
                ticker = yf.Ticker("^NSEI")
                info = ticker.info
                if info and (info.get('regularMarketPrice') or info.get('currentPrice')):
                    return {
                        'symbol': 'NIFTY',
                        'name': 'Nifty 50',
                        'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                        'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                        'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                        'source': 'yahoo_finance_index'
                    }
            except Exception:
                pass
        
        for suffix in ['.NS', '.BO']:
            try:
                ticker = yf.Ticker(f"{symbol}{suffix}")
                info = ticker.info
                
                if info and (info.get('regularMarketPrice') or info.get('currentPrice')):
                    return {
                        'symbol': symbol,
                        'name': info.get('longName', info.get('shortName', symbol)),
                        'business_summary': info.get('longBusinessSummary'),
                        'city': info.get('city'),
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown'),
                        'market_cap': info.get('marketCap', 0),
                        'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                        'pe_ratio': info.get('trailingPE'),
                        'pb_ratio': info.get('priceToBook'),
                        'dividend_yield': (info.get('dividendYield', 0) or 0) * 100,
                        'eps': info.get('trailingEps'),
                        'book_value': info.get('bookValue'),
                        'roe': (info.get('returnOnEquity') or 0) * 100 if info.get('returnOnEquity') else None,
                        'debt_to_equity': info.get('debtToEquity'),
                        'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                        'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                        'beta': info.get('beta'),
                        'source': f'yahoo_finance{suffix}'
                    }
            except Exception as e:
                logger.debug(f"Yahoo info failed for {symbol}{suffix}: {e}")
                continue
        
        return None

    def get_price_history(self, symbol: str, years: int = 10) -> Optional[pd.DataFrame]:
        """Get price history from Yahoo Finance."""
        yf = self._get_yf()
        if not yf:
            return None
        
        # Special handling for NIFTY index
        target_symbols = []
        if symbol.upper() == 'NIFTY':
            target_symbols.append(('^NSEI', 'yahoo_finance_index'))
        else:
            for suffix in ['.NS', '.BO']:
                target_symbols.append((f"{symbol}{suffix}", f"yahoo_finance{suffix}"))
        
        for ticker_symbol, source_name in target_symbols:
            try:
                ticker = yf.Ticker(ticker_symbol)
                df = ticker.history(period=f"{years}y")
                
                if df.empty:
                    df = ticker.history(period="max")
                
                if not df.empty:
                    df = df.reset_index()
                    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
                    
                    if 'date' in df.columns:
                        try:
                            if hasattr(df['date'].dt, 'tz') and df['date'].dt.tz is not None:
                                df['date'] = df['date'].dt.tz_localize(None)
                        except Exception:
                            pass
                        df['date'] = pd.to_datetime(df['date'])
                    
                    df['symbol'] = symbol
                    df['source'] = source_name
                    
                    required = ['date', 'open', 'high', 'low', 'close', 'volume', 'source']
                    if all(c in df.columns for c in required):
                        logger.info(f"Got {len(df)} Yahoo records for {symbol}")
                        return df[required]
                        
            except Exception as e:
                logger.debug(f"Yahoo price failed for {ticker_symbol}: {e}")
                continue
        
        return None


class JugaadDataSource:
    """Jugaad Data source (optional online fallback)."""

    def __init__(self):
        self._stock_df = None
        self._index_df = None

    def _get_modules(self):
        if self._stock_df is None and self._index_df is None:
            try:
                from jugaad_data.nse import stock_df, index_df
                self._stock_df = stock_df
                self._index_df = index_df
            except Exception:
                self._stock_df = None
                self._index_df = None
        return self._stock_df, self._index_df

    def get_price_history(self, symbol: str, years: int = 10) -> Optional[pd.DataFrame]:
        stock_df, index_df = self._get_modules()
        if not stock_df and not index_df:
            return None

        end_date = datetime.now().date()
        start_date = (datetime.now() - timedelta(days=years * 365)).date()

        try:
            if symbol.upper().startswith("NIFTY") and index_df:
                df = index_df(symbol=symbol, from_date=start_date, to_date=end_date)
            elif stock_df:
                df = stock_df(symbol=symbol, from_date=start_date, to_date=end_date)
            else:
                return None
        except Exception as e:
            logger.debug(f"Jugaad data failed for {symbol}: {e}")
            return None

        if df is None or df.empty:
            return None

        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        df["symbol"] = symbol.upper()
        df["source"] = "jugaad_data"

        required = ["symbol", "date", "open", "high", "low", "close", "volume", "source"]
        missing = [c for c in required if c not in df.columns]
        for col in missing:
            df[col] = None

        logger.info(f"Got {len(df)} Jugaad records for {symbol}")
        return df[required].sort_values("date")


class NSEToolsSource:
    """NSE Tools for real-time quotes and stock list."""
    
    def __init__(self):
        self._nse = None
    
    def _get_nse(self):
        if self._nse is None:
            try:
                from nsetools import Nse
                self._nse = Nse()
            except:
                pass
        return self._nse
    
    def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        nse = self._get_nse()
        if not nse:
            return None
        
        try:
            quote = nse.get_quote(symbol)
            if quote:
                return {
                    'symbol': symbol,
                    'name': quote.get('companyName', symbol),
                    'current_price': quote.get('lastPrice', 0),
                    'pe_ratio': quote.get('pe'),
                    'fifty_two_week_high': quote.get('high52'),
                    'fifty_two_week_low': quote.get('low52'),
                    'source': 'nse_tools'
                }
        except Exception as e:
            logger.debug(f"NSE Tools failed for {symbol}: {e}")
        
        return None
    
    def get_all_stock_codes(self) -> Dict[str, str]:
        nse = self._get_nse()
        if nse:
            try:
                return nse.get_stock_codes()
            except:
                pass
        return {}


class DataSourceManager:
    """Manages data sources with local-first priority."""
    
    def __init__(self, offline_mode: bool = True):
        self.offline_mode = offline_mode
        self.local = LocalDataSource()
        self.yahoo = YahooFinanceSource()
        self.jugaad = JugaadDataSource()
        self.nse_tools = NSEToolsSource()

        cache_config = get_config('cache', {}) or {}
        memory_cache = cache_config.get('memory', {}) or {}
        info_limit = memory_cache.get('stock_info_max_entries', 2048)
        price_limit = memory_cache.get('price_history_max_entries', 128)
        self._info_cache = LRUCache(info_limit)
        self._price_cache = LRUCache(price_limit)
        self._stock_list: Optional[pd.DataFrame] = None
        
        # Check if local data exists
        if self.local.has_data():
            logger.info("Local data available - running in offline mode")
        else:
            logger.warning("No local data found - online mode required")
            self.offline_mode = False
    
    def get_available_sources(self) -> List[str]:
        """Get list of available data sources."""
        sources = []
        
        if self.local.has_data():
            sources.append('local_data')
        
        if not self.offline_mode:
            try:
                import yfinance
                sources.append('yahoo_finance')
            except:
                pass

            try:
                import jugaad_data
                sources.append('jugaad_data')
            except:
                pass
            
            try:
                from nsetools import Nse
                sources.append('nse_tools')
            except:
                pass
        
        return sources
    
    def get_stock_info(self, symbol: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Get stock info - local first, then online."""
        cache_key = symbol.upper()
        
        if use_cache and cache_key in self._info_cache:
            return self._info_cache.get(cache_key)
        
        # Try local first
        info = self.local.get_stock_info(symbol)
        
        # Try online if no local data and not in offline mode
        if not info and not self.offline_mode:
            info = self.yahoo.get_stock_info(symbol)
            if not info:
                info = self.nse_tools.get_stock_info(symbol)
        
        if info:
            self._info_cache.set(cache_key, info)
            logger.info(f"Got info for {symbol} from {info.get('source', 'unknown')}")
        
        return info
    
    def get_price_history(self, symbol: str, years: int = 10, 
                          use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Get price history - local first, then online."""
        cache_key = symbol.upper()
        
        if use_cache and cache_key in self._price_cache:
            cached_entry = self._price_cache.get(cache_key)
            if cached_entry:
                cached, cached_years = cached_entry
                if cached is not None and not cached.empty and cached_years >= years:
                    return self._filter_price_history(cached, years)
        
        # Try local first
        logger.info(f"Fetching price history for {symbol}...")
        df = self.local.get_price_history(symbol, years)
        
        # Try online if no local data and not in offline mode
        if (df is None or df.empty) and not self.offline_mode:
            logger.info(f"No local data, trying online for {symbol}...")
            df = self.yahoo.get_price_history(symbol, years)

            if df is None or df.empty:
                df = self.jugaad.get_price_history(symbol, years)
        
        if df is not None and not df.empty:
            self._price_cache.set(cache_key, (df, years))
            return self._filter_price_history(df, years)
        
        logger.warning(f"No price history found for {symbol}")
        return None

    def get_index_history(self, index_symbol: str, years: int = 30) -> Optional[pd.DataFrame]:
        """Get index history from local sources."""
        index_symbol = index_symbol.upper().strip()
        df = self.local.get_index_history(index_symbol, years)
        if df is not None and not df.empty:
            return df
        return None
    
    def get_stock_cagr(self, symbol: str, window_years: int) -> Optional[float]:
        """Get cached stock CAGR from local DB (if available)."""
        return self.local.get_asset_cagr(symbol, window_years, asset_type="stock")
    
    def get_index_cagr(self, index_symbol: str, window_years: int) -> Optional[float]:
        """Get cached index CAGR from local DB (if available)."""
        return self.local.get_asset_cagr(index_symbol, window_years, asset_type="index")
    
    def get_financials(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get financial statements."""
        local_fin = self.local.get_financials(symbol)
        if local_fin:
            return local_fin

        # Try Yahoo Finance if online
        if not self.offline_mode:
            try:
                import yfinance as yf
                for suffix in ['.NS', '.BO']:
                    try:
                        ticker = yf.Ticker(f"{symbol}{suffix}")
                        income_stmt = ticker.financials.T if ticker.financials is not None else pd.DataFrame()
                        balance_sheet = ticker.balance_sheet.T if ticker.balance_sheet is not None else pd.DataFrame()
                        cash_flow = ticker.cashflow.T if ticker.cashflow is not None else pd.DataFrame()
                        dividends = ticker.dividends if ticker.dividends is not None else pd.Series(dtype=float)
                        result = {
                            'income_statement': income_stmt,
                            'balance_sheet': balance_sheet,
                            'cash_flow': cash_flow,
                            'dividends': dividends,
                            'source': f'yahoo_finance{suffix}'
                        }
                        self._cache_financials(symbol, result)
                        return result
                    except:
                        continue
            except:
                pass
        
        return {
            'income_statement': pd.DataFrame(),
            'balance_sheet': pd.DataFrame(),
            'cash_flow': pd.DataFrame(),
            'dividends': pd.Series(dtype=float),
            'source': 'none'
        }

    def _cache_financials(self, symbol: str, financials: Dict[str, Any]) -> None:
        """Persist live financials to offline cache."""
        try:
            FINANCIALS_DIR.mkdir(parents=True, exist_ok=True)
            payload = {
                "symbol": symbol,
                "downloaded_at": datetime.now().isoformat(),
                "income_statement": self._df_to_payload(financials.get("income_statement")),
                "balance_sheet": self._df_to_payload(financials.get("balance_sheet")),
                "cash_flow": self._df_to_payload(financials.get("cash_flow")),
                "dividends": self._series_to_payload(financials.get("dividends")),
                "source": financials.get("source", "yahoo_finance"),
            }
            fin_file = FINANCIALS_DIR / f"{symbol}.json"
            with open(fin_file, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to cache financials for {symbol}: {e}")

    @staticmethod
    def _df_to_payload(df: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        if df is None or df.empty:
            return None
        try:
            df = df.copy()
            df.index = df.index.astype(str)
            df.columns = [str(c) for c in df.columns]
            return df.to_dict(orient="split")
        except Exception:
            return None

    @staticmethod
    def _series_to_payload(series: Optional[pd.Series]) -> Optional[Dict[str, Any]]:
        if series is None or series.empty:
            return None
        try:
            series = series.dropna()
            if series.empty:
                return None
            return {
                "index": [str(i) for i in series.index],
                "values": [float(v) for v in series.values],
            }
        except Exception:
            return None
    
    def get_all_stocks(self, force_refresh: bool = False) -> pd.DataFrame:
        """Get complete list of stocks."""
        if self._stock_list is not None and not force_refresh:
            return self._stock_list
        
        # Try local symbols first
        local_symbols = self.local.get_all_symbols()
        if local_symbols:
            self._stock_list = pd.DataFrame({
                'symbol': local_symbols,
                'name': local_symbols,  # Will be updated when info is fetched
                'exchange': 'NSE'
            })
            logger.info(f"Got {len(local_symbols)} symbols from local storage")
            return self._stock_list
        
        # Try NSE Tools
        if not self.offline_mode:
            codes = self.nse_tools.get_all_stock_codes()
            if codes:
                self._stock_list = pd.DataFrame([
                    {'symbol': k, 'name': v, 'exchange': 'NSE'}
                    for k, v in codes.items() if k != 'SYMBOL'
                ])
                logger.info(f"Got {len(self._stock_list)} symbols from NSE Tools")
                return self._stock_list
        
        return pd.DataFrame(columns=['symbol', 'name', 'exchange'])

    def get_all_indices(self) -> List[str]:
        """Get list of available indices."""
        return self.local.get_all_indices()
    
    def search_stocks(self, query: str, limit: int = 20) -> pd.DataFrame:
        """Search stocks by symbol or name."""
        stocks = self.get_all_stocks()
        if stocks.empty:
            return pd.DataFrame()
        
        query = query.upper()
        symbol_match = stocks[stocks['symbol'].str.upper().str.startswith(query)]
        name_match = stocks[stocks['name'].str.upper().str.contains(query, na=False)]
        
        result = pd.concat([symbol_match, name_match]).drop_duplicates(subset=['symbol'])
        return result.head(limit)
    
    def refresh_data(self, symbol: str) -> Tuple[bool, str]:
        """Force refresh data from online sources."""
        if self.offline_mode:
            logger.info("Offline mode enabled; attempting direct refresh anyway.")
        
        # Clear caches
        cache_key = symbol.upper()
        if cache_key in self._info_cache:
            self._info_cache.delete(cache_key)
        
        if cache_key in self._price_cache:
            self._price_cache.delete(cache_key)
        
        # Refresh local files from Yahoo Finance (gap fill)
        try:
            refresh_result = refresh_symbol_data(symbol)
        except Exception as e:
            return False, f"Refresh failed: {e}"

        # Refetch (now that local is updated)
        info = self.get_stock_info(symbol, use_cache=False)
        prices = self.get_price_history(symbol, use_cache=False)
        
        if info and prices is not None and not prices.empty:
            new_records = refresh_result.get("new_records", 0)
            if refresh_result.get("updated"):
                return True, f"Data refreshed: +{new_records} new records"
            return True, f"Data refreshed: {len(prices)} price records"
        elif info:
            return True, "Stock info refreshed, but no price history available"
        else:
            return False, "Failed to refresh data from online sources"
    
    def clear_cache(self):
        """Clear all caches."""
        self._info_cache.clear()
        self._price_cache.clear()

    def _filter_price_history(self, df: pd.DataFrame, years: int) -> pd.DataFrame:
        """Filter cached price history to the requested window."""
        if df is None or df.empty:
            return df
        if "date" not in df.columns:
            return df
        try:
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            cutoff = datetime.now() - timedelta(days=years * 365)
            return df[df["date"] >= cutoff].sort_values("date")
        except Exception:
            return df
