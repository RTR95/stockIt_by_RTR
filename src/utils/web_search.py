"""
Web Search Utility for Stocron.
Fetches recent news and context for stocks.
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class WebSearch:
    """
    Simple web search wrapper to fetch news context.
    """
    
    def __init__(self):
        self._ddgs = None
        
    def _get_ddgs(self):
        if self._ddgs is None:
            try:
                from duckduckgo_search import DDGS
                self._ddgs = DDGS()
            except ImportError:
                logger.warning("duckduckgo-search not installed. Run: pip install duckduckgo-search")
                self._ddgs = False
        return self._ddgs

    def search_news(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Search for news articles.
        
        Args:
            query: Search query (e.g., "TCS stock news")
            max_results: Number of results to return
            
        Returns:
            List of dicts with 'title', 'link', 'date', 'snippet'
        """
        ddgs = self._get_ddgs()
        if not ddgs:
            return []
            
        try:
            results = ddgs.news(query, max_results=max_results)
            return [
                {
                    'title': r.get('title'),
                    'link': r.get('url'),
                    'date': r.get('date'),
                    'snippet': r.get('body')
                }
                for r in results
            ]
        except Exception as e:
            logger.warning(f"News search failed: {e}")
            return []

    def get_stock_news(self, symbol: str, company_name: str = "") -> str:
        """
        Get a summarized string of recent news for a stock.
        """
        query = f"{company_name or symbol} stock news india"
        news_items = self.search_news(query, max_results=3)
        
        if not news_items:
            return "No recent news found."
            
        summary = "Recent News:\n"
        for item in news_items:
            date_str = item['date'][:10] if item.get('date') else ""
            summary += f"- [{date_str}] {item['title']}: {item['snippet']}\n"
            
        return summary
