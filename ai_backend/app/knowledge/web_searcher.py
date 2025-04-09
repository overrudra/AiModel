import requests
import json
import time
import os
import re
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class WebSearcher:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        self.cache_file = os.path.join(self.cache_dir, "web_cache.json")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_cache()

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search multiple sources and return combined results"""
        results = []
        
        # Try DuckDuckGo
        ddg_results = self._search_duckduckgo(query)
        if ddg_results:
            results.extend(ddg_results)
        
        # Filter and rank results
        ranked_results = self._rank_results(query, results)
        
        # Return top results after filtering
        return ranked_results[:max_results]

    def _search_duckduckgo(self, query: str) -> List[Dict[str, str]]:
        """Search DuckDuckGo with improved result extraction"""
        try:
            cache_key = f"ddg_{query.lower()}"
            if cache_key in self.cache:
                cached = self.cache[cache_key]
                if time.time() - cached["time"] < 3600:  # 1 hour cache
                    return cached["results"]

            # Add exact match requirement for proper names
            if self._is_proper_name(query):
                search_query = f'"{query}"'  # Force exact match
            else:
                search_query = query

            url = f"https://duckduckgo.com/html/?q={search_query}"
            response = self.session.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                results = []
                
                for result in soup.select('.result__body'):
                    title = result.select_one('.result__title')
                    snippet = result.select_one('.result__snippet')
                    link = result.select_one('.result__url')
                    
                    if title and snippet:
                        text = snippet.get_text().strip()
                        if self._is_valid_result(query, text):
                            result_dict = {
                                'title': title.get_text().strip(),
                                'content': text,
                                'url': link.get_text().strip() if link else '',
                                'score': self._calculate_relevance(query, text)
                            }
                            results.append(result_dict)
                
                if results:
                    self.cache[cache_key] = {
                        "time": time.time(),
                        "results": results
                    }
                    self._save_cache()
                
                return results
                
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            
        return []

    def _is_proper_name(self, query: str) -> bool:
        """Check if query looks like a proper name"""
        words = query.strip().split()
        return len(words) >= 2 and all(word[0].isupper() for word in words if word)

    def _is_valid_result(self, query: str, text: str) -> bool:
        """Validate if the result is relevant and well-formed"""
        if len(text) < 50:  # Too short to be meaningful
            return False
            
        # Check for gibberish
        if re.search(r'[^\x20-\x7E\s]', text):  # Non-ASCII characters
            return False
            
        # Ensure some query terms are present
        query_terms = set(query.lower().split())
        text_terms = set(text.lower().split())
        matches = query_terms.intersection(text_terms)
        return len(matches) >= len(query_terms) * 0.5  # At least 50% of query terms present

    def _calculate_relevance(self, query: str, text: str) -> float:
        """Calculate relevance score for ranking"""
        score = 0.0
        query_terms = query.lower().split()
        
        # Check for exact phrase match
        if query.lower() in text.lower():
            score += 3.0
            
        # Check for individual term matches near the start
        first_sentence = text.split('.')[0].lower()
        for term in query_terms:
            if term in first_sentence:
                score += 1.0
                
        # Check for proper sentence structure
        if text[0].isupper() and text.endswith(('.', '!', '?')):
            score += 0.5
            
        return score

    def _rank_results(self, query: str, results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Rank and filter results by relevance"""
        if not results:
            return []
            
        # Sort by score
        ranked = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
        
        # Filter duplicates and low quality results
        seen_content = set()
        filtered = []
        for result in ranked:
            content = result['content']
            if content not in seen_content and self._is_valid_result(query, content):
                seen_content.add(content)
                filtered.append(result)
                
        return filtered

    def _load_cache(self):
        """Load search cache"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
            else:
                self.cache = {}
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            self.cache = {}

    def _save_cache(self):
        """Save search cache"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")