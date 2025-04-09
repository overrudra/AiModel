import requests
from bs4 import BeautifulSoup
import wikipedia
import time
from typing import List, Dict
import concurrent.futures
import json
import os
from urllib.parse import quote
import re
from .domains import DOMAIN_DEFINITIONS, format_domain_response
from .common_knowledge import get_definition, find_item
from .company_info import get_company_response
from ..config import OFFLINE_MODE, SYSTEM_CONFIG
from .learned_data import LearnedKnowledge

__all__ = ['KnowledgeRetriever']

class KnowledgeRetriever:
    def __init__(self):
        self.search_cache = {}
        self.cache_timeout = 3600  # 1 hour cache
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        self.cache_file = os.path.join(self.cache_dir, "knowledge_cache.json")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_cache()
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.learned = LearnedKnowledge()

    def _load_cache(self):
        """Load cached results from file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cached_data = json.load(f)
                    self.search_cache = {k: v for k, v in cached_data.items() 
                                      if time.time() - v['time'] < self.cache_timeout}
        except:
            self.search_cache = {}

    def _save_cache(self):
        """Save cache to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.search_cache, f)
        except:
            pass

    def search_wikipedia(self, query: str) -> str:
        """Enhanced Wikipedia search"""
        try:
            # Clean query
            query = re.sub(r'[^\w\s]', '', query)
            query = ' '.join(query.split())  # Normalize whitespace
            
            # Try exact search first
            try:
                page = wikipedia.page(query, auto_suggest=False)
                summary = page.summary[:500]  # Get shorter summary
                return self.clean_content(summary)
            except wikipedia.DisambiguationError as e:
                # Try first suggested page
                try:
                    page = wikipedia.page(e.options[0])
                    return self.clean_content(page.summary[:500])
                except:
                    pass
            except:
                # Try search
                results = wikipedia.search(query, results=3)
                for result in results:
                    try:
                        page = wikipedia.page(result)
                        return self.clean_content(page.summary[:500])
                    except:
                        continue
            return ""
        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return ""

    def search_web(self, query: str) -> List[str]:
        """Enhanced web search using multiple sources"""
        results = []
        
        # DuckDuckGo search
        try:
            url = f"https://lite.duckduckgo.com/lite?q={quote(query)}"
            response = self.session.get(url, headers=self.headers, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            for snippet in soup.find_all('a', {'class': 'result-snippet'})[:5]:
                results.append(snippet.text.strip())
        except:
            pass

        # Google search alternative
        try:
            url = f"https://www.google.com/search?q={quote(query)}"
            response = self.session.get(url, headers=self.headers, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')
            for snippet in soup.find_all('div', {'class': 'VwiC3b'})[:5]:
                results.append(snippet.text.strip())
        except:
            pass

        return list(set(results))  # Remove duplicates

    def search_news(self, query: str) -> List[str]:
        """Search recent news articles"""
        try:
            url = f"https://news.google.com/rss/search?q={quote(query)}"
            response = self.session.get(url, headers=self.headers, timeout=5)
            soup = BeautifulSoup(response.text, 'xml')
            results = []
            for item in soup.find_all('item')[:3]:
                title = item.title.text
                desc = item.description.text
                results.append(f"{title} - {desc}")
            return results
        except:
            return []

    def clean_content(self, text: str) -> str:
        """Clean and validate response"""
        if not text or len(text.split()) < 3:
            return ""
            
        # Remove special characters and normalize
        text = re.sub(r'[^\x20-\x7E\s]', '', text)
        text = re.sub(r'W+', '', text)
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        text = ' '.join(text.split())
        
        # Validate final text
        if len(text.split()) < 3 or not re.match(r'^[a-zA-Z0-9\s.,!?-]+$', text):
            return ""
            
        return text.strip()

    def extract_relevant_info(self, text: str, query: str) -> str:
        """Extract most relevant sentences"""
        sentences = text.split('.')
        query_words = set(query.lower().split())
        relevant = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # Check if sentence contains query terms
            if any(word in sentence.lower() for word in query_words):
                relevant.append(sentence)
        
        # Return most relevant sentences or first few if none are relevant
        return '. '.join(relevant[:3] if relevant else sentences[:2]) + '.'

    def extract_domain_info(self, text: str, query: str) -> str:
        """Extract domain-specific information"""
        # Remove HTML and clean text
        text = self.clean_content(text)
        
        # Look for definition patterns
        patterns = [
            r'(\w+(?:\s+\w+)?) (?:is|refers to) (?:a|an|the) (?:field|domain|discipline|technology|science)',
            r'(?:field|domain|discipline) of (\w+(?:\s+\w+)?)',
            r'(\w+(?:\s+\w+)?) (?:technology|science|engineering)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip().title()
        
        return None

    def identify_domain(self, query: str) -> str:
        """Identify the domain from the query"""
        query = query.lower().strip()
        
        # Direct matches
        for domain_key in DOMAIN_DEFINITIONS:
            if domain_key in query:
                return format_domain_response(domain_key)
                
        # Look for domain names in definitions
        for domain_key, domain in DOMAIN_DEFINITIONS.items():
            if domain['name'].lower() in query:
                return format_domain_response(domain_key)
        
        return None

    def get_knowledge(self, query: str) -> str:
        """Get knowledge with strict priority ordering"""
        try:
            # 1. First priority: Company/Identity info
            company_response = get_company_response(query)
            if company_response:
                return company_response

            # 2. Check cache
            cached = self._get_from_cache(query)
            if cached:
                return cached

            # 3. Try Wikipedia
            wiki_response = self.search_wikipedia(query)
            if wiki_response:
                self._add_to_cache(query, wiki_response)
                return wiki_response

            # 4. Fallback to web search
            if SYSTEM_CONFIG['use_internet']:
                web_response = self._get_online_knowledge(query)
                if web_response:
                    self._add_to_cache(query, web_response)
                    return web_response

            return "I apologize, I don't have information about that topic."

        except Exception as e:
            print(f"Knowledge retrieval error: {e}")
            return "I apologize, I'm having trouble retrieving that information."

    def _get_local_knowledge(self, query: str) -> str:
        """Try local knowledge sources"""
        # Always check company info first
        if company := get_company_response(query):
            return company
            
        # 1. Check cache
        if query in self.search_cache:
            if time.time() - self.search_cache[query]['time'] < self.cache_timeout:
                return self.search_cache[query]['result']
        
        # 2. Check learned knowledge
        if learned := self.learned.get_knowledge(query):
            return learned
            
        # 3. Try company info
        if company := get_company_response(query):
            return company
            
        # 4. Check common knowledge
        category, item = find_item(query)
        if category and item:
            if knowledge := get_definition(category, item):
                return knowledge
                
        # 5. Try domain definitions
        if domain := self.identify_domain(query):
            return domain
            
        return ""

    def _get_online_knowledge(self, query: str) -> str:
        """Search online sources and format response"""
        try:
            # Try Wikipedia first
            if wiki_info := self.search_wikipedia(query):
                # Extract most relevant parts
                sentences = wiki_info.split('.')[:3]
                return '. '.join(s.strip() for s in sentences if s.strip()) + '.'
                
            # Try web search
            if web_results := self.search_web(query):
                # Clean and format the best result
                best_result = self.clean_content(web_results[0])
                return '. '.join(best_result.split('.')[:2]) + '.'
                
        except Exception as e:
            print(f"Online search error: {str(e)}")
            
        return ""

    # ...rest of existing code...
