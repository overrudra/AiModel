import requests
import wikipedia
import time
import json
import os
import re
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from .domains import DOMAIN_DEFINITIONS, format_domain_response
from .common_knowledge import get_definition, find_item
from .company_info import get_company_response
from .learned_data import LearnedKnowledge
from .web_searcher import WebSearcher
from ..config import SYSTEM_CONFIG

class KnowledgeRetriever:
    def __init__(self):
        self.search_cache = {}
        self.cache_timeout = 3600  # 1 hour cache
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
        self.cache_file = os.path.join(self.cache_dir, "knowledge_cache.json")
        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_cache()
        self.web_searcher = WebSearcher()
        self.learned = LearnedKnowledge()

    def get_knowledge(self, query: str) -> str:
        """Get knowledge with improved accuracy and validation"""
        try:
            # Clean and normalize the query
            query = self._normalize_query(query)
            
            # 1. Check company info first
            if company_info := get_company_response(query):
                return self._validate_response(company_info)
                
            # 2. Check learned knowledge
            if learned := self.learned.get_knowledge(query):
                return self._validate_response(learned)
                
            # 3. Try Wikipedia with error handling and multiple attempts
            try:
                if wiki_info := self.search_wikipedia(query):
                    validated_info = self._validate_response(wiki_info)
                    if validated_info:
                        # Store validated info in learned knowledge
                        self.learned.add_knowledge(query, validated_info, "wikipedia")
                        return validated_info
            except Exception as e:
                print(f"Wikipedia search error: {e}")
                
            # 4. Try web search with improved results
            if web_results := self.web_searcher.search(query):
                best_result = web_results[0]['content']
                validated_result = self._validate_response(best_result)
                if validated_result:
                    # Store validated result in learned knowledge
                    self.learned.add_knowledge(query, validated_result, "web")
                    return validated_result
                
            # 5. Check common knowledge
            category, item = find_item(query)
            if category and item:
                if knowledge := get_definition(category, item):
                    return self._validate_response(knowledge)
                    
            return "I apologize, I don't have accurate information about that topic."
            
        except Exception as e:
            print(f"Knowledge retrieval error: {e}")
            return "Sorry, I encountered an error retrieving that information."

    def search_wikipedia(self, query: str) -> str:
        """Enhanced Wikipedia search with better accuracy"""
        try:
            # Try exact match first
            try:
                page = wikipedia.page(query, auto_suggest=False)
                summary = page.summary
                if summary:
                    # Take first two sentences for conciseness
                    sentences = re.split(r'(?<=[.!?])\s+', summary)
                    return ' '.join(sentences[:2])
            except wikipedia.exceptions.DisambiguationError as e:
                # Try the first suggested match
                if e.options:
                    try:
                        page = wikipedia.page(e.options[0], auto_suggest=False)
                        sentences = re.split(r'(?<=[.!?])\s+', page.summary)
                        return ' '.join(sentences[:2])
                    except:
                        pass
            except wikipedia.exceptions.PageError:
                # Try search
                search_results = wikipedia.search(query)
                if search_results:
                    try:
                        page = wikipedia.page(search_results[0], auto_suggest=False)
                        sentences = re.split(r'(?<=[.!?])\s+', page.summary)
                        return ' '.join(sentences[:2])
                    except:
                        pass
                        
        except Exception as e:
            print(f"Wikipedia error: {e}")
            
        return ""

    def _normalize_query(self, query: str) -> str:
        """Normalize query for better matching"""
        # Convert to lowercase
        query = query.lower()
        # Remove extra whitespace
        query = ' '.join(query.split())
        # Remove question words at start
        question_words = ['who', 'what', 'where', 'when', 'why', 'how', 'is', 'are', 'was', 'were']
        words = query.split()
        if words and words[0] in question_words:
            query = ' '.join(words[1:])
        return query

    def _validate_response(self, text: str) -> str:
        """Validate and clean response text"""
        if not text:
            return ""
            
        # Remove any markdown or HTML
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove non-ASCII characters
        text = re.sub(r'[^\x20-\x7E\s]', '', text)
        
        # Ensure proper sentence structure
        sentences = re.split(r'(?<=[.!?])\s+', text)
        valid_sentences = []
        
        for sentence in sentences:
            # Check if sentence looks valid
            if (len(sentence) > 10 and  # Long enough
                sentence[0].isupper() and  # Starts with capital
                sentence.strip().endswith(('.', '!', '?')) and  # Proper ending
                not re.search(r'([A-Za-z])\1{2,}', sentence)):  # No character repetition
                valid_sentences.append(sentence)
                
        if not valid_sentences:
            return ""
            
        return ' '.join(valid_sentences)

    def _load_cache(self):
        """Load cached results"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.search_cache = json.load(f)
        except:
            self.search_cache = {}

    def _save_cache(self):
        """Save cache to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.search_cache, f, indent=2)
        except Exception as e:
            print(f"Cache save error: {e}")
