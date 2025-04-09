import json
import os
from typing import Dict, Optional
from datetime import datetime

class LearnedKnowledge:
    def __init__(self):
        self.data_dir = "knowledge_data"
        self.data_file = os.path.join(self.data_dir, "learned_knowledge.json")
        os.makedirs(self.data_dir, exist_ok=True)
        self.knowledge_base = self._load_knowledge()
        
    def _load_knowledge(self) -> Dict:
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}
        
    def _save_knowledge(self):
        try:
            with open(self.data_file, 'w') as f:
                json.dump(self.knowledge_base, f, indent=2)
        except Exception as e:
            print(f"Error saving knowledge: {e}")

    def add_knowledge(self, query: str, response: str, source: str = "online"):
        """Add new knowledge with metadata"""
        query = query.lower()
        self.knowledge_base[query] = {
            "response": response,
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "access_count": 0,
            "last_verified": datetime.now().isoformat()
        }
        self._save_knowledge()
        print(f"Added new knowledge: {query}")

    def get_knowledge(self, query: str) -> Optional[str]:
        """Get learned knowledge"""
        entry = self.knowledge_base.get(query.lower())
        if entry:
            entry["access_count"] = entry.get("access_count", 0) + 1
            self._save_knowledge()
            return entry["response"]
        return None
