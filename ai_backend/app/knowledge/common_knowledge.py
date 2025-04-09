from typing import Tuple, Optional
from .domains import GENERAL_CATEGORIES, DOMAIN_DEFINITIONS

def get_definition(category: str, item: str) -> Optional[str]:
    """Get definition for an item in a category"""
    if category in DOMAIN_DEFINITIONS:
        domain = DOMAIN_DEFINITIONS[category]
        if "definition" in domain:
            return domain["definition"]
        elif "short_def" in domain:
            return domain["short_def"]
    return None

def find_item(query: str) -> Tuple[Optional[str], Optional[str]]:
    """Find category and item from query"""
    query = query.lower()
    # First check direct domain matches
    for domain in DOMAIN_DEFINITIONS:
        if domain.lower() in query or DOMAIN_DEFINITIONS[domain]["name"].lower() in query:
            return domain, domain

    # Then check general categories
    for category, items in GENERAL_CATEGORIES.items():
        for item in items:
            if item.lower() in query:
                return category, item
                
    return None, None