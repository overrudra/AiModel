import re
from typing import Optional, List

def clean_text(text: str) -> str:
    """Clean and format text content"""
    # Remove HTML tags and URLs
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Remove special characters and clean up whitespace
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return ' '.join(text.split())

def extract_domain_name(text: str) -> str:
    """Extract main domain or course name from text"""
    # Common domain markers
    markers = ['technology', 'science', 'engineering', 'course', 'field', 'domain', 'area', 'discipline']
    
    # Look for patterns like "X is a field of" or "X is a technology"
    sentences = text.split('.')
    for sentence in sentences:
        sentence = sentence.lower().strip()
        if any(marker in sentence for marker in markers):
            # Extract the subject before the marker
            for marker in markers:
                if marker in sentence:
                    parts = sentence.split(marker)[0].split()
                    if parts:
                        # Get last noun phrase before marker
                        domain = ' '.join(parts[-2:])
                        return domain.strip().title()
    
    # Fallback: return first capitalized phrase
    words = text.split()
    for i, word in enumerate(words):
        if word[0].isupper() and i < len(words) - 1:
            return f"{word} {words[i+1]}".strip()
    
    return ""

def format_response(text: str, context: str = "") -> str:
    """Format response to be concise and readable"""
    text = clean_text(text)
    
    if context:
        # Extract only relevant context sentences
        context_parts = context.split('.')
        relevant_parts = [p for p in context_parts if len(p.strip()) > 0][:2]
        if relevant_parts:
            return f"{text}\n\nSource: {'. '.join(relevant_parts)}."
    
    return text

def extract_question_answer(text: str, context: str) -> str:
    """Extract direct answer from context based on question"""
    question = clean_text(text.lower())
    context = clean_text(context)
    
    # For "what is" questions, look for definition patterns
    if question.startswith("what is") or question.startswith("whats"):
        topic = question.replace("what is", "").replace("whats", "").strip()
        # Look for definition patterns in context
        patterns = [
            f"{topic} is",
            f"{topic} refers to",
            f"{topic} means",
            f"refers to {topic} as",
            topic
        ]
        
        for pattern in patterns:
            for sentence in context.split('.'):
                if pattern in sentence.lower():
                    return clean_text(sentence) + "."
                    
    return None
