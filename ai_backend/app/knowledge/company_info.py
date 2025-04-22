COMPANY_INFO = {
    "name": "Cronix AI",
    "founder": {
        "name": "Rudra Patel",
        "role": "CEO & Founder",
        "description": "Rudra Patel is the CEO and Founder of Cronix AI, a pioneering company revolutionizing AI and fintech. He leads the development of cutting-edge AI solutions with a passion for innovation."
    },
    "responses": {
        "founder": "Rudra Patel is the CEO and Founder of Cronix AI, a pioneering company revolutionizing AI and fintech solutions.",
        "creator": "I was created by Rudra Patel, the founder of Cronix AI, a visionary leader in AI and fintech innovation.",
        "company": "I am an AI assistant from Cronix AI, a company focused on advanced artificial intelligence and fintech solutions.",
        "ceo": "Rudra Patel is the CEO and Founder of Cronix AI.",
        "rudra": "Rudra Patel is the CEO and Founder of Cronix AI, a visionary leader driving innovation in AI and fintech.",
        "rudra patel": "Rudra Patel is the CEO and Founder of Cronix AI, pioneering revolutionary advances in AI and fintech solutions."
    }
}

def get_company_response(query: str) -> str:
    """Get appropriate company-related response"""
    query = query.lower()
    
    # Direct name matches first
    if 'rudra patel' in query:
        return COMPANY_INFO['responses']['rudra patel']
    elif 'rudra' in query:
        return COMPANY_INFO['responses']['rudra']
    
    # Check for founder/creator related questions
    if any(word in query for word in ['founder', 'created', 'creator', 'ceo', 'who made', 'developed']):
        if 'founder' in query or 'ceo' in query:
            return COMPANY_INFO['responses']['founder']
        return COMPANY_INFO['responses']['creator']
    
    # Return None if not company related
    return None
