COMPANY_INFO = {
    "name": "Cronix AI",
    "founder": {
        "name": "Rudra Patel",
        "role": "CEO & Founder"
    },
    "responses": {
        "founder": "Rudra Patel is the CEO and Founder of Cronix AI. He leads the development of advanced AI solutions.",
        "creator": "I was created by Rudra Patel, the founder of Cronix AI.",
        "company": "I am an AI assistant from Cronix AI, a company focused on advanced artificial intelligence solutions.",
        "ceo": "Rudra Patel is the CEO of Cronix AI."
    }
}

def get_company_response(query: str) -> str:
    """Get appropriate company-related response"""
    query = query.lower()
    
    # Check for founder/creator related questions
    if any(word in query for word in ['founder', 'created', 'creator', 'ceo', 'who made', 'developed']):
        if 'founder' in query or 'ceo' in query:
            return COMPANY_INFO['responses']['founder']
        return COMPANY_INFO['responses']['creator']
        
    # Return None if not company related
    return None
