COMPANY_INFO = {
    "name": "Cronix AI",
    "founder": {
        "name": "Rudra Patel",
        "role": "CEO & Founder",
        "description": "Rudra Patel is the CEO and Founder of Cronix AI, an innovative company focused on AI model development and fintech solutions. He leads the company's vision in developing advanced artificial intelligence technologies and financial technology solutions."
    },
    "responses": {
        "founder": "Rudra Patel is the CEO and Founder of Cronix AI, an innovative company focused on AI model development and fintech solutions.",
        "creator": "I was created by Rudra Patel, the founder of Cronix AI, a company specializing in AI and fintech solutions.",
        "company": "I am an AI assistant from Cronix AI, a company focused on advanced artificial intelligence and fintech solutions.",
        "ceo": "Rudra Patel is the CEO and Founder of Cronix AI.",
        "rudra": "Rudra Patel is the CEO and Founder of Cronix AI, leading the development of advanced AI models and fintech solutions.",
        "rudra patel": "Rudra Patel is the CEO and Founder of Cronix AI, an innovative company specializing in AI model development and fintech solutions."
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
