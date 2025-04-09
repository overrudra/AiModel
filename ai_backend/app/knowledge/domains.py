DOMAIN_DEFINITIONS = {
    "ai": {
        "name": "Artificial Intelligence",
        "short_def": "AI is the simulation of human intelligence by machines.",
        "definition": """Artificial Intelligence (AI) is a branch of computer science focused on creating intelligent machines that can simulate human intelligence. It encompasses machine learning, natural language processing, robotics, and expert systems. AI systems can learn from experience, adjust to new inputs, and perform human-like tasks.""",
        "key_concepts": ["Machine Learning", "Neural Networks", "Deep Learning", "Natural Language Processing", "Computer Vision"],
        "applications": ["Healthcare", "Finance", "Autonomous Vehicles", "Virtual Assistants", "Robotics"]
    },
    "mathematics": {
        "name": "Mathematics",
        "short_def": "Mathematics is the science of numbers, quantity, structure, and space.",
        "definition": """Mathematics is the systematic study of numbers, quantities, shapes, and patterns. It is a fundamental science that provides tools for other sciences, engineering, and everyday life. Mathematics includes various branches like algebra, geometry, calculus, and statistics.""",
        "branches": ["Algebra", "Geometry", "Calculus", "Statistics", "Number Theory"],
        "applications": ["Physics", "Engineering", "Economics", "Computer Science", "Data Analysis"]
    },
    "computer_science": {
        "name": "Computer Science",
        "short_def": "Computer Science is the study of computers and computational systems.",
        "definition": """Computer Science is the study of computers, computational systems, and their theoretical foundations. It encompasses both the theoretical and practical aspects of computation and information processing. The field includes programming, algorithms, data structures, and computer systems.""",
        "key_concepts": ["Programming", "Algorithms", "Data Structures", "Computer Architecture", "Software Engineering"],
        "applications": ["Software Development", "AI", "Cybersecurity", "Database Systems", "Networks"]
    }
}

GENERAL_CATEGORIES = {
    "food": ["cake", "bread", "pizza"],
    "science": ["physics", "chemistry", "biology"],
    "technology": ["computer", "internet", "software"],
    # Add more categories as needed
}

def format_domain_response(domain_key: str, include_details: bool = True) -> str:
    """Format a comprehensive response for a domain"""
    if domain_key not in DOMAIN_DEFINITIONS:
        return None
        
    domain = DOMAIN_DEFINITIONS[domain_key]
    
    if not include_details:
        return domain['short_def']
        
    response = [domain['definition']]
    
    if 'key_concepts' in domain:
        concepts = ', '.join(domain['key_concepts'])
        response.append(f"\nKey concepts: {concepts}")
    
    if 'applications' in domain:
        applications = ', '.join(domain['applications'])
        response.append(f"\nApplications: {applications}")
    
    if 'branches' in domain:
        branches = ', '.join(domain['branches'])
        response.append(f"\nBranches: {branches}")
    
    return '\n'.join(response)

def get_category(query: str) -> str:
    """Get the general category of a query"""
    query = query.lower()
    for category, items in GENERAL_CATEGORIES.items():
        if query in items:
            return category
    return None
