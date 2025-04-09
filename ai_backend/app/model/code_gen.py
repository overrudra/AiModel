def is_code_prompt(prompt: str):
    keywords = ["write a code", "generate python", "build function", "error in code", "fix my code"]
    return any(k in prompt.lower() for k in keywords)

def generate_code(prompt: str):
    return f"# Code generation logic based on: {prompt}\nprint('Hello, World!')"
