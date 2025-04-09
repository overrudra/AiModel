MODEL_CONFIG = {
    "vocab_size": 32000,        # Increased vocab size
    "block_size": 512,          # Larger context window
    "n_embd": 768,             # Larger embedding size
    "n_layer": 12,             # More transformer layers
    "n_head": 12,              # More attention heads
    "dropout": 0.1,
    "learning_rate": 1e-4,
    "batch_size": 32,
    "warmup_steps": 1000,
    "max_steps": 100000
}

GENERATION_CONFIG = {
    "max_length": 300,          # Longer responses
    "min_length": 10,
    "temperature": 0.7,         # More focused responses
    "top_p": 0.9,              # Nucleus sampling
    "top_k": 50,               # More diverse token selection
    "repetition_penalty": 1.2   # Avoid repetition
}
