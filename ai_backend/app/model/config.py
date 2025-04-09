MODEL_CONFIG = {
    "vocab_size": 270,          # Match actual vocab.json size
    "block_size": 64,          # Match saved checkpoint block_size
    "n_embd": 256,             # Keep embedding size
    "n_layer": 6,              # Keep number of layers
    "n_head": 8,              # Keep number of attention heads
    "dropout": 0.1,
    "learning_rate": 3e-4,     # Match training learning rate
    "batch_size": 32,          # Match training batch size
    "warmup_steps": 1000,
    "max_steps": 100000
}

GENERATION_CONFIG = {
    "max_length": 100,          # Response length
    "min_length": 10,
    "temperature": 0.7,         # More focused responses
    "top_p": 0.9,              # Nucleus sampling
    "top_k": 50,               # More diverse token selection
    "repetition_penalty": 1.2   # Avoid repetition
}
