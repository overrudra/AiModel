from tokenizers import ByteLevelBPETokenizer
import os

# Make sure output folder exists
os.makedirs("tokenizer", exist_ok=True)

# Initialize tokenizer
tokenizer = ByteLevelBPETokenizer()

# Train tokenizer
tokenizer.train(
    files="ai-backend/app/dataset.txt",
    vocab_size=3000,
    min_frequency=2,
    special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
)

# Save tokenizer model
tokenizer.save_model("tokenizer")
