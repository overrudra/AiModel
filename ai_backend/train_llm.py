import torch
import torch.nn as nn
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import Dataset, DataLoader
from app.char_dataset import CharDataset
from app.model.llm import TinyTransformer
from app.model.config import MODEL_CONFIG

# Load trained tokenizer
tokenizer = ByteLevelBPETokenizer(
    "tokenizer/vocab.json",
    "tokenizer/merges.txt"
)

# Use configuration from MODEL_CONFIG
vocab_size = tokenizer.get_vocab_size()
block_size = MODEL_CONFIG['block_size']
batch_size = MODEL_CONFIG['batch_size']
learning_rate = MODEL_CONFIG['learning_rate']
epochs = 10
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
with open("ai-backend/app/dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Encode entire dataset
encoded = tokenizer.encode(text).ids
print(f"[DEBUG] Total tokens: {len(encoded)}")

# Custom dataset using token ids
class TokenDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

dataset = TokenDataset(encoded, block_size)
print(f"[DEBUG] Dataset size after blocking: {len(dataset)}")

# Abort if not enough data
if len(dataset) == 0:
    raise ValueError("❌ Not enough data to create dataset. Please add more content to dataset.txt.")

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create model with consistent config
model = TinyTransformer.create_model(MODEL_CONFIG, device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    total_loss = 0
    model.train()
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"✅ Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

# Save model with full configuration
model.save_checkpoint("ai-backend/app/model/tiny_llm.pth")
print("✅ Model saved successfully!")
