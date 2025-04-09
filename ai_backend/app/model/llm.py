import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x, mask=None):
        attention_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = x + attention_output
        x = x + self.ffn(self.ln2(x))
        return x

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size=None, block_size=64, n_embd=256, n_layer=6, n_head=8, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size or 100  # Fallback vocab size
        self.block_size = block_size
        self.n_embd = n_embd
        
        # Embeddings
        self.token_emb = nn.Embedding(self.vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)
        ])
        
        # Output head
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, self.vocab_size, bias=False)
        
        # Initialize
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, idx, mask=None):
        """Forward pass with safe dimension handling"""
        b, t = idx.size()
        
        # Ensure sequence length doesn't exceed block size
        if t > self.block_size:
            idx = idx[:, :self.block_size]
            t = self.block_size
        
        # Token embeddings
        tok_emb = self.token_emb(idx)
        
        # Position embeddings with safe indexing
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).clamp(max=self.block_size-1)
        pos_emb = self.pos_emb(pos)
        
        # Add embeddings and process
        x = self.drop(tok_emb + pos_emb)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

    def generate(self, idx: torch.Tensor, max_tokens: int = 100, temperature: float = 0.8) -> torch.Tensor:
        """Generate text with safe token handling"""
        self.eval()
        batch_size = idx.size(0)
        generated = idx
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # Get last block_size tokens
                if generated.size(1) > self.block_size:
                    context = generated[:, -self.block_size:]
                else:
                    context = generated
                
                # Forward pass
                logits = self(context)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Stop if end token or max length
                if next_token[0].item() in {0, 3}:  # pad or end token
                    break
                    
                generated = torch.cat([generated, next_token], dim=1)
                
                # Safety check for max length
                if generated.size(1) >= self.block_size * 2:
                    break
        
        return generated

    @classmethod
    def create_model(cls, config: dict, device='cpu'):
        """Create new model with config"""
        config = {
            'vocab_size': config.get('vocab_size', 100),
            'block_size': config.get('block_size', 64),
            'n_embd': config.get('n_embd', 256),
            'n_layer': config.get('n_layer', 6),
            'n_head': config.get('n_head', 8),
            'dropout': config.get('dropout', 0.1)
        }
        return cls(**config).to(device)

    def save_checkpoint(self, path: str):
        """Save model state"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'vocab_size': self.vocab_size,
                'block_size': self.block_size,
                'n_embd': self.n_embd
            }
        }, path)

    def load_checkpoint(self, path: str, device='cpu'):
        """Load model state with size check"""
        try:
            if not os.path.exists(path):
                return False
                
            checkpoint = torch.load(path, map_location=device)
            if 'config' in checkpoint:
                # Verify sizes match
                if checkpoint['config']['vocab_size'] != self.vocab_size:
                    print("⚠️ Vocab size mismatch - skipping load")
                    return False
                    
            self.load_state_dict(checkpoint['model_state_dict'], strict=False)
            return True
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False
