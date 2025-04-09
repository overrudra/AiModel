import torch
import torch.nn as nn
import os
import json
from datetime import datetime
from typing import List, Dict
import logging
from ..model.config import MODEL_CONFIG  # Add this import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model, vocab_size, device, max_steps=100000, vocab=None, encode_fn=None):
        self.model = model
        self.vocab_size = vocab_size
        self.device = device
        self.vocab = vocab  # Add vocab dictionary
        self.encode_fn = encode_fn  # Add external encode function
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        self.training_data = []
        self.history_file = "training_history.json"
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=max_steps
        )
        
        if not self.vocab or not self.encode_fn:
            logger.warning("Vocab or encode_fn not provided - training will be limited")
        
    def add_conversation(self, prompt: str, response: str):
        """Add conversation with validation"""
        if not prompt or not response:
            logger.warning("Empty prompt or response, skipping")
            return
            
        try:
            # Validate that we can encode both prompt and response
            if self._encode_text(prompt) is None or self._encode_text(response) is None:
                logger.warning("Failed to encode prompt or response, skipping")
                return
                
            self.training_data.append({
                'prompt': prompt,
                'response': response,
                'timestamp': datetime.now().isoformat()
            })
            self._save_history()
            self._train_on_recent()
            
        except Exception as e:
            logger.error(f"Error adding conversation: {str(e)}")
    
    def _save_history(self):
        """Save training history"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.training_data[-1000:], f)  # Keep last 1000 conversations
        except Exception as e:
            logger.error(f"Failed to save training history: {e}")
    
    def _load_history(self):
        """Load training history"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    self.training_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load training history: {e}")
    
    def _train_on_recent(self, num_epochs: int = 3):
        """Fixed training with proper index handling"""
        if not self.training_data:
            return
            
        self.model.train()
        batch_size = min(len(self.training_data), MODEL_CONFIG['batch_size'])
        recent_data = self.training_data[-batch_size*4:]
        
        try:
            for epoch in range(num_epochs):
                total_loss = 0
                valid_batches = 0
                
                for i in range(0, len(recent_data), batch_size):
                    batch = recent_data[i:i + batch_size]
                    try:
                        # Process single sequences
                        for item in batch:
                            text = f"{item['prompt']} {item['response']}"
                            input_ids = self.encode_fn(text)
                            
                            if input_ids is None:
                                continue
                                
                            # Ensure sequence fits in model context
                            if input_ids.size(1) > self.model.block_size:
                                input_ids = input_ids[:, :self.model.block_size]
                            
                            # Need at least 2 tokens for training
                            if input_ids.size(1) < 2:
                                continue
                            
                            try:
                                # Forward pass with safe indexing
                                logits = self.model(input_ids)
                                
                                # Prepare targets (shift right by 1)
                                targets = input_ids.clone()
                                if logits.size(1) > 1:  # Need at least 2 tokens
                                    logits = logits[:, :-1, :]  # Remove last prediction
                                    targets = targets[:, 1:]    # Remove first token
                                    
                                    # Calculate loss
                                    loss = self.criterion(
                                        logits.reshape(-1, self.vocab_size),
                                        targets.reshape(-1)
                                    )
                                    
                                    # Backward pass
                                    loss.backward()
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                                    self.optimizer.step()
                                    self.optimizer.zero_grad()
                                    
                                    total_loss += loss.item()
                                    valid_batches += 1
                                    
                            except RuntimeError as e:
                                logger.warning(f"Runtime error in training: {e}")
                                continue
                                
                    except Exception as e:
                        logger.error(f"Batch error: {e}")
                        continue
                
                if valid_batches > 0:
                    avg_loss = total_loss / valid_batches
                    logger.info(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
                else:
                    logger.warning("No valid samples in epoch")
                    
        except Exception as e:
            logger.error(f"Training error: {e}")
        finally:
            self.model.eval()

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text with dimension checks"""
        try:
            if self.encode_fn is None:
                logger.error("No encode function provided")
                return None
                
            encoded = self.encode_fn(text)
            if not isinstance(encoded, torch.Tensor):
                logger.error(f"Encoder returned {type(encoded)}, expected torch.Tensor")
                return None
                
            if encoded.dim() == 3:
                encoded = encoded.squeeze(0)  # Remove batch dimension if present
            elif encoded.dim() == 1:
                encoded = encoded.unsqueeze(0)  # Add batch dimension if missing
                
            return encoded
            
        except Exception as e:
            logger.error(f"Encoding error: {str(e)}")
            return None
