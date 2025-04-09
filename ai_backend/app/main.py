from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import os
import json
import re
import numpy as np
from app.model.llm import TinyTransformer
from app.model.image_gen import is_image_prompt, generate_image
from app.model.code_gen import is_code_prompt, generate_code
from app.model.config import MODEL_CONFIG
from app.knowledge.retriever import KnowledgeRetriever
from app.training.trainer import ModelTrainer

# FastAPI setup
app = FastAPI(
    title="Cronix AI Chat API",
    description="AI Chat API with knowledge retrieval and learning capabilities",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    prompt: str
    api_key: str

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VALID_API_KEYS = {"test_key", "dev_key"}
DEFAULT_RESPONSE = "I apologize, but I'm having trouble generating a response right now."

# Helper functions
def encode_text(text: str) -> torch.Tensor:
    """Encode text to tensor"""
    tokens = [vocab['<s>']]  # Start token
    words = text.split()
    for word in words:
        if f'Ġ{word}' in vocab:
            tokens.append(vocab[f'Ġ{word}'])
        else:
            if tokens[-1] != vocab['<s>']:
                tokens.append(vocab['Ġ'])
            for char in word:
                tokens.append(vocab.get(char, vocab['<unk>']))
    tokens.append(vocab['</s>'])  # End token
    return torch.tensor([tokens], device=device)

def decode_text(token_ids: torch.Tensor) -> str:
    """Decode tensor to text"""
    text = []
    for tid in token_ids:
        token = id_to_token.get(tid.item(), '<unk>')
        if token.startswith('Ġ'):
            text.append(token[1:])
        else:
            text.append(token)
    return ''.join(text)

def clean_response(text: str) -> str:
    """Clean up model generated response"""
    # Remove special tokens
    text = re.sub(r'<[^>]+>', '', text)  # Remove all XML-like tags
    text = re.sub(r'[^\x20-\x7E\s]', '', text)  # Remove non-ASCII characters
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add spaces between camelCase
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)  # Add space between letters and numbers
    
    # Fix common issues
    text = re.sub(r'W+', '', text)  # Remove repeated W's
    text = re.sub(r'(.)\1{2,}', r'\1', text)  # Remove character repetitions
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    # Clean up the result
    text = text.strip()
    
    # Return default response if text is too messy
    if (len(text.split()) < 3 or 
        not re.match(r'^[a-zA-Z0-9\s.,!?-]+$', text) or
        len(text) < 10):
        return "I apologize, I don't have enough information about that topic."
        
    return text

def generate_text(prompt: str, max_tokens: int = 100) -> str:
    """Generate text with knowledge retrieval"""
    try:
        # First try online search
        response = knowledge_retriever.search_wikipedia(prompt)
        if response:
            return response

        # Then try web search
        web_results = knowledge_retriever.search_web(prompt)
        if web_results:
            return web_results[0]

        # Finally fallback to model generation
        input_ids = encode_text(prompt)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_tokens=max_tokens,
                temperature=0.7  # Lower temperature for more focused responses
            )
        response = decode_text(output_ids[0])
        response = clean_response(response)
        
        return response

    except Exception as e:
        print(f"Generation error: {e}")
        return DEFAULT_RESPONSE

def quick_train(model, responses):
    """Quick pre-training"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    for _ in range(20):
        for response in responses:
            try:
                input_ids = encode_text(response)
                if input_ids.size(1) > model.block_size:
                    input_ids = input_ids[:, :model.block_size]
                
                logits = model(input_ids)
                targets = input_ids.clone()
                
                loss = criterion(
                    logits.view(-1, model.vocab_size),
                    targets.view(-1)
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            except Exception as e:
                print(f"Training error: {e}")
                continue
    
    model.eval()
    return model

# Load vocab and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(os.path.join(BASE_DIR, "tokenizer", "vocab.json"), 'r') as f:
    vocab = json.load(f)
id_to_token = {v: k for k, v in vocab.items()}
VOCAB_SIZE = len(vocab)

# Common words
COMMON_WORDS = {
    'Hello': 300, 'Hi': 301, 'I': 302, 'am': 303, 'the': 304,
    'is': 305, 'are': 306, 'what': 307, 'how': 308, 'can': 309,
    'you': 310, 'help': 311, 'hello': 312, 'world': 313,
    'today': 324
}

# Training responses
COMMON_RESPONSES = [
    "Hello! How may I help you today?",
    "Hi there! What can I do for you?",
    "Hello! Nice to meet you.",
]

# Initialize model
try:
    model_path = os.path.join(BASE_DIR, "model", "tiny_llm.pth")
    print(f"Loading model from: {model_path}")
    
    config = {
        "vocab_size": VOCAB_SIZE,
        "block_size": 64,
        "n_embd": 256,
        "n_layer": 6,
        "n_head": 8,
        "dropout": 0.1
    }
    
    model = TinyTransformer.create_model(config, device)
    
    if os.path.exists(model_path):
        if model.load_checkpoint(model_path, device):
            print("✅ Loaded checkpoint")
        else:
            print("⚠️ Training new model")
            model = quick_train(model, COMMON_RESPONSES)
            model.save_checkpoint(model_path)
    else:
        print("Training new model...")
        model = quick_train(model, COMMON_RESPONSES)
        model.save_checkpoint(model_path)
    
    model.eval()
    
except Exception as e:
    print(f"❌ Model error: {str(e)}")
    raise RuntimeError(f"Failed to initialize model: {str(e)}")

# Initialize services
knowledge_retriever = KnowledgeRetriever()
trainer = ModelTrainer(
    model=model,
    vocab_size=VOCAB_SIZE,
    device=device,
    max_steps=MODEL_CONFIG['max_steps'],
    vocab=vocab,
    encode_fn=encode_text
)

# API endpoints
@app.get("/")
async def root():
    return {"status": "ok", "message": "API is running"}

@app.get("/api-key")
async def get_api_key():
    return {"api_key": "test_key"}

@app.post("/chat")
async def chat(request: ChatRequest):
    if request.api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        prompt = request.prompt.strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="Empty prompt")
            
        response = ""
        if is_image_prompt(prompt):
            response = generate_image(prompt)
        elif is_code_prompt(prompt):
            response = generate_code(prompt)
        else:
            response = generate_text(prompt)
            
            # Only train on good responses
            if len(response) > 10 and not response.startswith("I apologize"):
                knowledge_retriever.learned.add_knowledge(prompt, response, "generated")
                trainer.add_conversation(prompt, response)
            
        return {"response": response}
        
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
