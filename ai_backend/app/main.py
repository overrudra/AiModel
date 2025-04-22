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
from app.model.config import MODEL_CONFIG, GENERATION_CONFIG
from app.knowledge.retriever import KnowledgeRetriever
from app.training.trainer import ModelTrainer
from app.utils.text_processor import clean_text, format_response
from app.knowledge.company_info import get_company_response
from app.db.mongo import save_chat, get_user_preferences, save_user_preference
from app.db.schemas import RequestPayload, PersonalizationRequest
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        elif word in vocab:
            tokens.append(vocab[word])
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
            text.append(' ' + token[1:])
        else:
            text.append(token)
    return ''.join(text).strip()

def clean_response(text: str) -> str:
    """Clean up model generated response"""
    # Remove special tokens and normalize
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\x20-\x7E\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Check for garbled/nonsensical text
    if (len(text.split()) < 3 or  # Too short
        len(text) < 10 or  # Too short
        re.search(r'([a-zA-Z])\1{3,}', text) or  # Repeated characters
        text.count('?') > 3 or  # Too many question marks
        len(re.findall(r'[A-Z]{5,}', text)) > 0):  # Long uppercase sequences
        return "I apologize, but I need more context to provide an accurate response."
        
    return text.strip()

def generate_text(prompt: str, max_tokens: int = GENERATION_CONFIG['max_length']) -> str:
    """Generate text with knowledge retrieval and model generation"""
    try:
        # First try knowledge retrieval
        response = knowledge_retriever.get_knowledge(prompt)
        if response and not response.startswith("I apologize"):
            return format_response(response)

        # If no knowledge found, use model generation
        input_ids = encode_text(prompt)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_tokens=max_tokens,
                temperature=GENERATION_CONFIG['temperature']
            )
        response = decode_text(output_ids[0])
        return format_response(response)

    except Exception as e:
        logger.error(f"Generation error: {e}")
        return "I apologize, but I need more context to provide an accurate response."

# Load vocab and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_path = os.path.join(BASE_DIR, "tokenizer", "vocab.json")
logger.info(f"Loading vocabulary from {vocab_path}")

try:
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    id_to_token = {v: k for k, v in vocab.items()}
    VOCAB_SIZE = len(vocab)
    logger.info(f"Loaded vocabulary with {VOCAB_SIZE} tokens")
except Exception as e:
    logger.error(f"Failed to load vocabulary: {e}")
    raise RuntimeError("Could not load vocabulary")

# Initialize model
try:
    model_path = os.path.join(BASE_DIR, "model", "tiny_llm.pth")
    logger.info(f"Loading model from: {model_path}")
    
    # Create model with correct vocab size
    model = TinyTransformer.create_model({**MODEL_CONFIG, "vocab_size": VOCAB_SIZE}, device)
    
    if os.path.exists(model_path):
        if model.load_checkpoint(model_path, device):
            logger.info("✅ Model loaded successfully")
        else:
            logger.warning("⚠️ Failed to load checkpoint, training new model")
            model = TinyTransformer.create_model({**MODEL_CONFIG, "vocab_size": VOCAB_SIZE}, device)
    else:
        logger.warning("⚠️ No model found, creating new one")
        model = TinyTransformer.create_model({**MODEL_CONFIG, "vocab_size": VOCAB_SIZE}, device)
    
    model.eval()
    
except Exception as e:
    logger.error(f"❌ Model error: {str(e)}")
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
async def chat(request: RequestPayload):
    if request.api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        prompt = request.prompt.strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="Empty prompt")

        # Check if this is a preference setting request
        if "my" in prompt.lower() and any(pref in prompt.lower() for pref in ["favorite", "favourite", "prefer", "like"]):
            parts = prompt.lower().split(" is ")
            if len(parts) == 2:
                preference_value = parts[1].strip()
                preference_key = next((word for word in parts[0].split() if word in ["color", "colour", "food", "sport", "hobby"]), None)
                if preference_key:
                    preference_key = "favorite_" + preference_key.replace("colour", "color")
                    save_user_preference(request.email, preference_key, preference_value)
                    return {"response": f"I'll remember that your {preference_key.replace('favorite_', '')} is {preference_value}!"}

        # Get user preferences
        user_prefs = get_user_preferences(request.email)
            
        # Check if query is about user preferences
        if any(word in prompt.lower() for word in ["what", "tell"]) and "my" in prompt.lower() and any(pref in prompt.lower() for pref in ["favorite", "favourite", "prefer", "like"]):
            for key, value in user_prefs.items():
                if key != "_id" and key != "email" and key.replace("favorite_", "") in prompt.lower():
                    return {"response": f"Your {key.replace('favorite_', '')} is {value}"}

        # If no preference handling, proceed with normal response generation
        response = generate_text(prompt)
        cleaned = format_response(response)
        
        if not cleaned or cleaned.startswith("I apologize"):
            return {"response": "I apologize, but I need more context to provide an accurate response."}

        # Save chat history
        save_chat(request.email, prompt, cleaned)
            
        # Only train on good responses
        if len(cleaned) > 10 and not cleaned.startswith("I apologize"):
            knowledge_retriever.learned.add_knowledge(prompt, cleaned, "generated")
            trainer.add_conversation(prompt, cleaned)
            
        return {"response": cleaned}
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/preferences")
async def set_preference(request: PersonalizationRequest):
    if request.api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        save_user_preference(request.email, request.preference_key, request.preference_value)
        return {"message": f"Successfully saved preference {request.preference_key}"}
    except Exception as e:
        logger.error(f"Preference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
