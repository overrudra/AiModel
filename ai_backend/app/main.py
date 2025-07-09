from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import os
import json
import re
import numpy as np
from .model.llm import TinyTransformer
from .model.image_gen import is_image_prompt, generate_image
from .model.code_gen import is_code_prompt, generate_code
from .model.config import MODEL_CONFIG, GENERATION_CONFIG
from .knowledge.retriever import KnowledgeRetriever
from .training.trainer import ModelTrainer
from .utils.text_processor import clean_text, format_response
from .knowledge.company_info import get_company_response
from .db.mongo import save_chat, get_user_preferences, save_user_preference
from .db.schemas import RequestPayload, PersonalizationRequest
import logging
from transformers import pipeline

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

def clean_response(text: str, emotion: str = None) -> str:
    """Clean up model generated response, always filter for repeated/garbled text."""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\\x20-\x7E\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    # Always filter for repeated/garbled text
    if (
        re.search(r'([a-zA-Z])\1{3,}', text) or  # Repeated characters
        text.count('?') > 3 or
        len(re.findall(r'[A-Z]{5,}', text)) > 0 or
        re.search(r'(prom){3,}', text) or  # Repeated nonsense syllables
        re.search(r'(im){3,}', text) or  # Repeated nonsense syllables
        re.search(r'(\b\w{2,4}\b)(?:[\s,.:;!?-]*\1){2,}', text, re.IGNORECASE)  # Repeated 2-4 letter syllables
    ):
        return "I apologize, but I need more context to provide an accurate response."
    # For emotional prompts, allow short but not nonsense responses
    if emotion in ["happy", "love", "surprised", "sad", "angry", "fear"]:
        if len(text.strip()) < 5:
            return "I apologize, but I need more context to provide an accurate response."
        return text.strip()
    # For other cases, keep stricter checks
    if (len(text.split()) < 3 or
        len(text) < 10):
        return "I apologize, but I need more context to provide an accurate response."
    return text.strip()

# Fallback templates for all emotions
EMOTION_FALLBACKS = {
    "happy": "Congratulations! That's fantastic news!",
    "sad": "I'm sorry to hear that. Remember, every setback is a setup for a comeback. You did your best!",
    "angry": "I understand your frustration. Take a deep breath—things will get better.",
    "surprised": "Wow, that sounds surprising! Tell me more!",
    "fear": "It's okay to feel scared sometimes. You're not alone, and things will be alright.",
    "love": "That's wonderful! Cherish those feelings and moments.",
    "neutral": "Thank you for sharing. If you'd like to talk more, I'm here to listen."
}

def is_personal_or_emotional_prompt(prompt: str) -> bool:
    """Detect if the prompt is personal/emotional and should bypass knowledge retriever."""
    personal_keywords = [
        "i feel", "i am", "i'm", "my ", "i won", "i lost", "i passed", "i failed", "i got", "i have", "i did", "i was", "i will", "i want", "i need", "i hope", "i wish", "i love", "i hate", "i'm happy", "i'm sad", "i'm angry", "i'm excited", "i'm scared", "i'm nervous", "i'm proud", "i'm disappointed", "i'm surprised", "i'm grateful", "i'm thankful", "i'm sorry", "i'm worried", "i'm upset", "i'm thrilled", "i'm delighted", "i'm heartbroken", "i'm overjoyed", "i'm devastated", "i'm anxious", "i'm relieved", "i'm frustrated", "i'm confused", "i'm hopeful", "i'm motivated", "i'm inspired", "i'm discouraged", "i'm determined", "i'm confident", "i'm insecure", "i'm jealous", "i'm envious", "i'm embarrassed", "i'm ashamed", "i'm guilty", "i'm lonely", "i'm bored", "i'm tired", "i'm energetic", "i'm enthusiastic", "i'm optimistic", "i'm pessimistic", "i'm curious", "i'm interested", "i'm indifferent", "i'm apathetic", "i'm passionate", "i'm ambitious", "i'm content", "i'm satisfied", "i'm dissatisfied", "i'm fulfilled", "i'm unfulfilled", "i'm excited", "i'm nervous", "i'm scared", "i'm afraid", "i'm terrified", "i'm anxious", "i'm worried", "i'm concerned", "i'm stressed", "i'm overwhelmed", "i'm relaxed", "i'm calm", "i'm peaceful", "i'm tranquil", "i'm serene", "i'm happy", "i'm sad", "i'm angry", "i'm mad", "i'm furious", "i'm upset", "i'm disappointed", "i'm proud", "i'm grateful", "i'm thankful", "i'm sorry", "i'm heartbroken", "i'm overjoyed", "i'm devastated", "i'm relieved", "i'm frustrated", "i'm confused", "i'm hopeful", "i'm motivated", "i'm inspired", "i'm discouraged", "i'm determined", "i'm confident", "i'm insecure", "i'm jealous", "i'm envious", "i'm embarrassed", "i'm ashamed", "i'm guilty", "i'm lonely", "i'm bored", "i'm tired", "i'm energetic", "i'm enthusiastic", "i'm optimistic", "i'm pessimistic", "i'm curious", "i'm interested", "i'm indifferent", "i'm apathetic", "i'm passionate", "i'm ambitious", "i'm content", "i'm satisfied", "i'm dissatisfied", "i'm fulfilled", "i'm unfulfilled"
    ]
    prompt_lower = prompt.lower()
    return any(kw in prompt_lower for kw in personal_keywords)

def generate_text(prompt: str, emotion: str = None, max_tokens: int = GENERATION_CONFIG['max_length']) -> str:
    try:
        if emotion and emotion != 'neutral':
            emotion_instruction = {
                'happy': 'Congratulate the user warmly: ',
                'sad': 'Respond with empathy and support: ',
                'angry': 'Respond calmly and help diffuse anger: ',
                'surprised': 'Respond with excitement or curiosity: ',
                'fear': 'Respond with reassurance and calm: ',
                'love': 'Respond warmly and appreciatively: '
            }.get(emotion, '')
            prompt = emotion_instruction + prompt
        if is_personal_or_emotional_prompt(prompt):
            input_ids = encode_text(prompt)
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_tokens=max_tokens,
                    temperature=GENERATION_CONFIG['temperature']
                )
            response = decode_text(output_ids[0])
            cleaned = clean_response(response, emotion)
            # Fallback for all emotions
            if (not cleaned or cleaned.startswith("I apologize")):
                return EMOTION_FALLBACKS.get(emotion, EMOTION_FALLBACKS["neutral"])
            return cleaned
        response = knowledge_retriever.get_knowledge(prompt)
        if response and not response.startswith("I apologize"):
            return format_response(response)
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

# Load emotion/sentiment analysis pipeline
try:
    emotion_classifier = pipeline("sentiment-analysis")
except Exception as e:
    logger.error(f"Failed to load emotion classifier: {e}")
    emotion_classifier = None

def detect_emotion(text: str) -> str:
    """Detect emotion using ML model, fallback to keywords/sentiment if needed."""
    if emotion_classifier:
        try:
            result = emotion_classifier(text)
            label = result[0]['label'].lower()
            # Map common sentiment labels to emotions
            if 'positive' in label:
                return 'happy'
            elif 'negative' in label:
                return 'sad'
            elif 'neutral' in label:
                return 'neutral'
            else:
                return label
        except Exception as e:
            logger.warning(f"Emotion classifier error: {e}")
    # Fallback: keyword/sentiment logic
    emotions = {
        'happy': ['happy', 'glad', 'joy', 'excited', 'delighted', 'pleased', 'wonderful', 'great', 'awesome', 'fantastic', 'win', 'won', 'passed', 'success'],
        'sad': ['sad', 'unhappy', 'down', 'depressed', 'upset', 'cry', 'miserable', 'heartbroken', 'fail', 'failed', 'lost', 'lose'],
        'angry': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'rage', 'hate'],
        'surprised': ['surprised', 'amazed', 'astonished', 'shocked', 'wow'],
        'fear': ['afraid', 'scared', 'fear', 'terrified', 'nervous', 'anxious'],
        'love': ['love', 'like', 'adore', 'fond', 'cherish'],
        'neutral': []
    }
    text_lower = text.lower()
    for emotion, keywords in emotions.items():
        for word in keywords:
            if word in text_lower:
                return emotion
    # Fallback: simple sentiment analysis
    positive_words = ['good', 'great', 'awesome', 'fantastic', 'love', 'enjoy', 'pleased', 'happy', 'wonderful']
    negative_words = ['bad', 'sad', 'hate', 'angry', 'upset', 'terrible', 'awful', 'depressed', 'cry']
    pos = sum(word in text_lower for word in positive_words)
    neg = sum(word in text_lower for word in negative_words)
    if pos > neg:
        return 'happy'
    elif neg > pos:
        return 'sad'
    elif '?' in text_lower:
        return 'surprised'
    else:
        return 'neutral'

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
        emotion = detect_emotion(prompt)
        # Check if this is a preference setting request
        if "my" in prompt.lower() and any(pref in prompt.lower() for pref in ["favorite", "favourite", "prefer", "like"]):
            parts = prompt.lower().split(" is ")
            if len(parts) == 2:
                preference_value = parts[1].strip()
                preference_key = next((word for word in parts[0].split() if word in ["color", "colour", "food", "sport", "hobby"]), None)
                if preference_key:
                    preference_key = "favorite_" + preference_key.replace("colour", "color")
                    save_user_preference(request.email, preference_key, preference_value)
                    return {"response": f"I'll remember that your {preference_key.replace('favorite_', '')} is {preference_value}!", "emotion": emotion}
        user_prefs = get_user_preferences(request.email)
        if any(word in prompt.lower() for word in ["what", "tell"]) and "my" in prompt.lower() and any(pref in prompt.lower() for pref in ["favorite", "favourite", "prefer", "like"]):
            for key, value in user_prefs.items():
                if key != "_id" and key != "email" and key.replace("favorite_", "") in prompt.lower():
                    return {"response": f"Your {key.replace('favorite_', '')} is {value}", "emotion": emotion}
        # If personal/emotional prompt, use generate_text result directly (already cleaned/fallback)
        if is_personal_or_emotional_prompt(prompt):
            cleaned = generate_text(prompt, emotion, max_tokens=GENERATION_CONFIG['max_length'])
        else:
            response = generate_text(prompt, emotion)
            cleaned = format_response(response)
        if not cleaned or cleaned.startswith("I apologize"):
            return {"response": "I apologize, but I need more context to provide an accurate response.", "emotion": emotion}
        save_chat(request.email, prompt, cleaned)
        if len(cleaned) > 10 and not cleaned.startswith("I apologize"):
            knowledge_retriever.learned.add_knowledge(prompt, cleaned, "generated")
            trainer.add_conversation(prompt, cleaned)
        return {"response": cleaned, "emotion": emotion}
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
