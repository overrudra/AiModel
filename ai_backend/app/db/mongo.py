from pymongo import MongoClient
import os

client = MongoClient(os.getenv("MONGODB_URI", "mongodb+srv://Cronix:aKFdN9RZn9oawpHx@cronixai.ppyautd.mongodb.net/?retryWrites=true&w=majority&appName=CronixAi"))
db = client["ai_backend"]
history_collection = db["chat_history"]
users_collection = db["users"]
preferences_collection = db["user_preferences"]

def save_chat(email, prompt, response):
    history_collection.insert_one({
        "email": email,
        "prompt": prompt,
        "response": response
    })

def get_user_history(email):
    return list(history_collection.find({"email": email}))

def save_user_preference(email: str, key: str, value: str):
    """Save or update a user preference"""
    preferences_collection.update_one(
        {"email": email},
        {"$set": {key: value}},
        upsert=True
    )

def get_user_preferences(email: str) -> dict:
    """Get all preferences for a user"""
    prefs = preferences_collection.find_one({"email": email})
    return prefs if prefs else {}
