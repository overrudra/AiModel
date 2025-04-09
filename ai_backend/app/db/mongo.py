from pymongo import MongoClient
import os

client = MongoClient(os.getenv("MONGODB_URI", "mongodb+srv://Cronix:aKFdN9RZn9oawpHx@cronixai.ppyautd.mongodb.net/?retryWrites=true&w=majority&appName=CronixAi"))
db = client["ai_backend"]
history_collection = db["chat_history"]
users_collection = db["users"]

def save_chat(email, prompt, response):
    history_collection.insert_one({
        "email": email,
        "prompt": prompt,
        "response": response
    })

def get_user_history(email):
    return list(history_collection.find({"email": email}))
