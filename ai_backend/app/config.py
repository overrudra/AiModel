OFFLINE_MODE = False  # Changed to False to allow online search when needed

SYSTEM_CONFIG = {
    "use_internet": True,  # Always try internet if local knowledge fails
    "use_local_knowledge": True,
    "cache_responses": True,
    "max_cache_size": 10000,
    "fallback_to_online": True  # New setting
}