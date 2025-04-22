from pydantic import BaseModel
from typing import Optional

class RequestPayload(BaseModel):
    prompt: str
    api_key: str
    email: str

class PersonalizationRequest(BaseModel):
    email: str
    api_key: str
    preference_key: str
    preference_value: str
