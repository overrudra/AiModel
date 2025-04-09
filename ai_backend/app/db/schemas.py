from pydantic import BaseModel

class RequestPayload(BaseModel):
    prompt: str
    api_key: str
    email: str
