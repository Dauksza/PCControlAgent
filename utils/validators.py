"""
Input validation utilities
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator

class TaskRequest(BaseModel):
    """Validation model for task execution requests"""
    task: str = Field(..., min_length=1, max_length=10000)
    model: Optional[str] = None
    stream: bool = False
    max_iterations: Optional[int] = Field(None, ge=1, le=100)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=32000)

    @validator('task')
    def validate_task(cls, v):
        if not v.strip():
            raise ValueError("Task cannot be empty")
        return v.strip()

class MessageRequest(BaseModel):
    """Validation model for chat messages"""
    role: str = Field(..., pattern="^(user|assistant|system|tool)$")
    content: str = Field(..., min_length=1)
    
class EmbeddingRequest(BaseModel):
    """Validation model for embedding generation"""
    texts: List[str] = Field(..., min_items=1, max_items=100)
    model: str = "mistral-embed"
    
    @validator('texts')
    def validate_texts(cls, v):
        if not all(text.strip() for text in v):
            raise ValueError("All texts must be non-empty")
        return v

class ImageAnalysisRequest(BaseModel):
    """Validation model for image analysis"""
    image_path: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1, max_length=2000)
    model: str = "pixtral-12b"
    resize: bool = True

def validate_api_key(api_key: str) -> bool:
    """
    Validate Mistral API key format
    
    Args:
        api_key: API key to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not api_key:
        return False
    
    # Basic validation - adjust based on actual Mistral key format
    if len(api_key) < 10:
        return False
    
    return True

def validate_model_name(model: str, available_models: List[str]) -> bool:
    """
    Validate if model name is available
    
    Args:
        model: Model name to validate
        available_models: List of available model names
    
    Returns:
        True if valid, False otherwise
    """
    return model in available_models
