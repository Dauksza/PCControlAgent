"""
Configuration management for Mistral AI Agent Platform
"""
import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Mistral API Configuration
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
    MISTRAL_API_BASE: str = "https://api.mistral.ai/v1"
    
    # Model Configuration
    DEFAULT_MODEL: str = "mistral-large-2407"
    MAX_ITERATIONS: int = 50
    TIMEOUT: int = 30
    
    # Feature Flags
    ENABLE_STREAMING: bool = True
    ENABLE_VISION: bool = True
    ENABLE_CONVERSATIONS_API: bool = True
    ENABLE_PARALLEL_TOOLS: bool = True
    
    # Response Format
    DEFAULT_RESPONSE_FORMAT: str = "text"  # text, json_object, json_schema
    
    # Advanced Parameters
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 4096
    TOP_P: float = 1.0
    PRESENCE_PENALTY: float = 0.0
    FREQUENCY_PENALTY: float = 0.0
    
    # Caching
    MODELS_CACHE_TTL: int = 3600  # 1 hour
    ENABLE_REDIS_CACHE: bool = False
    REDIS_URL: str = "redis://localhost:6379"
    
    # Circuit Breaker
    CIRCUIT_BREAKER_THRESHOLD: int = 5
    CIRCUIT_BREAKER_TIMEOUT: int = 60
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "agent.log"
    
    # Tool Configuration
    ENABLE_WEB_SEARCH: bool = True
    ENABLE_CODE_EXECUTION: bool = True
    ENABLE_IMAGE_GENERATION: bool = True
    ENABLE_DOCUMENT_LIBRARY: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
