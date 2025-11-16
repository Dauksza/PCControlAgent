"""
Constants and model definitions
"""
from typing import Dict, Any, List

# Model Categories
MODEL_CATEGORIES = {
    "enterprise": ["mistral-large-2407", "mistral-large-2411"],
    "balanced": ["mistral-medium-2312"],
    "reasoning": ["magistral-medium-2506"],
    "efficient": ["open-mistral-nemo"],
    "code": ["codestral-2405", "codestral-2501"],
    "vision": ["pixtral-12b", "pixtral-large"],
    "embeddings": ["mistral-embed"]
}

# Hardcoded fallback models (if API fetch fails)
FALLBACK_MODELS = {
    "mistral-large-2407": {
        "id": "mistral-large-2407",
        "name": "Mistral Large 2407",
        "contextWindow": 128000,
        "inputCost": 0.004,
        "outputCost": 0.012,
        "category": "enterprise",
        "description": "Enterprise-grade flagship model",
        "capabilities": ["text", "function_calling", "json_mode"]
    },
    "mistral-medium-2312": {
        "id": "mistral-medium-2312",
        "name": "Mistral Medium 2312",
        "contextWindow": 128000,
        "inputCost": 0.0025,
        "outputCost": 0.0075,
        "category": "balanced",
        "description": "Balanced performance model",
        "capabilities": ["text", "function_calling"]
    },
    "magistral-medium-2506": {
        "id": "magistral-medium-2506",
        "name": "Magistral Medium 2506",
        "contextWindow": 128000,
        "inputCost": 0.002,
        "outputCost": 0.006,
        "category": "reasoning",
        "description": "Reasoning-specialized model",
        "capabilities": ["text", "reasoning", "function_calling"]
    },
    "open-mistral-nemo": {
        "id": "open-mistral-nemo",
        "name": "Mistral Nemo 12B",
        "contextWindow": 128000,
        "inputCost": 0.0003,
        "outputCost": 0.0003,
        "category": "efficient",
        "description": "Efficient 12B model",
        "capabilities": ["text", "function_calling"]
    },
    "codestral-2405": {
        "id": "codestral-2405",
        "name": "Codestral 2405",
        "contextWindow": 32000,
        "inputCost": 0.001,
        "outputCost": 0.003,
        "category": "code",
        "description": "Code specialist model",
        "capabilities": ["code", "fill_in_middle"]
    },
    "pixtral-12b": {
        "id": "pixtral-12b",
        "name": "Pixtral 12B",
        "contextWindow": 128000,
        "inputCost": 0.00015,
        "outputCost": 0.00015,
        "category": "vision",
        "description": "Multimodal vision model",
        "capabilities": ["vision", "text", "function_calling"]
    },
    "mistral-embed": {
        "id": "mistral-embed",
        "name": "Mistral Embed",
        "contextWindow": 8000,
        "inputCost": 0.0001,
        "outputCost": 0.0,
        "category": "embeddings",
        "description": "Embedding model for RAG",
        "capabilities": ["embeddings"]
    }
}

# Mistral Built-in Connectors
MISTRAL_CONNECTORS = [
    "web_search",
    "code_interpreter",
    "image_generation",
    "document_library"
]

# API Endpoints
ENDPOINTS = {
    "models": "/models",
    "chat": "/chat/completions",
    "conversations": "/conversations",
    "embeddings": "/embeddings",
    "fine_tuning": "/fine_tuning/jobs",
    "ocr": "/ocr"
}
