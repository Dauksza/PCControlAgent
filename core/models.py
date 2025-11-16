"""
Dynamic model fetching and management
"""
from typing import List, Dict, Any, Optional
from core.mistral_client import MistralClient
from config.constants import MODEL_CATEGORIES, FALLBACK_MODELS
from utils.logging_config import get_logger

logger = get_logger(__name__)

class ModelManager:
    """
    Manage Mistral models with dynamic fetching and categorization
    """
    
    def __init__(self, client: MistralClient):
        self.client = client
        self.models_cache: List[Dict[str, Any]] = []
    
    async def get_models(self, refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get all available models
        
        Args:
            refresh: Force refresh from API
        
        Returns:
            List of model dictionaries
        """
        if not self.models_cache or refresh:
            self.models_cache = await self.client.fetch_available_models()
        
        return self.models_cache
    
    async def get_model_by_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get specific model by ID
        
        Args:
            model_id: Model identifier
        
        Returns:
            Model dictionary or None
        """
        models = await self.get_models()
        for model in models:
            if model.get("id") == model_id:
                return model
        
        # Fallback to hardcoded models
        return FALLBACK_MODELS.get(model_id)
    
    async def get_models_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get models filtered by category
        
        Args:
            category: Model category (enterprise, balanced, code, vision, etc.)
        
        Returns:
            List of models in that category
        """
        if category not in MODEL_CATEGORIES:
            logger.warning(f"Unknown category: {category}")
            return []
        
        model_ids = MODEL_CATEGORIES[category]
        models = await self.get_models()
        
        return [m for m in models if m.get("id") in model_ids]
    
    async def get_vision_models(self) -> List[Dict[str, Any]]:
        """Get all vision-capable models"""
        return await self.get_models_by_category("vision")
    
    async def get_code_models(self) -> List[Dict[str, Any]]:
        """Get all code-specialized models"""
        return await self.get_models_by_category("code")
    
    async def get_embedding_models(self) -> List[Dict[str, Any]]:
        """Get all embedding models"""
        return await self.get_models_by_category("embeddings")
    
    def get_model_info(self, model_id: str) -> str:
        """
        Get formatted model information
        
        Args:
            model_id: Model identifier
        
        Returns:
            Formatted model information string
        """
        model = FALLBACK_MODELS.get(model_id)
        if not model:
            return f"Model {model_id} not found"
        
        info = f"""
Model: {model['name']}
ID: {model['id']}
Category: {model['category']}
Context Window: {model['contextWindow']} tokens
Input Cost: ${model['inputCost']}/1K tokens
Output Cost: ${model['outputCost']}/1K tokens
Description: {model['description']}
Capabilities: {', '.join(model['capabilities'])}
"""
        return info.strip()
