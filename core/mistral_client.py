"""Core Mistral AI client with full API support."""
from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx
from mistralai import Mistral

from config.constants import FALLBACK_MODELS
from config.settings import settings
from config.user_settings import get_api_key as load_stored_api_key
from utils.cache_manager import CacheManager
from utils.error_handling import CircuitBreaker, CircuitBreakerOpen, MistralAPIError
from utils.logging_config import get_logger

logger = get_logger(__name__)

class MistralClient:
    """
    Comprehensive Mistral AI API client with all features:
    - Dynamic model fetching
    - Streaming responses
    - JSON mode & JSON schema
    - Vision/multimodal
    - Conversations API
    - Embeddings
    - Fine-tuning
    - OCR
    - Built-in connectors
    - MCP tools
    - Parallel tool calling
    """
    
    def __init__(self, api_key: Optional[str] = None):
        resolved_key = api_key

        if not resolved_key:
            try:
                resolved_key = load_stored_api_key()
            except Exception as exc:  # pragma: no cover - safety log
                logger.warning("Failed to load API key from storage: %s", exc)

        if not resolved_key:
            resolved_key = settings.MISTRAL_API_KEY

        if not resolved_key:
            raise ValueError(
                "Mistral API key not provided. Set MISTRAL_API_KEY environment variable "
                "or store one via POST /api/settings/api-key."
            )

        self.api_key = resolved_key
        
        # Initialize Mistral SDK client
        self.client = Mistral(api_key=self.api_key)
        
        # HTTP client for custom endpoints
        self.http_client = httpx.AsyncClient(
            base_url=settings.MISTRAL_API_BASE,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=settings.TIMEOUT
        )
        
        # Cache manager
        self.cache = CacheManager()
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            threshold=settings.CIRCUIT_BREAKER_THRESHOLD,
            timeout=settings.CIRCUIT_BREAKER_TIMEOUT
        )
    
    async def fetch_available_models(self) -> List[Dict[str, Any]]:
        """
        Fetch all available models from Mistral API /v1/models endpoint
        with 1-hour caching
        """
        logger.info("[fetchAvailableModels] Checking cache...")
        
        # Check cache
        cached = self.cache.get("mistral_models")
        if cached:
            cache_time = cached.get("timestamp")
            if cache_time and (datetime.now() - datetime.fromisoformat(cache_time)).seconds < settings.MODELS_CACHE_TTL:
                logger.info(f"[fetchAvailableModels] Returning cached models: {len(cached['models'])} models")
                return cached["models"]
        
        logger.info("[fetchAvailableModels] Cache expired, fetching from API...")
        
        try:
            # Make request to /v1/models
            response = await self.http_client.get("/models")
            response.raise_for_status()
            
            data = response.json()
            models = data.get("data", [])
            
            logger.info(f"[fetchAvailableModels] Parsed models: {len(models)} models")
            
            # Cache the results
            cache_data = {
                "models": models,
                "timestamp": datetime.now().isoformat(),
                "ttl": settings.MODELS_CACHE_TTL
            }
            self.cache.set("mistral_models", cache_data)
            
            return models
            
        except httpx.HTTPError as error:
            logger.error(f"[fetchAvailableModels] API Error: {error}")
            logger.info("[fetchAvailableModels] Falling back to hardcoded models")
            return list(FALLBACK_MODELS.values())
        except Exception as error:
            logger.error(f"[fetchAvailableModels] Unexpected error: {error}")
            return list(FALLBACK_MODELS.values())
    
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: str = None,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        parallel_tool_calls: bool = True,
        temperature: float = None,
        max_tokens: int = None,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncIterator[Dict[str, Any]]]:
        """
        Enhanced chat completion with all Mistral API features
        """
        if not self.circuit_breaker.can_execute():
            raise CircuitBreakerOpen("Circuit breaker is open, too many failures")
        
        model = model or settings.DEFAULT_MODEL
        temperature = temperature or settings.TEMPERATURE
        max_tokens = max_tokens or settings.MAX_TOKENS
        
        request_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        # Add parallel tool calls only if tools are provided
        if tools:
            request_params["tools"] = tools
            request_params["parallel_tool_calls"] = parallel_tool_calls
        
        # Add response format (JSON mode)
        if response_format:
            request_params["response_format"] = response_format
        
        # Add advanced parameters
        if kwargs.get("presence_penalty"):
            request_params["presence_penalty"] = kwargs["presence_penalty"]
        if kwargs.get("frequency_penalty"):
            request_params["frequency_penalty"] = kwargs["frequency_penalty"]
        if kwargs.get("top_p"):
            request_params["top_p"] = kwargs["top_p"]
        if kwargs.get("seed"):
            request_params["seed"] = kwargs["seed"]
        if kwargs.get("prediction"):
            request_params["prediction"] = kwargs["prediction"]
        
        try:
            if stream:
                return self._stream_completion(request_params)
            else:
                response = self.client.chat.complete(**request_params)
                self.circuit_breaker.record_success()
                return response.model_dump()
                
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise MistralAPIError(f"Chat completion failed: {str(e)}")
    
    async def _stream_completion(self, params: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream chat completion token by token
        """
        try:
            stream = self.client.chat.stream(**params)
            for chunk in stream:
                yield chunk.model_dump()
        except Exception as e:
            raise MistralAPIError(f"Streaming failed: {str(e)}")
    
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()
