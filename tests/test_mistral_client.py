"""
Tests for Mistral client
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from core.mistral_client import MistralClient
from config.constants import FALLBACK_MODELS

@pytest.mark.asyncio
async def test_client_initialization():
    """Test client initialization"""
    client = MistralClient(api_key="test_key")
    assert client.api_key == "test_key"
    assert client.client is not None
    await client.close()

@pytest.mark.asyncio
async def test_fetch_available_models_from_cache():
    """Test fetching models from cache"""
    client = MistralClient(api_key="test_key")
    
    # Mock cache
    client.cache.set("mistral_models", {
        "models": [{"id": "test-model"}],
        "timestamp": "2025-01-01T00:00:00"
    })
    
    models = await client.fetch_available_models()
    assert len(models) > 0
    
    await client.close()

@pytest.mark.asyncio
async def test_fetch_available_models_fallback():
    """Test fallback to hardcoded models"""
    client = MistralClient(api_key="test_key")
    
    # Clear cache
    client.cache.delete("mistral_models")
    
    # Mock HTTP client to raise error
    with patch.object(client.http_client, 'get', side_effect=Exception("API error")):
        models = await client.fetch_available_models()
        
        # Should return fallback models
        assert len(models) == len(FALLBACK_MODELS)
    
    await client.close()

@pytest.mark.asyncio
async def test_chat_completion_basic():
    """Test basic chat completion"""
    client = MistralClient(api_key="test_key")
    
    # Mock the Mistral SDK client
    mock_response = Mock()
    mock_response.model_dump.return_value = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Test response"
            }
        }]
    }
    
    with patch.object(client.client.chat, 'complete', return_value=mock_response):
        response = await client.chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            model="mistral-large-2407"
        )
        
        assert response is not None
        assert "choices" in response
    
    await client.close()

def test_circuit_breaker():
    """Test circuit breaker functionality"""
    client = MistralClient(api_key="test_key")
    
    # Initially closed
    assert client.circuit_breaker.state == "closed"
    assert client.circuit_breaker.can_execute() == True
    
    # Record failures
    for _ in range(5):
        client.circuit_breaker.record_failure()
    
    # Should be open now
    assert client.circuit_breaker.state == "open"
    assert client.circuit_breaker.can_execute() == False
