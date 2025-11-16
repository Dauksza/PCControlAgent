"""
Integration tests for streaming functionality
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from core.streaming import StreamingHandler

@pytest.mark.asyncio
@pytest.mark.integration
async def test_streaming_handler():
    """Test streaming handler"""
    handler = StreamingHandler()
    
    # Mock streaming chunks
    async def mock_stream():
        chunks = [
            {
                "choices": [{
                    "delta": {"content": "Hello "},
                    "finish_reason": None
                }]
            },
            {
                "choices": [{
                    "delta": {"content": "World!"},
                    "finish_reason": None
                }]
            },
            {
                "choices": [{
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
        ]
        for chunk in chunks:
            yield chunk
    
    # Process stream
    results = []
    async for result in handler.process_stream(mock_stream()):
        results.append(result)
    
    # Verify results
    assert len(results) > 0
    assert handler.get_accumulated_content() == "Hello World!"

@pytest.mark.asyncio
@pytest.mark.integration
async def test_streaming_with_tool_calls():
    """Test streaming with tool calls"""
    handler = StreamingHandler()
    
    async def mock_stream():
        chunks = [
            {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "id": "call_1",
                            "function": {
                                "name": "test_tool",
                                "arguments": "{}"
                            }
                        }]
                    },
                    "finish_reason": None
                }]
            },
            {
                "choices": [{
                    "delta": {},
                    "finish_reason": "tool_calls"
                }]
            }
        ]
        for chunk in chunks:
            yield chunk
    
    results = []
    async for result in handler.process_stream(mock_stream()):
        results.append(result)
    
    assert len(handler.get_accumulated_tool_calls()) > 0
