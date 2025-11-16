"""
Streaming response handler
"""
from typing import AsyncIterator, Dict, Any
from utils.logging_config import get_logger

logger = get_logger(__name__)

class StreamingHandler:
    """
    Handle streaming responses from Mistral API
    """
    
    def __init__(self):
        self.accumulated_content = ""
        self.accumulated_tool_calls = []
    
    async def process_stream(
        self,
        stream: AsyncIterator[Dict[str, Any]]
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process streaming chunks and yield formatted updates
        
        Args:
            stream: Async iterator of streaming chunks
        
        Yields:
            Processed chunk dictionaries
        """
        async for chunk in stream:
            # Extract delta from chunk
            choices = chunk.get("choices", [])
            if not choices:
                continue
            
            delta = choices[0].get("delta", {})
            
            # Handle content delta
            if "content" in delta and delta["content"]:
                content = delta["content"]
                self.accumulated_content += content
                
                yield {
                    "type": "content",
                    "content": content,
                    "accumulated": self.accumulated_content
                }
            
            # Handle tool calls delta
            if "tool_calls" in delta:
                tool_calls = delta["tool_calls"]
                self.accumulated_tool_calls.extend(tool_calls)
                
                yield {
                    "type": "tool_calls",
                    "tool_calls": tool_calls,
                    "accumulated": self.accumulated_tool_calls
                }
            
            # Handle finish reason
            finish_reason = choices[0].get("finish_reason")
            if finish_reason:
                yield {
                    "type": "finish",
                    "finish_reason": finish_reason,
                    "final_content": self.accumulated_content,
                    "final_tool_calls": self.accumulated_tool_calls
                }
    
    def reset(self):
        """Reset accumulated state"""
        self.accumulated_content = ""
        self.accumulated_tool_calls = []
    
    def get_accumulated_content(self) -> str:
        """Get accumulated content"""
        return self.accumulated_content
    
    def get_accumulated_tool_calls(self) -> list:
        """Get accumulated tool calls"""
        return self.accumulated_tool_calls
