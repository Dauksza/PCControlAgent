"""Enhanced Mistral AI client with full feature support"""
from mistralai import Mistral
from typing import List, Dict, Any, Optional, AsyncIterator

class EnhancedMistralClient:
    def __init__(self):
        self.client = None
    
    def init_client(self, api_key: str):
        self.client = Mistral(api_key=api_key)
        return self.client
    
    async def stream_chat(self, model: str, messages: List[Dict], api_key: str, **kwargs):
        client = self.init_client(api_key)
        for chunk in client.chat.stream(model=model, messages=messages, **kwargs):
            if chunk.data.choices and chunk.data.choices[0].delta.content:
                yield chunk.data.choices[0].delta.content
    
    def chat_with_tools(self, model: str, messages: List[Dict], tools: List[Dict], api_key: str, **kwargs):
        client = self.init_client(api_key)
        return client.chat.complete(model=model, messages=messages, tools=tools, **kwargs)
    
    def chat_with_web_search(self, model: str, messages: List[Dict], api_key: str, **kwargs):
        tools = [{"type": "web_search", "web_search": {}}]
        return self.chat_with_tools(model, messages, tools, api_key, **kwargs)
