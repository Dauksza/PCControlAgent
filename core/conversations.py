"""
Mistral Conversations API implementation with persistent memory
"""
import json
from typing import Dict, List, Optional, Any
import httpx
from config.settings import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

class ConversationsManager:
    """
    Manage conversations with persistent memory and branching
    using Mistral's /v1/conversations endpoint
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = settings.MISTRAL_API_BASE
        self.http_client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=30
        )
    
    async def create_conversation(
        self,
        agent_id: str,
        initial_messages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new conversation
        POST /v1/conversations
        """
        payload = {
            "agent_id": agent_id,
            "inputs": initial_messages,
            "metadata": metadata or {}
        }
        
        try:
            response = await self.http_client.post("/conversations", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to create conversation: {e}")
            raise
    
    async def append_to_conversation(
        self,
        conversation_id: str,
        messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Continue an existing conversation
        POST /v1/conversations/{id}/append
        """
        payload = {"inputs": messages}
        
        try:
            response = await self.http_client.post(
                f"/conversations/{conversation_id}/append",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to append to conversation: {e}")
            raise
    
    async def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Retrieve conversation history
        GET /v1/conversations/{id}
        """
        try:
            response = await self.http_client.get(f"/conversations/{conversation_id}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to get conversation: {e}")
            raise
    
    async def branch_conversation(
        self,
        conversation_id: str,
        from_message_index: int,
        new_messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a new conversation branch from a specific point
        """
        # Get original conversation
        original = await self.get_conversation(conversation_id)
        
        # Extract messages up to branch point
        messages_before_branch = original["messages"][:from_message_index]
        
        # Create new conversation with branched path
        return await self.create_conversation(
            agent_id=original["agent_id"],
            initial_messages=messages_before_branch + new_messages,
            metadata={"branched_from": conversation_id, "branch_point": from_message_index}
        )
    
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()
