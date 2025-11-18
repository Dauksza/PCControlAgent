"""REST API routes."""
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from agent.orchestrator import AgentOrchestrator
from core.mistral_client import MistralClient
from core.models import ModelManager
from config.settings import settings
from config.user_settings import (
    delete_api_key as delete_stored_api_key,
    get_api_key as load_stored_api_key,
    save_api_key as store_api_key,
)
from database import get_session
from models.conversation import Conversation, ConversationThread
from utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()

_client: Optional[MistralClient] = None
_model_manager: Optional[ModelManager] = None
_orchestrator: Optional[AgentOrchestrator] = None


def _clear_cached_clients() -> None:
    global _client, _model_manager, _orchestrator
    _client = None
    _model_manager = None
    _orchestrator = None


def _get_mistral_client() -> MistralClient:
    global _client
    if _client is None:
        try:
            _client = MistralClient()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    return _client


def _get_model_manager() -> ModelManager:
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager(_get_mistral_client())
    return _model_manager


def _get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        try:
            _orchestrator = AgentOrchestrator()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    return _orchestrator

# Request models
class TaskRequest(BaseModel):
    task: str
    model: Optional[str] = None
    stream: bool = False
    max_iterations: Optional[int] = None

class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class ConversationCreateRequest(BaseModel):
    title: Optional[str] = Field(default=None, description="Conversation title")
    initial_thread_name: Optional[str] = Field(default=None, description="Optional name for the first thread")


class ThreadCreateRequest(BaseModel):
    conversation_id: str
    name: Optional[str] = Field(default=None, description="Thread display name")


class ThreadRenameRequest(BaseModel):
    name: str = Field(..., description="New thread name")


class ApiKeyRequest(BaseModel):
    api_key: str = Field(..., min_length=10, description="Mistral API key to store securely")


class FrontendMessageRequest(BaseModel):
    conversation_id: str
    thread_id: str
    content: str = Field(..., min_length=1)
    model: Optional[str] = None

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "Mistral AI Agent Platform"
    }


@router.post("/settings/api-key")
async def save_api_key(request: ApiKeyRequest):
    """Persist the Mistral API key in encrypted storage."""
    try:
        await run_in_threadpool(store_api_key, request.api_key.strip())
        _clear_cached_clients()
        return {"status": "saved"}
    except Exception as exc:
        logger.error("Failed to save API key: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to store API key")


@router.get("/settings/api-key")
async def has_api_key():
    """Check whether a Mistral API key has been stored."""
    try:
        api_key = await run_in_threadpool(load_stored_api_key)
        return {"has_key": bool(api_key)}
    except Exception as exc:
        logger.error("Failed to load API key: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to load API key status")


@router.delete("/settings/api-key", status_code=status.HTTP_200_OK)
async def remove_api_key():
    """Delete the stored API key."""
    try:
        removed = await run_in_threadpool(delete_stored_api_key)
        _clear_cached_clients()
        return {"removed": bool(removed)}
    except Exception as exc:
        logger.error("Failed to delete API key: %s", exc)
        raise HTTPException(status_code=500, detail="Unable to delete API key")


@router.get("/frontend/bootstrap")
async def frontend_bootstrap(db: Session = Depends(get_session)):
    """Aggregate initial data required by the React frontend."""
    conversations = (
        db.query(Conversation)
        .order_by(Conversation.updated_at.desc())
        .all()
    )
    conversation_payload = [conversation.to_dict() for conversation in conversations]

    active_thread = None
    for conversation in conversation_payload:
        threads = conversation.get('threads') or []
        if threads:
            active_thread = threads[-1]
            break

    try:
        stored_key = await run_in_threadpool(load_stored_api_key)
    except Exception:
        stored_key = None

    has_api_key = bool(stored_key or settings.MISTRAL_API_KEY)

    models = await _get_model_manager().get_models()
    model_ids = [model.get('id') for model in models if model.get('id')]

    return {
        "conversations": conversation_payload,
        "activeThread": active_thread,
        "hasApiKey": has_api_key,
        "models": model_ids,
    }


@router.post("/frontend/message")
async def frontend_message(request: FrontendMessageRequest):
    """Simple chat endpoint tailored for the new frontend experience."""
    client = _get_mistral_client()
    try:
        response = await client.chat_completion(
            messages=[{"role": "user", "content": request.content}],
            model=request.model,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Frontend message failed: %s", exc)
        raise HTTPException(status_code=500, detail="Chat request failed")

    choice = response.get("choices", [{}])[0]
    message = choice.get("message", {})
    tool_calls = message.get("tool_calls", []) or []

    parsed_tool_calls = []
    for call in tool_calls:
        function = call.get("function") or {}
        arguments = function.get("arguments")
        try:
            parsed_args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            parsed_args = {"raw": arguments}

        parsed_tool_calls.append(
            {
                "id": call.get("id", str(uuid.uuid4())),
                "name": function.get("name", "unknown_tool"),
                "arguments": parsed_args or {},
            }
        )

    payload = {
        "message": {
            "id": str(uuid.uuid4()),
            "role": message.get("role", "assistant"),
            "content": message.get("content", ""),
            "createdAt": datetime.utcnow().isoformat(),
            "toolCalls": parsed_tool_calls,
        }
    }

    return payload


@router.post("/conversations/create")
async def create_conversation(
    request: ConversationCreateRequest,
    db: Session = Depends(get_session)
):
    """Create a new conversation with an optional initial thread."""
    title = (request.title or f"Conversation {datetime.utcnow():%Y-%m-%d %H:%M:%S}").strip()
    initial_thread_name = (request.initial_thread_name or "New Thread").strip()

    conversation = Conversation(title=title)
    db.add(conversation)
    db.flush()

    thread = ConversationThread(
        conversation_id=conversation.id,
        name=initial_thread_name or "New Thread"
    )
    db.add(thread)
    db.commit()
    db.refresh(conversation)
    db.refresh(thread)

    return {"conversation": conversation.to_dict()}


@router.get("/conversations/list")
async def list_conversations(db: Session = Depends(get_session)):
    """Return all conversations ordered by last update."""
    conversations = (
        db.query(Conversation)
        .order_by(Conversation.updated_at.desc())
        .all()
    )
    return {
        "count": len(conversations),
        "conversations": [c.to_dict(include_threads=False) for c in conversations]
    }


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str, db: Session = Depends(get_session)):
    """Fetch a conversation and its threads."""
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id)
        .first()
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"conversation": conversation.to_dict()}


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, db: Session = Depends(get_session)):
    """Delete a conversation and its threads."""
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id)
        .first()
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    db.delete(conversation)
    db.commit()
    return {"status": "deleted", "conversation_id": conversation_id}


@router.post("/threads/create")
async def create_thread(request: ThreadCreateRequest, db: Session = Depends(get_session)):
    """Create a new thread inside an existing conversation."""
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == request.conversation_id)
        .first()
    )
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    thread_name = (request.name or "New Thread").strip()
    thread = ConversationThread(conversation_id=conversation.id, name=thread_name or "New Thread")
    db.add(thread)

    # Touch conversation timestamp
    conversation.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(thread)

    return {"thread": thread.to_dict()}


@router.put("/threads/{thread_id}/rename")
async def rename_thread(thread_id: str, request: ThreadRenameRequest, db: Session = Depends(get_session)):
    """Rename an existing thread."""
    thread = (
        db.query(ConversationThread)
        .filter(ConversationThread.id == thread_id)
        .first()
    )
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    thread.name = request.name.strip() or thread.name
    thread.updated_at = datetime.utcnow()
    if thread.conversation:
        thread.conversation.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(thread)

    return {"thread": thread.to_dict()}

@router.get("/models")
async def list_models():
    """
    Get all available Mistral models from API
    """
    try:
        model_manager = _get_model_manager()
        models = await model_manager.get_models()
        return {
            "models": models,
            "count": len(models),
            "cached": bool(model_manager.models_cache)
        }
    except Exception as e:
        logger.error(f"Failed to fetch models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """
    Get information about a specific model
    """
    try:
        model = await _get_model_manager().get_model_by_id(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        return model
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/execute")
async def execute_task(request: TaskRequest):
    """
    Execute an agent task
    """
    try:
        logger.info(f"Executing task: {request.task[:100]}...")
        orchestrator = _get_orchestrator()

        result = await orchestrator.execute_task(
            task_description=request.task,
            model=request.model,
            max_iterations=request.max_iterations,
            stream=request.stream
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
async def chat_completion(request: ChatRequest):
    """
    Direct chat completion endpoint
    """
    try:
        response = await _get_mistral_client().chat_completion(
            messages=request.messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_execution_history(limit: int = 10):
    """
    Get agent execution history
    """
    try:
        orchestrator = _get_orchestrator()
        history = orchestrator.execution_history[-limit:]
        return {
            "history": history,
            "total": len(orchestrator.execution_history),
            "returned": len(history)
        }
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tools")
async def list_tools():
    """
    List all registered tools
    """
    try:
        orchestrator = _get_orchestrator()
        tools = orchestrator.tool_registry.list_tools()
        return {
            "tools": tools,
            "count": len(tools)
        }
    except Exception as e:
        logger.error(f"Failed to list tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))
