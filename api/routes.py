"""
REST API routes
"""
from fastapi import APIRouter, HTTPException
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from core.mistral_client import MistralClient
from agent.orchestrator import AgentOrchestrator
from core.models import ModelManager
from config.settings import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Initialize clients
client = MistralClient(api_key=settings.MISTRAL_API_KEY)
orchestrator = AgentOrchestrator(api_key=settings.MISTRAL_API_KEY)
model_manager = ModelManager(client)

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

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "Mistral AI Agent Platform"
    }

@router.get("/models")
async def list_models():
    """
    Get all available Mistral models from API
    """
    try:
        models = await client.fetch_available_models()
        return {
            "models": models,
            "count": len(models),
            "cached": True
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
        model = await model_manager.get_model_by_id(model_id)
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
        response = await client.chat_completion(
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
        tools = orchestrator.tool_registry.list_tools()
        return {
            "tools": tools,
            "count": len(tools)
        }
    except Exception as e:
        logger.error(f"Failed to list tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))
