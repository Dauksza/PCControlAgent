"""
WebSocket endpoints for streaming
"""
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Any
import json
from agent.orchestrator import AgentOrchestrator
from config.settings import settings
from utils.logging_config import get_logger

logger = get_logger(__name__)

class WebSocketManager:
    """
    Manage WebSocket connections for streaming execution
    """
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.orchestrator = AgentOrchestrator(api_key=settings.MISTRAL_API_KEY)
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_json(self, websocket: WebSocket, data: Dict[str, Any]):
        """Send JSON data to WebSocket"""
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
    
    async def execute_task_streaming(
        self,
        websocket: WebSocket,
        task: str,
        model: str = None
    ):
        """
        Execute task with streaming updates via WebSocket
        """
        try:
            # Send initial status
            await self.send_json(websocket, {
                "type": "status",
                "message": "Starting task execution...",
                "task": task
            })
            
            # Execute task (simplified - full streaming would require more work)
            result = await self.orchestrator.execute_task(
                task_description=task,
                model=model,
                stream=False
            )
            
            # Send completion
            await self.send_json(websocket, {
                "type": "complete",
                "result": result
            })
            
        except Exception as e:
            logger.error(f"WebSocket task execution failed: {e}")
            await self.send_json(websocket, {
                "type": "error",
                "message": str(e)
            })

# Global WebSocket manager instance
ws_manager = WebSocketManager()

async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for streaming execution
    """
    await ws_manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                task = message.get("task")
                model = message.get("model")
                
                if not task:
                    await ws_manager.send_json(websocket, {
                        "type": "error",
                        "message": "Task is required"
                    })
                    continue
                
                # Execute task with streaming
                await ws_manager.execute_task_streaming(websocket, task, model)
                
            except json.JSONDecodeError:
                await ws_manager.send_json(websocket, {
                    "type": "error",
                    "message": "Invalid JSON format"
                })
                
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)
