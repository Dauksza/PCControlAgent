"""
Main FastAPI application
"""
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from api.routes import router
from api.websocket import websocket_endpoint
from config.settings import settings
from utils.logging_config import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle management for the application
    """
    logger.info("Starting Mistral AI Agent Platform...")
    logger.info(f"Environment: {settings.LOG_LEVEL}")
    logger.info(f"Default Model: {settings.DEFAULT_MODEL}")
    yield
    logger.info("Shutting down Mistral AI Agent Platform...")

app = FastAPI(
    title="Mistral AI Agent Platform",
    description="Advanced autonomous agent with full Mistral API capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api", tags=["api"])

# WebSocket endpoint
@app.websocket("/ws/execute")
async def websocket_execute_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming execution"""
    await websocket_endpoint(websocket)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Mistral AI Agent Platform",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/api/health"
    }

@app.get("/docs-redirect")
async def docs_redirect():
    """Redirect to API documentation"""
    return {
        "message": "Visit /docs for interactive API documentation",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

if __name__ == "__main__":
    logger.info("="*50)
    logger.info("🚀 Starting Mistral AI Agent Platform")
    logger.info("="*50)
    logger.info(f"API Documentation: http://localhost:8000/docs")
    logger.info(f"ReDoc: http://localhost:8000/redoc")
    logger.info(f"Health Check: http://localhost:8000/api/health")
    logger.info("="*50)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )
