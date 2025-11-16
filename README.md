# 🚀 Mistral AI Autonomous Agent Platform

Advanced autonomous agent platform utilizing 100% of Mistral AI's API capabilities.

## ✨ Features

- **Dynamic Model Selection** - Fetches available models from Mistral API endpoint
- **Vision/Multimodal Support** - Pixtral models for image analysis and OCR
- **Streaming Responses** - Token-by-token updates
- **Built-in Connectors** - web_search, code_interpreter, image_generation, document_library
- **Conversations API** - Persistent memory with branching support
- **Embeddings API** - RAG and semantic search capabilities
- **JSON Mode & Schema** - Structured output enforcement
- **Parallel Tool Calling** - Independent tools run simultaneously
- **Fine-tuning Workflow** - Custom model training support
- **MCP Integration** - Model Context Protocol support
- **Never-Stop-Short Logic** - Verification before completion
- **Circuit Breaker Pattern** - Robust error handling

## 📋 Prerequisites

- Python 3.9 or higher
- Mistral API key ([Get one here](https://console.mistral.ai/))
- pip or poetry for package management

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd PCControlAgent
```

### 2. Run Setup Script

```bash
python setup.py
```

This will:
- Create a virtual environment
- Install all dependencies
- Create necessary directories
- Generate .env file

### 3. Configure Environment

Edit `.env` file and add your Mistral API key:

```bash
MISTRAL_API_KEY=your_actual_api_key_here
```

### 4. Activate Virtual Environment

**Linux/Mac:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 5. Run the Application

```bash
python app.py
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

Once running, visit:
- **API Docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Key Endpoints

#### GET /models
List all available Mistral models

```bash
curl http://localhost:8000/models
```

#### POST /execute
Execute an agent task

```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Search for latest AI developments and summarize",
    "model": "mistral-large-2407",
    "stream": false
  }'
```

#### WS /ws/execute
WebSocket endpoint for streaming execution

## 🎨 Alternative UIs

### Streamlit UI

```bash
streamlit run ui/streamlit_app.py
```

### Gradio UI

```bash
python ui/gradio_app.py
```

## 🏗️ Project Structure

```
PCControlAgent/
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── README.md                 # This file
├── blueprint.md              # Detailed implementation guide
├── app.py                    # FastAPI main application
├── setup.py                  # Automated setup script
├── config/                   # Configuration management
│   ├── __init__.py
│   ├── settings.py          # Settings with Pydantic
│   └── constants.py         # Model definitions, API endpoints
├── core/                     # Core Mistral API integration
│   ├── __init__.py
│   ├── mistral_client.py   # Main Mistral API client
│   ├── models.py           # Dynamic model fetching
│   ├── conversations.py    # Conversations API
│   ├── embeddings.py       # Embeddings & RAG
│   ├── vision.py           # Vision/multimodal
│   ├── fine_tuning.py      # Fine-tuning workflows
│   ├── streaming.py        # Streaming handler
│   └── mcp.py              # Model Context Protocol
├── agent/                    # Agent orchestration
│   ├── __init__.py
│   ├── orchestrator.py     # Main execution loop
│   ├── tool_registry.py    # Tool management
│   ├── task_decomposer.py  # Task breakdown
│   └── completion_detector.py  # Task completion logic
├── tools/                    # Tool implementations
│   ├── __init__.py
│   ├── base_tool.py
│   ├── web_search.py
│   ├── code_execution.py
│   ├── image_generation.py
│   ├── document_library.py
│   ├── browser_automation.py
│   ├── ocr_tool.py
│   └── custom_tools.py
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── error_handling.py
│   ├── logging_config.py
│   ├── cache_manager.py
│   └── validators.py
├── api/                      # API routes
│   ├── __init__.py
│   ├── routes.py
│   └── websocket.py
├── ui/                       # User interfaces
│   ├── __init__.py
│   ├── streamlit_app.py
│   └── gradio_app.py
└── tests/                    # Test suite
    ├── __init__.py
    ├── test_mistral_client.py
    ├── test_agent.py
    ├── test_tools.py
    └── integration/
        ├── test_full_workflow.py
        └── test_streaming.py
```

## 🧪 Testing

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=. --cov-report=html
```

## 🔧 Development

### Code Formatting

```bash
black .
```

### Linting

```bash
ruff check .
```

### Type Checking

```bash
mypy .
```

## 📖 Usage Examples

### Example 1: Simple Task

```python
import requests

response = requests.post(
    "http://localhost:8000/execute",
    json={
        "task": "What are the latest developments in AI?",
        "model": "mistral-large-2407"
    }
)

print(response.json())
```

### Example 2: Vision Task

```python
task = """
Analyze this product image and provide:
1. Product description
2. Key features
3. Target audience
"""

response = requests.post(
    "http://localhost:8000/execute",
    json={
        "task": task,
        "model": "pixtral-12b"
    }
)
```

### Example 3: Multi-Step Workflow

```python
task = """
1. Search for Python async best practices
2. Write a code example demonstrating async/await
3. Explain the benefits
"""

response = requests.post(
    "http://localhost:8000/execute",
    json={
        "task": task,
        "model": "mistral-large-2407",
        "max_iterations": 20
    }
)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🔗 Resources

- [Mistral AI Documentation](https://docs.mistral.ai/)
- [Mistral API Reference](https://docs.mistral.ai/api/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 💬 Support

For issues and questions, please open an issue on GitHub.

## 🌟 Acknowledgments

Built with Mistral AI's powerful API and following best practices for autonomous agent development.
