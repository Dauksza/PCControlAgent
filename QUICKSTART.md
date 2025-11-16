# Mistral AI Agent Platform - Quick Start Guide

## 🎯 Overview

This is a complete implementation of an advanced autonomous agent platform that leverages 100% of Mistral AI's API capabilities. The platform includes:

- Dynamic model selection and management
- Vision/multimodal support with Pixtral models
- Streaming responses
- Built-in connectors (web_search, code_interpreter, etc.)
- Conversations API with persistent memory
- Embeddings API for RAG/semantic search
- Autonomous task execution with never-stop-short logic
- FastAPI backend with WebSocket support
- Multiple UI options (Streamlit and Gradio)

## 🚀 Quick Setup (5 Minutes)

### Step 1: Run the Setup Script

```bash
python setup.py
```

This will:
- Create a virtual environment
- Install all dependencies
- Create necessary directories
- Generate `.env` file

### Step 2: Configure Your API Key

Edit the `.env` file and add your Mistral API key:

```bash
MISTRAL_API_KEY=your_actual_api_key_here
```

Get your API key from: https://console.mistral.ai/

### Step 3: Activate Virtual Environment

**Linux/Mac:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### Step 4: Run the Application

```bash
python app.py
```

The API will be available at: http://localhost:8000

## 📚 Documentation

### API Documentation
Once running, visit:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Key Endpoints

#### GET /api/health
Health check endpoint

#### GET /api/models
List all available Mistral models

#### POST /api/execute
Execute an autonomous agent task

Example request:
```json
{
  "task": "Search for the latest AI developments and summarize",
  "model": "mistral-large-2407",
  "max_iterations": 50
}
```

#### POST /api/chat
Direct chat completion

#### WS /ws/execute
WebSocket endpoint for streaming execution

## 🎨 UI Options

### Option 1: Streamlit UI (Recommended for beginners)

```bash
streamlit run ui/streamlit_app.py
```

Visit: http://localhost:8501

### Option 2: Gradio UI (More interactive)

```bash
python ui/gradio_app.py
```

Visit: http://localhost:7860

## 🧪 Testing

Run all tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=. --cov-report=html
```

Run only unit tests (exclude integration tests):
```bash
pytest -m "not integration"
```

## 📦 Project Structure

```
PCControlAgent/
├── config/           # Configuration management
├── core/             # Core Mistral API integration
├── agent/            # Agent orchestration logic
├── tools/            # Tool implementations
├── utils/            # Utility functions
├── api/              # REST API & WebSocket
├── ui/               # User interfaces (Streamlit & Gradio)
├── tests/            # Test suite
├── app.py            # Main FastAPI application
├── setup.py          # Setup script
└── requirements.txt  # Python dependencies
```

## 🔧 Configuration

All configuration is in `.env` file. Key settings:

```bash
# API Configuration
MISTRAL_API_KEY=your_key_here
DEFAULT_MODEL=mistral-large-2407
MAX_ITERATIONS=50

# Feature Flags
ENABLE_STREAMING=True
ENABLE_VISION=True
ENABLE_PARALLEL_TOOLS=True

# Logging
LOG_LEVEL=INFO
```

## 🛠️ Available Tools

The platform includes several built-in tools:

1. **web_search** - Search the web for current information
2. **code_interpreter** - Execute Python, JavaScript, or Bash code
3. **calculator** - Perform mathematical calculations
4. **image_generation** - Generate images (stub, needs integration)
5. **document_library** - Access document library (stub)
6. **browser_automation** - Browser tasks (stub)
7. **ocr_tool** - Extract text from images (stub)

## 💡 Usage Examples

### Example 1: Simple Task

```python
import requests

response = requests.post(
    "http://localhost:8000/api/execute",
    json={
        "task": "What are the latest developments in AI?",
        "model": "mistral-large-2407"
    }
)

print(response.json())
```

### Example 2: Multi-Step Workflow

```python
task = """
1. Search for Python async/await best practices
2. Summarize the top 5 practices
3. Provide a code example demonstrating each practice
"""

response = requests.post(
    "http://localhost:8000/api/execute",
    json={
        "task": task,
        "model": "mistral-large-2407",
        "max_iterations": 30
    }
)
```

### Example 3: Using Vision Model

```python
response = requests.post(
    "http://localhost:8000/api/execute",
    json={
        "task": "Analyze this image and describe what you see",
        "model": "pixtral-12b"
    }
)
```

## 🔐 Security Notes

1. **Never commit your `.env` file** - It's in `.gitignore` by default
2. **Code execution tool** - Uses subprocess with timeout, but needs proper sandboxing for production
3. **API key** - Keep your Mistral API key secure
4. **CORS** - Currently allows all origins, restrict in production

## 🐛 Troubleshooting

### "MISTRAL_API_KEY not provided"
- Make sure you've created the `.env` file and added your API key
- Check that the virtual environment is activated

### "Failed to fetch models"
- Check your internet connection
- Verify your API key is valid
- The platform will fall back to hardcoded models if API fetch fails

### Import errors
- Make sure virtual environment is activated
- Run: `pip install -r requirements.txt`

### Tests failing
- Some tests require mocking - this is expected
- Run: `pytest -m "not integration"` to skip integration tests

## 📈 Performance Optimization

1. **Enable caching** - Set `ENABLE_REDIS_CACHE=True` and configure Redis
2. **Adjust iterations** - Lower `MAX_ITERATIONS` for faster responses
3. **Model selection** - Use smaller models like `open-mistral-nemo` for faster responses
4. **Parallel tools** - Keep `ENABLE_PARALLEL_TOOLS=True` for concurrent execution

## 🤝 Contributing

To extend the platform:

1. **Add new tools** - Create a new file in `tools/` extending `BaseTool`
2. **Register tools** - Add to `orchestrator._register_default_tools()`
3. **Add endpoints** - Extend `api/routes.py`
4. **Write tests** - Add tests in `tests/`

## 📖 Additional Resources

- [Mistral AI Documentation](https://docs.mistral.ai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Gradio Documentation](https://www.gradio.app/docs/)

## 🆘 Support

For issues:
1. Check the logs in `logs/agent.log`
2. Review API documentation at `/docs`
3. Check configuration in `.env`
4. Review the blueprint.md for detailed implementation details

## 📝 License

This project is provided as-is for educational and development purposes.

---

**Happy Coding! 🚀**

For detailed implementation information, see `blueprint.md`.
