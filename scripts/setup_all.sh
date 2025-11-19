#!/bin/bash
set -e

echo "🚀 PCControlAgent Complete Setup & Installation"
echo "=" x 60

# 1. Python dependencies
echo "
📦 Installing Python dependencies..."
pip install -q mistralai neo4j httpx python-dotenv fastapi uvicorn websockets

# 2. Frontend dependencies
echo "
📦 Installing Frontend dependencies..."
cd frontend && npm install && cd ..

# 3. Setup Neo4j with Docker
echo "
🗄️ Setting up Neo4j database..."
if command -v docker &> /dev/null; then
    docker-compose up -d neo4j
    echo "✅ Neo4j started on ports 7474 (HTTP) and 7687 (Bolt)"
else
    echo "⚠️ Docker not found. Skipping Neo4j setup."
fi

# 4. Install MCP servers
echo "
🔧 Installing MCP servers..."
npx -y @modelcontextprotocol/create-server filesystem
echo "✅ MCP filesystem server installed"

# 5. Initialize memory graph
echo "
🧠 Initializing memory graph..."
python3 << PYINIT
from core.memory_graph import MemoryGraph
try:
    mg = MemoryGraph()
    mg.connect()
    print("✅ Memory graph connected")
    mg.close()
except Exception as e:
    print(f"⚠️ Memory graph: {e}")
PYINIT

# 6. Run tests
echo "
🧪 Running tests..."
pytest tests/ -v || echo "⚠️ Some tests failed"

echo "
" x 2
echo "✅ Setup Complete!"
echo "
Next steps:"
echo "  1. Set MISTRAL_API_KEY in .env file"
echo "  2. Run: ./scripts/start_all.sh"
echo "  3. Open browser to http://localhost:8000"
