#!/bin/bash

echo "🚀 Starting PCControlAgent..."

# Start backend
echo "Starting backend server..."
uvicorn app:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Start frontend
echo "Starting frontend..."
cd frontend && npm run dev &
FRONTEND_PID=$!

echo "
✅ Services started!"
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:4173"
echo "
Press Ctrl+C to stop all services"

# Wait and cleanup
trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait
