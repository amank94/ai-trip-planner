#!/bin/bash

# AI Trip Planner - Server Startup Script
# This script helps start the server with proper permissions

echo "🚀 Starting AI Trip Planner Server..."
echo ""

# Navigate to project root
cd "$(dirname "$0")"

# Activate virtual environment
if [ -f "backend/.venv/bin/activate" ]; then
    source backend/.venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found at backend/.venv"
    exit 1
fi

# Try different ports if one fails
PORTS=(8000 3000 5000 8080 8888)

for PORT in "${PORTS[@]}"; do
    echo ""
    echo "🔄 Trying port $PORT..."
    cd backend
    
    # Try to start server
    python -m uvicorn main:app --host 127.0.0.1 --port $PORT 2>&1 | head -20 &
    SERVER_PID=$!
    
    # Wait a moment to see if it starts
    sleep 3
    
    # Check if process is still running
    if kill -0 $SERVER_PID 2>/dev/null; then
        echo ""
        echo "✅ Server started successfully!"
        echo "📡 Server running on: http://localhost:$PORT"
        echo "📊 API docs: http://localhost:$PORT/docs"
        echo "🏥 Health check: http://localhost:$PORT/health"
        echo ""
        echo "Press Ctrl+C to stop the server"
        wait $SERVER_PID
        exit 0
    else
        echo "❌ Port $PORT failed, trying next..."
        kill $SERVER_PID 2>/dev/null
    fi
done

echo ""
echo "❌ Could not start server on any port"
echo "📖 Please check FIX_PERMISSIONS.md for help with macOS permissions"
exit 1
