#!/bin/bash

# Startup script for Transformer Encoder Web Visualizer

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║         Transformer Encoder Web Visualizer                ║"
echo "║              Starting Server...                            ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠${NC} Virtual environment not found. Please run ./run.sh first."
    exit 1
fi

# Activate virtual environment
echo -e "${BLUE}▶${NC} Activating virtual environment..."
source venv/bin/activate

# Check if Flask dependencies are installed
echo -e "${BLUE}▶${NC} Checking dependencies..."
python -c "import flask, flask_socketio" 2>/dev/null

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠${NC} Installing web visualizer dependencies..."
    pip install flask flask-cors flask-socketio python-socketio simple-websocket -q
    echo -e "${GREEN}✓${NC} Dependencies installed"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo ""
echo -e "${GREEN}🚀 Starting Web Visualizer...${NC}"
echo ""
echo -e "   ${BLUE}URL:${NC} http://localhost:5000"
echo -e "   ${BLUE}Press${NC} Ctrl+C to stop the server"
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""

# Start the server
cd viz_app && python app.py
