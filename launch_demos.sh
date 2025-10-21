#!/bin/bash
# Launch script for CellMorphNet demo applications

echo "======================================================================"
echo "🚀 CellMorphNet Demo Launcher"
echo "======================================================================"
echo ""

# Check if model checkpoint exists
if [ ! -f "models/bloodmnist_full_exp/checkpoints/best.pth" ]; then
    echo "❌ Error: Model checkpoint not found!"
    echo "   Expected: models/bloodmnist_full_exp/checkpoints/best.pth"
    echo ""
    echo "   Please train a model first:"
    echo "   python src/train.py --data_dir data/raw/bloodmnist_full --num_classes 8 --epochs 50"
    exit 1
fi

echo "✅ Model checkpoint found: models/bloodmnist_full_exp/checkpoints/best.pth"
echo ""

# Menu
echo "Choose demo to launch:"
echo "  1) Streamlit Web UI (Interactive interface)"
echo "  2) FastAPI REST API (API server)"
echo "  3) Both (Streamlit on :8501, FastAPI on :8000)"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "🌐 Starting Streamlit Web UI..."
        echo "   URL: http://localhost:8501"
        echo "   Press Ctrl+C to stop"
        echo ""
        streamlit run demos/streamlit_app.py
        ;;
    2)
        echo ""
        echo "🔌 Starting FastAPI Server..."
        echo "   API URL: http://localhost:8000"
        echo "   Docs: http://localhost:8000/docs"
        echo "   Press Ctrl+C to stop"
        echo ""
        python demos/fastapi_server.py
        ;;
    3)
        echo ""
        echo "🚀 Starting both applications..."
        echo "   Streamlit UI: http://localhost:8501"
        echo "   FastAPI Docs: http://localhost:8000/docs"
        echo "   Press Ctrl+C to stop"
        echo ""
        # Start FastAPI in background
        python demos/fastapi_server.py &
        FASTAPI_PID=$!
        sleep 3
        # Start Streamlit (foreground)
        streamlit run demos/streamlit_app.py
        # Kill FastAPI when Streamlit stops
        kill $FASTAPI_PID 2>/dev/null
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac
