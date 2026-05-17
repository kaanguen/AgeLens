#!/usr/bin/env python
"""
AgeLens Backend Server
Starts the FastAPI server with the age and gender prediction models
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run the FastAPI app
from app import app
import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("🔮 Starting AgeLens FastAPI Server")
    print("=" * 60)
    print("\n📍 API will be available at: http://localhost:8000")
    print("📍 Health check: http://localhost:8000/health")
    print("\n⏳ Loading models on startup...")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
