"""Start the FastAPI server."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import uvicorn

if __name__ == "__main__":
    print("Starting FastAPI server on http://127.0.0.1:8000")
    print("Press Ctrl+C to stop")
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=True)
