"""
Run the FastAPI server:

  uvicorn run_api:app --reload --host 0.0.0.0 --port 8000

Or:

  python -m uvicorn run_api:app --reload --host 127.0.0.1 --port 8000
"""

from finance_tracker.api.main import app

__all__ = ["app"]
