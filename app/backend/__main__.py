"""Run API: python -m app.backend (from repo root: ayurvedic-health-assistant)."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.backend:app", host="127.0.0.1", port=8000, reload=False)
