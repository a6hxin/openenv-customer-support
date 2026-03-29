"""
spaces_app.py — Hugging Face Spaces entry point.

Mounts the FastAPI OpenEnv API and serves a browser-based interactive
demo UI at the root. The demo lets visitors run any of the 3 tasks
step-by-step directly in their browser without writing any code.

This file is only used when deploying to HF Spaces (SDK: docker).
The Dockerfile CMD points to app.main:app directly for production use.
For Spaces with the Gradio/static SDK you would use this file instead.
"""
from app.main import app  # re-export the FastAPI app

# HF Spaces docker SDK just needs a callable ASGI app named `app`.
# uvicorn is started by the Dockerfile CMD.
__all__ = ["app"]
