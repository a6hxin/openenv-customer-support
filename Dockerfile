# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# HuggingFace Spaces runs as user 1000
RUN useradd -m -u 1000 appuser
WORKDIR /app

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=appuser:appuser . .

# HuggingFace Spaces requires port 7860
ENV PORT=7860
EXPOSE 7860

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
