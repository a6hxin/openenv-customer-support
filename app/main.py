"""
OpenEnv Customer Support Resolution — FastAPI application.

Exposes the full OpenEnv contract:
  POST /reset          — start a new episode
  POST /step           — take an action
  GET  /state          — inspect current state
  GET  /tasks          — list tasks and action schema
  POST /grader         — score a completed episode
  GET  /baseline       — run all baseline agents and return scores
  GET  /health         — liveness check
"""
from __future__ import annotations
import asyncio
from typing import Any, Dict, List

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.models import (
    Action, ResetRequest, ResetResult, StepResult,
    EnvironmentState, TaskInfo, GraderRequest, GraderResult,
    BaselineResult
)
from app.environment import SupportEnvironment, TASK_META, TASK_MODULES

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Support Resolution — OpenEnv",
    description=(
        "A real-world OpenEnv environment where an AI agent handles customer "
        "support tickets across billing, technical, and security domains. "
        "Built for the OpenEnv challenge by Meta & Hugging Face."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static demo UI if present
_static_dir = Path(__file__).parent.parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# Global environment instance (single-session for Spaces demo)
_env = SupportEnvironment()


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Serve the interactive demo UI."""
    html_path = _static_dir / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>OpenEnv Customer Support Resolution</h1><p>API docs: <a href='/docs'>/docs</a></p>")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok", "version": "1.0.0", "environment": "customer-support-resolution"}


# ── OpenEnv Core API ──────────────────────────────────────────────────────────
@app.post("/reset", response_model=ResetResult, tags=["openenv"])
async def reset(request: ResetRequest):
    """
    Reset the environment and start a new episode.
    Pass `task_id` (one of: billing_dispute_easy, technical_outage_medium,
    security_incident_hard) and an optional `seed` for reproducibility.
    """
    try:
        result = _env.reset(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult, tags=["openenv"])
async def step(action: Action):
    """
    Submit an action and advance the environment by one step.
    Returns the next observation, reward, done flag, and info dict.
    """
    try:
        result = _env.step(action)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=EnvironmentState, tags=["openenv"])
async def state():
    """
    Return the full current environment state including episode log.
    """
    return _env.state()


# ── Tasks ─────────────────────────────────────────────────────────────────────
@app.get("/tasks", tags=["openenv"])
async def list_tasks():
    """
    List all available tasks with their descriptions, difficulty,
    and the full action schema (fields required for a step() action).
    """
    return {
        "tasks": [t.model_dump() for t in TASK_META.values()],
        "total": len(TASK_META),
    }


# ── Grader ────────────────────────────────────────────────────────────────────
@app.post("/grader", response_model=GraderResult, tags=["openenv"])
async def grade(request: GraderRequest):
    """
    Score a completed episode against the task's rubric.
    Pass `task_id` and the `episode_log` from the episode.
    Returns score (0.0–1.0), breakdown, feedback, and pass/fail.
    """
    if request.task_id not in TASK_MODULES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id. Valid: {list(TASK_MODULES.keys())}"
        )
    module = TASK_MODULES[request.task_id]
    result = module.grade(request.episode_log)
    return GraderResult(
        task_id=request.task_id,
        score=result["score"],
        breakdown=result["breakdown"],
        feedback=result["feedback"],
        passed=result["passed"],
    )


# ── Baseline ──────────────────────────────────────────────────────────────────
@app.get("/baseline", response_model=BaselineResult, tags=["openenv"])
async def baseline():
    """
    Run the scripted baseline agent on all 3 tasks and return scores.
    The baseline uses deterministic action sequences (seed=42).
    """
    from baseline import run_baseline_agent
    scores: Dict[str, float] = {}
    details: Dict[str, Any] = {}

    for task_id in TASK_META:
        env = SupportEnvironment()
        score, log = run_baseline_agent(env, task_id, seed=42)
        scores[task_id] = score
        details[task_id] = {
            "steps": len(log),
            "score": score,
            "passed": score >= TASK_META[task_id].reward_threshold,
        }

    mean_score = round(sum(scores.values()) / len(scores), 4)
    return BaselineResult(scores=scores, mean_score=mean_score, details=details)
