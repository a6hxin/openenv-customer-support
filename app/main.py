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
from typing import Any, Dict, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.models import (
    Action, ResetRequest, ResetResult, StepResult,
    EnvironmentState, GraderRequest, GraderResult,
    BaselineResult
)
from app.environment import SupportEnvironment, TASK_META, TASK_MODULES

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Support Resolution — OpenEnv",
    description=(
        "A real-world OpenEnv environment where an AI agent handles customer "
        "support tickets across billing, technical, and security domains."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static demo UI
_static_dir = Path(__file__).parent.parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# Global environment instance
_env = SupportEnvironment()


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    html_path = _static_dir / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>OpenEnv Customer Support Resolution</h1><p><a href='/docs'>API Docs</a></p>")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok", "version": "1.0.0", "environment": "customer-support-resolution"}


# ── OpenEnv Core API ──────────────────────────────────────────────────────────
@app.post("/reset", tags=["openenv"])
async def reset(request: ResetRequest):
    """Reset the environment and start a new episode."""
    try:
        result = _env.reset(request)
        # Return plain dict to avoid serialization issues
        obs = result.observation
        return {
            "task_id": result.task_id,
            "task_description": result.task_description,
            "observation": {
                "ticket_id": obs.ticket_id,
                "customer_tier": obs.customer_tier,
                "issue_category": obs.issue_category,
                "description": obs.description,
                "priority": obs.priority,
                "conversation_history": [
                    {"role": m.role, "content": m.content, "timestamp": m.timestamp}
                    for m in obs.conversation_history
                ],
                "current_step": obs.current_step,
                "max_steps": obs.max_steps,
                "resolved": obs.resolved,
                "sla_breached": obs.sla_breached,
                "metadata": obs.metadata,
            },
            "info": result.info,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", tags=["openenv"])
async def step(action: Action):
    """Submit an action and advance the environment by one step."""
    try:
        result = _env.step(action)
        obs = result.observation
        return {
            "observation": {
                "ticket_id": obs.ticket_id,
                "customer_tier": obs.customer_tier,
                "issue_category": obs.issue_category,
                "description": obs.description,
                "priority": obs.priority,
                "conversation_history": [
                    {"role": m.role, "content": m.content, "timestamp": m.timestamp}
                    for m in obs.conversation_history
                ],
                "current_step": obs.current_step,
                "max_steps": obs.max_steps,
                "resolved": obs.resolved,
                "sla_breached": obs.sla_breached,
                "metadata": obs.metadata,
            },
            "reward": result.reward,
            "done": result.done,
            "truncated": result.truncated,
            "info": result.info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", tags=["openenv"])
async def state():
    """Return the full current environment state."""
    s = _env.state()
    obs = s.current_observation
    return {
        "session_id": s.session_id,
        "task_id": s.task_id,
        "current_observation": {
            "ticket_id": obs.ticket_id,
            "customer_tier": obs.customer_tier,
            "issue_category": obs.issue_category,
            "description": obs.description,
            "priority": obs.priority,
            "conversation_history": [
                {"role": m.role, "content": m.content, "timestamp": m.timestamp}
                for m in obs.conversation_history
            ],
            "current_step": obs.current_step,
            "max_steps": obs.max_steps,
            "resolved": obs.resolved,
            "sla_breached": obs.sla_breached,
            "metadata": obs.metadata,
        } if obs else None,
        "step_count": s.step_count,
        "total_reward": s.total_reward,
        "done": s.done,
        "episode_log": s.episode_log,
    }


# ── Tasks ─────────────────────────────────────────────────────────────────────
@app.get("/tasks", tags=["openenv"])
async def list_tasks():
    """List all available tasks with action schema."""
    return {
        "tasks": [t.model_dump() for t in TASK_META.values()],
        "total": len(TASK_META),
    }


# ── Grader ────────────────────────────────────────────────────────────────────
@app.post("/grader", tags=["openenv"])
async def grade(request: GraderRequest):
    """Score a completed episode."""
    if request.task_id not in TASK_MODULES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id. Valid: {list(TASK_MODULES.keys())}"
        )
    try:
        module = TASK_MODULES[request.task_id]
        result = module.grade(request.episode_log)
        return {
            "task_id": request.task_id,
            "score": result["score"],
            "breakdown": result["breakdown"],
            "feedback": result["feedback"],
            "passed": result["passed"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Baseline ──────────────────────────────────────────────────────────────────
@app.get("/baseline", tags=["openenv"])
async def baseline():
    """Run the scripted baseline agent on all 3 tasks."""
    try:
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
        return {"scores": scores, "mean_score": mean_score, "details": details}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
