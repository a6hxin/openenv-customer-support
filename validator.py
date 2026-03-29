"""
validator.py — Pre-submission checklist for OpenEnv environments.

Run this before submitting:
    python validator.py

All checks must pass (exit 0) or you are disqualified.
"""
from __future__ import annotations
import sys
import json
import yaml
import importlib
from pathlib import Path

PASS = "✓"
FAIL = "✗"
WARN = "⚠"
results = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    symbol = PASS if condition else FAIL
    msg = f"  {symbol} {name}"
    if detail:
        msg += f"\n      {detail}"
    print(msg)
    results.append(condition)
    return condition


def section(title: str):
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


# ── 1. File structure ─────────────────────────────────────────────────────────
section("1. Required files")
check("openenv.yaml exists", Path("openenv.yaml").exists())
check("Dockerfile exists", Path("Dockerfile").exists())
check("requirements.txt exists", Path("requirements.txt").exists())
check("baseline.py exists", Path("baseline.py").exists())
check("README.md exists", Path("README.md").exists())
check("app/main.py exists", Path("app/main.py").exists())
check("app/models.py exists", Path("app/models.py").exists())
check("app/environment.py exists", Path("app/environment.py").exists())

# ── 2. openenv.yaml spec ──────────────────────────────────────────────────────
section("2. openenv.yaml compliance")
try:
    with open("openenv.yaml") as f:
        spec = yaml.safe_load(f)
    check("Valid YAML", True)
    check("Has 'name' field", "name" in spec)
    check("Has 'version' field", "version" in spec)
    check("Has 'observation_space'", "observation_space" in spec)
    check("Has 'action_space'", "action_space" in spec)
    check("Has 'tasks'", "tasks" in spec)
    check("Has 3+ tasks", len(spec.get("tasks", [])) >= 3,
          f"Found: {len(spec.get('tasks', []))}")
    check("Has 'endpoints'", "endpoints" in spec)
    for ep in ["reset", "step", "state", "tasks", "grader", "baseline"]:
        check(f"  endpoint '{ep}' listed", ep in spec.get("endpoints", {}))
except Exception as e:
    check("openenv.yaml parseable", False, str(e))

# ── 3. Typed models ───────────────────────────────────────────────────────────
section("3. Typed models")
try:
    from app.models import (
        Observation, Action, StepResult, ResetRequest,
        ResetResult, EnvironmentState, TaskInfo, GraderRequest,
        GraderResult, BaselineResult, ActionType
    )
    check("All model classes importable", True)
    check("Action has action_type field", hasattr(Action, "model_fields") and
          "action_type" in Action.model_fields)
    check("StepResult has reward field", "reward" in StepResult.model_fields)
    check("StepResult has done field", "done" in StepResult.model_fields)
    check("Observation has current_step", "current_step" in Observation.model_fields)

    # Validate model instantiation
    obs = Observation(
        ticket_id="TEST-001",
        customer_tier="pro",
        issue_category="billing",
        description="test",
        priority="medium",
    )
    check("Observation instantiates correctly", obs.ticket_id == "TEST-001")

    action = Action(action_type=ActionType.LOOKUP_ACCOUNT, payload={"account_id": "ACC-1"})
    check("Action instantiates correctly", action.action_type == ActionType.LOOKUP_ACCOUNT)

except Exception as e:
    check("Models importable", False, str(e))

# ── 4. Environment API ────────────────────────────────────────────────────────
section("4. Environment step()/reset()/state() API")
try:
    from app.environment import SupportEnvironment, TASK_META
    from app.models import ResetRequest, Action, ActionType

    env = SupportEnvironment()
    check("SupportEnvironment instantiates", True)

    # reset()
    r = env.reset(ResetRequest(task_id="billing_dispute_easy", seed=42))
    check("reset() returns ResetResult", r is not None)
    check("reset() observation has ticket_id", bool(r.observation.ticket_id))

    # state()
    s = env.state()
    check("state() returns EnvironmentState", s is not None)
    check("state() session_id set", bool(s.session_id))

    # step()
    step_r = env.step(Action(action_type=ActionType.LOOKUP_ACCOUNT,
                             payload={"account_id": "ACC-7842"}))
    check("step() returns StepResult", step_r is not None)
    check("step() reward in [0,1]", 0.0 <= step_r.reward <= 1.0,
          f"reward={step_r.reward}")
    check("step() has done flag", isinstance(step_r.done, bool))

except Exception as e:
    check("Environment API functional", False, str(e))

# ── 5. All 3 tasks present with graders ───────────────────────────────────────
section("5. Tasks and graders")
task_ids = ["billing_dispute_easy", "technical_outage_medium", "security_incident_hard"]
for tid in task_ids:
    try:
        mod = importlib.import_module(f"tasks.{tid}")
        check(f"Task '{tid}' importable", True)
        check(f"  has make_initial_observation()", hasattr(mod, "make_initial_observation"))
        check(f"  has step()", hasattr(mod, "step"))
        check(f"  has grade()", hasattr(mod, "grade"))

        # Run a minimal episode and grade
        env = SupportEnvironment()
        from app.models import ResetRequest
        env.reset(ResetRequest(task_id=tid, seed=0))
        env.step(Action(action_type=ActionType.CLOSE_TICKET, payload={}))
        grade_r = env.grade()
        score = grade_r["score"]
        check(f"  grade() returns score in [0,1]", 0.0 <= score <= 1.0,
              f"score={score}")
    except Exception as e:
        check(f"Task '{tid}' functional", False, str(e))

# ── 6. Baseline script ────────────────────────────────────────────────────────
section("6. Baseline inference")
try:
    from baseline import run_baseline_agent
    check("baseline.py importable", True)
    scores = {}
    for tid in task_ids:
        env = SupportEnvironment()
        score, log = run_baseline_agent(env, tid, seed=42)
        scores[tid] = score
        check(
            f"  baseline score [{tid}] in (0,1]",
            0.0 < score <= 1.0,
            f"score={score:.4f}"
        )
    mean = sum(scores.values()) / len(scores)
    check(f"  mean baseline score > 0.5", mean > 0.5, f"mean={mean:.4f}")
except Exception as e:
    check("baseline runs without error", False, str(e))

# ── 7. FastAPI routes ─────────────────────────────────────────────────────────
section("7. FastAPI routes registered")
try:
    from app.main import app as fastapi_app
    routes = {r.path for r in fastapi_app.routes}
    for ep in ["/health", "/reset", "/step", "/state", "/tasks", "/grader", "/baseline"]:
        check(f"  route '{ep}' registered", ep in routes)
except Exception as e:
    check("FastAPI app importable", False, str(e))

# ── Summary ───────────────────────────────────────────────────────────────────
section("Summary")
total = len(results)
passed = sum(results)
failed = total - passed
print(f"\n  Passed: {passed}/{total}")
if failed > 0:
    print(f"  Failed: {failed}/{total}")
    print(f"\n  {FAIL} PRE-SUBMISSION VALIDATION FAILED — fix issues above before submitting.")
    sys.exit(1)
else:
    print(f"\n  {PASS} ALL CHECKS PASSED — ready to submit!")
    sys.exit(0)
