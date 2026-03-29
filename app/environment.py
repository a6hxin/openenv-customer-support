"""
Environment engine — manages sessions, dispatches to task modules,
accumulates episode logs, and enforces the OpenEnv step/reset/state contract.
"""
from __future__ import annotations
import random
import uuid
from typing import Any, Dict, Optional

from app.models import (
    Action, Observation, ResetRequest, ResetResult,
    StepResult, EnvironmentState, TaskInfo, ACTION_SCHEMA
)

# ── Task registry ─────────────────────────────────────────────────────────────
from tasks import billing_dispute_easy, technical_outage_medium, security_incident_hard

TASK_MODULES = {
    "billing_dispute_easy": billing_dispute_easy,
    "technical_outage_medium": technical_outage_medium,
    "security_incident_hard": security_incident_hard,
}

TASK_META = {
    "billing_dispute_easy": TaskInfo(
        id="billing_dispute_easy",
        name="Billing Dispute Resolution (Easy)",
        description=(
            "Resolve a straightforward billing dispute where a customer was "
            "double-charged. The agent must verify the charge, apply the refund "
            "policy, and close the ticket."
        ),
        difficulty="easy",
        max_steps=10,
        reward_threshold=0.7,
        action_schema=ACTION_SCHEMA,
    ),
    "technical_outage_medium": TaskInfo(
        id="technical_outage_medium",
        name="Technical Outage Triage (Medium)",
        description=(
            "Diagnose and resolve a connectivity issue affecting a Pro customer. "
            "Requires checking system status, gathering diagnostics, and applying "
            "the correct fix or escalation path."
        ),
        difficulty="medium",
        max_steps=15,
        reward_threshold=0.65,
        action_schema=ACTION_SCHEMA,
    ),
    "security_incident_hard": TaskInfo(
        id="security_incident_hard",
        name="Security Incident Response (Hard)",
        description=(
            "Handle a suspected account compromise. Follow the full security "
            "runbook: verify identity, freeze account, reset credentials, notify "
            "the customer, and document the incident — all within SLA."
        ),
        difficulty="hard",
        max_steps=20,
        reward_threshold=0.6,
        action_schema=ACTION_SCHEMA,
    ),
}


class SupportEnvironment:
    """
    Single-session environment instance.
    One instance per HTTP session (stored in app state).
    """

    def __init__(self):
        self.session_id: str = str(uuid.uuid4())
        self.task_id: Optional[str] = None
        self.current_obs: Optional[Observation] = None
        self.step_count: int = 0
        self.total_reward: float = 0.0
        self.done: bool = False
        self.episode_log: list = []
        self._rng: random.Random = random.Random()
        self._task_module = None

    # ── reset() ──────────────────────────────────────────────────────────────
    def reset(self, request: ResetRequest) -> ResetResult:
        if request.task_id not in TASK_MODULES:
            raise ValueError(
                f"Unknown task_id '{request.task_id}'. "
                f"Valid tasks: {list(TASK_MODULES.keys())}"
            )

        self.task_id = request.task_id
        self._task_module = TASK_MODULES[request.task_id]
        self._rng = random.Random(request.seed)

        seed = request.seed
        self.current_obs = self._task_module.make_initial_observation(seed=seed)
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        self.episode_log = []
        self.session_id = str(uuid.uuid4())

        task_info = TASK_META[request.task_id]

        return ResetResult(
            observation=self.current_obs,
            task_id=self.task_id,
            task_description=task_info.description,
            info={"session_id": self.session_id, "seed": seed},
        )

    # ── step() ────────────────────────────────────────────────────────────────
    def step(self, action: Action) -> StepResult:
        if self.current_obs is None:
            raise RuntimeError("Call reset() before step().")
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        prev_obs = self.current_obs
        new_obs, reward, done, truncated, info = self._task_module.step(
            prev_obs, action, self._rng
        )

        self.total_reward += reward
        self.step_count += 1
        self.done = done or truncated
        self.current_obs = new_obs

        log_entry = {
            "step": self.step_count,
            "action": {"action_type": action.action_type.value, "payload": action.payload},
            "reward": reward,
            "done": self.done,
            "observation": new_obs.model_dump(),
        }
        self.episode_log.append(log_entry)

        return StepResult(
            observation=new_obs,
            reward=reward,
            done=self.done,
            truncated=truncated,
            info={**info, "total_reward": self.total_reward, "step": self.step_count},
        )

    # ── state() ───────────────────────────────────────────────────────────────
    def state(self) -> EnvironmentState:
        return EnvironmentState(
            session_id=self.session_id,
            task_id=self.task_id,
            current_observation=self.current_obs,
            step_count=self.step_count,
            total_reward=self.total_reward,
            done=self.done,
            episode_log=self.episode_log,
        )

    # ── grade() ───────────────────────────────────────────────────────────────
    def grade(self) -> Dict[str, Any]:
        if not self.task_id:
            raise RuntimeError("No task loaded. Call reset() first.")
        result = self._task_module.grade(self.episode_log)
        return result
