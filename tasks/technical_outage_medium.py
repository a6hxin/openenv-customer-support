"""
Task: technical_outage_medium
Difficulty: Medium
Scenario: A Pro customer can't connect to the API. Agent must check system
status, gather diagnostics, determine root cause (regional outage vs config
error), apply the right fix or escalate appropriately.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import random

from app.models import (
    Observation, Action, ActionType, CustomerTier,
    IssueCategory, Priority, Message
)

TASK_ID = "technical_outage_medium"
MAX_STEPS = 15

SCENARIOS = [
    {
        "root_cause": "regional_outage",
        "affected_region": "eu-west-1",
        "affected_service": "api_gateway",
        "customer_region": "eu-west-1",
        "correct_fix": "escalate",
        "description": (
            "Our application has been getting 503 errors from your API for "
            "the past 2 hours. All our requests are timing out. This is "
            "blocking our production deployment. Please help urgently!"
        ),
    },
    {
        "root_cause": "misconfiguration",
        "affected_region": None,
        "affected_service": "auth",
        "customer_region": "us-east-1",
        "correct_fix": "apply_fix",
        "description": (
            "We suddenly can't authenticate to your API. We're getting "
            "401 Unauthorized errors even though our API key hasn't changed. "
            "This started about an hour ago after our team rotated some settings."
        ),
    },
]


def make_initial_observation(seed: Optional[int] = None) -> Observation:
    rng = random.Random(seed)
    scenario = rng.choice(SCENARIOS)

    return Observation(
        ticket_id="TKT-TECH-042",
        customer_tier=CustomerTier.PRO,
        issue_category=IssueCategory.TECHNICAL,
        description=scenario["description"],
        priority=Priority.HIGH,
        conversation_history=[
            Message(role="customer", content=scenario["description"])
        ],
        current_step=0,
        max_steps=MAX_STEPS,
        resolved=False,
        metadata={
            "scenario": scenario,
            "root_cause": scenario["root_cause"],
            "correct_fix": scenario["correct_fix"],
            "account_id": "ACC-3391",
            "customer_name": "DevOps Team @ NovaCorp",
            "account_looked_up": False,
            "status_checked": False,
            "diagnostics_gathered": False,
            "correct_action_taken": False,
            "customer_notified": False,
        }
    )


def step(
    obs: Observation,
    action: Action,
    rng: random.Random
) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
    meta = obs.metadata.copy()
    scenario = meta["scenario"]
    history = list(obs.conversation_history)
    reward = 0.0
    done = False
    truncated = False
    info: Dict[str, Any] = {}

    at = action.action_type
    payload = action.payload

    if at == ActionType.LOOKUP_ACCOUNT:
        if not meta["account_looked_up"]:
            meta["account_looked_up"] = True
            reward = 0.10
            history.append(Message(
                role="system",
                content=(
                    f"Account ACC-3391 loaded. Tier: PRO. Region: "
                    f"{scenario['customer_region']}. "
                    f"API key status: active. Last successful call: 3 hours ago."
                )
            ))
            info["account_verified"] = True

    elif at == ActionType.CHECK_SYSTEM_STATUS:
        if not meta["status_checked"]:
            meta["status_checked"] = True
            reward = 0.20
            if scenario["root_cause"] == "regional_outage":
                history.append(Message(
                    role="system",
                    content=(
                        f"Status page: DEGRADED. "
                        f"Service: {scenario['affected_service']} — "
                        f"Region: {scenario['affected_region']} — "
                        f"Status: OUTAGE. Incident #INC-8821 opened 2h ago. "
                        f"Engineering team is investigating."
                    )
                ))
                info["outage_detected"] = True
            else:
                history.append(Message(
                    role="system",
                    content=(
                        "Status page: ALL SYSTEMS OPERATIONAL. "
                        "No active incidents. auth service: healthy across all regions."
                    )
                ))
                info["no_outage"] = True

    elif at == ActionType.ASK_CLARIFICATION:
        q = payload.get("question", "")
        if q and not meta["diagnostics_gathered"]:
            meta["diagnostics_gathered"] = True
            reward = 0.15
            history.append(Message(role="agent", content=q))
            if scenario["root_cause"] == "misconfiguration":
                history.append(Message(
                    role="customer",
                    content=(
                        "We recently updated our API config — changed the "
                        "base URL from v1 to v2 endpoint but maybe we also "
                        "accidentally changed the auth header format."
                    )
                ))
            else:
                history.append(Message(
                    role="customer",
                    content=(
                        "We haven't changed anything on our end. "
                        "It started failing across all our servers at once."
                    )
                ))

    elif at == ActionType.APPLY_FIX:
        if scenario["correct_fix"] == "apply_fix":
            if meta.get("diagnostics_gathered") or meta.get("status_checked"):
                meta["correct_action_taken"] = True
                reward = 0.30
                history.append(Message(
                    role="system",
                    content=(
                        "Fix applied: auth configuration reset to v2 API defaults. "
                        "API key re-validated. Test request: HTTP 200 OK."
                    )
                ))
                info["fix_applied"] = True
            else:
                reward = 0.05
                history.append(Message(
                    role="system",
                    content="Fix attempted without full diagnostics. Partial success."
                ))
        else:
            # Wrong action — there's an infrastructure outage, agent can't fix it
            reward = -0.05
            history.append(Message(
                role="system",
                content=(
                    "Fix attempted but issue persists. "
                    "Root cause appears to be infrastructure-level."
                )
            ))
            info["wrong_action"] = "should have escalated"

    elif at == ActionType.ESCALATE:
        tier = payload.get("tier", 2)
        reason = payload.get("reason", "")
        if scenario["correct_fix"] == "escalate":
            if meta.get("status_checked"):
                meta["correct_action_taken"] = True
                reward = 0.30
                history.append(Message(
                    role="system",
                    content=(
                        f"Escalated to Tier-{tier} (Infrastructure). "
                        f"Linked to Incident #INC-8821. "
                        f"Customer will be updated via status page."
                    )
                ))
                info["escalated_correctly"] = True
            else:
                reward = 0.10
                history.append(Message(
                    role="system",
                    content="Escalated without first checking system status."
                ))
        else:
            reward = -0.05
            history.append(Message(
                role="system",
                content="Escalated, but this was resolvable at Tier-1 with a config fix."
            ))

    elif at == ActionType.SEND_RESPONSE:
        msg = payload.get("message", "")
        if meta["correct_action_taken"] and len(msg) > 20:
            meta["customer_notified"] = True
            reward = 0.15
            history.append(Message(role="agent", content=msg))
            history.append(Message(
                role="customer",
                content="Thank you for the update. We'll monitor and let you know."
            ))
        elif len(msg) > 10:
            reward = 0.05
            history.append(Message(role="agent", content=msg))

    elif at == ActionType.CLOSE_TICKET:
        if meta["correct_action_taken"] and meta["customer_notified"]:
            reward = 0.20
            done = True
            meta["closed"] = True
            info["resolved"] = True
        elif meta["correct_action_taken"]:
            reward = 0.10
            done = True
            meta["closed"] = True
            info["resolved"] = True
            info["note"] = "Closed without notifying customer"
        else:
            reward = -0.10
            history.append(Message(
                role="system",
                content="Cannot close: issue not yet resolved."
            ))

    next_step = obs.current_step + 1
    truncated = next_step >= MAX_STEPS and not done
    if truncated:
        done = True

    new_obs = obs.model_copy(update={
        "current_step": next_step,
        "conversation_history": history,
        "metadata": meta,
        "resolved": meta.get("closed", False),
        "sla_breached": truncated,
    })
    return new_obs, reward, done, truncated, info


def grade(episode_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    final_obs = episode_log[-1].get("observation", {}) if episode_log else {}
    meta = final_obs.get("metadata", {})

    account_verified = meta.get("account_looked_up", False)
    status_checked = meta.get("status_checked", False)
    diagnostics = meta.get("diagnostics_gathered", False)
    correct_action = meta.get("correct_action_taken", False)
    customer_notified = meta.get("customer_notified", False)
    closed = meta.get("closed", False)
    sla_breached = final_obs.get("sla_breached", False)
    steps_used = final_obs.get("current_step", MAX_STEPS)

    breakdown = {
        "account_verified": 0.10 if account_verified else 0.0,
        "status_checked": 0.15 if status_checked else 0.0,
        "diagnostics_gathered": 0.15 if diagnostics else 0.0,
        "correct_root_cause_action": 0.30 if correct_action else 0.0,
        "customer_notified": 0.15 if customer_notified else 0.0,
        "ticket_closed": 0.15 if closed else 0.0,
    }
    if sla_breached:
        breakdown["sla_breach_penalty"] = -0.10
    if steps_used <= 8:
        breakdown["efficiency_bonus"] = 0.05

    score = max(0.0, min(1.0, sum(breakdown.values())))

    feedback = []
    if not status_checked:
        feedback.append("System status was never checked — critical for outage triage.")
    if not correct_action:
        feedback.append("Wrong resolution path chosen for the root cause.")
    if not customer_notified:
        feedback.append("Customer was not kept informed.")
    if not feedback:
        feedback.append("Correct triage flow followed end-to-end.")

    return {
        "score": round(score, 4),
        "breakdown": breakdown,
        "feedback": " ".join(feedback),
        "passed": score >= 0.65,
    }
