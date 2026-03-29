"""
Task: security_incident_hard
Difficulty: Hard
Scenario: A customer reports their account may be compromised. The agent must
follow the security runbook in exact order:
  1. Verify customer identity
  2. Lookup account and flag it
  3. Freeze/lock the account
  4. Reset credentials
  5. Notify the customer with recovery steps
  6. Document and close within SLA

Wrong ordering, missing steps, or skipping identity verification are penalised.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import random

from app.models import (
    Observation, Action, ActionType, CustomerTier,
    IssueCategory, Priority, Message
)

TASK_ID = "security_incident_hard"
MAX_STEPS = 20

DESCRIPTION = (
    "URGENT: I think my account has been hacked. I received login notifications "
    "from an IP in a country I've never been to, and there are API calls being "
    "made that I didn't authorize. I need immediate help securing my account!"
)

# Correct runbook order (enforced for partial credit)
RUNBOOK = [
    "identity_verified",
    "account_looked_up",
    "account_frozen",
    "credentials_reset",
    "customer_notified",
    "incident_documented",
]


def make_initial_observation(seed: Optional[int] = None) -> Observation:
    rng = random.Random(seed)
    tier = rng.choice([CustomerTier.PRO, CustomerTier.ENTERPRISE])

    return Observation(
        ticket_id="TKT-SEC-007",
        customer_tier=tier,
        issue_category=IssueCategory.SECURITY,
        description=DESCRIPTION,
        priority=Priority.CRITICAL,
        conversation_history=[
            Message(role="customer", content=DESCRIPTION),
            Message(
                role="system",
                content=(
                    "⚠ SECURITY ALERT: Anomalous login detected 47 minutes ago. "
                    "Origin: 185.220.101.x (Tor exit node). "
                    "6 unauthorized API calls made. SLA for security incidents: 30 min."
                )
            ),
        ],
        current_step=0,
        max_steps=MAX_STEPS,
        resolved=False,
        metadata={
            "account_id": "ACC-0091",
            "customer_name": "Marcus Webb",
            "tier": tier.value,
            # Runbook completion flags
            "identity_verified": False,
            "account_looked_up": False,
            "account_frozen": False,
            "credentials_reset": False,
            "customer_notified": False,
            "incident_documented": False,
            # Ordering tracker
            "steps_completed_in_order": [],
            "out_of_order_actions": 0,
            "skipped_identity_check": False,
        }
    )


def _check_order(meta: Dict, current_step_name: str) -> bool:
    """Return True if this step comes after all required predecessors."""
    idx = RUNBOOK.index(current_step_name)
    for prereq in RUNBOOK[:idx]:
        if not meta.get(prereq, False):
            return False
    return True


def step(
    obs: Observation,
    action: Action,
    rng: random.Random
) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
    meta = obs.metadata.copy()
    history = list(obs.conversation_history)
    reward = 0.0
    done = False
    truncated = False
    info: Dict[str, Any] = {}

    at = action.action_type
    payload = action.payload

    # ── Identity verification (must be FIRST) ─────────────────────────────
    if at == ActionType.ASK_CLARIFICATION:
        q = payload.get("question", "")
        is_identity_q = any(kw in q.lower() for kw in [
            "verify", "identity", "confirm", "name", "email",
            "last", "account", "phone", "dob", "address"
        ])
        if is_identity_q and not meta["identity_verified"]:
            meta["identity_verified"] = True
            meta["steps_completed_in_order"].append("identity_verified")
            reward = 0.15
            history.append(Message(role="agent", content=q))
            history.append(Message(
                role="customer",
                content=(
                    "Sure. My name is Marcus Webb, email marcus.webb@novacorp.io, "
                    "last login before this was from NYC three days ago."
                )
            ))
            info["identity_verified"] = True
        elif not is_identity_q:
            reward = 0.02
            history.append(Message(role="agent", content=q))
            history.append(Message(role="customer", content="I'm not sure what you mean."))

    # ── Account lookup ────────────────────────────────────────────────────
    elif at == ActionType.LOOKUP_ACCOUNT:
        if not meta["identity_verified"]:
            meta["skipped_identity_check"] = True
            meta["out_of_order_actions"] += 1
            reward = -0.05
            history.append(Message(
                role="system",
                content="⚠ Policy violation: account data accessed before identity verified."
            ))
            info["policy_violation"] = "identity not verified"
        elif not meta["account_looked_up"]:
            meta["account_looked_up"] = True
            meta["steps_completed_in_order"].append("account_looked_up")
            reward = 0.10
            history.append(Message(
                role="system",
                content=(
                    "Account ACC-0091 loaded. Owner: Marcus Webb. "
                    "Status: ACTIVE. Tier: "
                    f"{meta['tier'].upper()}. "
                    "Recent anomaly: 6 API calls from Tor exit node. "
                    "Current sessions: 2 (1 suspicious)."
                )
            ))
            info["account_loaded"] = True

    # ── Freeze / lock account ──────────────────────────────────────────────
    elif at == ActionType.APPLY_FIX:
        fix_type = payload.get("fix_type", "")
        is_freeze = any(kw in fix_type.lower() for kw in ["freeze", "lock", "suspend", "block"])
        if is_freeze:
            if not meta["account_looked_up"]:
                reward = -0.05
                meta["out_of_order_actions"] += 1
                history.append(Message(
                    role="system",
                    content="Cannot freeze: account not yet loaded."
                ))
            elif not meta["account_frozen"]:
                in_order = _check_order(meta, "account_frozen")
                meta["account_frozen"] = True
                meta["steps_completed_in_order"].append("account_frozen")
                reward = 0.20 if in_order else 0.10
                history.append(Message(
                    role="system",
                    content=(
                        "Account ACC-0091 FROZEN. All active sessions terminated. "
                        "Suspicious session from 185.220.101.x killed. "
                        "API keys temporarily suspended."
                    )
                ))
                info["account_frozen"] = True
        else:
            reward = 0.0
            history.append(Message(
                role="system",
                content=f"Fix '{fix_type}' applied. No direct security impact."
            ))

    # ── Reset credentials ──────────────────────────────────────────────────
    elif at == ActionType.RESET_CREDENTIALS:
        if not meta["account_frozen"]:
            reward = -0.05
            meta["out_of_order_actions"] += 1
            history.append(Message(
                role="system",
                content=(
                    "⚠ Credentials reset without freezing account first. "
                    "Attacker may still have active session."
                )
            ))
            info["out_of_order"] = "should freeze before resetting credentials"
        elif not meta["credentials_reset"]:
            in_order = _check_order(meta, "credentials_reset")
            method = payload.get("method", "email")
            meta["credentials_reset"] = True
            meta["steps_completed_in_order"].append("credentials_reset")
            reward = 0.20 if in_order else 0.10
            history.append(Message(
                role="system",
                content=(
                    f"Credentials reset via {method}. New temp password generated. "
                    "All API keys rotated. MFA tokens invalidated. "
                    "Recovery email queued."
                )
            ))
            info["credentials_reset"] = True

    # ── Notify customer ────────────────────────────────────────────────────
    elif at == ActionType.SEND_RESPONSE:
        msg = payload.get("message", "")
        if meta["credentials_reset"] and not meta["customer_notified"]:
            # Check message quality: should include recovery steps
            has_recovery = any(kw in msg.lower() for kw in [
                "password", "reset", "link", "recover", "secure", "steps", "new"
            ])
            in_order = _check_order(meta, "customer_notified")
            meta["customer_notified"] = True
            meta["steps_completed_in_order"].append("customer_notified")
            reward = (0.15 if has_recovery else 0.08) if in_order else 0.05
            history.append(Message(role="agent", content=msg))
            history.append(Message(
                role="customer",
                content=(
                    "Thank you for acting so quickly. I just received the "
                    "password reset email. I'm enabling MFA immediately."
                )
            ))
            info["customer_notified"] = True
        elif len(msg) > 10:
            reward = 0.03
            history.append(Message(role="agent", content=msg))

    # ── Close ticket (also handles documentation) ──────────────────────────
    elif at == ActionType.CLOSE_TICKET:
        resolution = payload.get("resolution", "")
        has_documentation = len(resolution) > 50

        if not meta["credentials_reset"]:
            reward = -0.10
            history.append(Message(
                role="system",
                content="Cannot close security incident: credentials not reset."
            ))
        else:
            in_order = _check_order(meta, "incident_documented")
            meta["incident_documented"] = True
            meta["steps_completed_in_order"].append("incident_documented")

            doc_reward = 0.15 if has_documentation else 0.05
            reward = doc_reward if in_order else doc_reward * 0.5
            done = True
            meta["closed"] = True
            history.append(Message(
                role="system",
                content=(
                    "Security incident #SEC-007 closed and documented. "
                    "Incident report filed. Account unfrozen with new credentials. "
                    "Compliance team notified per security policy."
                )
            ))
            info["resolved"] = True
            info["documented"] = has_documentation

    elif at == ActionType.ESCALATE:
        tier_level = payload.get("tier", 2)
        if meta["account_frozen"] and not meta["credentials_reset"]:
            reward = 0.05  # acceptable to loop in security team
            history.append(Message(
                role="system",
                content=f"Security team (Tier-{tier_level}) notified. Co-handling incident."
            ))
        else:
            reward = 0.0

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

    completed = {step: meta.get(step, False) for step in RUNBOOK}
    steps_in_order = meta.get("steps_completed_in_order", [])
    out_of_order = meta.get("out_of_order_actions", 0)
    sla_breached = final_obs.get("sla_breached", False)
    skipped_identity = meta.get("skipped_identity_check", False)

    # Check if runbook was followed in correct order
    correct_order = steps_in_order == [s for s in RUNBOOK if completed[s]]

    breakdown = {
        "identity_verified": 0.15 if completed["identity_verified"] else 0.0,
        "account_looked_up": 0.10 if completed["account_looked_up"] else 0.0,
        "account_frozen": 0.20 if completed["account_frozen"] else 0.0,
        "credentials_reset": 0.20 if completed["credentials_reset"] else 0.0,
        "customer_notified": 0.15 if completed["customer_notified"] else 0.0,
        "incident_documented": 0.15 if completed["incident_documented"] else 0.0,
        "runbook_order_bonus": 0.05 if correct_order else 0.0,
    }
    if skipped_identity:
        breakdown["identity_skip_penalty"] = -0.10
    if out_of_order > 0:
        breakdown["out_of_order_penalty"] = -0.03 * min(out_of_order, 3)
    if sla_breached:
        breakdown["sla_breach_penalty"] = -0.10

    score = max(0.0, min(1.0, sum(breakdown.values())))

    feedback = []
    if not completed["identity_verified"]:
        feedback.append("Identity was never verified — critical security policy violation.")
    if skipped_identity:
        feedback.append("Account data was accessed before identity was confirmed.")
    if not completed["account_frozen"]:
        feedback.append("Account was never frozen, leaving attacker with potential access.")
    if not completed["credentials_reset"]:
        feedback.append("Credentials were not reset.")
    if not completed["incident_documented"]:
        feedback.append("Incident was not documented — required for compliance.")
    if not correct_order and any(completed.values()):
        feedback.append("Runbook steps were not completed in the correct order.")
    if not feedback:
        feedback.append("Security runbook followed correctly with proper documentation.")

    return {
        "score": round(score, 4),
        "breakdown": breakdown,
        "feedback": " ".join(feedback),
        "passed": score >= 0.60,
    }
