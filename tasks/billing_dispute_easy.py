"""
Task: billing_dispute_easy
Difficulty: Easy
Scenario: Customer was double-charged for their monthly subscription.
Agent must: verify account, confirm duplicate charge, issue refund, close ticket.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import random

from app.models import (
    Observation, Action, ActionType, CustomerTier,
    IssueCategory, Priority, Message
)


TASK_ID = "billing_dispute_easy"
MAX_STEPS = 10

SCENARIO = {
    "ticket_id": "TKT-BILL-001",
    "customer_name": "Sarah Chen",
    "account_id": "ACC-7842",
    "email": "sarah.chen@example.com",
    "tier": CustomerTier.PRO,
    "double_charge_amount": 49.99,
    "charge_dates": ["2024-11-01", "2024-11-01"],
    "description": (
        "Hi, I just noticed I was charged twice for my Pro subscription "
        "this month — both charges appeared on Nov 1st for $49.99 each. "
        "I only have one account. Please help!"
    ),
}


def make_initial_observation(seed: Optional[int] = None) -> Observation:
    rng = random.Random(seed)
    # Use a fixed amount so agent can read it from the description
    amount = SCENARIO["double_charge_amount"]
    desc = SCENARIO["description"]

    return Observation(
        ticket_id=SCENARIO["ticket_id"],
        customer_tier=SCENARIO["tier"],
        issue_category=IssueCategory.BILLING,
        description=desc,
        priority=Priority.MEDIUM,
        conversation_history=[
            Message(role="customer", content=desc)
        ],
        current_step=0,
        max_steps=MAX_STEPS,
        resolved=False,
        metadata={
            "account_id": SCENARIO["account_id"],
            "charge_amount": amount,
            "charges_count": 2,
            "customer_name": SCENARIO["customer_name"],
            "account_looked_up": False,
            "refund_issued": False,
            "customer_acknowledged": False,
        }
    )


def step(
    obs: Observation,
    action: Action,
    rng: random.Random
) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
    """
    Execute one step. Returns (new_obs, reward, done, truncated, info).
    Reward is incremental — partial credit for each correct action.
    """
    meta = obs.metadata.copy()
    history = list(obs.conversation_history)
    reward = 0.0
    done = False
    truncated = False
    info: Dict[str, Any] = {}

    at = action.action_type
    payload = action.payload

    if at == ActionType.LOOKUP_ACCOUNT:
        if not meta.get("account_looked_up"):
            meta["account_looked_up"] = True
            reward = 0.15
            history.append(Message(
                role="system",
                content=(
                    f"Account {meta['account_id']} found. "
                    f"Customer: {SCENARIO['customer_name']}, Tier: PRO. "
                    f"Billing records show 2 charges of ${meta['charge_amount']} "
                    f"on 2024-11-01. Duplicate confirmed."
                )
            ))
            info["account_verified"] = True
        else:
            reward = 0.0
            info["note"] = "Account already looked up"

    elif at == ActionType.ISSUE_REFUND:
        if not meta.get("account_looked_up"):
            reward = -0.05
            history.append(Message(
                role="system",
                content="Cannot issue refund: account not verified first."
            ))
            info["error"] = "must lookup_account first"
        elif meta.get("refund_issued"):
            reward = 0.0
            info["note"] = "Refund already issued"
        else:
            amount = payload.get("amount", 0)
            expected = meta["charge_amount"]
            if abs(amount - expected) < 0.01:
                meta["refund_issued"] = True
                reward = 0.35
                history.append(Message(
                    role="system",
                    content=f"Refund of ${expected} processed successfully. "
                            f"Transaction ID: REF-{rng.randint(10000,99999)}"
                ))
                info["refund_processed"] = True
            else:
                reward = 0.05
                history.append(Message(
                    role="system",
                    content=f"Refund amount ${amount} does not match duplicate "
                            f"charge amount ${expected}. Partial action recorded."
                ))
                info["warning"] = "Wrong refund amount"

    elif at == ActionType.SEND_RESPONSE:
        msg = payload.get("message", "")
        if meta.get("refund_issued") and len(msg) > 20:
            reward = 0.15
            meta["customer_acknowledged"] = True
            history.append(Message(role="agent", content=msg))
            history.append(Message(
                role="customer",
                content="Thank you so much! I can see the refund is on its way. "
                        "Really appreciate the quick help!"
            ))
        elif len(msg) > 10:
            reward = 0.05
            history.append(Message(role="agent", content=msg))
        else:
            reward = 0.0

    elif at == ActionType.ASK_CLARIFICATION:
        q = payload.get("question", "")
        if q:
            reward = 0.05
            history.append(Message(role="agent", content=q))
            history.append(Message(
                role="customer",
                content="Yes, both charges hit on the same day. "
                        "My bank shows two separate $49.99 debits."
            ))

    elif at == ActionType.CLOSE_TICKET:
        if meta.get("refund_issued") and meta.get("customer_acknowledged"):
            reward = 0.35
            done = True
            meta["closed"] = True
            history.append(Message(
                role="system",
                content="Ticket closed. Resolution: duplicate charge refunded. "
                        "CSAT survey sent."
            ))
            info["resolved"] = True
        elif meta.get("refund_issued"):
            reward = 0.15
            done = True
            meta["closed"] = True
            info["resolved"] = True
            info["note"] = "Closed without customer confirmation — minor deduction"
        else:
            reward = -0.1
            history.append(Message(
                role="system",
                content="Cannot close: underlying issue not resolved yet."
            ))
            info["error"] = "close attempted before resolution"

    elif at == ActionType.ESCALATE:
        # Wrong action for this easy case
        reward = -0.05
        history.append(Message(
            role="system",
            content="Escalation request submitted. Note: this issue was "
                    "resolvable at Tier-1."
        ))
        info["note"] = "unnecessary escalation"

    else:
        reward = 0.0
        info["note"] = f"Action {at} had no effect on this scenario"

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
    """
    Score a completed episode.
    Returns breakdown dict and final score in [0, 1].
    """
    actions_taken = [e["action"]["action_type"] for e in episode_log if "action" in e]
    final_obs = episode_log[-1].get("observation", {}) if episode_log else {}
    meta = final_obs.get("metadata", {})

    # Rubric
    account_verified = meta.get("account_looked_up", False)
    refund_issued = meta.get("refund_issued", False)
    customer_acked = meta.get("customer_acknowledged", False)
    ticket_closed = meta.get("closed", False)
    steps_used = final_obs.get("current_step", MAX_STEPS)
    sla_breached = final_obs.get("sla_breached", False)
    escalated_unnecessarily = "escalate" in actions_taken

    breakdown = {
        "account_verified": 0.20 if account_verified else 0.0,
        "refund_issued": 0.35 if refund_issued else 0.0,
        "customer_notified": 0.15 if customer_acked else 0.0,
        "ticket_closed": 0.20 if ticket_closed else 0.0,
        "efficiency_bonus": 0.10 if steps_used <= 5 else (0.05 if steps_used <= 7 else 0.0),
    }

    if escalated_unnecessarily:
        breakdown["unnecessary_escalation_penalty"] = -0.05
    if sla_breached:
        breakdown["sla_breach_penalty"] = -0.10

    score = max(0.0, min(1.0, sum(breakdown.values())))

    feedback_parts = []
    if not account_verified:
        feedback_parts.append("Agent never looked up the account before taking action.")
    if not refund_issued:
        feedback_parts.append("Refund was not issued.")
    if not customer_acked:
        feedback_parts.append("Customer was not notified of the resolution.")
    if not ticket_closed:
        feedback_parts.append("Ticket was not formally closed.")
    if escalated_unnecessarily:
        feedback_parts.append("Issue was needlessly escalated to Tier-2.")
    if not feedback_parts:
        feedback_parts.append("All billing dispute resolution steps completed correctly.")

    return {
        "score": round(score, 4),
        "breakdown": breakdown,
        "feedback": " ".join(feedback_parts),
        "passed": score >= 0.7,
    }
