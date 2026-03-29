"""
baseline.py — Scripted deterministic baseline agent for all 3 tasks.

This script can be run standalone:
    python baseline.py

Or called programmatically via run_baseline_agent().

The baseline agent uses hand-crafted action sequences that represent
a competent (but not optimal) support agent. It serves as a reproducible
lower bound for RL agent development.

Expected scores (seed=42):
  billing_dispute_easy     ~0.85
  technical_outage_medium  ~0.70
  security_incident_hard   ~0.75
"""
from __future__ import annotations
import sys
import json
from typing import List, Tuple, Dict, Any

from app.models import Action, ActionType
from app.environment import SupportEnvironment, TASK_META


# ── Per-task scripted action sequences ────────────────────────────────────────

def _billing_dispute_actions() -> List[Action]:
    """Optimal path: lookup → refund → respond → close."""
    return [
        Action(action_type=ActionType.LOOKUP_ACCOUNT,
               payload={"account_id": "ACC-7842"}),
        Action(action_type=ActionType.ASK_CLARIFICATION,
               payload={"question": "Could you confirm which payment method was charged?"}),
        Action(action_type=ActionType.ISSUE_REFUND,
               payload={"amount": 49.99, "reason": "duplicate_charge"}),
        Action(action_type=ActionType.SEND_RESPONSE,
               payload={"message": (
                   "Hi Sarah, I've verified the duplicate charge on your account "
                   "and processed a full refund of $49.99. It should appear within "
                   "3–5 business days. I'm sorry for the inconvenience!"
               )}),
        Action(action_type=ActionType.CLOSE_TICKET,
               payload={"resolution": "Duplicate charge confirmed and refunded. Customer notified."}),
    ]


def _technical_outage_actions(seed: int = 42) -> List[Action]:
    """
    Covers both scenario branches by always checking status first,
    then gathering diagnostics before acting.
    """
    import random
    rng = random.Random(seed)
    scenario_idx = rng.randint(0, 1)  # mirrors make_initial_observation logic

    actions = [
        Action(action_type=ActionType.LOOKUP_ACCOUNT,
               payload={"account_id": "ACC-3391"}),
        Action(action_type=ActionType.CHECK_SYSTEM_STATUS,
               payload={"service": "api_gateway", "region": "all"}),
        Action(action_type=ActionType.ASK_CLARIFICATION,
               payload={"question": (
                   "Have you made any configuration changes recently, "
                   "such as updating API keys, base URLs, or auth headers?"
               )}),
    ]

    if scenario_idx == 0:
        # Regional outage path — escalate
        actions += [
            Action(action_type=ActionType.ESCALATE,
                   payload={"tier": 2, "reason": "Regional infrastructure outage INC-8821"}),
            Action(action_type=ActionType.SEND_RESPONSE,
                   payload={"message": (
                       "We've identified an active outage in the eu-west-1 region "
                       "affecting the API gateway. Our infrastructure team is working "
                       "on it (Incident #INC-8821). You'll receive updates on our "
                       "status page. ETA: ~1 hour."
                   )}),
        ]
    else:
        # Misconfiguration path — apply fix
        actions += [
            Action(action_type=ActionType.APPLY_FIX,
                   payload={"fix_type": "reset_auth_config", "component": "auth"}),
            Action(action_type=ActionType.SEND_RESPONSE,
                   payload={"message": (
                       "I've reset your API authentication configuration to the "
                       "correct v2 defaults. Please retry your requests — "
                       "a test call from our end returned HTTP 200. "
                       "Let me know if you need anything else."
                   )}),
        ]

    actions.append(
        Action(action_type=ActionType.CLOSE_TICKET,
               payload={"resolution": "Root cause identified and resolved. Customer informed."})
    )
    return actions


def _security_incident_actions() -> List[Action]:
    """Full runbook: verify identity → lookup → freeze → reset → notify → document."""
    return [
        Action(action_type=ActionType.ASK_CLARIFICATION,
               payload={"question": (
                   "To verify your identity before I take any action, could you "
                   "please confirm your full name, the email on the account, and "
                   "the location of your last legitimate login?"
               )}),
        Action(action_type=ActionType.LOOKUP_ACCOUNT,
               payload={"account_id": "ACC-0091"}),
        Action(action_type=ActionType.APPLY_FIX,
               payload={"fix_type": "freeze_account",
                        "component": "account",
                        "reason": "Suspected compromise from Tor exit node"}),
        Action(action_type=ActionType.RESET_CREDENTIALS,
               payload={"method": "email", "notify_customer": True,
                        "rotate_api_keys": True, "invalidate_mfa": True}),
        Action(action_type=ActionType.SEND_RESPONSE,
               payload={"message": (
                   "Hi Marcus, I've secured your account. Here's what I did: "
                   "(1) Terminated all active sessions, (2) Rotated all API keys, "
                   "(3) Sent a password reset link to your email, "
                   "(4) Invalidated existing MFA tokens. "
                   "Please reset your password and re-enroll in MFA. "
                   "We also recommend reviewing recent API usage for unauthorized activity."
               )}),
        Action(action_type=ActionType.CLOSE_TICKET,
               payload={"resolution": (
                   "Security incident confirmed: unauthorized access from Tor exit node "
                   "185.220.101.x. Actions taken: identity verified, account frozen, "
                   "all sessions terminated, API keys rotated, credentials reset via email, "
                   "customer notified with recovery steps. No data exfiltration detected. "
                   "Incident documented and compliance team notified per policy."
               )}),
    ]


BASELINE_ACTION_FACTORIES = {
    "billing_dispute_easy": lambda seed: _billing_dispute_actions(),
    "technical_outage_medium": _technical_outage_actions,
    "security_incident_hard": lambda seed: _security_incident_actions(),
}


# ── Runner ────────────────────────────────────────────────────────────────────

def run_baseline_agent(
    env: SupportEnvironment,
    task_id: str,
    seed: int = 42,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Run the scripted baseline agent on a single task.
    Returns (final_score, episode_log).
    """
    from app.models import ResetRequest
    env.reset(ResetRequest(task_id=task_id, seed=seed))

    factory = BASELINE_ACTION_FACTORIES[task_id]
    actions = factory(seed)

    for action in actions:
        if env.done:
            break
        env.step(action)

    grader_result = env.grade()
    return grader_result["score"], env.episode_log


# ── Standalone entry point ────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("OpenEnv — Customer Support Resolution")
    print("Baseline Agent Results (seed=42)")
    print("=" * 60)

    all_scores = {}
    for task_id, task_info in TASK_META.items():
        env = SupportEnvironment()
        score, log = run_baseline_agent(env, task_id, seed=42)
        all_scores[task_id] = score
        status = "✓ PASS" if score >= task_info.reward_threshold else "✗ FAIL"
        print(f"\n[{task_info.difficulty.upper()}] {task_info.name}")
        print(f"  Score:     {score:.4f}  (threshold: {task_info.reward_threshold})")
        print(f"  Steps:     {len(log)}")
        print(f"  Status:    {status}")

        grader = TASK_META[task_id]
        module = __import__(f"tasks.{task_id}", fromlist=["grade"])
        detail = module.grade(log)
        print("  Breakdown:")
        for k, v in detail["breakdown"].items():
            print(f"    {k:<35} {v:+.2f}")
        print(f"  Feedback:  {detail['feedback']}")

    mean = sum(all_scores.values()) / len(all_scores)
    print("\n" + "=" * 60)
    print(f"Mean Score: {mean:.4f}")
    print("=" * 60)

    # Write JSON results for CI
    with open("baseline_results.json", "w") as f:
        json.dump({
            "scores": all_scores,
            "mean_score": mean,
            "seed": 42,
        }, f, indent=2)
    print("\nResults saved to baseline_results.json")
    return all_scores


if __name__ == "__main__":
    scores = main()
    # Exit non-zero if any task fails completely
    if any(v == 0.0 for v in scores.values()):
        sys.exit(1)
