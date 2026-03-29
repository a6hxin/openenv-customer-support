"""
demo.py — Interactive walkthrough demo for all 3 OpenEnv tasks.

Demonstrates the full step()/reset()/state() API loop with a human-readable
trace. Suitable for submission demo and quick manual verification.

Usage:
    python demo.py                 # run all tasks
    python demo.py --task billing  # run one task by keyword
    python demo.py --seed 0        # use a specific seed
"""
from __future__ import annotations
import argparse
import json
import sys
import textwrap
from typing import List

from app.models import Action, ActionType, ResetRequest
from app.environment import SupportEnvironment, TASK_META


# ── ANSI colours (graceful fallback if unsupported) ───────────────────────────
try:
    import os
    if os.name == "nt" or not sys.stdout.isatty():
        raise ImportError
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[32m"
    YELLOW = "\033[33m"
    RED    = "\033[31m"
    CYAN   = "\033[36m"
    GREY   = "\033[90m"
except ImportError:
    RESET = BOLD = GREEN = YELLOW = RED = CYAN = GREY = ""


def _c(color: str, text: str) -> str:
    return f"{color}{text}{RESET}"


def _wrap(text: str, indent: int = 6) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=78, initial_indent=prefix,
                         subsequent_indent=prefix)


def _print_obs(obs, step_num: int | None = None):
    label = f"Step {step_num}" if step_num is not None else "Initial"
    print(f"\n  {_c(BOLD, f'── Observation [{label}]')}")
    print(f"    ticket_id     : {obs.ticket_id}")
    print(f"    category      : {obs.issue_category}  |  priority: {obs.priority}")
    print(f"    customer_tier : {obs.customer_tier}")
    print(f"    step          : {obs.current_step}/{obs.max_steps}")
    print(f"    resolved      : {obs.resolved}  |  sla_breached: {obs.sla_breached}")

    # Show last message from conversation
    history = obs.conversation_history
    if history:
        last = history[-1]
        role_color = CYAN if last.role == "customer" else (
            YELLOW if last.role == "system" else GREEN
        )
        role_label = _c(role_color, f"[{last.role}]")
        print(f"    last msg      : {role_label}")
        print(_wrap(last.content, indent=18))


def _print_action(action: Action, step_num: int):
    payload_str = json.dumps(action.payload) if action.payload else "{}"
    print(f"\n  {_c(BOLD, f'── Action [{step_num}]')}")
    print(f"    action_type : {_c(YELLOW, action.action_type.value)}")
    if action.payload:
        print(_wrap(f"payload     : {payload_str}", indent=20))


def _print_step_result(result, total: float):
    reward_color = GREEN if result.reward >= 0.15 else (
        YELLOW if result.reward >= 0.01 else RED
    )
    print(f"    reward      : {_c(reward_color, f'{result.reward:+.3f}')}  "
          f"(cumulative: {total:.3f})")
    if result.info:
        flagged = {k: v for k, v in result.info.items()
                   if k not in ("total_reward", "step")}
        if flagged:
            print(f"    info        : {_c(GREY, json.dumps(flagged))}")
    if result.done:
        label = _c(RED, "TRUNCATED (SLA breach)") if result.truncated \
            else _c(GREEN, "DONE")
        print(f"    status      : {label}")


def _print_grade(result: dict, task_id: str):
    threshold = TASK_META[task_id].reward_threshold
    score = result["score"]
    passed = result["passed"]
    score_color = GREEN if passed else RED
    print(f"\n  {_c(BOLD, '── Grader Result')}")
    print(f"    score   : {_c(score_color, f'{score:.4f}')}  "
          f"(threshold: {threshold}  |  {'PASS ✓' if passed else 'FAIL ✗'})")
    print(f"    feedback: {result['feedback']}")
    print(f"\n  {_c(BOLD, 'Breakdown:')}")
    for k, v in result["breakdown"].items():
        bar = "█" * int(abs(v) * 20)
        color = GREEN if v >= 0 else RED
        print(f"    {k:<38} {_c(color, f'{v:+.2f}')}  {_c(GREY, bar)}")


# ── Action sequences for demo ─────────────────────────────────────────────────

DEMO_EPISODES = {
    "billing_dispute_easy": [
        Action(action_type=ActionType.LOOKUP_ACCOUNT,
               payload={"account_id": "ACC-7842"}),
        Action(action_type=ActionType.ASK_CLARIFICATION,
               payload={"question": "Can you confirm the dates of the two charges?"}),
        Action(action_type=ActionType.ISSUE_REFUND,
               payload={"amount": 49.99, "reason": "duplicate_charge"}),
        Action(action_type=ActionType.SEND_RESPONSE,
               payload={"message": (
                   "Hi Sarah! I've confirmed the duplicate charge on Nov 1st "
                   "and issued a full refund of $49.99. It should appear within "
                   "3–5 business days. Apologies for the inconvenience!"
               )}),
        Action(action_type=ActionType.CLOSE_TICKET,
               payload={"resolution": (
                   "Duplicate charge confirmed via billing records. "
                   "Full refund of $49.99 issued. Customer notified and satisfied."
               )}),
    ],

    "technical_outage_medium": [
        Action(action_type=ActionType.LOOKUP_ACCOUNT,
               payload={"account_id": "ACC-3391"}),
        Action(action_type=ActionType.CHECK_SYSTEM_STATUS,
               payload={"service": "api_gateway", "region": "all"}),
        Action(action_type=ActionType.ASK_CLARIFICATION,
               payload={"question": (
                   "Have you made any recent configuration changes, such as "
                   "updating base URLs, API key formats, or auth headers?"
               )}),
        # Branch A: apply fix (handles misconfiguration scenario)
        Action(action_type=ActionType.APPLY_FIX,
               payload={"fix_type": "reset_auth_config", "component": "auth"}),
        Action(action_type=ActionType.SEND_RESPONSE,
               payload={"message": (
                   "I've reset the auth configuration to the correct v2 defaults "
                   "and verified your API key. A test call returned HTTP 200. "
                   "Please retry — you should be unblocked now."
               )}),
        Action(action_type=ActionType.CLOSE_TICKET,
               payload={"resolution": "Auth misconfiguration identified and corrected. "
                                      "Connectivity confirmed. Customer notified."}),
    ],

    "security_incident_hard": [
        Action(action_type=ActionType.ASK_CLARIFICATION,
               payload={"question": (
                   "Before taking any action, I need to verify your identity. "
                   "Could you confirm your full name, account email address, "
                   "and the city of your last legitimate login?"
               )}),
        Action(action_type=ActionType.LOOKUP_ACCOUNT,
               payload={"account_id": "ACC-0091"}),
        Action(action_type=ActionType.APPLY_FIX,
               payload={"fix_type": "freeze_account",
                        "component": "account",
                        "reason": "Suspected unauthorised access from Tor exit node"}),
        Action(action_type=ActionType.RESET_CREDENTIALS,
               payload={"method": "email",
                        "notify_customer": True,
                        "rotate_api_keys": True,
                        "invalidate_mfa": True}),
        Action(action_type=ActionType.SEND_RESPONSE,
               payload={"message": (
                   "Hi Marcus — I've secured your account. Here's what was done: "
                   "(1) All active sessions terminated, (2) API keys rotated, "
                   "(3) Password reset link sent to your registered email, "
                   "(4) MFA tokens invalidated. "
                   "Please reset your password and re-enrol in MFA before "
                   "resuming use. We recommend reviewing recent API call logs "
                   "for any unauthorised activity."
               )}),
        Action(action_type=ActionType.CLOSE_TICKET,
               payload={"resolution": (
                   "Security incident confirmed: unauthorised access from Tor "
                   "exit node 185.220.101.x. Timeline: 47 mins from detection "
                   "to resolution (within SLA). Actions: identity verified, "
                   "account frozen, all sessions killed, API keys rotated, "
                   "credentials reset via email, MFA invalidated, customer "
                   "notified with recovery steps. No data exfiltration detected. "
                   "Compliance team notified. Incident documented."
               )}),
    ],
}


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_demo_episode(task_id: str, seed: int = 42) -> float:
    env = SupportEnvironment()
    task_info = TASK_META[task_id]

    print(f"\n{'═' * 70}")
    print(f"  {_c(BOLD, task_info.name)}")
    print(f"  difficulty: {task_info.difficulty.upper()}  |  "
          f"max_steps: {task_info.max_steps}  |  "
          f"threshold: {task_info.reward_threshold}  |  seed: {seed}")
    print(f"{'═' * 70}")

    reset_result = env.reset(ResetRequest(task_id=task_id, seed=seed))
    print(f"\n  {_c(GREY, 'Session ID: ' + env.session_id)}")
    _print_obs(reset_result.observation)

    actions = DEMO_EPISODES[task_id]
    cumulative = 0.0

    for i, action in enumerate(actions, 1):
        if env.done:
            break
        _print_action(action, i)
        result = env.step(action)
        cumulative += result.reward
        _print_step_result(result, cumulative)
        _print_obs(result.observation, step_num=i)

    # Grade
    grade = env.grade()
    _print_grade(grade, task_id)

    # State summary
    state = env.state()
    print(f"\n  {_c(BOLD, 'Episode Summary')}")
    print(f"    total steps   : {state.step_count}")
    print(f"    total reward  : {cumulative:.4f}")
    print(f"    grader score  : {grade['score']:.4f}")

    return grade["score"]


def main():
    parser = argparse.ArgumentParser(description="OpenEnv Customer Support Demo")
    parser.add_argument("--task", default="all",
                        help="Task keyword: billing / technical / security / all")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    keyword_map = {
        "billing": "billing_dispute_easy",
        "technical": "technical_outage_medium",
        "security": "security_incident_hard",
    }

    if args.task == "all":
        tasks = list(TASK_META.keys())
    elif args.task in keyword_map:
        tasks = [keyword_map[args.task]]
    elif args.task in TASK_META:
        tasks = [args.task]
    else:
        print(f"Unknown task '{args.task}'. Choose: billing / technical / security / all")
        sys.exit(1)

    all_scores = {}
    for task_id in tasks:
        score = run_demo_episode(task_id, seed=args.seed)
        all_scores[task_id] = score

    if len(all_scores) > 1:
        mean = sum(all_scores.values()) / len(all_scores)
        print(f"\n{'═' * 70}")
        print(f"  {_c(BOLD, 'Overall Results')}")
        for tid, s in all_scores.items():
            threshold = TASK_META[tid].reward_threshold
            status = _c(GREEN, "PASS ✓") if s >= threshold else _c(RED, "FAIL ✗")
            print(f"    {tid:<40} {s:.4f}  {status}")
        print(f"    {'Mean':<40} {mean:.4f}")
        print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
