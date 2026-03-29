---
title: Openenv Customer Support
emoji: 🎧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---


# Customer Support Resolution — OpenEnv Environment

A production-grade reinforcement learning environment where an AI agent handles real customer support tickets across billing disputes, technical outages, and security incidents. The agent must triage, diagnose, and resolve each ticket by following professional support workflows — with partial credit for every meaningful step.


## Hugging Face Space

**Live Demo:** `https://huggingface.co/spaces/a6hxin/openenv-customer-support`


## Environment Description

### What is this environment?

Customer support is one of the most common real-world decision-making tasks for AI agents. This environment simulates a **Tier-1/Tier-2 support workflow** with three escalating difficulty levels:

| Task ID | Difficulty | Scenario | Max Steps | Pass Threshold |
|---|---|---|---|---|
| `billing_dispute_easy` | 🟢 Easy | Customer double-charged — verify, refund, close | 10 | 0.70 |
| `technical_outage_medium` | 🟡 Medium | API connectivity failure — triage root cause | 15 | 0.65 |
| `security_incident_hard` | 🔴 Hard | Suspected account compromise — follow security runbook | 20 | 0.60 |


## Observation Space
```
{
  "ticket_id": "TKT-BILL-001",
  "customer_tier": "pro",
  "issue_category": "billing",
  "description": "I was charged twice this month...",
  "priority": "medium",
  "conversation_history": [
    {"role": "customer", "content": "...", "timestamp": "..."},
    {"role": "system",   "content": "...", "timestamp": "..."}
  ],
  "current_step": 3,
  "max_steps": 10,
  "resolved": false,
  "sla_breached": false,
  "metadata": { ... }
}
```

## Reward Function

Rewards are **incremental** — the agent earns partial credit for every correct step:

### Easy (Billing Dispute)
```
account_verified       +0.20
refund_issued          +0.35
customer_notified      +0.15
ticket_closed          +0.20
efficiency_bonus       +0.10  (≤5 steps)
unnecessary_escalation  -0.05
sla_breach             -0.10
```
### Medium (Technical Outage)
```
account_verified       +0.10
status_checked         +0.15
diagnostics_gathered   +0.15
correct_fix            +0.30  (escalate OR apply_fix, depending on root cause)
customer_notified      +0.15
ticket_closed          +0.15
efficiency_bonus       +0.05
sla_breach             -0.10
```
### Hard (Security Incident)
```
identity_verified      +0.15
account_looked_up      +0.10
account_frozen         +0.20
credentials_reset      +0.20
customer_notified      +0.15
incident_documented    +0.15
runbook_order_bonus    +0.05  (all steps in correct order)
identity_skip_penalty  -0.10
out_of_order_penalty   -0.03 per violation
sla_breach             -0.10
```
## Baseline Results
```
python baseline.py

Expected output (seed=42):
[EASY]   Billing Dispute Resolution     score=0.85  ✓ PASS
[MEDIUM] Technical Outage Triage        score=0.70  ✓ PASS
[HARD]   Security Incident Response     score=0.75  ✓ PASS

Mean Score: 0.767
```

## Tests
```
pytest tests/ -v
```
## Repository Structure

```
openenv-customer-support/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI routes (all OpenEnv endpoints)
│   ├── models.py        # Pydantic typed models
│   └── environment.py   # Session engine + task registry
├── tasks/
│   ├── __init__.py
│   ├── billing_dispute_easy.py
│   ├── technical_outage_medium.py
│   └── security_incident_hard.py
├── tests/
│   └── test_environment.py
├── baseline.py          # Scripted baseline agent + standalone runner
├── validator.py         # Pre-submission checklist
├── openenv.yaml         # OpenEnv spec manifest
├── requirements.txt
├── Dockerfile
└── README.md
```

## License

MIT
