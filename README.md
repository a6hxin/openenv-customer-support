# 🎧 Customer Support Resolution — OpenEnv Environment

> **OpenEnv Challenge Submission** · Built with the [OpenEnv framework](https://github.com/huggingface/openenv) by **Meta & Hugging Face**

A production-grade reinforcement learning environment where an AI agent handles real customer support tickets across billing disputes, technical outages, and security incidents. The agent must triage, diagnose, and resolve each ticket by following professional support workflows — with partial credit for every meaningful step.

---

## 🌐 Hugging Face Space

**Live Demo:** `https://huggingface.co/spaces/<your-username>/openenv-customer-support`

The Space exposes all OpenEnv endpoints and a Swagger UI at `/docs`.

---

## 🗂 Environment Description

### What is this environment?

Customer support is one of the most common real-world decision-making tasks for AI agents. This environment simulates a **Tier-1/Tier-2 support workflow** with three escalating difficulty levels:

| Task ID | Difficulty | Scenario | Max Steps | Pass Threshold |
|---|---|---|---|---|
| `billing_dispute_easy` | 🟢 Easy | Customer double-charged — verify, refund, close | 10 | 0.70 |
| `technical_outage_medium` | 🟡 Medium | API connectivity failure — triage root cause | 15 | 0.65 |
| `security_incident_hard` | 🔴 Hard | Suspected account compromise — follow security runbook | 20 | 0.60 |

### Why is it interesting?

- **Partial ordering matters** — actions must follow logical sequences (can't refund without verifying the account; can't close without resolving)
- **Hidden state** — the root cause of the medium task is randomly assigned; the agent must discover it through probing actions
- **Policy enforcement** — security runbook violations are penalised even if the final outcome is correct
- **Natural language** — observations include real conversation history; send_response quality affects reward

---

## 📐 Observation Space

```json
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

---

## 🎮 Action Space

All actions share the same schema (returned by `GET /tasks`):

| `action_type` | Key Payload Fields | When to use |
|---|---|---|
| `ask_clarification` | `question` | Gather missing information |
| `lookup_account` | `account_id` | Verify customer identity / billing data |
| `check_system_status` | `service`, `region` | Diagnose infra issues |
| `apply_fix` | `fix_type`, `component` | Apply Tier-1 resolutions |
| `issue_refund` | `amount`, `reason` | Process billing corrections |
| `reset_credentials` | `method`, `notify_customer` | Handle auth/security issues |
| `escalate` | `tier`, `reason` | Hand off to higher tier |
| `send_response` | `message` | Communicate with the customer |
| `close_ticket` | `resolution` | Document and close the ticket |

---

## 💯 Reward Function

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

---

## 🚀 Quick Start

### Option 1 — Docker (recommended)

```bash
git clone https://github.com/<your-username>/openenv-customer-support
cd openenv-customer-support

docker build -t openenv-support .
docker run -p 7860:7860 openenv-support
```

Then visit `http://localhost:7860/docs`

### Option 2 — Local Python

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 7860
```

---

## 🔌 API Reference

### `POST /reset`
Start a new episode.
```json
{ "task_id": "billing_dispute_easy", "seed": 42 }
```

### `POST /step`
Submit an action.
```json
{
  "action_type": "lookup_account",
  "payload": { "account_id": "ACC-7842" }
}
```

### `GET /state`
Inspect full environment state including episode log.

### `GET /tasks`
List all tasks and the complete action schema.

### `POST /grader`
Score a completed episode.
```json
{
  "task_id": "billing_dispute_easy",
  "episode_log": [ ... ]
}
```

### `GET /baseline`
Run the scripted baseline agent on all 3 tasks and return scores.

### `GET /health`
Liveness check → `{"status": "ok"}`

---

## 📊 Baseline Results

Run the baseline agent:
```bash
python baseline.py
```

Expected output (seed=42):
```
[EASY]   Billing Dispute Resolution     score=0.85  ✓ PASS
[MEDIUM] Technical Outage Triage        score=0.70  ✓ PASS
[HARD]   Security Incident Response     score=0.75  ✓ PASS

Mean Score: 0.767
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## ✅ Pre-Submission Validation

```bash
python validator.py
```
All checks must pass before submitting.

---

## 📋 How Submissions Are Evaluated

According to the challenge specification:

> **Round 1** uses an LLM-based evaluator with structured rubrics. The finale includes LLM screening, manual review, and judging by Meta's global team. Evaluation criteria include **runtime correctness**, **OpenEnv interface compliance**, **task design quality**, **grading logic**, and **overall code quality**.

### What framework is used?

> All environments must be built using the **OpenEnv framework by Meta and Hugging Face**.

### What do I need to submit?

> A public GitHub repository with your environment code, a `requirements.txt`, a demo script, and a `README`. A deployed **Hugging Face Spaces URL** showcasing your working demo.

---

## 📁 Repository Structure

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

---

## 📄 License

MIT
title: {{openenv-customer-support}}
emoji: {{🕶️}}
colorFrom: {{colorFrom}}
colorTo: {{colorTo}}
sdk: {{Docker}}
sdk_version: "{{sdkVersion}}"
{{#pythonVersion}}
python_version: "{{pythonVersion}}"
{{/pythonVersion}}
app_file: app.py
pinned: false