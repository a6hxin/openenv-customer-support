"""
tests/test_http_api.py — Live HTTP integration tests.

Requires the server to be running:
    uvicorn app.main:app --port 7860

Run with:
    pytest tests/test_http_api.py -v --base-url http://localhost:7860

Or against the deployed HF Space:
    pytest tests/test_http_api.py -v --base-url https://<space>.hf.space
"""
from __future__ import annotations
import pytest
import httpx
import json


def pytest_addoption(parser):
    parser.addoption(
        "--base-url",
        default="http://localhost:7860",
        help="Base URL of the running OpenEnv server",
    )


@pytest.fixture(scope="session")
def base_url(request):
    return request.config.getoption("--base-url").rstrip("/")


@pytest.fixture(scope="session")
def client(base_url):
    with httpx.Client(base_url=base_url, timeout=30.0) as c:
        yield c


# ── Health ─────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_body(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "environment" in data


# ── /tasks ─────────────────────────────────────────────────────────────────────

class TestTasks:
    def test_tasks_returns_200(self, client):
        r = client.get("/tasks")
        assert r.status_code == 200

    def test_tasks_has_three(self, client):
        data = client.get("/tasks").json()
        assert data["total"] == 3

    def test_tasks_ids(self, client):
        data = client.get("/tasks").json()
        ids = {t["id"] for t in data["tasks"]}
        assert "billing_dispute_easy" in ids
        assert "technical_outage_medium" in ids
        assert "security_incident_hard" in ids

    def test_tasks_action_schema(self, client):
        data = client.get("/tasks").json()
        for task in data["tasks"]:
            schema = task["action_schema"]
            assert "properties" in schema
            assert "action_type" in schema["properties"]
            assert "enum" in schema["properties"]["action_type"]

    def test_tasks_difficulty_range(self, client):
        data = client.get("/tasks").json()
        difficulties = {t["difficulty"] for t in data["tasks"]}
        assert difficulties == {"easy", "medium", "hard"}


# ── /reset ─────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_billing(self, client):
        r = client.post("/reset", json={"task_id": "billing_dispute_easy", "seed": 42})
        assert r.status_code == 200
        data = r.json()
        assert data["task_id"] == "billing_dispute_easy"
        obs = data["observation"]
        assert obs["ticket_id"] == "TKT-BILL-001"
        assert obs["current_step"] == 0
        assert obs["resolved"] is False

    def test_reset_technical(self, client):
        r = client.post("/reset", json={"task_id": "technical_outage_medium", "seed": 1})
        assert r.status_code == 200
        data = r.json()
        assert data["observation"]["issue_category"] == "technical"

    def test_reset_security(self, client):
        r = client.post("/reset", json={"task_id": "security_incident_hard", "seed": 42})
        assert r.status_code == 200
        data = r.json()
        assert data["observation"]["priority"] == "critical"

    def test_reset_invalid_task(self, client):
        r = client.post("/reset", json={"task_id": "nonexistent"})
        assert r.status_code == 400

    def test_reset_reproducible(self, client):
        r1 = client.post("/reset", json={"task_id": "billing_dispute_easy", "seed": 99})
        r2 = client.post("/reset", json={"task_id": "billing_dispute_easy", "seed": 99})
        assert r1.json()["observation"]["description"] == r2.json()["observation"]["description"]

    def test_reset_missing_task_id(self, client):
        r = client.post("/reset", json={"seed": 42})
        assert r.status_code == 422  # Pydantic validation error


# ── /step ──────────────────────────────────────────────────────────────────────

class TestStep:
    def setup_method(self, method):
        """Each test method gets a fresh episode."""
        pass

    def test_step_lookup_account(self, client):
        client.post("/reset", json={"task_id": "billing_dispute_easy", "seed": 42})
        r = client.post("/step", json={
            "action_type": "lookup_account",
            "payload": {"account_id": "ACC-7842"}
        })
        assert r.status_code == 200
        data = r.json()
        assert "reward" in data
        assert "done" in data
        assert "observation" in data
        assert data["observation"]["current_step"] == 1
        assert -1.0 <= data["reward"] <= 1.0

    def test_step_reward_positive_for_correct_action(self, client):
        client.post("/reset", json={"task_id": "billing_dispute_easy", "seed": 42})
        r = client.post("/step", json={
            "action_type": "lookup_account",
            "payload": {"account_id": "ACC-7842"}
        })
        assert r.json()["reward"] > 0

    def test_step_invalid_action_type(self, client):
        client.post("/reset", json={"task_id": "billing_dispute_easy", "seed": 42})
        r = client.post("/step", json={"action_type": "do_magic"})
        assert r.status_code == 422

    def test_full_billing_episode(self, client):
        client.post("/reset", json={"task_id": "billing_dispute_easy", "seed": 42})
        actions = [
            {"action_type": "lookup_account", "payload": {"account_id": "ACC-7842"}},
            {"action_type": "issue_refund",   "payload": {"amount": 49.99, "reason": "duplicate_charge"}},
            {"action_type": "send_response",  "payload": {"message": "Refund of $49.99 issued. Should arrive in 3-5 business days."}},
            {"action_type": "close_ticket",   "payload": {"resolution": "Duplicate charge refunded. Customer notified."}},
        ]
        total_reward = 0.0
        for act in actions:
            r = client.post("/step", json=act)
            assert r.status_code == 200
            total_reward += r.json()["reward"]
            if r.json()["done"]:
                break
        assert total_reward > 0
        assert r.json()["done"] is True

    def test_step_conversation_grows(self, client):
        client.post("/reset", json={"task_id": "billing_dispute_easy", "seed": 42})
        r0 = client.post("/step", json={"action_type": "ask_clarification",
                                         "payload": {"question": "Can you confirm the dates?"}})
        r1 = client.post("/step", json={"action_type": "lookup_account",
                                         "payload": {"account_id": "ACC-7842"}})
        h0 = len(r0.json()["observation"]["conversation_history"])
        h1 = len(r1.json()["observation"]["conversation_history"])
        assert h1 > h0


# ── /state ─────────────────────────────────────────────────────────────────────

class TestState:
    def test_state_before_reset(self, client):
        # state always returns 200 (even with no active episode)
        r = client.get("/state")
        assert r.status_code == 200

    def test_state_has_session_id(self, client):
        client.post("/reset", json={"task_id": "billing_dispute_easy", "seed": 42})
        data = client.get("/state").json()
        assert "session_id" in data
        assert data["session_id"] != ""

    def test_state_step_count_increments(self, client):
        client.post("/reset", json={"task_id": "billing_dispute_easy", "seed": 42})
        client.post("/step", json={"action_type": "lookup_account", "payload": {}})
        client.post("/step", json={"action_type": "ask_clarification",
                                    "payload": {"question": "Test?"}})
        data = client.get("/state").json()
        assert data["step_count"] == 2

    def test_state_episode_log(self, client):
        client.post("/reset", json={"task_id": "billing_dispute_easy", "seed": 42})
        client.post("/step", json={"action_type": "lookup_account", "payload": {}})
        data = client.get("/state").json()
        assert len(data["episode_log"]) == 1
        assert "action" in data["episode_log"][0]
        assert "reward" in data["episode_log"][0]


# ── /grader ────────────────────────────────────────────────────────────────────

class TestGrader:
    def _get_episode_log(self, client, task_id, actions):
        client.post("/reset", json={"task_id": task_id, "seed": 42})
        for act in actions:
            r = client.post("/step", json=act)
            if r.json()["done"]:
                break
        return client.get("/state").json()["episode_log"]

    def test_grader_billing_full_episode(self, client):
        log = self._get_episode_log(client, "billing_dispute_easy", [
            {"action_type": "lookup_account",  "payload": {"account_id": "ACC-7842"}},
            {"action_type": "issue_refund",    "payload": {"amount": 49.99, "reason": "dup"}},
            {"action_type": "send_response",   "payload": {"message": "Refund issued and confirmed!"}},
            {"action_type": "close_ticket",    "payload": {"resolution": "Duplicate charge refunded."}},
        ])
        r = client.post("/grader", json={
            "task_id": "billing_dispute_easy",
            "episode_log": log,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["task_id"] == "billing_dispute_easy"
        assert 0.0 <= data["score"] <= 1.0
        assert data["score"] >= 0.70
        assert data["passed"] is True
        assert "breakdown" in data
        assert "feedback" in data

    def test_grader_score_range(self, client):
        """Empty episode should score 0.0."""
        r = client.post("/grader", json={
            "task_id": "billing_dispute_easy",
            "episode_log": [],
        })
        assert r.status_code == 200
        assert r.json()["score"] == 0.0

    def test_grader_invalid_task(self, client):
        r = client.post("/grader", json={
            "task_id": "nonexistent",
            "episode_log": [],
        })
        assert r.status_code == 400

    def test_grader_all_tasks_return_valid_scores(self, client):
        for task_id in ["billing_dispute_easy", "technical_outage_medium", "security_incident_hard"]:
            client.post("/reset", json={"task_id": task_id, "seed": 42})
            log = client.get("/state").json()["episode_log"]
            r = client.post("/grader", json={"task_id": task_id, "episode_log": log})
            assert r.status_code == 200
            score = r.json()["score"]
            assert 0.0 <= score <= 1.0, f"{task_id}: score {score} out of range"


# ── /baseline ──────────────────────────────────────────────────────────────────

class TestBaseline:
    def test_baseline_returns_200(self, client):
        r = client.get("/baseline", timeout=120.0)
        assert r.status_code == 200

    def test_baseline_has_all_tasks(self, client):
        data = client.get("/baseline", timeout=120.0).json()
        assert "billing_dispute_easy" in data["scores"]
        assert "technical_outage_medium" in data["scores"]
        assert "security_incident_hard" in data["scores"]

    def test_baseline_scores_in_range(self, client):
        data = client.get("/baseline", timeout=120.0).json()
        for task_id, score in data["scores"].items():
            assert 0.0 <= score <= 1.0, f"{task_id}: {score} out of range"

    def test_baseline_mean_score(self, client):
        data = client.get("/baseline", timeout=120.0).json()
        assert "mean_score" in data
        assert 0.0 <= data["mean_score"] <= 1.0
        # verify mean is computed correctly
        scores = list(data["scores"].values())
        expected_mean = round(sum(scores) / len(scores), 4)
        assert abs(data["mean_score"] - expected_mean) < 0.001
