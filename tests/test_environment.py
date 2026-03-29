"""
tests/test_environment.py — Full test suite for OpenEnv compliance.
Run with: pytest tests/ -v
"""
import pytest
from app.models import Action, ActionType, ResetRequest
from app.environment import SupportEnvironment, TASK_META


@pytest.fixture
def env():
    return SupportEnvironment()


# ── reset() ───────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_billing(self, env):
        r = env.reset(ResetRequest(task_id="billing_dispute_easy", seed=42))
        assert r.observation.ticket_id == "TKT-BILL-001"
        assert r.observation.current_step == 0
        assert not r.observation.resolved

    def test_reset_technical(self, env):
        r = env.reset(ResetRequest(task_id="technical_outage_medium", seed=42))
        assert r.observation.issue_category == "technical"

    def test_reset_security(self, env):
        r = env.reset(ResetRequest(task_id="security_incident_hard", seed=42))
        assert r.observation.priority == "critical"

    def test_reset_invalid_task(self, env):
        with pytest.raises(ValueError):
            env.reset(ResetRequest(task_id="nonexistent_task"))

    def test_reset_clears_state(self, env):
        env.reset(ResetRequest(task_id="billing_dispute_easy", seed=1))
        env.step(Action(action_type=ActionType.LOOKUP_ACCOUNT))
        env.reset(ResetRequest(task_id="billing_dispute_easy", seed=1))
        assert env.step_count == 0
        assert env.total_reward == 0.0

    def test_seed_reproducibility(self, env):
        r1 = env.reset(ResetRequest(task_id="billing_dispute_easy", seed=99))
        obs1 = r1.observation.model_dump()
        env2 = SupportEnvironment()
        r2 = env2.reset(ResetRequest(task_id="billing_dispute_easy", seed=99))
        obs2 = r2.observation.model_dump()
        assert obs1["description"] == obs2["description"]


# ── step() ────────────────────────────────────────────────────────────────────

class TestStep:
    def test_step_before_reset_raises(self, env):
        with pytest.raises(RuntimeError):
            env.step(Action(action_type=ActionType.LOOKUP_ACCOUNT))

    def test_step_returns_step_result(self, env):
        env.reset(ResetRequest(task_id="billing_dispute_easy", seed=42))
        r = env.step(Action(action_type=ActionType.LOOKUP_ACCOUNT,
                            payload={"account_id": "ACC-7842"}))
        assert r is not None
        assert 0.0 <= r.reward <= 1.0
        assert isinstance(r.done, bool)
        assert r.observation.current_step == 1

    def test_reward_range(self, env):
        env.reset(ResetRequest(task_id="billing_dispute_easy", seed=42))
        for _ in range(5):
            r = env.step(Action(action_type=ActionType.ASK_CLARIFICATION,
                                payload={"question": "Test?"}))
            assert 0.0 <= r.reward <= 1.0

    def test_step_increments_counter(self, env):
        env.reset(ResetRequest(task_id="technical_outage_medium", seed=42))
        for i in range(3):
            env.step(Action(action_type=ActionType.ASK_CLARIFICATION,
                            payload={"question": "Hello?"}))
        assert env.step_count == 3

    def test_episode_terminates_on_close(self, env):
        env.reset(ResetRequest(task_id="billing_dispute_easy", seed=42))
        env.step(Action(action_type=ActionType.LOOKUP_ACCOUNT))
        env.step(Action(action_type=ActionType.ISSUE_REFUND, payload={"amount": 49.99}))
        env.step(Action(action_type=ActionType.SEND_RESPONSE,
                        payload={"message": "Refund processed, you should see it in 3-5 days."}))
        r = env.step(Action(action_type=ActionType.CLOSE_TICKET,
                            payload={"resolution": "Duplicate charge refunded."}))
        assert r.done

    def test_step_after_done_raises(self, env):
        env.reset(ResetRequest(task_id="billing_dispute_easy", seed=42))
        # Force done
        for _ in range(15):
            if env.done:
                break
            env.step(Action(action_type=ActionType.CLOSE_TICKET,
                            payload={"resolution": "done"}))
        with pytest.raises(RuntimeError):
            env.step(Action(action_type=ActionType.LOOKUP_ACCOUNT))


# ── state() ───────────────────────────────────────────────────────────────────

class TestState:
    def test_state_before_reset(self, env):
        s = env.state()
        assert s.task_id is None
        assert s.step_count == 0

    def test_state_after_reset(self, env):
        env.reset(ResetRequest(task_id="billing_dispute_easy", seed=42))
        s = env.state()
        assert s.task_id == "billing_dispute_easy"
        assert s.session_id != ""

    def test_state_episode_log_grows(self, env):
        env.reset(ResetRequest(task_id="billing_dispute_easy", seed=42))
        env.step(Action(action_type=ActionType.LOOKUP_ACCOUNT))
        env.step(Action(action_type=ActionType.ISSUE_REFUND, payload={"amount": 49.99}))
        s = env.state()
        assert len(s.episode_log) == 2


# ── Graders ───────────────────────────────────────────────────────────────────

class TestGraders:
    def _run_task(self, task_id, actions, seed=42):
        env = SupportEnvironment()
        env.reset(ResetRequest(task_id=task_id, seed=seed))
        for a in actions:
            if env.done:
                break
            env.step(a)
        return env.grade(), env.episode_log

    def test_billing_perfect_score(self):
        actions = [
            Action(action_type=ActionType.LOOKUP_ACCOUNT, payload={"account_id": "ACC-7842"}),
            Action(action_type=ActionType.ISSUE_REFUND, payload={"amount": 49.99, "reason": "dup"}),
            Action(action_type=ActionType.SEND_RESPONSE,
                   payload={"message": "Refund of $49.99 has been issued to your account."}),
            Action(action_type=ActionType.CLOSE_TICKET,
                   payload={"resolution": "Duplicate charge confirmed and refunded."}),
        ]
        result, _ = self._run_task("billing_dispute_easy", actions)
        assert result["score"] >= 0.70
        assert 0.0 <= result["score"] <= 1.0

    def test_billing_no_actions_low_score(self):
        result, _ = self._run_task("billing_dispute_easy", [])
        assert result["score"] == 0.0

    def test_security_full_runbook(self):
        actions = [
            Action(action_type=ActionType.ASK_CLARIFICATION,
                   payload={"question": "Please verify your identity — name and email?"}),
            Action(action_type=ActionType.LOOKUP_ACCOUNT, payload={"account_id": "ACC-0091"}),
            Action(action_type=ActionType.APPLY_FIX,
                   payload={"fix_type": "freeze_account"}),
            Action(action_type=ActionType.RESET_CREDENTIALS,
                   payload={"method": "email", "notify_customer": True}),
            Action(action_type=ActionType.SEND_RESPONSE,
                   payload={"message": "Your account is secure. Please reset your password via the link sent."}),
            Action(action_type=ActionType.CLOSE_TICKET,
                   payload={"resolution": (
                       "Confirmed compromise from Tor exit node. "
                       "Identity verified, account frozen, sessions terminated, "
                       "API keys rotated, credentials reset, customer notified."
                   )}),
        ]
        result, _ = self._run_task("security_incident_hard", actions)
        assert result["score"] >= 0.60
        assert 0.0 <= result["score"] <= 1.0

    def test_security_skip_identity_penalised(self):
        # Go straight to lookup without verifying identity
        actions = [
            Action(action_type=ActionType.LOOKUP_ACCOUNT, payload={"account_id": "ACC-0091"}),
            Action(action_type=ActionType.RESET_CREDENTIALS,
                   payload={"method": "email"}),
            Action(action_type=ActionType.CLOSE_TICKET,
                   payload={"resolution": "Fixed."}),
        ]
        result, _ = self._run_task("security_incident_hard", actions)
        # Should be penalised for skipping identity check
        assert result["score"] < 0.70

    def test_all_grader_scores_in_range(self):
        for task_id in TASK_META:
            env = SupportEnvironment()
            env.reset(ResetRequest(task_id=task_id, seed=42))
            env.step(Action(action_type=ActionType.ASK_CLARIFICATION,
                            payload={"question": "Can you describe the issue?"}))
            result = env.grade()
            assert 0.0 <= result["score"] <= 1.0, (
                f"Task {task_id} score out of range: {result['score']}"
            )


# ── Task registry ─────────────────────────────────────────────────────────────

class TestTaskRegistry:
    def test_all_three_tasks_present(self):
        assert len(TASK_META) == 3

    def test_task_ids(self):
        assert "billing_dispute_easy" in TASK_META
        assert "technical_outage_medium" in TASK_META
        assert "security_incident_hard" in TASK_META

    def test_task_info_fields(self):
        for tid, info in TASK_META.items():
            assert info.id == tid
            assert info.max_steps > 0
            assert 0.0 < info.reward_threshold <= 1.0
            assert info.difficulty in ("easy", "medium", "hard")
            assert "action_type" in info.action_schema["properties"]

    def test_difficulty_progression(self):
        difficulties = {tid: info.difficulty for tid, info in TASK_META.items()}
        assert difficulties["billing_dispute_easy"] == "easy"
        assert difficulties["technical_outage_medium"] == "medium"
        assert difficulties["security_incident_hard"] == "hard"
