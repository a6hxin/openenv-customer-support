"""
Typed models for the Customer Support Resolution OpenEnv environment.
All models use Pydantic v2 for strict validation.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid
from datetime import datetime


# ─── Enums ────────────────────────────────────────────────────────────────────

class CustomerTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class IssueCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    FEATURE_REQUEST = "feature_request"
    SECURITY = "security"


class ActionType(str, Enum):
    ASK_CLARIFICATION = "ask_clarification"
    LOOKUP_ACCOUNT = "lookup_account"
    APPLY_FIX = "apply_fix"
    ESCALATE = "escalate"
    SEND_RESPONSE = "send_response"
    CLOSE_TICKET = "close_ticket"
    ISSUE_REFUND = "issue_refund"
    RESET_CREDENTIALS = "reset_credentials"
    CHECK_SYSTEM_STATUS = "check_system_status"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ─── Sub-models ───────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str  # "customer", "agent", "system"
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class AccountInfo(BaseModel):
    account_id: str
    customer_name: str
    tier: CustomerTier
    email: str
    subscription_start: str
    last_payment: Optional[float] = None
    payment_status: str = "current"
    flags: List[str] = Field(default_factory=list)


class SystemStatus(BaseModel):
    service: str
    status: str  # "operational", "degraded", "outage"
    affected_regions: List[str] = Field(default_factory=list)
    incident_id: Optional[str] = None


# ─── Core API models ──────────────────────────────────────────────────────────

class Observation(BaseModel):
    """The observation returned to the agent on each step."""
    ticket_id: str
    customer_tier: CustomerTier
    issue_category: IssueCategory
    description: str
    priority: Priority
    conversation_history: List[Message] = Field(default_factory=list)
    current_step: int = 0
    max_steps: int = 15
    resolved: bool = False
    sla_breached: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    """An action submitted by the agent."""
    action_type: ActionType
    payload: Dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    """Result of calling step().
    
    Note: per-step `reward` may be negative (penalty for wrong actions).
    The grader's final `score` is always clamped to [0.0, 1.0].
    """
    observation: Observation
    reward: float = Field(ge=-1.0, le=1.0)
    done: bool
    truncated: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
    """Request body for reset()."""
    task_id: str
    seed: Optional[int] = None


class ResetResult(BaseModel):
    """Result of calling reset()."""
    observation: Observation
    task_id: str
    task_description: str
    info: Dict[str, Any] = Field(default_factory=dict)


class TaskInfo(BaseModel):
    """Metadata about a task, including action schema."""
    id: str
    name: str
    description: str
    difficulty: TaskDifficulty
    max_steps: int
    reward_threshold: float
    action_schema: Dict[str, Any]


class GraderRequest(BaseModel):
    """Request to grade a completed episode."""
    task_id: str
    episode_log: List[Dict[str, Any]]


class GraderResult(BaseModel):
    """Result of the grader."""
    task_id: str
    score: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float]
    feedback: str
    passed: bool


class BaselineResult(BaseModel):
    """Aggregated baseline scores."""
    scores: Dict[str, float]
    mean_score: float
    details: Dict[str, Any]


class EnvironmentState(BaseModel):
    """Full internal state (returned by GET /state)."""
    session_id: str
    task_id: Optional[str]
    current_observation: Optional[Observation]
    step_count: int
    total_reward: float
    done: bool
    episode_log: List[Dict[str, Any]]


# ─── Action schema (exposed via /tasks) ───────────────────────────────────────

ACTION_SCHEMA = {
    "type": "object",
    "required": ["action_type"],
    "properties": {
        "action_type": {
            "type": "string",
            "enum": [a.value for a in ActionType],
            "description": "The type of action to perform"
        },
        "payload": {
            "type": "object",
            "description": "Action-specific parameters",
            "examples": {
                "ask_clarification": {"question": "Can you describe the exact error message?"},
                "lookup_account": {"account_id": "ACC-12345"},
                "apply_fix": {"fix_type": "clear_cache", "component": "auth"},
                "escalate": {"tier": 2, "reason": "Requires infrastructure access"},
                "send_response": {"message": "I've resolved your issue. Please let me know if it persists."},
                "close_ticket": {"resolution": "Refund issued and confirmed by customer"},
                "issue_refund": {"amount": 29.99, "reason": "duplicate_charge"},
                "reset_credentials": {"method": "email", "notify_customer": True},
                "check_system_status": {"service": "auth", "region": "us-east-1"}
            }
        }
    }
}
