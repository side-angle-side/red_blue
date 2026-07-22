from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from red_blue_platform.config import AppConfig
from red_blue_platform.db import Database


AgentRole = Literal["full", "red", "blue"]


class AgentContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    db: Database
    config: AppConfig
    role: AgentRole = "full"
    red_client: Any | None = None
    red_backend: str = "fallback"
    blue_client: Any | None = None
    blue_backend: str = "fallback"
    evaluator_client: Any | None = None
    evaluator_backend: str = "fallback"
    default_limit: int = 20
    blue_memory_top_k: int = 3
    detector_evolution_batch: int = 3
    detector_holdout_limit: int = 1
    default_export_path: Path = Path("data/agent_blue_input.jsonl")
    defense_success_labels: set[str] = Field(default_factory=set)
    success_labels: set[str] = Field(default_factory=set)


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    ok: bool
    message: str
    data: dict[str, Any] = Field(default_factory=dict)

    def to_observation(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "message": self.message,
            "data": self.data,
        }


class AgentAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: str
    args: dict[str, Any] = Field(default_factory=dict)
    thought: str = ""

    @field_validator("action")
    @classmethod
    def action_must_not_be_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("action must not be empty")
        return normalized


class AgentStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    thought: str
    action: str
    args: dict[str, Any]
    observation: ToolResult


class AgentRunResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    final_message: str
    steps: list[AgentStep]
    state: dict[str, Any]


class ObserveStateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GenerateRedArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    limit: int | None = Field(default=None, gt=0)


class EvolveRedArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    limit: int | None = Field(default=None, gt=0)
    variants_per_attack: int | None = Field(default=None, gt=0)
    include_already_evolved: bool = False


class ExportBlueArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    out: str | None = None
    evolved_only: bool = False
    all: bool = False


class AttackReportArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunBlueArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    limit: int | None = Field(default=None, gt=0)


class EvaluateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    limit: int | None = Field(default=None, gt=0)


class EvolveBlueArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    limit: int | None = Field(default=None, gt=0)
    holdout_limit: int | None = Field(default=None, gt=0)


class FinalArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = "Agent run finished."
