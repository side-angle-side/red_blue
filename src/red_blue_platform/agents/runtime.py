from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from red_blue_platform.agents.base import AgentContext, AgentRole, AgentRunResult
from red_blue_platform.agents.react import ReActAgent
from red_blue_platform.agents.tools import tool_registry
from red_blue_platform.config import AppConfig
from red_blue_platform.db import Database
from red_blue_platform.llm import LocalTransformersClient, OpenAICompatibleClient


BackendName = Literal["fallback", "api", "local-transformers"]


class AgentRuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: AgentRole = "full"
    planner_backend: BackendName = "fallback"
    red_backend: BackendName = "fallback"
    blue_backend: BackendName = "fallback"
    evaluator_backend: BackendName = "fallback"
    model: str | None = None
    device_map: str = "auto"
    max_new_tokens: int = Field(default=256, gt=0)
    default_limit: int = Field(default=20, gt=0)
    blue_memory_top_k: int = Field(default=3, gt=0)
    detector_evolution_batch: int = Field(default=3, gt=0)
    detector_holdout_limit: int = Field(default=1, gt=0)
    default_export_path: Path = Path("data/agent_blue_input.jsonl")
    defense_success_labels: set[str] = Field(default_factory=set)
    success_labels: set[str] = Field(default_factory=set)


class AgentRuntime:
    def __init__(self, agent: ReActAgent):
        self.agent = agent

    def run(self, goal: str, max_steps: int = 8) -> AgentRunResult:
        return self.agent.run(goal=goal, max_steps=max_steps)


class AgentFactory:
    def __init__(self, config: AppConfig, db: Database):
        self.config = config
        self.db = db

    def create(self, runtime_config: AgentRuntimeConfig) -> AgentRuntime:
        planner_client = self._build_client(
            backend=runtime_config.planner_backend,
            model=runtime_config.model,
            device_map=runtime_config.device_map,
            max_new_tokens=runtime_config.max_new_tokens,
        )
        red_client = self._build_client(
            backend=runtime_config.red_backend,
            model=runtime_config.model,
            device_map=runtime_config.device_map,
            max_new_tokens=runtime_config.max_new_tokens,
        )
        blue_client = self._build_client(
            backend=runtime_config.blue_backend,
            model=runtime_config.model,
            device_map=runtime_config.device_map,
            max_new_tokens=runtime_config.max_new_tokens,
        )
        evaluator_client = self._build_client(
            backend=runtime_config.evaluator_backend,
            model=runtime_config.model,
            device_map=runtime_config.device_map,
            max_new_tokens=runtime_config.max_new_tokens,
        )
        context = AgentContext(
            db=self.db,
            config=self.config,
            role=runtime_config.role,
            red_client=red_client,
            red_backend=runtime_config.red_backend,
            blue_client=blue_client,
            blue_backend=runtime_config.blue_backend,
            evaluator_client=evaluator_client,
            evaluator_backend=runtime_config.evaluator_backend,
            default_limit=runtime_config.default_limit,
            blue_memory_top_k=runtime_config.blue_memory_top_k,
            detector_evolution_batch=runtime_config.detector_evolution_batch,
            detector_holdout_limit=runtime_config.detector_holdout_limit,
            default_export_path=runtime_config.default_export_path,
            defense_success_labels=runtime_config.defense_success_labels,
            success_labels=runtime_config.success_labels,
        )
        agent = ReActAgent(
            context=context,
            planner_client=planner_client,
            tools=tool_registry(runtime_config.role),
        )
        return AgentRuntime(agent)

    def _build_client(
        self,
        backend: BackendName,
        model: str | None,
        device_map: str,
        max_new_tokens: int,
    ):
        if backend == "fallback":
            return None
        if backend == "api":
            return OpenAICompatibleClient(self.config.api)
        selected_model = model or self.config.api.model
        if not selected_model:
            raise ValueError("--model is required when using local-transformers")
        return LocalTransformersClient(selected_model, device_map=device_map, max_new_tokens=max_new_tokens)
