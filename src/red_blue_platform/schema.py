from __future__ import annotations

from dataclasses import dataclass
from typing import Any


RED_KEYS = (
    "Data_ID",
    "Seed_Prompt",
    "Wrapped_Prompt",
    "Strategy",
    "Parent_Data_ID",
    "Generation",
    "Evolution_Reason",
)
BLUE_KEYS = (
    *RED_KEYS,
    "Response",
    "Blue_Decision",
    "Blue_Memory_Top_Score",
    "Blue_Memory_Matches",
    "Blue_Detector_Version",
)
EVAL_KEYS = (*BLUE_KEYS, "Eval_Result")


@dataclass(frozen=True)
class Seed:
    source: str
    source_id: str
    seed_prompt: str
    strategy: str = ""
    category: str = ""


@dataclass(frozen=True)
class AttackRecord:
    data_id: str
    seed_prompt: str
    wrapped_prompt: str
    strategy: str
    response: str | None = None
    eval_result: str | None = None
    eval_reason: str | None = None
    eval_confidence: float | None = None
    parent_data_id: str | None = None
    generation: int = 0
    evolution_reason: str | None = None
    blue_decision: str | None = None
    blue_memory_top_score: float | None = None
    blue_memory_matches: str | None = None
    blue_detector_version_id: str | None = None

    def red_dict(self) -> dict[str, Any]:
        out = {
            "Data_ID": self.data_id,
            "Seed_Prompt": self.seed_prompt,
            "Wrapped_Prompt": self.wrapped_prompt,
            "Strategy": self.strategy,
        }
        if self.parent_data_id is not None:
            out["Parent_Data_ID"] = self.parent_data_id
            out["Generation"] = self.generation
        if self.evolution_reason is not None:
            out["Evolution_Reason"] = self.evolution_reason
        return out

    def blue_dict(self) -> dict[str, Any]:
        out = self.red_dict()
        out["Response"] = self.response
        if self.blue_decision is not None:
            out["Blue_Decision"] = self.blue_decision
        if self.blue_memory_top_score is not None:
            out["Blue_Memory_Top_Score"] = self.blue_memory_top_score
        if self.blue_memory_matches is not None:
            out["Blue_Memory_Matches"] = self.blue_memory_matches
        if self.blue_detector_version_id is not None:
            out["Blue_Detector_Version"] = self.blue_detector_version_id
        return out

    def eval_dict(self) -> dict[str, Any]:
        out = self.blue_dict()
        out["Eval_Result"] = self.eval_result
        if self.eval_reason is not None:
            out["Eval_Reason"] = self.eval_reason
        if self.eval_confidence is not None:
            out["Eval_Confidence"] = self.eval_confidence
        return out
