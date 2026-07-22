from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from red_blue_platform.agents.base import (
    AgentContext,
    AgentRole,
    AttackReportArgs,
    EvolveBlueArgs,
    EvolveRedArgs,
    EvaluateArgs,
    ExportBlueArgs,
    FinalArgs,
    GenerateRedArgs,
    ObserveStateArgs,
    RunBlueArgs,
    ToolResult,
)
from red_blue_platform.blue_memory import (
    build_blue_experience,
    evolve_detector_prompt,
    run_layered_blue_defense,
)
from red_blue_platform.evaluation import evaluate_response, fallback_evaluate_response
from red_blue_platform.io import write_jsonl
from red_blue_platform.red_team import fallback_evolve_records, fallback_records, evolve_records, generate_records


ToolHandler = Callable[[BaseModel, AgentContext], ToolResult]


class AgentTool:
    def __init__(
        self,
        name: str,
        description: str,
        args_model: type[BaseModel],
        handler: ToolHandler,
    ):
        self.name = name
        self.description = description
        self.args_model = args_model
        self.handler = handler

    def run(self, args: dict[str, Any], context: AgentContext) -> ToolResult:
        try:
            validated_args = self.args_model.model_validate(args)
        except ValidationError as exc:
            return ToolResult(
                ok=False,
                message=f"Invalid args for {self.name}: {exc.errors()}",
                data={"errors": exc.errors()},
            )
        return self.handler(validated_args, context)

    def description_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": self.args_model.model_json_schema(),
        }


def tool_registry(role: AgentRole = "full") -> dict[str, AgentTool]:
    all_tools = {
        "observe_state": AgentTool(
            name="observe_state",
            description="Read a compact database snapshot before deciding the next step.",
            args_model=ObserveStateArgs,
            handler=observe_state,
        ),
        "generate_red": AgentTool(
            name="generate_red",
            description="Generate initial red-team attack rows from seed rows without attacks.",
            args_model=GenerateRedArgs,
            handler=generate_red,
        ),
        "evolve_red": AgentTool(
            name="evolve_red",
            description="Generate stronger child rows from attacks whose Eval_Result means defense success.",
            args_model=EvolveRedArgs,
            handler=evolve_red,
        ),
        "evolve_blue": AgentTool(
            name="evolve_blue",
            description="Evolve the blue detector prompt from attack-success experience memories.",
            args_model=EvolveBlueArgs,
            handler=evolve_blue,
        ),
        "export_blue": AgentTool(
            name="export_blue",
            description="Export rows for blue-team testing.",
            args_model=ExportBlueArgs,
            handler=export_blue,
        ),
        "run_blue": AgentTool(
            name="run_blue",
            description="Generate blue-team defensive responses for attack rows that have no response.",
            args_model=RunBlueArgs,
            handler=run_blue,
        ),
        "evaluate": AgentTool(
            name="evaluate",
            description="Evaluate completed blue-team responses that do not yet have Eval_Result.",
            args_model=EvaluateArgs,
            handler=evaluate,
        ),
        "attack_report": AgentTool(
            name="attack_report",
            description="Summarize attack success rate from Eval_Result labels.",
            args_model=AttackReportArgs,
            handler=attack_report,
        ),
        "final": AgentTool(
            name="final",
            description="Stop the agent and return a user-facing summary.",
            args_model=FinalArgs,
            handler=final,
        ),
    }
    role_tools = {
        "full": {
            "observe_state",
            "generate_red",
            "evolve_red",
            "evolve_blue",
            "export_blue",
            "run_blue",
            "evaluate",
            "attack_report",
            "final",
        },
        "red": {"observe_state", "generate_red", "evolve_red", "attack_report", "final"},
        "blue": {"observe_state", "evolve_blue", "export_blue", "run_blue", "evaluate", "attack_report", "final"},
    }[role]
    return {name: tool for name, tool in all_tools.items() if name in role_tools}


def tool_descriptions(tools: dict[str, AgentTool]) -> list[dict[str, Any]]:
    return [tool.description_dict() for tool in tools.values()]


def observe_state(args: ObserveStateArgs, context: AgentContext) -> ToolResult:
    snapshot = state_snapshot(context)
    return ToolResult(ok=True, message="Observed current database state.", data=snapshot)


def state_snapshot(context: AgentContext) -> dict[str, Any]:
    with context.db.connect() as conn:
        seed_count = int(conn.execute("SELECT COUNT(*) FROM seeds").fetchone()[0])
        attack_count = int(conn.execute("SELECT COUNT(*) FROM attacks").fetchone()[0])
        seeds_pending_red = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM seeds s
                WHERE NOT EXISTS (
                  SELECT 1 FROM attacks a
                  WHERE a.source = s.source AND a.source_id = s.source_id
                )
                """
            ).fetchone()[0]
        )
        missing_response = int(
            conn.execute("SELECT COUNT(*) FROM attacks WHERE response IS NULL").fetchone()[0]
        )
        missing_eval = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM attacks
                WHERE response IS NOT NULL
                  AND TRIM(response) != ''
                  AND (eval_result IS NULL OR TRIM(eval_result) = '')
                """
            ).fetchone()[0]
        )
        evolved_pending_blue = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM attacks
                WHERE parent_data_id IS NOT NULL
                  AND response IS NULL
                """
            ).fetchone()[0]
        )
        latest_generation = int(
            conn.execute("SELECT COALESCE(MAX(generation), 0) FROM attacks").fetchone()[0] or 0
        )
        eval_results = {
            row["value"]: int(row["count"])
            for row in conn.execute(
                """
                SELECT COALESCE(eval_result, '') AS value, COUNT(*) AS count
                FROM attacks
                GROUP BY value
                """
            ).fetchall()
        }
        blue_memory_count = int(conn.execute("SELECT COUNT(*) FROM blue_experiences").fetchone()[0])
        blue_memory_pending_detector = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM blue_experiences
                WHERE detector_training_used = 0
                  AND detector_holdout_used = 0
                """
            ).fetchone()[0]
        )
        active_row = conn.execute(
            """
            SELECT version_id
            FROM blue_detector_versions
            WHERE is_active = 1
            ORDER BY generation DESC, created_at DESC
            LIMIT 1
            """
        ).fetchone()
        active_blue_detector_version = active_row["version_id"] if active_row else "blue-v0"

    defended_candidates = len(
        context.db.fetch_attacks_for_evolution(
            defense_success_labels=context.defense_success_labels,
            limit=100000,
        )
    )
    return {
        "seed_count": seed_count,
        "attack_count": attack_count,
        "seeds_pending_red": seeds_pending_red,
        "missing_response": missing_response,
        "missing_eval": missing_eval,
        "defended_candidates": defended_candidates,
        "evolved_pending_blue": evolved_pending_blue,
        "latest_generation": latest_generation,
        "eval_results": eval_results,
        "blue_memory_count": blue_memory_count,
        "blue_memory_pending_detector": blue_memory_pending_detector,
        "active_blue_detector_version": active_blue_detector_version,
    }


def generate_red(args: GenerateRedArgs, context: AgentContext) -> ToolResult:
    limit = args.limit or context.default_limit
    red_cfg = context.config.red_team
    variants_per_seed = _positive_int(red_cfg.get("variants_per_seed"), 2)
    strategies = red_cfg.get("strategies", [])
    inserted = 0
    failed = 0
    for seed in context.db.fetch_seeds_for_generation(limit):
        seed_dict = dict(seed)
        try:
            records = (
                fallback_records(seed_dict, strategies, variants_per_seed)
                if context.red_backend == "fallback"
                else generate_records(context.red_client, seed_dict, strategies, variants_per_seed)
            )
        except Exception as exc:
            failed += 1
            context.db.insert_generation_failure(
                seed_dict,
                f"Red-team generation failed: {type(exc).__name__}",
            )
            continue
        for record in records:
            context.db.insert_attack(seed["source"], seed["source_id"], record)
            inserted += 1
    return ToolResult(
        ok=failed == 0,
        message=f"Generated {inserted} red-team rows.",
        data={"inserted": inserted, "failed": failed, "limit": limit},
    )


def evolve_red(args: EvolveRedArgs, context: AgentContext) -> ToolResult:
    limit = args.limit or context.default_limit
    variants_per_attack = args.variants_per_attack or 1
    include_already_evolved = args.include_already_evolved
    candidates = context.db.fetch_attacks_for_evolution(
        defense_success_labels=context.defense_success_labels,
        limit=limit,
        include_already_evolved=include_already_evolved,
    )
    inserted = 0
    failed = 0
    for attack in candidates:
        attack_dict = dict(attack)
        try:
            records = (
                fallback_evolve_records(attack_dict, variants_per_attack)
                if context.red_backend == "fallback"
                else evolve_records(context.red_client, attack_dict, variants_per_attack)
            )
        except Exception as exc:
            failed += 1
            context.db.insert_evolution_failure(
                attack_dict,
                f"Red-team evolution failed: {type(exc).__name__}",
            )
            continue
        for record in records:
            context.db.insert_attack(attack["source"], attack["source_id"], record)
            inserted += 1
    return ToolResult(
        ok=failed == 0,
        message=f"Evolved {inserted} red-team rows from {len(candidates)} defended attacks.",
        data={
            "inserted": inserted,
            "source_rows": len(candidates),
            "failed": failed,
            "limit": limit,
        },
    )


def export_blue(args: ExportBlueArgs, context: AgentContext) -> ToolResult:
    out = Path(str(args.out or context.default_export_path))
    if not out.is_absolute():
        out = (context.config.path.parent / out).resolve()
    evolved_only = args.evolved_only
    include_all = args.all
    records = context.db.export_attacks(only_missing_response=not include_all, evolved_only=evolved_only)
    count = write_jsonl(out, (record.blue_dict() for record in records))
    return ToolResult(
        ok=True,
        message=f"Exported {count} rows to {out}.",
        data={"count": count, "out": str(out), "evolved_only": evolved_only, "all": include_all},
    )


def run_blue(args: RunBlueArgs, context: AgentContext) -> ToolResult:
    limit = args.limit or context.default_limit
    inserted = 0
    failed = 0
    for attack in context.db.fetch_attacks_for_blue_response(limit):
        try:
            attack_dict = dict(attack)
            memories = context.db.search_blue_experiences(
                attack_dict["wrapped_prompt"],
                top_k=context.blue_memory_top_k,
            )
            detector = context.db.active_blue_detector()
            blue_result = run_layered_blue_defense(
                client=context.blue_client,
                attack=attack_dict,
                memories=memories,
                detector_prompt=detector["system_prompt"],
                backend=context.blue_backend,
            )
            context.db.update_response(
                attack["data_id"],
                blue_result["response"],
                blue_decision=blue_result["decision"],
                blue_memory_top_score=blue_result["top_score"],
                blue_memory_matches=_json_dumps(blue_result["memories"]),
                blue_detector_version_id=detector["version_id"],
            )
            inserted += 1
        except Exception as exc:
            failed += 1
            context.db.mark_attack_processing_failed(
                str(attack["data_id"]),
                f"Blue-team response failed: {type(exc).__name__}",
            )
    return ToolResult(
        ok=failed == 0,
        message=f"Generated {inserted} blue-team responses.",
        data={"updated": inserted, "failed": failed, "limit": limit},
    )


def evaluate(args: EvaluateArgs, context: AgentContext) -> ToolResult:
    limit = args.limit or context.default_limit
    updated = 0
    failed = 0
    for attack in context.db.fetch_attacks_for_evaluation(limit):
        try:
            result = (
                fallback_evaluate_response(dict(attack))
                if context.evaluator_backend == "fallback"
                else evaluate_response(context.evaluator_client, dict(attack))
            )
            context.db.update_eval(
                attack["data_id"],
                result["eval_result"],
                result["eval_reason"],
                result["eval_confidence"],
            )
            if _normalize_label(result["eval_result"]) in context.success_labels:
                attack_dict = {**dict(attack), **result}
                context.db.insert_blue_experience(
                    attack["data_id"],
                    build_blue_experience(attack_dict),
                )
                context.db.insert_replay_attack(attack_dict)
            updated += 1
        except Exception as exc:
            failed += 1
            context.db.mark_attack_processing_failed(
                str(attack["data_id"]),
                f"Evaluation failed: {type(exc).__name__}",
            )
    return ToolResult(
        ok=failed == 0,
        message=f"Evaluated {updated} blue-team responses.",
        data={"updated": updated, "failed": failed, "limit": limit},
    )


def evolve_blue(args: EvolveBlueArgs, context: AgentContext) -> ToolResult:
    limit = args.limit or context.detector_evolution_batch
    holdout_limit = args.holdout_limit or context.detector_holdout_limit
    training_rows, holdout_rows = context.db.pending_blue_experience_split(limit, holdout_limit)
    lessons = [dict(row) for row in training_rows]
    holdouts = [dict(row) for row in holdout_rows]
    if len(lessons) < limit or len(holdouts) < holdout_limit:
        return ToolResult(
            ok=True,
            message=(
                "Blue detector evolution skipped; "
                f"training={len(lessons)}/{limit}, holdout={len(holdouts)}/{holdout_limit}."
            ),
            data={
                "created": False,
                "training_available": len(lessons),
                "holdout_available": len(holdouts),
                "limit": limit,
                "holdout_limit": holdout_limit,
            },
        )
    parent = context.db.active_blue_detector()
    evolved = evolve_detector_prompt(
        client=context.blue_client,
        current_prompt=parent["system_prompt"],
        lessons=lessons,
        backend=context.blue_backend,
    )
    generation = int(parent["generation"]) + 1
    version_id = f"blue-v{generation}-{len(evolved['system_prompt'])}-{len(lessons)}"
    memory_ids = [str(row["memory_id"]) for row in lessons]
    holdout_ids = [str(row["memory_id"]) for row in holdouts]
    successful_holdouts = _validate_evolved_detector_holdouts(
        context=context,
        detector_prompt=evolved["system_prompt"],
        holdouts=holdouts,
    )
    activated = successful_holdouts == len(holdouts)
    context.db.insert_blue_detector_version(
        version_id=version_id,
        parent_version_id=parent["version_id"],
        generation=generation,
        system_prompt=evolved["system_prompt"],
        evolution_reason=evolved["evolution_reason"],
        training_memory_ids=memory_ids,
        holdout_memory_ids=holdout_ids,
        activate=activated,
    )
    status = "activated" if activated else "rejected"
    return ToolResult(
        ok=True,
        message=(
            f"Evolved blue detector {version_id} {status}; "
            f"holdout={successful_holdouts}/{len(holdouts)}."
        ),
        data={
            "created": True,
            "activated": activated,
            "version_id": version_id,
            "training_memory_ids": memory_ids,
            "holdout_memory_ids": holdout_ids,
            "successful_holdouts": successful_holdouts,
        },
    )


def attack_report(args: AttackReportArgs, context: AgentContext) -> ToolResult:
    report = context.db.attack_success_stats(success_labels=context.success_labels)
    return ToolResult(ok=True, message="Computed attack report.", data=report)


def final(args: FinalArgs, context: AgentContext) -> ToolResult:
    summary = args.summary
    return ToolResult(ok=True, message=summary, data={"summary": summary})


def _positive_int(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _normalize_label(value: str) -> str:
    return value.strip().casefold().replace("-", "_").replace(" ", "_")


def _validate_evolved_detector_holdouts(
    *,
    context: AgentContext,
    detector_prompt: str,
    holdouts: list[dict[str, Any]],
) -> int:
    successes = 0
    for holdout in holdouts:
        attack = {
            "data_id": holdout["data_id"],
            "seed_prompt": "",
            "wrapped_prompt": holdout["lesson_text"],
            "strategy": holdout["attack_pattern"],
            "category": holdout["intent"],
        }
        blue_result = run_layered_blue_defense(
            client=context.blue_client,
            attack=attack,
            memories=[],
            detector_prompt=detector_prompt,
            backend=context.blue_backend,
        )
        evaluation_input = {
            **attack,
            "response": blue_result["response"],
        }
        evaluation = (
            fallback_evaluate_response(evaluation_input)
            if context.evaluator_backend == "fallback"
            else evaluate_response(context.evaluator_client, evaluation_input)
        )
        successes += int(_normalize_label(evaluation["eval_result"]) in context.defense_success_labels)
    return successes
