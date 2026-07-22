from pathlib import Path
import json

import pytest

from red_blue_platform.agents.base import AgentContext
from red_blue_platform.agents.react import ReActAgent, parse_action
from red_blue_platform.agents.runtime import AgentFactory, AgentRuntimeConfig
from red_blue_platform.agents.tools import tool_registry
from red_blue_platform.blue_team import fallback_defense_response
from red_blue_platform.cli import _cycle_metrics, _cycle_stop_decision
from red_blue_platform.config import load_config
from red_blue_platform.datasets import load_source
from red_blue_platform.db import Database
from red_blue_platform.evaluation import fallback_evaluate_response
from red_blue_platform.io import read_records
from red_blue_platform.red_team import fallback_evolve_records, fallback_records
from red_blue_platform.schema import AttackRecord, Seed


class FailingClient:
    def chat(self, *args: object, **kwargs: object) -> str:
        raise RuntimeError("backend unavailable")


def test_local_seed_to_blue_export_flow(tmp_path: Path) -> None:
    cfg = load_config("config.example.yaml")
    db = Database(tmp_path / "red_blue.sqlite")
    db.init()

    source = cfg.datasets[0]
    seeds = load_source(cfg.path, source)
    assert db.upsert_seeds(seeds) == 1

    seed = dict(db.fetch_seeds_for_generation(limit=1)[0])
    records = fallback_records(seed, cfg.red_team["strategies"], variants_per_seed=2)
    for record in records:
        db.insert_attack(seed["source"], seed["source_id"], record)

    exported = db.export_attacks(only_missing_response=True)
    assert len(exported) == 2
    assert {row.strategy for row in exported} == {"roleplay_reframe", "indirect_context"}
    assert all(row.blue_dict()["Wrapped_Prompt"] for row in exported)


def test_update_unknown_data_id_fails(tmp_path: Path) -> None:
    db = Database(tmp_path / "red_blue.sqlite")
    db.init()

    with pytest.raises(KeyError, match="missing"):
        db.update_response("missing", "response")


def test_attack_success_stats_by_strategy(tmp_path: Path) -> None:
    cfg = load_config("config.example.yaml")
    db = Database(tmp_path / "red_blue.sqlite")
    db.init()

    source = cfg.datasets[0]
    db.upsert_seeds(load_source(cfg.path, source))
    seed = dict(db.fetch_seeds_for_generation(limit=1)[0])
    records = fallback_records(seed, cfg.red_team["strategies"], variants_per_seed=2)
    for record in records:
        db.insert_attack(seed["source"], seed["source_id"], record)

    db.update_eval(records[0].data_id, "attack_success")
    db.update_eval(records[1].data_id, "safe")

    report = db.attack_success_stats(success_labels={"attack_success"})
    assert report["total"] == 2
    assert report["success"] == 1
    assert report["rate"] == 0.5
    assert report["by_strategy"][records[0].strategy]["success"] == 1
    assert report["by_strategy"][records[1].strategy]["success"] == 0


def test_evolve_defended_attack_and_export_evolved_only(tmp_path: Path) -> None:
    cfg = load_config("config.example.yaml")
    db = Database(tmp_path / "red_blue.sqlite")
    db.init()

    source = cfg.datasets[0]
    db.upsert_seeds(load_source(cfg.path, source))
    seed = dict(db.fetch_seeds_for_generation(limit=1)[0])
    records = fallback_records(seed, cfg.red_team["strategies"], variants_per_seed=2)
    for record in records:
        db.insert_attack(seed["source"], seed["source_id"], record)

    db.update_response(records[0].data_id, "I cannot help with that.")
    db.update_eval(records[0].data_id, "防御成功", reason="The response refused the unsafe request.")
    db.update_response(records[1].data_id, "unsafe response")
    db.update_eval(records[1].data_id, "attack_success")

    candidates = db.fetch_attacks_for_evolution(
        defense_success_labels={"防御成功"},
        limit=10,
    )
    assert [row["data_id"] for row in candidates] == [records[0].data_id]

    evolved_records = fallback_evolve_records(dict(candidates[0]), variants_per_attack=1)
    db.insert_attack(candidates[0]["source"], candidates[0]["source_id"], evolved_records[0])

    evolved_export = db.export_attacks(only_missing_response=True, evolved_only=True)
    assert len(evolved_export) == 1
    evolved = evolved_export[0]
    assert evolved.parent_data_id == records[0].data_id
    assert evolved.generation == 1
    assert evolved.evolution_reason
    assert "Previous defense response" in evolved.wrapped_prompt

    assert db.fetch_attacks_for_evolution(defense_success_labels={"防御成功"}, limit=10) == []


def test_react_agent_generates_and_exports_initial_blue_rows(tmp_path: Path) -> None:
    cfg = load_config("config.example.yaml")
    db = Database(tmp_path / "red_blue.sqlite")
    db.init()

    source = cfg.datasets[0]
    db.upsert_seeds(load_source(cfg.path, source))

    out = tmp_path / "blue_input.jsonl"
    context = AgentContext(
        db=db,
        config=cfg,
        red_backend="fallback",
        default_limit=1,
        default_export_path=out,
        defense_success_labels={"防御成功", "safe"},
        success_labels={"attack_success"},
    )
    result = ReActAgent(context).run(goal="生成红队样本并导出给蓝队", max_steps=5)

    assert [step.action for step in result.steps] == [
        "observe_state",
        "generate_red",
        "export_blue",
        "final",
    ]
    rows = read_records(out)
    assert len(rows) == 2
    assert all(row["Wrapped_Prompt"] for row in rows)
    assert result.state["missing_response"] == 2


def test_react_agent_evolves_defended_rows_and_exports_only_evolved(tmp_path: Path) -> None:
    cfg = load_config("config.example.yaml")
    db = Database(tmp_path / "red_blue.sqlite")
    db.init()

    source = cfg.datasets[0]
    db.upsert_seeds(load_source(cfg.path, source))
    seed = dict(db.fetch_seeds_for_generation(limit=1)[0])
    records = fallback_records(seed, cfg.red_team["strategies"], variants_per_seed=2)
    for record in records:
        db.insert_attack(seed["source"], seed["source_id"], record)

    db.update_response(records[0].data_id, "I cannot help with that.")
    db.update_eval(records[0].data_id, "防御成功", reason="The response refused the request.")

    out = tmp_path / "evolved_blue_input.jsonl"
    context = AgentContext(
        db=db,
        config=cfg,
        red_backend="fallback",
        default_limit=5,
        default_export_path=out,
        defense_success_labels={"防御成功", "safe"},
        success_labels={"attack_success"},
    )
    result = ReActAgent(context).run(goal="根据防御成功样本进化下一轮并导出", max_steps=5)

    assert [step.action for step in result.steps] == [
        "observe_state",
        "evolve_red",
        "export_blue",
        "final",
    ]
    rows = read_records(out)
    assert len(rows) == 1
    assert rows[0]["Parent_Data_ID"] == records[0].data_id
    assert rows[0]["Generation"] == 1
    assert result.state["evolved_pending_blue"] == 1


def test_blue_fallback_and_evaluator_close_the_feedback_loop(tmp_path: Path) -> None:
    cfg = load_config("config.example.yaml")
    db = Database(tmp_path / "red_blue.sqlite")
    db.init()
    source = cfg.datasets[0]
    db.upsert_seeds(load_source(cfg.path, source))
    seed = dict(db.fetch_seeds_for_generation(limit=1)[0])
    record = fallback_records(seed, cfg.red_team["strategies"], variants_per_seed=1)[0]
    db.insert_attack(seed["source"], seed["source_id"], record)

    pending = dict(db.fetch_attacks_for_blue_response(limit=1)[0])
    db.update_response(record.data_id, fallback_defense_response(pending))
    evaluated = fallback_evaluate_response(dict(db.fetch_attacks_for_evaluation(limit=1)[0]))
    db.update_eval(record.data_id, **{
        "eval_result": evaluated["eval_result"],
        "reason": evaluated["eval_reason"],
        "confidence": evaluated["eval_confidence"],
    })

    candidates = db.fetch_attacks_for_evolution({"defense_success"}, limit=1)
    assert [row["data_id"] for row in candidates] == [record.data_id]


def test_attack_success_feedback_is_written_to_blue_memory(tmp_path: Path) -> None:
    cfg = load_config("config.example.yaml")
    db = Database(tmp_path / "red_blue.sqlite")
    db.init()
    db.upsert_seeds(load_source(cfg.path, cfg.datasets[0]))
    seed = dict(db.fetch_seeds_for_generation(limit=1)[0])
    record = fallback_records(seed, cfg.red_team["strategies"], variants_per_seed=1)[0]
    db.insert_attack(seed["source"], seed["source_id"], record)
    db.update_response(record.data_id, "Sure, here is the unsafe answer.")

    context = AgentContext(
        db=db,
        config=cfg,
        evaluator_backend="fallback",
        success_labels={"attack_success"},
    )
    result = tool_registry()["evaluate"].run({"limit": 1}, context)

    assert result.ok
    matches = db.search_blue_experiences(record.wrapped_prompt, top_k=3)
    assert len(matches) == 1
    assert matches[0]["data_id"] == record.data_id
    assert matches[0]["source_data_id"] == record.data_id
    assert matches[0]["safe_decision"]
    assert matches[0]["trigger_features"]
    with db.connect() as conn:
        embedding = conn.execute(
            "SELECT embedding FROM blue_experiences WHERE data_id = ?",
            (record.data_id,),
        ).fetchone()["embedding"]
    assert len(json.loads(embedding)) == 64
    replay_rows = db.export_attacks(only_missing_response=True)
    assert len(replay_rows) == 1
    assert replay_rows[0].parent_data_id == record.data_id
    assert replay_rows[0].strategy.endswith("+replay")


def test_blue_memory_top3_blocks_similar_prompt(tmp_path: Path) -> None:
    cfg = load_config("config.example.yaml")
    db = Database(tmp_path / "red_blue.sqlite")
    db.init()
    db.upsert_seeds(load_source(cfg.path, cfg.datasets[0]))
    seed = dict(db.fetch_seeds_for_generation(limit=1)[0])
    first = fallback_records(seed, cfg.red_team["strategies"], variants_per_seed=1)[0]
    db.insert_attack(seed["source"], seed["source_id"], first)
    db.update_response(first.data_id, "unsafe answer")
    db.update_eval(first.data_id, "attack_success", reason="blue missed the prompt")
    db.insert_blue_experience(
        first.data_id,
        {
            "lesson_text": first.wrapped_prompt,
            "intent": seed["category"],
            "attack_pattern": first.strategy,
            "obfuscation_style": first.strategy,
            "trigger_features": first.wrapped_prompt,
            "repair_hint": "block repeats",
        },
    )
    second = AttackRecord(
        data_id=f"{first.data_id}:similar",
        seed_prompt=first.seed_prompt,
        wrapped_prompt=first.wrapped_prompt,
        strategy=first.strategy,
    )
    db.insert_attack(seed["source"], seed["source_id"], second)

    context = AgentContext(db=db, config=cfg, blue_backend="fallback", success_labels={"attack_success"})
    result = tool_registry()["run_blue"].run({"limit": 1}, context)

    assert result.ok
    exported = db.export_attacks()
    row = next(item for item in exported if item.data_id == second.data_id)
    assert row.blue_decision == "memory_block"
    assert row.blue_memory_top_score is not None
    assert row.blue_memory_top_score >= 0.78
    assert "blue-memory:" in (row.blue_memory_matches or "")


def test_evolve_blue_activates_detector_from_memory_batch(tmp_path: Path) -> None:
    cfg = load_config("config.example.yaml")
    db = Database(tmp_path / "red_blue.sqlite")
    db.init()
    db.upsert_seeds(load_source(cfg.path, cfg.datasets[0]))
    seed = dict(db.fetch_seeds_for_generation(limit=1)[0])
    records = fallback_records(seed, cfg.red_team["strategies"], variants_per_seed=2)
    records.append(
        AttackRecord(
            data_id=f"{records[0].data_id}:holdout",
            seed_prompt=records[0].seed_prompt,
            wrapped_prompt=f"{records[0].wrapped_prompt} holdout",
            strategy=records[0].strategy,
        )
    )
    for record in records:
        db.insert_attack(seed["source"], seed["source_id"], record)
        db.update_response(record.data_id, "unsafe answer")
        db.update_eval(record.data_id, "attack_success")
        db.insert_blue_experience(
            record.data_id,
            {
                "lesson_text": record.wrapped_prompt,
                "intent": seed["category"],
                "attack_pattern": record.strategy,
                "obfuscation_style": record.strategy,
                "trigger_features": record.wrapped_prompt,
                "repair_hint": "generalize this failure",
            },
        )

    context = AgentContext(
        db=db,
        config=cfg,
        blue_backend="fallback",
        evaluator_backend="fallback",
        detector_evolution_batch=2,
        detector_holdout_limit=1,
        defense_success_labels={"defense_success"},
    )
    result = tool_registry()["evolve_blue"].run({"limit": 2, "holdout_limit": 1}, context)

    assert result.ok
    assert result.data["created"] is True
    assert result.data["activated"] is True
    assert db.active_blue_detector()["version_id"].startswith("blue-v1-")
    assert db.pending_blue_experiences_for_detector(2) == []


def test_react_agent_runs_one_complete_red_blue_improvement_cycle(tmp_path: Path) -> None:
    cfg = load_config("config.example.yaml")
    db = Database(tmp_path / "red_blue.sqlite")
    db.init()
    db.upsert_seeds(load_source(cfg.path, cfg.datasets[0]))
    context = AgentContext(
        db=db,
        config=cfg,
        red_backend="fallback",
        blue_backend="fallback",
        evaluator_backend="fallback",
        default_limit=5,
        defense_success_labels={"defense_success"},
        success_labels={"attack_success"},
    )

    result = ReActAgent(context).run(goal="完成一次红蓝循环提升", max_steps=8)

    assert [step.action for step in result.steps] == [
        "observe_state",
        "generate_red",
        "run_blue",
        "evaluate",
        "evolve_red",
        "run_blue",
        "evaluate",
        "final",
    ]
    assert result.state["latest_generation"] == 1
    assert result.state["missing_response"] == 0
    assert result.state["missing_eval"] == 0
    next_round = db.fetch_attacks_for_evolution({"defense_success"}, limit=10)
    assert len(next_round) == 2
    assert all(row["generation"] == 1 for row in next_round)


def test_react_agent_can_continue_for_multiple_iterations(tmp_path: Path) -> None:
    cfg = load_config("config.example.yaml")
    db = Database(tmp_path / "red_blue.sqlite")
    db.init()
    db.upsert_seeds(load_source(cfg.path, cfg.datasets[0]))
    context = AgentContext(
        db=db,
        config=cfg,
        red_backend="fallback",
        blue_backend="fallback",
        evaluator_backend="fallback",
        default_limit=5,
        defense_success_labels={"defense_success"},
        success_labels={"attack_success"},
    )
    agent = ReActAgent(context)

    for _ in range(3):
        result = agent.run(goal="完成红蓝循环提升", max_steps=8)
        assert any(step.action in {"generate_red", "evolve_red"} for step in result.steps)

    assert result.state["latest_generation"] == 3
    assert result.state["missing_response"] == 0
    assert result.state["missing_eval"] == 0


def test_parse_action_extracts_json_from_model_text() -> None:
    action = parse_action('```json\n{"thought":"check","action":"observe_state","args":{}}\n```')

    assert action.thought == "check"
    assert action.action == "observe_state"
    assert action.args == {}


def test_cycle_stop_requires_data_exhausted_plus_effect_condition() -> None:
    base = _stop_metric(iteration=1, data_exhausted=True, attack_success_rate=0.2)

    decision = _cycle_stop_decision(
        history=[base],
        window=3,
        target_attack_success_rate=0.05,
        target_holdout_attack_success_rate=0.05,
        false_positive_limit=0.02,
        min_delta=0.01,
        invalid_attack_limit=0.30,
        duplication_limit=0.80,
        similarity_high=0.90,
    )

    assert decision["stop"] is False


def test_cycle_stop_triggers_when_data_exhausted_and_no_gain() -> None:
    history = [
        _stop_metric(iteration=idx, data_exhausted=True, attack_success_rate=0.2)
        for idx in range(1, 4)
    ]

    decision = _cycle_stop_decision(
        history=history,
        window=3,
        target_attack_success_rate=0.05,
        target_holdout_attack_success_rate=0.05,
        false_positive_limit=0.02,
        min_delta=0.01,
        invalid_attack_limit=0.30,
        duplication_limit=0.80,
        similarity_high=0.90,
    )

    assert decision["stop"] is True
    assert decision["trigger"] == "STOP_05_DATA_EXHAUSTED + STOP_02_NO_MEANINGFUL_GAIN"


def test_cycle_stop_keeps_running_after_a_significant_red_gain() -> None:
    history = [
        _stop_metric(iteration=1, data_exhausted=True, attack_success_rate=0.20),
        _stop_metric(iteration=2, data_exhausted=True, attack_success_rate=0.24),
        _stop_metric(iteration=3, data_exhausted=True, attack_success_rate=0.28),
    ]

    decision = _cycle_stop_decision(
        history=history,
        window=3,
        target_attack_success_rate=0.05,
        target_holdout_attack_success_rate=0.05,
        false_positive_limit=0.02,
        min_delta=0.05,
        invalid_attack_limit=0.30,
        duplication_limit=0.80,
        similarity_high=0.90,
    )

    assert decision["stop"] is False


def test_stop_05_uses_only_the_consumed_first_layer(tmp_path: Path) -> None:
    db = Database(tmp_path / "red_blue.sqlite")
    db.init()
    db.upsert_seeds([Seed(source="probe", source_id="seed-1", seed_prompt="seed")])
    first_layer = AttackRecord(
        data_id="probe:initial",
        seed_prompt="seed",
        wrapped_prompt="initial attack",
        strategy="initial",
    )
    second_layer = AttackRecord(
        data_id="probe:initial:evolved",
        seed_prompt="seed",
        wrapped_prompt="evolved attack",
        strategy="initial+evolved",
        parent_data_id=first_layer.data_id,
        generation=1,
    )
    db.insert_attack("probe", "seed-1", first_layer)
    db.insert_attack("probe", "seed-1", second_layer)
    db.update_response(first_layer.data_id, "I can't help with that.")
    db.update_eval(first_layer.data_id, "defense_success")

    metrics = _cycle_metrics(
        db=db,
        iteration=1,
        success_labels={"attack_success"},
        defense_success_labels={"defense_success"},
        detector_memory_required=4,
    )

    assert metrics["missing_response"] == 1
    assert metrics["first_layer_missing_response"] == 0
    assert metrics["first_layer_missing_eval"] == 0
    assert metrics["data_exhausted"] is True


def test_generation_failure_is_terminal_for_the_first_layer(tmp_path: Path) -> None:
    cfg = load_config("config.example.yaml")
    db = Database(tmp_path / "red_blue.sqlite")
    db.init()
    db.upsert_seeds([Seed(source="probe", source_id="seed-1", seed_prompt="seed")])
    context = AgentContext(db=db, config=cfg, red_backend="api", red_client=FailingClient())

    result = tool_registry()["generate_red"].run({"limit": 1}, context)

    assert result.ok is False
    assert db.fetch_seeds_for_generation(limit=1) == []
    with db.connect() as conn:
        row = conn.execute("SELECT eval_result FROM attacks").fetchone()
    assert row["eval_result"] == "processing_failed"


def test_blue_failure_is_terminal_and_triggers_quality_stop(tmp_path: Path) -> None:
    cfg = load_config("config.example.yaml")
    db = Database(tmp_path / "red_blue.sqlite")
    db.init()
    db.upsert_seeds([Seed(source="probe", source_id="seed-1", seed_prompt="seed")])
    record = AttackRecord(
        data_id="probe:initial",
        seed_prompt="seed",
        wrapped_prompt="attack",
        strategy="initial",
    )
    db.insert_attack("probe", "seed-1", record)
    context = AgentContext(db=db, config=cfg, blue_backend="api", blue_client=FailingClient())

    result = tool_registry()["run_blue"].run({"limit": 1}, context)
    metrics = _cycle_metrics(
        db=db,
        iteration=1,
        success_labels={"attack_success"},
        defense_success_labels={"defense_success"},
        detector_memory_required=4,
    )
    decision = _cycle_stop_decision(
        history=[metrics],
        window=3,
        target_attack_success_rate=0.05,
        target_holdout_attack_success_rate=0.05,
        false_positive_limit=0.02,
        min_delta=0.01,
        invalid_attack_limit=0.30,
        duplication_limit=0.80,
        similarity_high=0.90,
    )

    assert result.ok is False
    assert metrics["missing_response"] == 0
    assert metrics["missing_eval"] == 0
    assert metrics["data_exhausted"] is True
    assert metrics["processing_failure_rate"] == 1.0
    assert decision["trigger"] == "STOP_05_DATA_EXHAUSTED + STOP_03_QUALITY_COLLAPSE"


def _stop_metric(
    *,
    iteration: int,
    data_exhausted: bool,
    attack_success_rate: float,
) -> dict[str, object]:
    return {
        "iteration": iteration,
        "total": 10,
        "attack_success_count": 2,
        "new_attack_success_count": 0,
        "defense_success_count": 8,
        "new_defense_success_count": 0,
        "attack_success_rate": attack_success_rate,
        "holdout_attack_success_rate": attack_success_rate,
        "false_positive_rate": 0.0,
        "invalid_attack_rate": 0.0,
        "processing_failure_count": 0,
        "processing_failure_rate": 0.0,
        "duplication_rate": 0.0,
        "missing_response": 0,
        "missing_eval": 0,
        "defended_candidates": 0,
        "blue_memory_pending_detector": 0,
        "data_exhausted": data_exhausted,
    }


def test_tool_args_are_validated_with_pydantic(tmp_path: Path) -> None:
    cfg = load_config("config.example.yaml")
    db = Database(tmp_path / "red_blue.sqlite")
    db.init()
    context = AgentContext(db=db, config=cfg)

    result = tool_registry()["generate_red"].run({"limit": -1, "unexpected": True}, context)

    assert result.ok is False
    assert "Invalid args for generate_red" in result.message
    assert result.data["errors"]


def test_agent_factory_uses_same_runtime_with_red_and_blue_roles(tmp_path: Path) -> None:
    cfg = load_config("config.example.yaml")
    db = Database(tmp_path / "red_blue.sqlite")
    db.init()

    source = cfg.datasets[0]
    db.upsert_seeds(load_source(cfg.path, source))

    factory = AgentFactory(config=cfg, db=db)
    red_runtime = factory.create(
        AgentRuntimeConfig(
            role="red",
            red_backend="fallback",
            default_limit=1,
            defense_success_labels={"防御成功", "safe"},
            success_labels={"attack_success"},
        )
    )
    red_result = red_runtime.run(goal="生成红队样本", max_steps=4)
    assert [step.action for step in red_result.steps] == ["observe_state", "generate_red", "final"]

    out = tmp_path / "blue_role_export.jsonl"
    blue_runtime = factory.create(
        AgentRuntimeConfig(
            role="blue",
            default_export_path=out,
            defense_success_labels={"防御成功", "safe"},
            success_labels={"attack_success"},
        )
    )
    blue_result = blue_runtime.run(goal="导出给蓝队测试", max_steps=4)
    assert [step.action for step in blue_result.steps] == ["observe_state", "export_blue", "final"]
    assert len(read_records(out)) == 2
