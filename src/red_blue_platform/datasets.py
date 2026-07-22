from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .io import read_records
from .schema import Seed


def load_source(config_path: Path, source_config: dict[str, Any]) -> list[Seed]:
    source_type = source_config.get("type")
    if source_type == "local":
        return list(_load_local(config_path, source_config))
    if source_type == "huggingface":
        return list(_load_huggingface(source_config))
    raise ValueError(f"Unsupported dataset source type: {source_type!r}")


def _load_local(config_path: Path, source_config: dict[str, Any]) -> Iterable[Seed]:
    path = Path(source_config["path"])
    if not path.is_absolute():
        path = config_path.parent / path
    records = read_records(path)
    yield from _normalize_records(source_config, records)


def _load_huggingface(source_config: dict[str, Any]) -> Iterable[Seed]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the optional dependency with: pip install -e '.[datasets]'") from exc

    dataset = load_dataset(
        source_config["path"],
        source_config.get("subset"),
        split=source_config.get("split", "train"),
    )
    limit = source_config.get("limit")
    if limit:
        dataset = dataset.select(range(min(int(limit), len(dataset))))
    yield from _normalize_records(source_config, dataset)


def _normalize_records(source_config: dict[str, Any], records: Iterable[dict[str, Any]]) -> Iterable[Seed]:
    fields = source_config.get("fields", {})
    source = source_config["name"]
    id_field = fields.get("id", "id")
    prompt_field = fields.get("prompt", "prompt")
    strategy_field = fields.get("strategy", "strategy")
    category_field = fields.get("category", "category")

    for idx, record in enumerate(records):
        prompt = str(record.get(prompt_field) or "").strip()
        if not prompt:
            continue
        source_id = str(record.get(id_field) or f"{source}:{idx}")
        strategy = _stringify_metadata(record.get(strategy_field, ""))
        category = _stringify_metadata(record.get(category_field, ""))
        yield Seed(
            source=source,
            source_id=source_id,
            seed_prompt=prompt,
            strategy=strategy,
            category=category,
        )


def _stringify_metadata(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)

