from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    api_key_env: str
    model: str
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    request_delay_seconds: float = 0.0


@dataclass(frozen=True)
class AppConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def db_path(self) -> Path:
        configured = self.raw.get("database", {}).get("path", "data/red_blue.sqlite")
        return (self.path.parent / configured).resolve()

    @property
    def api(self) -> ApiConfig:
        api = self.raw.get("api", {})
        return ApiConfig(
            base_url=api.get("base_url", "https://api.openai.com/v1"),
            api_key_env=api.get("api_key_env", "OPENAI_API_KEY"),
            model=api.get("model", "gpt-4.1-mini"),
            timeout_seconds=int(api.get("timeout_seconds", 60)),
            max_retries=int(api.get("max_retries", 3)),
            retry_backoff_seconds=float(api.get("retry_backoff_seconds", 2.0)),
            request_delay_seconds=float(api.get("request_delay_seconds", 0.0)),
        )

    @property
    def red_team(self) -> dict[str, Any]:
        return self.raw.get("red_team", {})

    @property
    def datasets(self) -> list[dict[str, Any]]:
        return list(self.raw.get("datasets", []))


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig(path=config_path, raw=raw)
