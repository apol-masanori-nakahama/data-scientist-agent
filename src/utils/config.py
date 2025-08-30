from __future__ import annotations
import os
from dataclasses import dataclass


def _get_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppConfig:
    anthropic_api_key: str | None
    openai_api_key: str | None
    provider: str
    model: str
    max_tokens: int
    fast_test: bool
    no_multiproc: bool
    s3_bucket: str | None
    s3_prefix: str
    insight_rounds: int

    @staticmethod
    def load() -> "AppConfig":
        return AppConfig(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            provider=os.getenv("LLM_PROVIDER", "anthropic"),
            model=os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            fast_test=_get_bool("FAST_TEST", False),
            no_multiproc=_get_bool("NO_MULTIPROC", False),
            s3_bucket=os.getenv("S3_BUCKET"),
            s3_prefix=os.getenv("S3_PREFIX", "data-scientist-agent/"),
            insight_rounds=int(os.getenv("INSIGHT_ROUNDS", "3")),
        )

