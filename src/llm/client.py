# src/llm/client.py
from __future__ import annotations
from typing import List, Dict, Any, cast
import os
from dotenv import load_dotenv

load_dotenv()

def _normalize_anthropic_model(name: str) -> str:
    n = name.strip().lower()
    aliases = {
        "claude-4-sonnet": "claude-sonnet-4-20250514",
        "claude-4-sonnet-latest": "claude-sonnet-4-20250514",
        "claude-sonnet-4": "claude-sonnet-4-20250514",
        "claude-sonnet-4-latest": "claude-sonnet-4-20250514",
    }
    return aliases.get(n, name)

class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None, temperature: float = 0.2):
        self.provider = provider or os.getenv("LLM_PROVIDER", "anthropic").lower()
        default_model = "claude-sonnet-4-20250514" if self.provider == "anthropic" else "gpt-4o-mini"
        selected = model or os.getenv("LLM_MODEL", default_model)
        if self.provider == "anthropic":
            selected = _normalize_anthropic_model(selected)
        self.model = selected
        self.temperature = temperature
        self._client: Any = None
        if self.provider == "openai":
            try:
                key = os.getenv("OPENAI_API_KEY")
                if not key:
                    self._client = None
                else:
                    from openai import OpenAI
                    self._client = OpenAI(api_key=key)
            except Exception:
                self._client = None
        elif self.provider == "anthropic":
            try:
                key = os.getenv("ANTHROPIC_API_KEY")
                if not key:
                    self._client = None
                else:
                    import anthropic
                    self._client = anthropic.Anthropic(api_key=key)
            except Exception:
                self._client = None

    def chat(self, messages: List[Dict[str, str]], tools: Any = None, tool_choice: Any = None) -> str:
        if self._client is None:
            raise RuntimeError("LLM client not available")
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))
        if self.provider == "openai":
            resp = cast(Any, self._client).chat.completions.create(  # type: ignore[reportUnknownMemberType]
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice
            )
            return resp.choices[0].message.content or ""
        if self.provider == "anthropic":
            # Convert OpenAI-style messages to Anthropic format
            sys_prompt = "\n".join(m["content"] for m in messages if m["role"]=="system") if messages else None
            user_messages = [m for m in messages if m["role"]!="system"]
            resp = cast(Any, self._client).messages.create(  # type: ignore[reportUnknownMemberType]
                model=self.model,
                max_tokens=max_tokens,
                temperature=self.temperature,
                system=sys_prompt,
                messages=[{"role": m["role"], "content": m["content"]} for m in user_messages]
            )
            # Flatten text content
            parts = []
            for c in resp.content:
                if getattr(c, "type", "text") == "text":
                    parts.append(getattr(c, "text", ""))
                elif isinstance(c, dict):
                    parts.append(c.get("text", ""))
            return "\n".join([p for p in parts if p])
        raise NotImplementedError("Unsupported provider")