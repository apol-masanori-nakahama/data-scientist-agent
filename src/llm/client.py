# src/llm/client.py
from __future__ import annotations
from typing import List, Dict, Any, cast, Callable
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
        # Default model per provider (OpenAI -> GPT-5)
        default_model = "claude-sonnet-4-20250514" if self.provider == "anthropic" else "gpt-5"
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

    def is_ready(self, deep: bool = False) -> bool:
        if self._client is None:
            return False
        if not deep:
            return True
        try:
            if self.provider == "openai":
                try:
                    _ = getattr(self._client, "models").retrieve(self.model)  # type: ignore[attr-defined]
                    return True
                except Exception:
                    try:
                        resp = getattr(self._client, "responses").create(  # type: ignore[attr-defined]
                            model=self.model,
                            input="ping",
                            max_output_tokens=1,
                        )
                        return bool(resp)
                    except Exception:
                        return False
            elif self.provider == "anthropic":
                try:
                    _ = getattr(self._client, "messages").create(  # type: ignore[attr-defined]
                        model=self.model,
                        max_tokens=1,
                        temperature=0.0,
                        messages=[{"role": "user", "content": "ping"}],
                    )
                    return True
                except Exception:
                    return False
            return True
        except Exception:
            return False

    def chat(self, messages: List[Dict[str, str]], tools: Any = None, tool_choice: Any = None) -> str:
        if self._client is None:
            raise RuntimeError("LLM client not available")
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))
        if self.provider == "openai":
            try:
                # Try Chat Completions first (works for many models)
                token_arg_name = "max_completion_tokens" if str(self.model).lower().startswith("gpt-5") else "max_tokens"
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    token_arg_name: max_tokens,
                }
                if not str(self.model).lower().startswith("gpt-5"):
                    kwargs["temperature"] = self.temperature
                if tools is not None:
                    kwargs["tools"] = tools
                if tool_choice is not None:
                    kwargs["tool_choice"] = tool_choice
                resp = cast(Any, self._client).chat.completions.create(  # type: ignore[reportUnknownMemberType]
                    **kwargs
                )
                return resp.choices[0].message.content or ""
            except Exception:
                # Fallback to Responses API (common for GPT-5 series)
                def _flatten_msgs(msgs: List[Dict[str, str]]) -> str:
                    parts = []
                    for m in msgs:
                        role = m.get("role", "user")
                        content = m.get("content", "")
                        if role == "system":
                            parts.append(f"[system]\n{content}")
                        elif role == "user":
                            parts.append(f"[user]\n{content}")
                        else:
                            parts.append(f"[{role}]\n{content}")
                    return "\n\n".join(parts)
                flat = _flatten_msgs(messages)
                # Prefer `max_output_tokens` in Responses API (some SDKs accept either name)
                try:
                    resp = cast(Any, self._client).responses.create(  # type: ignore[reportUnknownMemberType]
                        model=self.model,
                        input=flat,
                        max_output_tokens=max_tokens,
                    )
                except Exception:
                    # Last fallback without token cap
                    resp = cast(Any, self._client).responses.create(  # type: ignore[reportUnknownMemberType]
                        model=self.model,
                        input=flat,
                    )
                # Extract text robustly
                if hasattr(resp, "output_text"):
                    return cast(Any, resp).output_text  # type: ignore[return-value]
                data = getattr(resp, "output", None) or getattr(resp, "data", None) or []
                if isinstance(data, list) and data:
                    first = data[0]
                    # SDK shape may vary
                    if isinstance(first, dict):
                        content = first.get("content")
                        if isinstance(content, list) and content and isinstance(content[0], dict):
                            return str(content[0].get("text", ""))
                    else:
                        # Try attribute style
                        content = getattr(first, "content", None)
                        if isinstance(content, list) and content:
                            return str(getattr(content[0], "text", ""))
                return ""
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

    def chat_stream(self, messages: List[Dict[str, str]], on_delta: Callable[[str], None], tools: Any = None, tool_choice: Any = None) -> str:
        """Stream tokens for the given messages. Calls on_delta(text_chunk) repeatedly.

        Returns the final text. Falls back to non-streaming if provider/SDK doesn't support it.
        """
        if self._client is None:
            # No client -> no streaming possible
            full = self.chat(messages, tools=tools, tool_choice=tool_choice)
            if full:
                on_delta(full)
            return full
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))
        acc: list[str] = []
        def _push(txt: str) -> None:
            if not txt:
                return
            acc.append(txt)
            try:
                on_delta(txt)
            except Exception:
                # UI update failures shouldn't break streaming
                pass
        if self.provider == "openai":
            # Prefer Chat Completions streaming
            try:
                token_arg_name = "max_completion_tokens" if str(self.model).lower().startswith("gpt-5") else "max_tokens"
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    token_arg_name: max_tokens,
                    "stream": True,
                }
                if not str(self.model).lower().startswith("gpt-5"):
                    kwargs["temperature"] = self.temperature
                if tools is not None:
                    kwargs["tools"] = tools
                if tool_choice is not None:
                    kwargs["tool_choice"] = tool_choice
                stream = self._client.chat.completions.create(**kwargs)  # type: ignore[reportUnknownMemberType]
                for ev in stream:  # type: ignore[reportUnknownVariableType]
                    try:
                        choice = getattr(ev, "choices", None)
                        if choice and len(choice) > 0:
                            delta = getattr(choice[0], "delta", None)
                            if delta is not None:
                                # delta.content may be str or list
                                content = getattr(delta, "content", None)
                                if isinstance(content, str):
                                    _push(content)
                                elif isinstance(content, list):
                                    for c in content:
                                        if isinstance(c, dict) and c.get("type") == "text":
                                            _push(str(c.get("text", "")))
                    except Exception:
                        continue
                return "".join(acc)
            except Exception:
                # Try Responses streaming if available
                try:
                    # Flatten messages first
                    def _flatten_msgs(msgs: List[Dict[str, str]]) -> str:
                        parts = []
                        for m in msgs:
                            role = m.get("role", "user")
                            content = m.get("content", "")
                            if role == "system":
                                parts.append(f"[system]\n{content}")
                            elif role == "user":
                                parts.append(f"[user]\n{content}")
                            else:
                                parts.append(f"[{role}]\n{content}")
                        return "\n\n".join(parts)
                    flat = _flatten_msgs(messages)
                    stream_ctx = getattr(self._client, "responses").stream  # type: ignore[attr-defined]
                    # Some SDKs use context manager API
                    with stream_ctx(model=self.model, input=flat) as stream:  # type: ignore[reportUnknownMemberType]
                        for event in stream:  # type: ignore[reportUnknownVariableType]
                            try:
                                et = getattr(event, "type", "")
                                if et.endswith("output_text.delta"):
                                    delta = getattr(event, "delta", "")
                                    _push(str(delta))
                            except Exception:
                                continue
                        try:
                            final = stream.get_final_response()  # type: ignore[attr-defined]
                            # Best effort to extract final text
                            if hasattr(final, "output_text"):
                                text = getattr(final, "output_text")
                                if text:
                                    return str(text)
                        except Exception:
                            pass
                    return "".join(acc)
                except Exception:
                    # Fall back to non-streaming
                    full = self.chat(messages, tools=tools, tool_choice=tool_choice)
                    _push(full)
                    return full
        elif self.provider == "anthropic":
            # Convert OpenAI-style messages to Anthropic format and stream
            try:
                sys_prompt = "\n".join(m["content"] for m in messages if m["role"]=="system") if messages else None
                user_messages = [m for m in messages if m["role"]!="system"]
                stream_ctx = getattr(self._client, "messages").stream  # type: ignore[attr-defined]
                with stream_ctx(  # type: ignore[reportUnknownMemberType]
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    system=sys_prompt,
                    messages=[{"role": m["role"], "content": m["content"]} for m in user_messages]
                ) as stream:
                    try:
                        for text in stream.text_stream:  # type: ignore[reportUnknownMemberType]
                            _push(str(text))
                    except Exception:
                        # Some SDK versions provide event stream instead
                        for ev in stream:  # type: ignore[reportUnknownVariableType]
                            try:
                                if getattr(ev, "type", "") == "content_block_delta":
                                    _push(str(getattr(ev, "delta", {}).get("text", "")))
                            except Exception:
                                continue
                    try:
                        final_msg = stream.get_final_message()  # type: ignore[attr-defined]
                        # Flatten final content
                        parts: list[str] = []
                        for c in getattr(final_msg, "content", []) or []:
                            if getattr(c, "type", "text") == "text":
                                parts.append(getattr(c, "text", ""))
                            elif isinstance(c, dict):
                                parts.append(str(c.get("text", "")))
                        if parts:
                            return "".join(parts)
                    except Exception:
                        pass
                    return "".join(acc)
            except Exception:
                # Fall back to non-streaming
                full = self.chat(messages, tools=tools, tool_choice=tool_choice)
                _push(full)
                return full
        # Unsupported provider -> non-streaming fallback
        full = self.chat(messages, tools=tools, tool_choice=tool_choice)
        _push(full)
        return full
