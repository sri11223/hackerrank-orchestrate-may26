"""Robust LLM client wrappers for structured support triage calls."""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from .config import Settings, get_settings


Provider = Literal["openai", "groq"]


class LLMConfigurationError(RuntimeError):
    """Raised when a requested provider is unavailable or misconfigured."""


class LLMResponseError(RuntimeError):
    """Raised when a provider returns unusable output."""


class StrictJSONError(LLMResponseError):
    """Raised when strict JSON parsing fails."""


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class LLMResult:
    provider: Provider
    model: str
    content: str
    input_tokens: int
    output_tokens: int
    attempts: int
    parsed_json: Any | None = None
    cache_hit: bool = False


class TokenCounter:
    """Token counter with tiktoken support and a deterministic fallback."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self._encoding = None
        try:
            import tiktoken  # type: ignore

            try:
                self._encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                self._encoding = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            self._encoding = None

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding is not None:
            return len(self._encoding.encode(text))
        # Explicit deterministic fallback: rough English-token approximation.
        return max(1, len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)))

    def count_messages(self, messages: Sequence[ChatMessage]) -> int:
        # Chat message metadata overhead is small but real; keep it explicit.
        return sum(self.count_text(message.content) + 4 for message in messages) + 2


class StrictJSONParser:
    """Parse JSON only when the response is exactly a JSON object or array."""

    _json_like = re.compile(r"^\s*(\{.*\}|\[.*\])\s*$", re.DOTALL)
    _fenced_json = re.compile(r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$", re.DOTALL)

    @classmethod
    def parse(cls, text: str) -> Any:
        cleaned = cls.clean(text)
        if not cls._json_like.match(cleaned):
            raise StrictJSONError("LLM response was not a standalone JSON object or array")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise StrictJSONError(f"Invalid JSON from LLM: {exc}") from exc

    @classmethod
    def clean(cls, text: str) -> str:
        """Strip common markdown JSON fences before strict parsing."""

        cleaned = (text or "").strip().lstrip("\ufeff")
        match = cls._fenced_json.match(cleaned)
        if match:
            return match.group(1).strip()

        # Some providers occasionally emit uppercase/language-tagged fences with
        # stray whitespace after the closing marker. Keep this conservative:
        # remove only a leading fence line and a final closing fence.
        lines = cleaned.splitlines()
        if len(lines) >= 3 and re.match(r"^\s*```(?:[A-Za-z0-9_-]+)?\s*$", lines[0]):
            closing_index = len(lines) - 1
            while closing_index > 0 and not lines[closing_index].strip():
                closing_index -= 1
            if lines[closing_index].strip() == "```":
                return "\n".join(lines[1:closing_index]).strip()
        return cleaned


class LLMClient:
    """Small wrapper around OpenAI and Groq chat-completion APIs."""

    def __init__(self, settings: Settings | None = None, seed: int = 7) -> None:
        self.settings = settings or get_settings()
        self._rng = random.Random(seed)
        self._openai_client: Any | None = None
        self._groq_client: Any | None = None

    def count_tokens(self, messages: Sequence[ChatMessage], model: str | None = None) -> int:
        return TokenCounter(model or self.settings.openai_model).count_messages(messages)

    def chat_json(
        self,
        provider: Provider,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> LLMResult:
        return self.chat(
            provider,
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            strict_json=True,
        )

    def chat(
        self,
        provider: Provider,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
        strict_json: bool = False,
    ) -> LLMResult:
        resolved_model = model or self._default_model(provider)
        input_tokens = self.count_tokens(messages, resolved_model)
        cache_key = self._cache_key(
            provider=provider,
            model=resolved_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            strict_json=strict_json,
        )
        cached = self._read_cache(
            cache_key=cache_key,
            provider=provider,
            model=resolved_model,
            input_tokens=input_tokens,
            strict_json=strict_json,
        )
        if cached is not None:
            return cached

        last_error: Exception | None = None

        for attempt in range(1, self.settings.max_retries + 1):
            try:
                content, usage_in, usage_out = self._call_provider(
                    provider=provider,
                    model=resolved_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    strict_json=strict_json,
                )
                output_tokens = usage_out or TokenCounter(resolved_model).count_text(content)
                parsed = StrictJSONParser.parse(content) if strict_json else None
                result = LLMResult(
                    provider=provider,
                    model=resolved_model,
                    content=content,
                    input_tokens=usage_in or input_tokens,
                    output_tokens=output_tokens,
                    attempts=attempt,
                    parsed_json=parsed,
                )
                self._write_cache(cache_key, result)
                return result
            except LLMConfigurationError:
                raise
            except StrictJSONError as exc:
                last_error = exc
                if attempt >= self.settings.max_retries:
                    break
                self._sleep_before_retry(attempt)
            except Exception as exc:  # Provider SDKs expose different retryable exception types.
                last_error = exc
                if attempt >= self.settings.max_retries:
                    break
                self._sleep_before_retry(attempt)

        raise LLMResponseError(
            f"{provider} call failed after {self.settings.max_retries} attempts: {last_error}"
        ) from last_error

    def _cache_key(
        self,
        *,
        provider: Provider,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float,
        max_tokens: int,
        strict_json: bool,
    ) -> str:
        payload = {
            "provider": provider,
            "model": model,
            "messages": [message.as_dict() for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "strict_json": strict_json,
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        return self.settings.llm_cache_dir / f"{cache_key}.json"

    def _read_cache(
        self,
        *,
        cache_key: str,
        provider: Provider,
        model: str,
        input_tokens: int,
        strict_json: bool,
    ) -> LLMResult | None:
        if not self.settings.llm_cache_enabled:
            return None
        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            content = str(payload["content"])
            parsed = StrictJSONParser.parse(content) if strict_json else None
            return LLMResult(
                provider=provider,
                model=model,
                content=content,
                input_tokens=int(payload.get("input_tokens") or input_tokens),
                output_tokens=int(payload.get("output_tokens") or TokenCounter(model).count_text(content)),
                attempts=0,
                parsed_json=parsed,
                cache_hit=True,
            )
        except Exception:
            return None

    def _write_cache(self, cache_key: str, result: LLMResult) -> None:
        if not self.settings.llm_cache_enabled:
            return
        try:
            cache_dir = self.settings.llm_cache_dir
            cache_dir.mkdir(parents=True, exist_ok=True)
            path = self._cache_path(cache_key)
            temp_path = path.with_suffix(".tmp")
            temp_path.write_text(
                json.dumps(
                    {
                        "provider": result.provider,
                        "model": result.model,
                        "content": result.content,
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            temp_path.replace(path)
        except OSError:
            return

    def _default_model(self, provider: Provider) -> str:
        if provider == "openai":
            return self.settings.openai_model
        if provider == "groq":
            return self.settings.groq_model
        raise LLMConfigurationError(f"Unsupported provider: {provider}")

    def _sleep_before_retry(self, attempt: int) -> None:
        base = self.settings.retry_base_delay_seconds * (2 ** (attempt - 1))
        jitter = self._rng.uniform(0, self.settings.retry_base_delay_seconds)
        delay = min(self.settings.retry_max_delay_seconds, base + jitter)
        time.sleep(delay)

    def _call_provider(
        self,
        *,
        provider: Provider,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float,
        max_tokens: int,
        strict_json: bool,
    ) -> tuple[str, int | None, int | None]:
        if provider == "openai":
            return self._call_openai(model, messages, temperature, max_tokens, strict_json)
        if provider == "groq":
            return self._call_groq(model, messages, temperature, max_tokens, strict_json)
        raise LLMConfigurationError(f"Unsupported provider: {provider}")

    def _call_openai(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float,
        max_tokens: int,
        strict_json: bool,
    ) -> tuple[str, int | None, int | None]:
        if not self.settings.openai_api_key:
            raise LLMConfigurationError("OPENAI_API_KEY is not set")
        if self._openai_client is None:
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as exc:
                raise LLMConfigurationError("Install the openai package to use OpenAI") from exc
            self._openai_client = OpenAI(
                api_key=self.settings.openai_api_key,
                timeout=self.settings.request_timeout_seconds,
            )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [message.as_dict() for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if strict_json:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._openai_client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        output_tokens = getattr(usage, "completion_tokens", None) if usage else None
        return content.strip(), input_tokens, output_tokens

    def _call_groq(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        temperature: float,
        max_tokens: int,
        strict_json: bool,
    ) -> tuple[str, int | None, int | None]:
        if not self.settings.groq_api_key:
            raise LLMConfigurationError("GROQ_API_KEY is not set")
        if self._groq_client is None:
            try:
                from groq import Groq  # type: ignore
            except ImportError as exc:
                raise LLMConfigurationError("Install the groq package to use Groq") from exc
            self._groq_client = Groq(
                api_key=self.settings.groq_api_key,
                timeout=self.settings.request_timeout_seconds,
            )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [message.as_dict() for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if strict_json:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._groq_client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        output_tokens = getattr(usage, "completion_tokens", None) if usage else None
        return content.strip(), input_tokens, output_tokens
