"""Robust LLM client wrappers for structured support triage calls."""

from __future__ import annotations

import json
import random
import re
import time
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

    @classmethod
    def parse(cls, text: str) -> Any:
        if not cls._json_like.match(text):
            raise StrictJSONError("LLM response was not a standalone JSON object or array")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise StrictJSONError(f"Invalid JSON from LLM: {exc}") from exc


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
                return LLMResult(
                    provider=provider,
                    model=resolved_model,
                    content=content,
                    input_tokens=usage_in or input_tokens,
                    output_tokens=output_tokens,
                    attempts=attempt,
                    parsed_json=parsed,
                )
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
