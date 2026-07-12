"""Shared OpenAI-compatible LLM client with deterministic mock mode."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None


def approx_tokens(text: str) -> int:
    return max(1, math.ceil(len(text or "") / 4))


@dataclass
class LLMUsage:
    api_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_clock_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return int(self.input_tokens + self.output_tokens)

    def reset(self) -> None:
        self.api_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.wall_clock_seconds = 0.0
        self.errors = []

    def as_dict(self) -> Dict[str, Any]:
        return {
            "api_calls": self.api_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "wall_clock_seconds": round(self.wall_clock_seconds, 6),
            "errors": list(self.errors),
        }


class PeopleJoinLLMClient:
    """Thin chat-completions client; empty API key forces mock mode."""

    def __init__(
        self,
        *,
        mode: str = "mock",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.2,
        timeout: float = 60.0,
        seed: int = 0,
        allow_mock_fallback: Optional[bool] = None,
    ):
        key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.mode = mode if mode in {"mock", "api"} else "mock"
        if self.mode == "api" and not key:
            self.mode = "mock"
        self.api_key = key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.seed = int(seed)
        # Formal API runs must not silently become mock; mock mode may use fallbacks.
        if allow_mock_fallback is None:
            allow_mock_fallback = self.mode == "mock"
        self.allow_mock_fallback = bool(allow_mock_fallback)
        self.usage = LLMUsage()
        self._client = None
        if self.mode == "api":
            if httpx is None:
                raise ModuleNotFoundError("httpx is required for PeopleJoin API mode")
            self._client = httpx.Client(timeout=timeout)

    @property
    def mock_mode(self) -> bool:
        return self.mode == "mock"

    def reset_usage(self) -> None:
        self.usage.reset()

    def complete(
        self,
        system: str,
        messages: List[Dict[str, str]],
        *,
        mock_fn=None,
        max_retries: int = 3,
    ) -> str:
        """Return assistant text; counts tokens/time for both mock and API."""
        seeded_system = f"{system}\n\nExperiment seed: {self.seed}."
        start = time.perf_counter()
        last_exc: Optional[Exception] = None
        output = ""
        try:
            if self.mock_mode:
                if mock_fn is None:
                    raise ValueError("mock_fn is required in mock mode")
                output = str(mock_fn(seeded_system, messages))
            else:
                delay = 1.5
                attempt_errors: List[str] = []
                for attempt in range(max(1, int(max_retries))):
                    try:
                        output = self._call_api(seeded_system, messages)
                        last_exc = None
                        attempt_errors = []
                        break
                    except Exception as exc:
                        last_exc = exc
                        detail = str(exc)
                        if hasattr(exc, "response") and exc.response is not None:
                            try:
                                detail = f"{exc} | body={exc.response.text[:400]}"
                            except Exception:
                                pass
                        attempt_errors.append(
                            f"attempt={attempt + 1}/{max_retries} {type(exc).__name__}: {detail[:400]}"
                        )
                        if attempt + 1 >= max_retries:
                            break
                        time.sleep(delay)
                        delay = min(delay * 2.0, 20.0)
                if attempt_errors:
                    self.usage.errors.extend(attempt_errors)
                if last_exc is not None:
                    raise last_exc
        except Exception as exc:
            self.usage.errors.append(f"{type(exc).__name__}: {str(exc)[:300]}")
            if self.allow_mock_fallback and mock_fn is not None:
                output = str(mock_fn(seeded_system, messages))
            else:
                elapsed = time.perf_counter() - start
                self.usage.api_calls += 1
                self.usage.wall_clock_seconds += elapsed
                raise
        elapsed = time.perf_counter() - start
        self.usage.api_calls += 1
        self.usage.wall_clock_seconds += elapsed
        text_in = seeded_system + "\n" + "\n".join(str(m.get("content", "")) for m in messages)
        self.usage.input_tokens += approx_tokens(text_in)
        self.usage.output_tokens += approx_tokens(output)
        return output

    def _call_api(self, system: str, messages: List[Dict[str, str]]) -> str:
        assert self._client is not None
        # Sanitize empty contents which some providers reject with 400.
        clean_messages = []
        for m in messages:
            content = m.get("content", "")
            if content is None:
                content = ""
            content = str(content)
            if not content.strip():
                content = "(empty)"
            clean_messages.append({"role": m.get("role", "user"), "content": content})
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        # OpenRouter recommends these attribution headers; omitting them can 403.
        if "openrouter.ai" in self.base_url:
            headers["HTTP-Referer"] = os.environ.get(
                "OPENROUTER_HTTP_REFERER",
                "https://github.com/clawbot-matching",
            )
            headers["X-Title"] = os.environ.get(
                "OPENROUTER_X_TITLE",
                "coweaver-peoplejoin",
            )
        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}] + clean_messages,
                "temperature": self.temperature,
            },
        )
        if resp.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"Client error '{resp.status_code}' for url '{resp.request.url}' | body={resp.text[:500]}",
                request=resp.request,
                response=resp,
            )
        data = resp.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        if "content" in data:
            content = data["content"]
            if isinstance(content, list):
                return content[0].get("text", "")
            if isinstance(content, str):
                return content
        raise KeyError("Unsupported chat completion response shape")
