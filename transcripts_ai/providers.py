"""Replaceable AI-provider layer.

Roles, not models, are the unit of configuration: the engine asks for the
"extractor" or "summarizer" role and the registry maps it to a provider+model
from environment variables. One HTTP client speaks the OpenAI-compatible
chat-completions dialect used by OpenAI, Ollama, llama.cpp server and LM
Studio, so cloud → local is a config change, not a code change.

Every call goes through ``call_role``: strict JSON parsing + schema validation,
one bounded retry carrying the validation error, and a typed failure the
pipeline routes to the review queue. Invalid output can cost one retry; it can
never crash the pipeline or produce unvalidated data.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

ROLES = ("extractor", "verifier", "adjudicator", "summarizer", "reviewer", "answerer")


class ProviderError(RuntimeError):
    """Transport-level failure (network, HTTP, auth)."""


class ValidationFailed(RuntimeError):
    """Output failed schema validation after the bounded retry."""

    def __init__(self, message: str, attempts: list[str]):
        super().__init__(message)
        self.attempts = attempts


@dataclass
class ProviderResponse:
    text: str
    model: str
    provider: str
    usage: dict[str, int] = field(default_factory=dict)
    request_id: str | None = None


class ChatProvider(Protocol):
    provider_name: str
    model: str

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 2000,
        temperature: float = 0.0,
    ) -> ProviderResponse: ...


# ---------------------------------------------------------------------------
# OpenAI-compatible HTTP provider (OpenAI / Ollama / llama.cpp / LM Studio)
# ---------------------------------------------------------------------------

class OpenAICompatProvider:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str = "",
        provider_name: str = "openai",
        timeout: float = 120.0,
        opener: Callable[..., Any] | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.provider_name = provider_name
        self.timeout = timeout
        self._opener = opener or urllib.request.urlopen

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 2000,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - passthrough detail
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ProviderError(f"HTTP {exc.code} from {self.provider_name}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(f"{self.provider_name} request failed: {exc}") from exc
        try:
            text = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"malformed completion payload: {body!r:.300}") from exc
        usage = body.get("usage") or {}
        return ProviderResponse(
            text=text,
            model=body.get("model", self.model),
            provider=self.provider_name,
            usage={k: v for k, v in usage.items() if isinstance(v, int)},
            request_id=body.get("id"),
        )


# ---------------------------------------------------------------------------
# Fake provider for tests and dry runs
# ---------------------------------------------------------------------------

class FakeProvider:
    """Scripted provider: returns queued responses in order, records calls."""

    provider_name = "fake"

    def __init__(self, responses: list[str] | None = None, model: str = "fake-model"):
        self.model = model
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    def queue(self, *responses: str) -> "FakeProvider":
        self.responses.extend(responses)
        return self

    def complete(self, *, system: str, user: str, max_tokens: int = 2000,
                 temperature: float = 0.0) -> ProviderResponse:
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        if not self.responses:
            raise ProviderError("FakeProvider has no queued responses")
        return ProviderResponse(
            text=self.responses.pop(0), model=self.model, provider=self.provider_name
        )


# ---------------------------------------------------------------------------
# Role registry
# ---------------------------------------------------------------------------

@dataclass
class RoleConfig:
    provider: str   # "openai" | "local" | "fake"
    model: str


def parse_role_spec(spec: str) -> RoleConfig:
    provider, _, model = spec.partition(":")
    provider = provider.strip().lower()
    model = model.strip()
    if provider not in {"openai", "local", "fake"} or not model:
        raise ProviderError(
            f"invalid role spec {spec!r}; expected 'openai:<model>' | 'local:<model>' | 'fake:<model>'"
        )
    return RoleConfig(provider=provider, model=model)


class RoleRegistry:
    """Maps role names to providers from env (AI_ROLE_EXTRACTOR=...)."""

    def __init__(
        self,
        env: dict[str, str] | None = None,
        *,
        overrides: dict[str, ChatProvider] | None = None,
    ):
        self.env = dict(os.environ if env is None else env)
        self.overrides = dict(overrides or {})
        self._cache: dict[str, ChatProvider] = {}

    def provider_for(self, role: str) -> ChatProvider:
        if role not in ROLES:
            raise ProviderError(f"unknown role {role!r}")
        if role in self.overrides:
            return self.overrides[role]
        if role in self._cache:
            return self._cache[role]
        spec = self.env.get(f"AI_ROLE_{role.upper()}")
        if not spec:
            raise ProviderError(
                f"no provider configured for role {role!r}; set AI_ROLE_{role.upper()}"
            )
        config = parse_role_spec(spec)
        if config.provider == "openai":
            provider: ChatProvider = OpenAICompatProvider(
                model=config.model,
                base_url=self.env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                api_key=self.env.get("OPENAI_API_KEY", ""),
                provider_name="openai",
            )
        elif config.provider == "local":
            provider = OpenAICompatProvider(
                model=config.model,
                base_url=self.env.get("LOCAL_AI_BASE_URL", "http://127.0.0.1:11434/v1"),
                api_key=self.env.get("LOCAL_AI_API_KEY", ""),
                provider_name="local",
            )
        else:  # fake
            provider = FakeProvider(model=config.model)
        self._cache[role] = provider
        return provider


# ---------------------------------------------------------------------------
# Structured calls with validation + bounded retry
# ---------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.S)


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array in a response (tolerates fences)."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(text)
        if match:
            return json.loads(match.group(0))
        raise


Validator = Callable[[Any], list[str]]  # returns problems; empty = valid


def call_role(
    provider: ChatProvider,
    *,
    system: str,
    user: str,
    validator: Validator,
    max_tokens: int = 2000,
    usage_sink: Callable[[ProviderResponse], None] | None = None,
) -> tuple[Any, ProviderResponse]:
    """One structured call: parse -> validate -> at most one corrective retry.

    Raises ValidationFailed after the retry; the caller routes that to the
    review queue. Usage is reported to the sink before parsing so malformed
    paid responses are still accounted for.
    """
    attempts: list[str] = []
    prompt = user
    for attempt in range(2):
        response = provider.complete(system=system, user=prompt, max_tokens=max_tokens)
        if usage_sink is not None:
            usage_sink(response)
        attempts.append(response.text)
        try:
            payload = extract_json(response.text)
        except json.JSONDecodeError as exc:
            problems = [f"response was not valid JSON: {exc}"]
        else:
            problems = validator(payload)
            if not problems:
                return payload, response
        if attempt == 0:
            prompt = (
                f"{user}\n\nYour previous reply was rejected for these reasons:\n- "
                + "\n- ".join(problems)
                + "\nReply again with ONLY corrected JSON."
            )
    raise ValidationFailed("; ".join(problems), attempts)
