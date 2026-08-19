"""Teacher adapters: cloud models as advisors, never as authorities.

Layer 2a of docs/TRAINING.md. A *teacher* is any chat model that can be
asked for an opinion — GPT, Claude, Gemini, or the local Llama on the
3080 box. Teachers are optional accelerators: nothing in the engine
requires them, no teacher output ever reaches campaign memory, and a
missing API key just means that teacher sits out.

Configuration is one env var, mirroring the engine's role specs::

    TEACHERS="openai:gpt-5-mini,anthropic:claude-sonnet-5,gemini:gemini-2.5-pro"

Providers and their keys:

- ``openai:<model>``     — OPENAI_API_KEY, OPENAI_BASE_URL to override
- ``anthropic:<model>``  — ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL
- ``gemini:<model>``     — GEMINI_API_KEY (or GOOGLE_API_KEY), GEMINI_BASE_URL
- ``local:<model>``      — LOCAL_AI_BASE_URL (Ollama/llama.cpp; no key needed)
- ``fake:<script>``      — deterministic, for tests and dry runs; e.g.
  ``fake:accept:Ghomra`` always votes accept-with-Ghomra,
  ``fake:reject`` always rejects, ``fake:garbage`` replies non-JSON.

All adapters implement the engine's ``ChatProvider`` protocol, so the
same validated-JSON call path (``providers.call_role``) serves both. A
teacher that errors or fails validation is recorded as an abstention by
the panel layer — it can never crash a run. Retry policy for transient
failures belongs to the overnight loop (a later layer), not here.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from ..providers import ChatProvider, OpenAICompatProvider, ProviderError, ProviderResponse

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
ANTHROPIC_VERSION = "2023-06-01"

TEACHER_PROVIDERS = ("openai", "anthropic", "gemini", "local", "fake")


# ---------------------------------------------------------------------------
# Shared HTTP plumbing (same style as providers.OpenAICompatProvider)
# ---------------------------------------------------------------------------

def _post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    timeout: float,
    opener: Callable[..., Any],
    provider_name: str,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with opener(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # pragma: no cover - passthrough detail
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ProviderError(f"HTTP {exc.code} from {provider_name}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ProviderError(f"{provider_name} request failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Anthropic Messages API
# ---------------------------------------------------------------------------

def anthropic_request(
    *, base_url: str, api_key: str, model: str, system: str, user: str,
    max_tokens: int, temperature: float,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Pure request shaping: (url, headers, payload). Key goes in a header,
    never the URL, so it cannot leak into logs or history files."""
    url = f"{base_url.rstrip('/')}/v1/messages"
    headers = {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION}
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    return url, headers, payload


def parse_anthropic_response(body: dict[str, Any], *, fallback_model: str) -> ProviderResponse:
    try:
        blocks = body["content"]
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ProviderError(f"malformed anthropic payload: {body!r:.300}") from exc
    usage = body.get("usage") or {}
    return ProviderResponse(
        text=text,
        model=body.get("model", fallback_model),
        provider="anthropic",
        usage={k: v for k, v in usage.items() if isinstance(v, int)},
        request_id=body.get("id"),
    )


class AnthropicProvider:
    provider_name = "anthropic"

    def __init__(
        self, *, model: str, api_key: str,
        base_url: str = DEFAULT_ANTHROPIC_BASE_URL,
        timeout: float = 120.0, opener: Callable[..., Any] | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self._opener = opener or urllib.request.urlopen

    def complete(self, *, system: str, user: str, max_tokens: int = 2000,
                 temperature: float = 0.0) -> ProviderResponse:
        url, headers, payload = anthropic_request(
            base_url=self.base_url, api_key=self.api_key, model=self.model,
            system=system, user=user, max_tokens=max_tokens, temperature=temperature,
        )
        body = _post_json(url, headers, payload, timeout=self.timeout,
                          opener=self._opener, provider_name=self.provider_name)
        return parse_anthropic_response(body, fallback_model=self.model)


# ---------------------------------------------------------------------------
# Google Gemini generateContent API
# ---------------------------------------------------------------------------

def gemini_request(
    *, base_url: str, api_key: str, model: str, system: str, user: str,
    max_tokens: int, temperature: float,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Pure request shaping: (url, headers, payload). Key in header only."""
    url = f"{base_url.rstrip('/')}/models/{model}:generateContent"
    headers = {"x-goog-api-key": api_key}
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    return url, headers, payload


def parse_gemini_response(body: dict[str, Any], *, fallback_model: str) -> ProviderResponse:
    candidates = body.get("candidates") or []
    if not candidates:
        feedback = body.get("promptFeedback") or {}
        raise ProviderError(f"gemini returned no candidates: {feedback!r:.300}")
    try:
        parts = candidates[0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError(f"malformed gemini payload: {body!r:.300}") from exc
    usage = body.get("usageMetadata") or {}
    return ProviderResponse(
        text=text,
        model=body.get("modelVersion", fallback_model),
        provider="gemini",
        usage={k: v for k, v in usage.items() if isinstance(v, int)},
        request_id=body.get("responseId"),
    )


class GeminiProvider:
    provider_name = "gemini"

    def __init__(
        self, *, model: str, api_key: str,
        base_url: str = DEFAULT_GEMINI_BASE_URL,
        timeout: float = 120.0, opener: Callable[..., Any] | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self._opener = opener or urllib.request.urlopen

    def complete(self, *, system: str, user: str, max_tokens: int = 2000,
                 temperature: float = 0.0) -> ProviderResponse:
        url, headers, payload = gemini_request(
            base_url=self.base_url, api_key=self.api_key, model=self.model,
            system=system, user=user, max_tokens=max_tokens, temperature=temperature,
        )
        body = _post_json(url, headers, payload, timeout=self.timeout,
                          opener=self._opener, provider_name=self.provider_name)
        return parse_gemini_response(body, fallback_model=self.model)


# ---------------------------------------------------------------------------
# Deterministic fake teacher (tests, dry runs, and offline demos)
# ---------------------------------------------------------------------------

class ScriptedTeacherProvider:
    """A teacher whose whole personality is its model string.

    - ``accept[:choice]`` — always votes accept, with the given choice
    - ``reject``          — always votes reject
    - ``uncertain``       — always votes uncertain
    - ``garbage``         — replies non-JSON (exercises the abstain path)
    - ``answer:<text>``   — answers scorecard questions with <text>
    """

    provider_name = "fake"

    def __init__(self, script: str):
        self.model = f"fake:{script}"
        self.script = script
        self.calls = 0

    def complete(self, *, system: str, user: str, max_tokens: int = 2000,
                 temperature: float = 0.0) -> ProviderResponse:
        self.calls += 1
        kind, _, payload = self.script.partition(":")
        if kind == "accept":
            text = json.dumps({"verdict": "accept", "choice": payload,
                               "reason": "scripted accept"})
        elif kind == "reject":
            text = json.dumps({"verdict": "reject", "choice": "",
                               "reason": "scripted reject"})
        elif kind == "uncertain":
            text = json.dumps({"verdict": "uncertain", "choice": "",
                               "reason": "scripted uncertainty"})
        elif kind == "answer":
            text = json.dumps({"answer": payload})
        else:  # garbage or anything unrecognised
            text = "Hmm, I would have to think about that one."
        return ProviderResponse(text=text, model=self.model, provider="fake")


# ---------------------------------------------------------------------------
# Teacher discovery from the environment
# ---------------------------------------------------------------------------

@dataclass
class Teacher:
    name: str                     # display id: "openai", "anthropic", ...
    provider: ChatProvider
    max_tokens: int = 600         # verdicts are tiny; keep spend bounded
    usage_totals: dict[str, int] = field(default_factory=dict)

    @property
    def model(self) -> str:
        return self.provider.model

    def record_usage(self, response: ProviderResponse) -> None:
        for key, value in response.usage.items():
            self.usage_totals[key] = self.usage_totals.get(key, 0) + value


@dataclass(frozen=True)
class SkippedTeacher:
    spec: str
    reason: str


def _build_teacher(provider_kind: str, model: str, env: dict[str, str]) -> ChatProvider:
    if provider_kind == "openai":
        key = env.get("OPENAI_API_KEY", "")
        if not key:
            raise ProviderError("OPENAI_API_KEY is not set")
        return OpenAICompatProvider(
            model=model, api_key=key, provider_name="openai",
            base_url=env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
    if provider_kind == "anthropic":
        key = env.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        return AnthropicProvider(
            model=model, api_key=key,
            base_url=env.get("ANTHROPIC_BASE_URL", DEFAULT_ANTHROPIC_BASE_URL),
        )
    if provider_kind == "gemini":
        key = env.get("GEMINI_API_KEY", "") or env.get("GOOGLE_API_KEY", "")
        if not key:
            raise ProviderError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")
        return GeminiProvider(
            model=model, api_key=key,
            base_url=env.get("GEMINI_BASE_URL", DEFAULT_GEMINI_BASE_URL),
        )
    if provider_kind == "local":
        return OpenAICompatProvider(
            model=model, provider_name="local",
            base_url=env.get("LOCAL_AI_BASE_URL", "http://127.0.0.1:11434/v1"),
            api_key=env.get("LOCAL_AI_API_KEY", ""),
        )
    if provider_kind == "fake":
        return ScriptedTeacherProvider(model)
    raise ProviderError(
        f"unknown teacher provider {provider_kind!r}; expected one of {TEACHER_PROVIDERS}"
    )


def discover_teachers(
    env: dict[str, str], *, spec: str | None = None,
) -> tuple[list[Teacher], list[SkippedTeacher]]:
    """Build every teacher named in ``spec`` (default: env TEACHERS).

    A teacher with a missing key is *skipped with a reason*, never an
    error: the panel runs with whoever showed up. Malformed specs are
    also skipped so one typo cannot take down the whole panel.
    """
    raw = spec if spec is not None else env.get("TEACHERS", "")
    teachers: list[Teacher] = []
    skipped: list[SkippedTeacher] = []
    names_seen: dict[str, int] = {}
    for part in (p.strip() for p in raw.split(",")):
        if not part:
            continue
        provider_kind, _, model = part.partition(":")
        provider_kind = provider_kind.strip().lower()
        model = model.strip()
        if provider_kind not in TEACHER_PROVIDERS or not model:
            skipped.append(SkippedTeacher(
                spec=part,
                reason=f"expected '<provider>:<model>' with provider in {TEACHER_PROVIDERS}",
            ))
            continue
        try:
            provider = _build_teacher(provider_kind, model, env)
        except ProviderError as exc:
            skipped.append(SkippedTeacher(spec=part, reason=str(exc)))
            continue
        base = (f"fake-{model.partition(':')[0]}" if provider_kind == "fake"
                else provider_kind)
        names_seen[base] = names_seen.get(base, 0) + 1
        name = base if names_seen[base] == 1 else f"{base}-{names_seen[base]}"
        teachers.append(Teacher(name=name, provider=provider))
    return teachers, skipped
