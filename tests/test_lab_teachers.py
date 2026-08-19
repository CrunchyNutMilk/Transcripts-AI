"""Layer 2a: teacher adapters — request shaping, response parsing, discovery."""
import json

import pytest

from transcripts_ai.lab.teachers import (
    ANTHROPIC_VERSION,
    AnthropicProvider,
    GeminiProvider,
    ScriptedTeacherProvider,
    anthropic_request,
    discover_teachers,
    gemini_request,
    parse_anthropic_response,
    parse_gemini_response,
)
from transcripts_ai.providers import ProviderError


class TestAnthropicShape:
    def test_request_shape(self):
        url, headers, payload = anthropic_request(
            base_url="https://api.anthropic.com", api_key="sk-test",
            model="claude-sonnet-5", system="sys", user="usr",
            max_tokens=600, temperature=0.0,
        )
        assert url == "https://api.anthropic.com/v1/messages"
        assert headers["x-api-key"] == "sk-test"
        assert headers["anthropic-version"] == ANTHROPIC_VERSION
        assert "sk-test" not in url        # key never rides in the URL
        assert payload["system"] == "sys"
        assert payload["messages"] == [{"role": "user", "content": "usr"}]
        assert payload["max_tokens"] == 600

    def test_parse_response(self):
        body = {
            "id": "msg_1", "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "hello "},
                        {"type": "text", "text": "there"}],
            "usage": {"input_tokens": 10, "output_tokens": 4},
        }
        response = parse_anthropic_response(body, fallback_model="m")
        assert response.text == "hello there"
        assert response.usage == {"input_tokens": 10, "output_tokens": 4}
        assert response.provider == "anthropic"

    def test_parse_malformed_raises(self):
        with pytest.raises(ProviderError, match="malformed anthropic"):
            parse_anthropic_response({"content": "not-a-list"}, fallback_model="m")


class TestGeminiShape:
    def test_request_shape(self):
        url, headers, payload = gemini_request(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key="g-key", model="gemini-2.5-pro", system="sys", user="usr",
            max_tokens=600, temperature=0.0,
        )
        assert url.endswith("/models/gemini-2.5-pro:generateContent")
        assert headers["x-goog-api-key"] == "g-key"
        assert "g-key" not in url          # key never rides in the URL
        assert payload["system_instruction"]["parts"] == [{"text": "sys"}]
        assert payload["contents"][0]["parts"] == [{"text": "usr"}]
        assert payload["generationConfig"]["maxOutputTokens"] == 600

    def test_parse_response(self):
        body = {
            "candidates": [{"content": {"parts": [{"text": "an"}, {"text": "swer"}]}}],
            "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 3},
            "modelVersion": "gemini-2.5-pro",
        }
        response = parse_gemini_response(body, fallback_model="m")
        assert response.text == "answer"
        assert response.usage["promptTokenCount"] == 12

    def test_no_candidates_raises(self):
        with pytest.raises(ProviderError, match="no candidates"):
            parse_gemini_response({"promptFeedback": {"blockReason": "SAFETY"}},
                                  fallback_model="m")


class TestHttpAdapters:
    """The providers over a stubbed opener — no network anywhere."""

    class _StubResponse:
        def __init__(self, body):
            self._body = json.dumps(body).encode()

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def test_anthropic_complete_round_trip(self):
        captured = {}

        def opener(request, timeout):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data)
            return self._StubResponse({
                "content": [{"type": "text", "text": "ok"}], "usage": {},
            })

        provider = AnthropicProvider(model="claude-sonnet-5", api_key="k",
                                     opener=opener)
        response = provider.complete(system="s", user="u", max_tokens=99)
        assert response.text == "ok"
        assert captured["url"].endswith("/v1/messages")
        assert captured["body"]["max_tokens"] == 99

    def test_gemini_complete_round_trip(self):
        def opener(request, timeout):
            return self._StubResponse({
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            })

        provider = GeminiProvider(model="gemini-2.5-pro", api_key="k",
                                  opener=opener)
        assert provider.complete(system="s", user="u").text == "ok"


class TestScriptedTeacher:
    @pytest.mark.parametrize("script,verdict,choice", [
        ("accept:Ghomra", "accept", "Ghomra"),
        ("reject", "reject", ""),
        ("uncertain", "uncertain", ""),
    ])
    def test_verdict_scripts(self, script, verdict, choice):
        reply = ScriptedTeacherProvider(script).complete(system="s", user="u")
        payload = json.loads(reply.text)
        assert (payload["verdict"], payload["choice"]) == (verdict, choice)

    def test_garbage_is_not_json(self):
        reply = ScriptedTeacherProvider("garbage").complete(system="s", user="u")
        with pytest.raises(json.JSONDecodeError):
            json.loads(reply.text)

    def test_answer_script(self):
        reply = ScriptedTeacherProvider("answer:Ghomra").complete(system="s", user="u")
        assert json.loads(reply.text) == {"answer": "Ghomra"}


class TestDiscovery:
    def test_full_panel_with_keys(self):
        env = {
            "TEACHERS": "openai:gpt-5-mini, anthropic:claude-sonnet-5, gemini:gemini-2.5-pro, local:llama3.1",
            "OPENAI_API_KEY": "a", "ANTHROPIC_API_KEY": "b", "GEMINI_API_KEY": "c",
        }
        teachers, skipped = discover_teachers(env)
        assert [t.name for t in teachers] == ["openai", "anthropic", "gemini", "local"]
        assert not skipped

    def test_missing_key_skips_with_reason_not_error(self):
        env = {"TEACHERS": "openai:gpt-5-mini,fake:accept:Ghomra"}
        teachers, skipped = discover_teachers(env)
        assert [t.name for t in teachers] == ["fake-accept"]
        assert len(skipped) == 1
        assert "OPENAI_API_KEY" in skipped[0].reason

    def test_google_api_key_fallback(self):
        env = {"TEACHERS": "gemini:gemini-2.5-pro", "GOOGLE_API_KEY": "g"}
        teachers, _ = discover_teachers(env)
        assert teachers and teachers[0].name == "gemini"

    def test_malformed_spec_skipped(self):
        teachers, skipped = discover_teachers({"TEACHERS": "chatgpt-4,anthropic:"})
        assert not teachers and len(skipped) == 2

    def test_duplicate_providers_get_distinct_names(self):
        env = {"TEACHERS": "fake:accept:A,fake:accept:B"}
        teachers, _ = discover_teachers(env)
        assert [t.name for t in teachers] == ["fake-accept", "fake-accept-2"]

    def test_explicit_spec_overrides_env(self):
        teachers, _ = discover_teachers({"TEACHERS": "fake:reject"},
                                        spec="fake:uncertain")
        assert teachers[0].model == "fake:uncertain"

    def test_empty_env_is_no_teachers(self):
        assert discover_teachers({}) == ([], [])
