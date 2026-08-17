import io
import json

import pytest

from transcripts_ai.providers import (
    FakeProvider,
    OpenAICompatProvider,
    ProviderError,
    RoleRegistry,
    ValidationFailed,
    call_role,
    extract_json,
    parse_role_spec,
)


class TestRoleSpec:
    def test_parse(self):
        config = parse_role_spec("local:qwen3-32b")
        assert config.provider == "local"
        assert config.model == "qwen3-32b"

    @pytest.mark.parametrize("bad", ["", "openai", "weird:m", ":model"])
    def test_rejects_bad_specs(self, bad):
        with pytest.raises(ProviderError):
            parse_role_spec(bad)


class TestRegistry:
    def test_env_selection_and_cache(self):
        registry = RoleRegistry(
            {"AI_ROLE_EXTRACTOR": "fake:test-model"}
        )
        p1 = registry.provider_for("extractor")
        assert p1.model == "test-model"
        assert registry.provider_for("extractor") is p1

    def test_local_provider_base_url(self):
        registry = RoleRegistry(
            {
                "AI_ROLE_SUMMARIZER": "local:llama3",
                "LOCAL_AI_BASE_URL": "http://127.0.0.1:8080/v1",
            }
        )
        provider = registry.provider_for("summarizer")
        assert isinstance(provider, OpenAICompatProvider)
        assert provider.base_url == "http://127.0.0.1:8080/v1"
        assert provider.provider_name == "local"

    def test_missing_role_is_explicit_error(self):
        with pytest.raises(ProviderError, match="AI_ROLE_VERIFIER"):
            RoleRegistry({}).provider_for("verifier")

    def test_unknown_role_rejected(self):
        with pytest.raises(ProviderError):
            RoleRegistry({}).provider_for("wizard")

    def test_override_wins(self):
        fake = FakeProvider(["{}"])
        registry = RoleRegistry({}, overrides={"extractor": fake})
        assert registry.provider_for("extractor") is fake


class TestOpenAICompatTransport:
    def make_provider(self, body: dict):
        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def opener(request, timeout):
            opener.request = request
            return _Resp(json.dumps(body).encode())

        provider = OpenAICompatProvider(
            model="m", base_url="http://x/v1", api_key="k", opener=opener
        )
        return provider, opener

    def test_success_and_usage(self):
        provider, opener = self.make_provider(
            {
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                "model": "m-2026",
                "id": "req_1",
            }
        )
        response = provider.complete(system="s", user="u")
        assert response.text == "hi"
        assert response.usage == {"prompt_tokens": 10, "completion_tokens": 2}
        assert response.request_id == "req_1"
        sent = json.loads(opener.request.data)
        assert sent["messages"][0]["role"] == "system"
        assert opener.request.headers["Authorization"] == "Bearer k"

    def test_malformed_payload_is_provider_error(self):
        provider, _ = self.make_provider({"nope": True})
        with pytest.raises(ProviderError):
            provider.complete(system="s", user="u")


class TestExtractJson:
    def test_plain(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_embedded(self):
        assert extract_json('Sure! Here it is: {"a": [1, 2]} hope that helps') == {"a": [1, 2]}

    def test_garbage_raises(self):
        with pytest.raises(json.JSONDecodeError):
            extract_json("no json here")


class TestCallRole:
    def validator(self, payload):
        problems = []
        if not isinstance(payload, dict) or "facts" not in payload:
            problems.append("missing 'facts' key")
        return problems

    def test_valid_first_try(self):
        fake = FakeProvider(['{"facts": []}'])
        payload, response = call_role(
            fake, system="s", user="u", validator=self.validator
        )
        assert payload == {"facts": []}
        assert len(fake.calls) == 1

    def test_retry_with_error_feedback_then_success(self):
        fake = FakeProvider(['{"wrong": true}', '{"facts": [1]}'])
        payload, _ = call_role(fake, system="s", user="u", validator=self.validator)
        assert payload == {"facts": [1]}
        assert len(fake.calls) == 2
        assert "missing 'facts' key" in fake.calls[1]["user"]

    def test_two_failures_raise_validation_failed(self):
        fake = FakeProvider(["not json at all", '{"still": "wrong"}'])
        with pytest.raises(ValidationFailed) as exc:
            call_role(fake, system="s", user="u", validator=self.validator)
        assert len(exc.value.attempts) == 2

    def test_usage_reported_even_for_invalid_output(self):
        seen = []
        fake = FakeProvider(["junk", "junk2"])
        with pytest.raises(ValidationFailed):
            call_role(
                fake, system="s", user="u", validator=self.validator,
                usage_sink=seen.append,
            )
        assert len(seen) == 2
