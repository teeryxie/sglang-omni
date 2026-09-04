# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path

import av
import pytest

from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.socialomni import SocialOmniLevel1Sample, SocialOmniLevel2Sample
from benchmarks.eval import benchmark_omni_socialomni as entrypoint
from benchmarks.tasks.socialomni import (
    JUDGE_MAX_TOKENS,
    JUDGE_PARSE_ATTEMPTS,
    LEVEL1_MAX_TOKENS,
    LEVEL2_RESPONSE_MAX_TOKENS,
    LEVEL2_WHEN_MAX_TOKENS,
    JudgeSpec,
    bounded_map,
    build_ffmpeg_prefix_command,
    build_judge_prompt,
    build_response_prompt,
    build_when_prompt,
    create_video_prefix,
    judge_payload,
    load_judge_config,
    model_payload,
    parse_choice,
    parse_judge_score,
    parse_when,
    request_chat_completion,
    resolve_ffmpeg_executable,
    run_judges,
)


def _level1(path: str = "/tmp/video.mp4") -> SocialOmniLevel1Sample:
    return SocialOmniLevel1Sample(
        "one", path, "Who?", ("one", "two", "three", "four"), "A", "speaker_visible"
    )


def _level2(index: int = 0) -> SocialOmniLevel2Sample:
    return SocialOmniLevel2Sample(
        str(index),
        "/tmp/video.mp4",
        3.0,
        "Alex",
        "Should Alex speak now?",
        "What should Alex say?",
        "NO",
        "private reference response",
        "private reference transcript",
    )


def _config(**overrides) -> entrypoint.SocialOmniEvalConfig:
    values = {
        "dataset_root": ".",
        "model": "qwen3-omni",
        "model_revision": "model-revision",
        "launch_command": "python -m sglang_omni.cli serve ...",
        "base_url": "http://localhost:8000",
        "level": "level1",
        "judge_config": None,
        "prefix_cache_dir": "cache",
        "mini": False,
        "max_samples": None,
        "max_concurrency": 1,
        "timeout_s": 30,
        "output_dir": "results",
    }
    values.update(overrides)
    return entrypoint.SocialOmniEvalConfig(**values)


def test_model_prompts_do_not_leak_reference_material() -> None:
    sample = _level2()
    when = build_when_prompt(sample)
    response = build_response_prompt(sample)
    for secret in (sample.reference_context, sample.reference_response):
        assert secret not in when
        assert secret not in response
    judge = build_judge_prompt(sample, "candidate")
    assert sample.reference_context in judge
    assert sample.reference_response in judge


def test_model_payload_uses_native_video_with_embedded_audio() -> None:
    payload = model_payload("qwen3-omni", "prompt", "/tmp/prefix.mp4", 8)
    assert payload["videos"] == ["/tmp/prefix.mp4"]
    assert payload["use_audio_in_video"] is True
    assert payload["modalities"] == ["text"]


def test_judge_payload_allows_reasoning_before_score() -> None:
    judge = JudgeSpec(
        "gemini-2.5-pro", "gemini-2.5-pro", "http://localhost:8000", None, 1
    )
    assert judge_payload(judge, "prompt")["max_tokens"] == JUDGE_MAX_TOKENS == 8192


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Answer: A", "A"), ("\\boxed{B}", "B"), ("C", "C"), ("A or B", "")],
)
def test_choice_parser_is_strict(raw: str, expected: str) -> None:
    assert parse_choice(raw, ("A", "B", "C", "D")) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Answer: A", "YES"), ("Answer: B", "NO"), ("YES", "YES"), ("maybe", "")],
)
def test_when_parser(raw: str, expected: str) -> None:
    assert parse_when(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"), [("75", 75), ("Score: 25", 25), ("75 or 100", None)]
)
def test_judge_score_parser(raw: str, expected: int | None) -> None:
    assert parse_judge_score(raw) == expected


def test_judge_config_has_only_fixed_public_fields(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SECRET_VALUE", "must-not-appear")
    path = tmp_path / "judges.json"
    path.write_text(
        json.dumps(
            {
                "judges": [
                    {
                        "name": name,
                        "model": name,
                        "base_url": "http://localhost:8000",
                        "api_key_env": "SECRET_VALUE" if index == 0 else None,
                        "max_concurrency": 1,
                    }
                    for index, name in enumerate(
                        ("gpt-4o", "gemini-2.5-pro", "qwen3-omni")
                    )
                ]
            }
        ),
        encoding="utf-8",
    )
    public = [judge.public_dict() for judge in load_judge_config(path)]
    assert "must-not-appear" not in json.dumps(public)
    assert public[0]["api_key_env"] == "SECRET_VALUE"


@pytest.mark.asyncio
async def test_bounded_map_limits_active_workers() -> None:
    active = 0
    maximum = 0

    async def worker(value: int) -> int:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        return value

    assert await bounded_map(list(range(20)), worker, 3) == list(range(20))
    assert maximum == 3


class _Response:
    def __init__(self, status: int = 400, body: str = "specific failure body"):
        self.status = status
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def text(self) -> str:
        return self.body


class _Session:
    def __init__(self, *responses: _Response):
        self.responses = list(responses) or [_Response()]
        self.calls = 0

    def post(self, *_args, **_kwargs):
        response = self.responses[self.calls]
        self.calls += 1
        return response


@pytest.mark.asyncio
async def test_http_error_body_is_preserved() -> None:
    result = await request_chat_completion(
        _Session(),  # type: ignore[arg-type]
        api_url="http://example/v1/chat/completions",
        payload={},
        request_id="request",
    )
    assert not result.is_success
    assert "HTTP 400" in result.error
    assert "specific failure body" in result.error


@pytest.mark.parametrize("status", [429, 501, 507])
@pytest.mark.asyncio
async def test_retryable_http_error_is_retried(monkeypatch, status: int) -> None:
    session = _Session(
        _Response(status, "retry"),
        _Response(
            200,
            json.dumps({"choices": [{"message": {"content": "Answer: A"}}]}),
        ),
    )

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("benchmarks.tasks.socialomni.asyncio.sleep", no_sleep)
    result = await request_chat_completion(
        session,  # type: ignore[arg-type]
        api_url="http://example/v1/chat/completions",
        payload={},
        request_id="request",
    )
    assert result.is_success
    assert session.calls == 2


@pytest.mark.asyncio
async def test_non_retryable_http_error_is_not_retried() -> None:
    session = _Session(_Response(400, "bad request"))
    result = await request_chat_completion(
        session,  # type: ignore[arg-type]
        api_url="http://example/v1/chat/completions",
        payload={},
        request_id="request",
    )
    assert not result.is_success
    assert session.calls == 1


@pytest.mark.asyncio
async def test_malformed_success_response_does_not_escape() -> None:
    result = await request_chat_completion(
        _Session(_Response(200, "[]")),  # type: ignore[arg-type]
        api_url="http://example/v1/chat/completions",
        payload={},
        request_id="request",
        max_attempts=1,
    )
    assert not result.is_success
    assert "invalid JSON response object" in result.error


@pytest.mark.asyncio
async def test_judges_preserve_raw_results(monkeypatch) -> None:
    sample = _level2()
    record = {
        "sample_id": sample.sample_id,
        "gold_when": "YES",
        "gold_response": "candidate",
        "gold_response_success": True,
        "gold_judge_scores": {},
        "judge_results": {},
    }
    judges = [
        JudgeSpec(name, name, "http://localhost:8000", None, 1)
        for name in ("gpt-4o", "gemini-2.5-pro", "qwen3-omni")
    ]

    async def fake_request(*_args, request_id: str, **_kwargs):
        return RequestResult(
            request_id=request_id, text="75", is_success=True, latency_s=0.1
        )

    monkeypatch.setattr(
        "benchmarks.tasks.socialomni.request_chat_completion", fake_request
    )
    _, failures = await run_judges([sample], [record], judges, timeout_s=30)
    assert not failures
    assert record["gold_judge_scores"] == {name.name: 75 for name in judges}
    result = record["judge_results"]["gpt-4o"]
    assert set(result) == {
        "score",
        "raw_response",
        "is_success",
        "latency_s",
        "prompt_tokens",
        "completion_tokens",
        "error",
    }
    assert result["raw_response"] == "75"


@pytest.mark.asyncio
async def test_judges_preserve_invalid_raw_score_and_error(monkeypatch) -> None:
    sample = _level2()
    record = {
        "sample_id": sample.sample_id,
        "gold_when": "YES",
        "gold_response": "candidate",
        "gold_response_success": True,
        "gold_judge_scores": {},
        "judge_results": {},
    }
    judges = [JudgeSpec("gpt-4o", "gpt-4o", "http://localhost:8000", None, 1)]

    calls = 0

    async def fake_request(*_args, request_id: str, **_kwargs):
        nonlocal calls
        calls += 1
        return RequestResult(request_id=request_id, text="Score: 80", is_success=True)

    monkeypatch.setattr(
        "benchmarks.tasks.socialomni.request_chat_completion", fake_request
    )
    _, failures = await run_judges([sample], [record], judges, timeout_s=30)
    result = record["judge_results"]["gpt-4o"]
    assert len(failures) == 1
    assert result["score"] is None
    assert result["raw_response"] == "Score: 80"
    assert result["is_success"] is False
    assert "invalid judge score" in result["error"]
    assert calls == JUDGE_PARSE_ATTEMPTS


@pytest.mark.asyncio
async def test_judge_parse_retry_stops_after_valid_score(monkeypatch) -> None:
    sample = _level2()
    record = {
        "sample_id": sample.sample_id,
        "gold_when": "YES",
        "gold_response": "candidate",
        "gold_response_success": True,
        "gold_judge_scores": {},
        "judge_results": {},
    }
    judges = [JudgeSpec("gpt-4o", "gpt-4o", "http://localhost:8000", None, 1)]
    responses = iter(("not a score", "75"))

    async def fake_request(*_args, request_id: str, **_kwargs):
        return RequestResult(
            request_id=request_id,
            text=next(responses),
            is_success=True,
            latency_s=0.1,
            prompt_tokens=2,
            completion_tokens=1,
        )

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(
        "benchmarks.tasks.socialomni.request_chat_completion", fake_request
    )
    monkeypatch.setattr("benchmarks.tasks.socialomni.asyncio.sleep", no_sleep)
    results, failures = await run_judges([sample], [record], judges, timeout_s=30)
    assert not failures
    assert record["gold_judge_scores"] == {"gpt-4o": 75}
    assert results[0].latency_s == pytest.approx(0.2)
    assert results[0].prompt_tokens == 4
    assert results[0].completion_tokens == 2


def test_prefix_command_reencodes_video_and_audio(tmp_path: Path) -> None:
    command = build_ffmpeg_prefix_command(
        "ffmpeg", tmp_path / "source.mp4", 1.25, tmp_path / "prefix.mp4"
    )
    assert "-c:v" in command and "libx264" in command
    assert "-c:a" in command and "aac" in command
    assert "copy" not in command
    assert command[command.index("-t") + 1] == "1.250000"


@pytest.mark.asyncio
async def test_prefix_media_ends_at_query_time(tmp_path: Path) -> None:
    ffmpeg = resolve_ffmpeg_executable()
    if not ffmpeg:
        pytest.skip("ffmpeg is unavailable")
    source = tmp_path / "source.mp4"
    process = await asyncio.create_subprocess_exec(
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=size=64x64:rate=10:duration=2",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=2",
        "-shortest",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-y",
        str(source),
    )
    assert await process.wait() == 0
    prefix = await create_video_prefix(source, 0.75, tmp_path / "cache")
    with av.open(str(prefix)) as container:
        assert container.duration is not None
        assert container.duration / av.time_base <= 0.85


@pytest.mark.asyncio
async def test_level2_automatically_derives_first_200_view(monkeypatch) -> None:
    samples = [_level2(index) for index in range(209)]
    records = [
        {
            "sample_id": sample.sample_id,
            "gold_when": "NO",
            "predicted_when": "NO",
            "when_success": True,
            "gold_response": "",
            "gold_response_success": None,
            "gold_judge_scores": {},
            "judge_results": {},
        }
        for sample in samples
    ]
    monkeypatch.setattr(
        entrypoint, "load_socialomni_level2_samples", lambda *_a, **_k: samples
    )
    monkeypatch.setattr(
        entrypoint,
        "inspect_socialomni_dataset",
        lambda *_a, **_k: {
            "expected_huggingface_revision": "revision",
            "metadata_sha256": {},
            "evaluation_input_sha256": "digest",
            "metadata_matches_expected_revision": True,
        },
    )
    monkeypatch.setattr(
        entrypoint,
        "collect_benchmark_provenance",
        lambda **_kwargs: {"repository": {"commit": "abc", "dirty": False}},
    )

    async def fake_model(*_args, **_kwargs):
        return records, []

    monkeypatch.setattr(entrypoint, "run_level2_model", fake_model)
    config = entrypoint.SocialOmniEvalConfig(
        dataset_root=".",
        model="qwen3-omni",
        model_revision="model-revision",
        launch_command="python -m sglang_omni.cli serve ...",
        base_url="http://localhost:8000",
        level="level2",
        judge_config=None,
        prefix_cache_dir="cache",
        mini=False,
        max_samples=None,
        max_concurrency=1,
        timeout_s=30,
        output_dir="results",
    )
    result = await entrypoint.run_socialomni(config)
    assert result["paper_core_200"]["sample_count"] == 200
    assert result["paper_core_200"]["when"]["total_samples"] == 200
    assert result["summary"]["status"] == "incomplete"
    assert result["summary"]["formal_status"] == "incomplete"


@pytest.mark.asyncio
async def test_invalid_judge_config_fails_before_level2_requests(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "judges.json"
    config_path.write_text('{"judges": []}', encoding="utf-8")
    called = False

    async def fake_model(*_args, **_kwargs):
        nonlocal called
        called = True
        return [], []

    monkeypatch.setattr(entrypoint, "run_level2_model", fake_model)
    config = _config(level="level2", judge_config=str(config_path))
    with pytest.raises(ValueError, match="exactly three judges"):
        await entrypoint.run_socialomni(config)
    assert not called


@pytest.mark.parametrize(
    (
        "sample_count",
        "mini",
        "dirty",
        "metadata_matches",
        "model_revision",
        "launch_command",
        "expected",
    ),
    [
        (2_000, False, False, True, "model-revision", "serve command", "complete"),
        (2_000, False, True, True, "model-revision", "serve command", "incomplete"),
        (2_000, False, False, False, "model-revision", "serve command", "incomplete"),
        (2, True, False, True, "model-revision", "serve command", "incomplete"),
        (2_000, False, False, True, None, "serve command", "incomplete"),
        (2_000, False, False, True, "model-revision", None, "incomplete"),
    ],
)
@pytest.mark.asyncio
async def test_formal_status_requires_clean_repository_and_expected_metadata(
    monkeypatch,
    sample_count: int,
    mini: bool,
    dirty: bool,
    metadata_matches: bool,
    model_revision: str | None,
    launch_command: str | None,
    expected: str,
) -> None:
    samples = [
        SocialOmniLevel1Sample(
            str(index),
            "/tmp/video.mp4",
            "Who?",
            ("one", "two", "three", "four"),
            "A",
            "speaker_visible" if index else "visibility_mismatch",
        )
        for index in range(sample_count)
    ]

    class FakeRunner:
        wall_clock_s = 1.0

        def __init__(self, _config):
            pass

        async def run(self, current_samples, _send):
            return [
                RequestResult(
                    request_id=sample.sample_id,
                    text="Answer: A",
                    is_success=True,
                )
                for sample in current_samples
            ]

    monkeypatch.setattr(entrypoint, "BenchmarkRunner", FakeRunner)
    monkeypatch.setattr(
        entrypoint, "load_socialomni_level1_samples", lambda *_a, **_k: samples
    )
    monkeypatch.setattr(
        entrypoint,
        "inspect_socialomni_dataset",
        lambda *_a, **_k: {
            "expected_huggingface_revision": "revision",
            "metadata_sha256": {},
            "evaluation_input_sha256": "digest",
            "metadata_matches_expected_revision": metadata_matches,
        },
    )

    def fake_provenance(**kwargs):
        assert kwargs["dataset_revision"] is None
        assert kwargs["model_revision"] == model_revision
        assert kwargs["launch_command"] == launch_command
        return {"repository": {"commit": "abc", "dirty": dirty}}

    monkeypatch.setattr(entrypoint, "collect_benchmark_provenance", fake_provenance)
    result = await entrypoint.run_socialomni(
        _config(
            mini=mini,
            model_revision=model_revision,
            launch_command=launch_command,
        )
    )
    assert result["summary"]["status"] == "complete"
    assert result["summary"]["formal_status"] == expected
    assert result["config"]["generation"] == {
        "temperature": 0.0,
        "stream": False,
        "level1_max_tokens": LEVEL1_MAX_TOKENS,
        "level2_when_max_tokens": LEVEL2_WHEN_MAX_TOKENS,
        "level2_response_max_tokens": LEVEL2_RESPONSE_MAX_TOKENS,
        "judge_max_tokens": JUDGE_MAX_TOKENS,
    }


@pytest.mark.asyncio
async def test_level2_formal_status_requires_full_three_judge_run(monkeypatch) -> None:
    samples = [_level2(index) for index in range(209)]
    records = [
        {
            "sample_id": sample.sample_id,
            "gold_when": "YES",
            "predicted_when": "YES",
            "when_success": True,
            "when_raw_response": "Answer: A",
            "gold_response": "candidate",
            "gold_response_success": True,
            "gold_judge_scores": {},
            "judge_results": {},
        }
        for sample in samples
    ]
    judges = [
        JudgeSpec(name, name, "http://localhost:8000", None, 1)
        for name in ("gpt-4o", "gemini-2.5-pro", "qwen3-omni")
    ]

    monkeypatch.setattr(
        entrypoint, "load_socialomni_level2_samples", lambda *_a, **_k: samples
    )
    monkeypatch.setattr(entrypoint, "load_judge_config", lambda _path: judges)
    monkeypatch.setattr(
        entrypoint,
        "inspect_socialomni_dataset",
        lambda *_a, **_k: {
            "expected_huggingface_revision": "revision",
            "verification_scope": "metadata_only",
            "metadata_sha256": {},
            "evaluation_input_sha256": "digest",
            "metadata_matches_expected_revision": True,
        },
    )
    monkeypatch.setattr(
        entrypoint,
        "collect_benchmark_provenance",
        lambda **_kwargs: {"repository": {"commit": "abc", "dirty": False}},
    )

    async def fake_model(*_args, **_kwargs):
        return records, []

    async def fake_judges(_samples, current, _judges, **_kwargs):
        for record in current:
            record["gold_judge_scores"] = {judge.name: 75 for judge in judges}
        return [], []

    monkeypatch.setattr(entrypoint, "run_level2_model", fake_model)
    monkeypatch.setattr(entrypoint, "run_judges", fake_judges)

    result = await entrypoint.run_socialomni(
        _config(level="level2", judge_config="judges.json")
    )

    assert result["summary"]["status"] == "complete"
    assert result["summary"]["formal_status"] == "complete"


def test_paper_core_judge_completeness_is_independent() -> None:
    scores = {name: 75 for name in ("gpt-4o", "gemini-2.5-pro", "qwen3-omni")}
    records = [
        {
            "gold_when": "YES",
            "gold_response": "candidate",
            "gold_response_success": True,
            "gold_judge_scores": dict(scores),
        }
        for _ in range(209)
    ]
    records[-1]["gold_judge_scores"].pop("gpt-4o")
    assert entrypoint._judges_complete(records[:200], True)
    assert not entrypoint._judges_complete(records, True)


def test_invalid_score_is_not_judge_complete() -> None:
    record = {
        "gold_when": "YES",
        "gold_response": "candidate",
        "gold_response_success": True,
        "gold_judge_scores": {
            "gpt-4o": 75,
            "gemini-2.5-pro": 80,
            "qwen3-omni": 75,
        },
    }
    assert not entrypoint._judges_complete([record], True)
