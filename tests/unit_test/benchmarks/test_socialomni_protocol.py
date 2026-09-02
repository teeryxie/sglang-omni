from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

import benchmarks.tasks.socialomni as socialomni_tasks
from benchmarks.dataset.socialomni import SocialOmniLevel1Sample, SocialOmniLevel2Sample
from benchmarks.eval.benchmark_omni_socialomni import RunArtifacts
from benchmarks.tasks.socialomni import (
    ChatResult,
    JudgeSpec,
    bounded_map,
    build_ffmpeg_prefix_command,
    build_judge_prompt,
    build_level1_prompt,
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
    run_level2_judge_phase,
)
from benchmarks.tasks.video_understanding import make_video_send_fn


def _level1(video: Path) -> SocialOmniLevel1Sample:
    return SocialOmniLevel1Sample(
        sample_id="l1",
        video_path=str(video),
        question="Who is speaking?",
        options=["Alice", "Bob", "Carol", "Dan"],
        answer="B",
        visibility="speaker_visible",
        prompt="unused",
        asr_content="SECRET ASR",
    )


def _level2(video: Path) -> SocialOmniLevel2Sample:
    return SocialOmniLevel2Sample(
        sample_id="l2",
        video_path=str(video),
        target_participant="Alice",
        timestamp="1.25",
        timestamp_s=1.25,
        gold_when="YES",
        reference_response="SECRET REFERENCE",
        reference_context="SECRET ASR",
        question_when="Should Alice speak now?",
        question_how="What should Alice say?",
    )


def test_model_prompts_do_not_leak_reference_material(tmp_path: Path) -> None:
    sample1 = _level1(tmp_path / "one.mp4")
    sample2 = _level2(tmp_path / "two.mp4")

    prompts = [
        build_level1_prompt(sample1),
        build_when_prompt(sample2),
        build_response_prompt(sample2),
    ]

    assert all("SECRET" not in prompt for prompt in prompts)
    judge_prompt = build_judge_prompt(sample2, "candidate")
    assert "SECRET ASR" in judge_prompt
    assert "SECRET REFERENCE" in judge_prompt
    payload = model_payload("model", prompts[1], sample2.video_path, max_tokens=8)
    assert payload["use_audio_in_video"] is True


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Answer: C", "C"), ("choice is B", "B"), ("unknown", "")],
)
def test_choice_parsing(raw: str, expected: str) -> None:
    assert parse_choice(raw, ("A", "B", "C", "D")) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Answer: A", "YES"), ("NO", "NO"), ("perhaps", "")],
)
def test_when_parsing(raw: str, expected: str) -> None:
    assert parse_when(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("75", 75), ("Score: 25", 25), ("80", None), ("75 or 100", None)],
)
def test_judge_score_is_strict(raw: str, expected: int | None) -> None:
    assert parse_judge_score(raw) == expected


def test_bounded_map_never_exceeds_worker_limit() -> None:
    active = 0
    maximum = 0

    async def worker(value: int) -> int:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.005)
        active -= 1
        return value * 2

    result = asyncio.run(bounded_map(list(range(30)), worker, 3))

    assert result == [value * 2 for value in range(30)]
    assert maximum == 3


def test_judge_phase_enforces_global_request_limit(monkeypatch, tmp_path: Path) -> None:
    active = 0
    maximum = 0

    async def fake_request(*_args, **kwargs) -> ChatResult:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.005)
        active -= 1
        return ChatResult(
            request_id=kwargs["request_id"],
            text="75",
            is_success=True,
            latency_s=0.005,
        )

    monkeypatch.setattr(socialomni_tasks, "request_chat_completion", fake_request)
    samples = {
        f"sample-{index}": SocialOmniLevel2Sample(
            sample_id=f"sample-{index}",
            video_path=str(tmp_path / f"{index}.mp4"),
            target_participant="Alice",
            timestamp="1.0",
            timestamp_s=1.0,
            gold_when="YES",
            reference_response="reference",
            reference_context="context",
            question_when="Should Alice speak now?",
            question_how="What should Alice say?",
        )
        for index in range(4)
    }
    model_records = [
        {
            "sample_id": sample_id,
            "gold_when": "YES",
            "gold_response": "candidate",
            "prefix_path": sample.video_path,
        }
        for sample_id, sample in samples.items()
    ]
    judges = [
        JudgeSpec(name=name, model=name, base_url="http://example", max_concurrency=4)
        for name in ("gpt-4o", "gemini-2.5-pro", "qwen3-omni")
    ]

    results = asyncio.run(
        run_level2_judge_phase(
            samples,
            model_records,
            judges=judges,
            max_concurrency=2,
            max_attempts=1,
            timeout_s=1,
        )
    )

    assert maximum == 2
    assert len(results) == 4
    assert all(len(result["gold_judge_scores"]) == 3 for result in results)


class _Response:
    def __init__(self, status: int, body: dict | str) -> None:
        self.status = status
        self.body = body if isinstance(body, str) else json.dumps(body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def text(self) -> str:
        return self.body


class _Session:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = iter(responses)
        self.last_json: dict | None = None

    def post(self, *_args, **kwargs) -> _Response:
        self.last_json = kwargs.get("json")
        return next(self.responses)


def test_retry_records_http_failure_and_success() -> None:
    attempts: list[dict] = []
    session = _Session(
        [
            _Response(503, {"detail": "busy"}),
            _Response(200, {"choices": [{"message": {"content": "OK"}}]}),
        ]
    )

    result = asyncio.run(
        request_chat_completion(
            session,  # type: ignore[arg-type]
            api_url="http://example/v1/chat/completions",
            payload={},
            request_id="request",
            phase="test",
            max_attempts=2,
            retry_backoff_s=0,
            attempt_hook=attempts.append,
        )
    )

    assert result.is_success is True
    assert [attempt["success"] for attempt in attempts] == [False, True]
    assert "HTTP 503" in attempts[0]["error"]


def test_video_helper_sends_embedded_audio_flag_and_preserves_http_error(
    tmp_path: Path,
) -> None:
    sample = _level1(tmp_path / "video.mp4")
    session = _Session([_Response(503, {"detail": "overloaded"})])
    send = make_video_send_fn(
        "qwen3-omni",
        "http://example/v1/chat/completions",
        use_audio_in_video=True,
    )

    result = asyncio.run(send(session, sample))  # type: ignore[arg-type]

    assert session.last_json is not None
    assert session.last_json["use_audio_in_video"] is True
    assert result.error == 'HTTP 503: {"detail": "overloaded"}'


def test_judge_config_records_env_name_not_secret(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SOCIALOMNI_TEST_KEY", "top-secret")
    path = tmp_path / "judges.json"
    path.write_text(
        json.dumps(
            {
                "judges": [
                    {
                        "name": name,
                        "model": name,
                        "base_url": f"https://{name}.example/v1",
                        "api_key_env": "SOCIALOMNI_TEST_KEY",
                        "max_tokens": 512 if name == "gemini-2.5-pro" else 8,
                    }
                    for name in ("gpt-4o", "gemini-2.5-pro", "qwen3-omni")
                ]
            }
        ),
        encoding="utf-8",
    )

    specs = load_judge_config(path)
    serialized = json.dumps([spec.public_dict() for spec in specs])

    assert "SOCIALOMNI_TEST_KEY" in serialized
    assert "top-secret" not in serialized
    assert [spec.max_tokens for spec in specs] == [8, 512, 8]


def test_judge_phase_uses_per_judge_token_budget(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, int] = {}

    async def fake_request(*_args, **kwargs) -> ChatResult:
        observed[kwargs["payload"]["model"]] = kwargs["payload"]["max_tokens"]
        return ChatResult(
            request_id=kwargs["request_id"],
            text="75",
            is_success=True,
            latency_s=0.001,
        )

    monkeypatch.setattr(socialomni_tasks, "request_chat_completion", fake_request)
    sample = _level2(tmp_path / "video.mp4")
    judges = [
        JudgeSpec(
            name=name,
            model=name,
            base_url="http://example",
            max_tokens=max_tokens,
        )
        for name, max_tokens in (
            ("gpt-4o", 8),
            ("gemini-2.5-pro", 512),
            ("qwen3-omni", 16),
        )
    ]

    asyncio.run(
        run_level2_judge_phase(
            {sample.sample_id: sample},
            [
                {
                    "sample_id": sample.sample_id,
                    "gold_when": "YES",
                    "gold_response": "candidate",
                    "prefix_path": sample.video_path,
                }
            ],
            judges=judges,
            max_concurrency=3,
            max_attempts=1,
            timeout_s=1,
        )
    )

    assert observed == {
        "gpt-4o": 8,
        "gemini-2.5-pro": 512,
        "qwen3-omni": 16,
    }


def test_judge_payload_only_sends_server_path_when_explicit() -> None:
    text_only = JudgeSpec(
        name="gpt-4o", model="gpt-4o", base_url="https://example.test"
    )
    local_video = JudgeSpec(
        name="qwen3-omni",
        model="qwen3-omni",
        base_url="http://localhost:8000",
        video_input="server-path",
    )

    remote_payload = judge_payload(
        text_only, "prompt", "/local/prefix.mp4", max_tokens=8
    )
    local_payload = judge_payload(
        local_video, "prompt", "/local/prefix.mp4", max_tokens=8
    )

    assert "videos" not in remote_payload
    assert local_payload["videos"] == ["/local/prefix.mp4"]
    assert local_payload["use_audio_in_video"] is True


def test_resume_requires_identical_manifest(tmp_path: Path) -> None:
    artifacts = RunArtifacts(tmp_path / "run")
    artifacts.prepare({"model": "one"}, {"repository": {}}, resume=False)
    artifacts.prepare({"model": "one"}, {"repository": {}}, resume=True)

    with pytest.raises(ValueError, match="fingerprint"):
        artifacts.prepare({"model": "two"}, {"repository": {}}, resume=True)


def test_ffmpeg_command_reencodes_instead_of_stream_copy(tmp_path: Path) -> None:
    command = build_ffmpeg_prefix_command(
        "ffmpeg", tmp_path / "input.mp4", 1.25, tmp_path / "output.mp4"
    )

    assert "copy" not in command
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-c:a") + 1] == "aac"
    assert command[command.index("-t") + 1] == "1.250000"


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required",
)
def test_prefix_media_ends_at_requested_time(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=30:duration=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ],
        check=True,
    )

    prefix = asyncio.run(create_video_prefix(source, 0.75, tmp_path / "cache"))
    duration = float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(prefix),
            ],
            text=True,
        ).strip()
    )

    assert 0.70 <= duration <= 0.80
