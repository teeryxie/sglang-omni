from __future__ import annotations

import asyncio
import base64
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

import benchmarks.eval.benchmark_omni_socialomni as socialomni_benchmark
import benchmarks.tasks.socialomni as socialomni_tasks
from benchmarks.dataset.socialomni import (
    SocialOmniDatasetInfo,
    SocialOmniLevel1Sample,
    SocialOmniLevel2Sample,
)
from benchmarks.eval.benchmark_omni_socialomni import RunArtifacts, SocialOmniEvalConfig
from benchmarks.tasks.socialomni import (
    ChatResult,
    JudgeSpec,
    ModelSpec,
    _client_session,
    bounded_map,
    build_ffmpeg_prefix_command,
    build_judge_prompt,
    build_level1_prompt,
    build_response_prompt,
    build_when_prompt,
    create_video_prefix,
    extract_inline_video_frames,
    judge_payload,
    load_judge_config,
    model_payload,
    parse_choice,
    parse_judge_score,
    parse_when,
    request_chat_completion,
    request_upload_av_completion,
    resolve_ffmpeg_executable,
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
    assert "$LETTER" not in prompts[0]
    judge_prompt = build_judge_prompt(sample2, "candidate")
    assert "SECRET ASR" in judge_prompt
    assert "SECRET REFERENCE" in judge_prompt
    payload = model_payload("model", prompts[1], sample2.video_path, max_tokens=8)
    assert payload["use_audio_in_video"] is True

    visual_prompts = [
        build_level1_prompt(sample1, visual_only=True),
        build_when_prompt(sample2, visual_only=True),
        build_response_prompt(sample2, visual_only=True),
    ]
    assert all("audio" not in prompt.lower() for prompt in visual_prompts)
    assert all(
        "video-frame" in prompt.lower() or "video frames" in prompt.lower()
        for prompt in visual_prompts
    )


def test_inline_frame_payload_uses_openai_image_content() -> None:
    payload = model_payload(
        "gemini-2.5-pro",
        "question",
        "/private/video.mp4",
        max_tokens=32,
        frame_data_uris=("data:image/jpeg;base64,AAAA",),
    )

    assert "videos" not in payload
    assert "use_audio_in_video" not in payload
    assert payload["messages"][0]["content"] == [
        {"type": "text", "text": "question"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,AAAA"},
        },
    ]


def test_model_spec_validates_paper_media_transport() -> None:
    ModelSpec(
        model="gemini-2.5-pro",
        base_url="https://example.test/v1",
        video_input="inline-frames",
    ).validate()
    with pytest.raises(ValueError, match="video_input"):
        ModelSpec(
            model="model", base_url="http://example", video_input="unknown"
        ).validate()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Answer: C", "C"),
        ("choice is B", "B"),
        ("D", "D"),
        ("C. Alice is speaking.", "C"),
        ("A: The first option is correct.", "A"),
        (r"The final answer is $\boxed{D}$", "D"),
        ("Answer: B\nFinal Answer: C", ""),
        (r"Answer: B, but $\boxed{C}$", ""),
        ("Because Alice is visible", ""),
        ("unknown", ""),
    ],
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


def test_judge_phase_reuses_partial_scores(monkeypatch, tmp_path: Path) -> None:
    requested: list[str] = []
    journal: list[dict] = []

    async def fake_request(*_args, **kwargs) -> ChatResult:
        requested.append(kwargs["payload"]["model"])
        return ChatResult(
            request_id=kwargs["request_id"],
            text="75",
            is_success=True,
            latency_s=0.001,
            status_code=200,
        )

    monkeypatch.setattr(socialomni_tasks, "request_chat_completion", fake_request)
    sample = _level2(tmp_path / "video.mp4")
    judges = [
        JudgeSpec(name=name, model=name, base_url="http://example")
        for name in ("gpt-4o", "gemini-2.5-pro", "qwen3-omni")
    ]
    existing = {
        "record_type": "judge_score",
        "phase": "level2_judge",
        "sample_id": sample.sample_id,
        "judge": "gpt-4o",
        "score": 50,
        "detail": {"raw_response": "50", "attempt": 1},
        "error": "",
    }

    results = asyncio.run(
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
            existing_judge_records=[existing],
            result_hook=journal.append,
        )
    )

    assert set(requested) == {"gemini-2.5-pro", "qwen3-omni"}
    assert results[0]["gold_judge_scores"] == {
        "gpt-4o": 50,
        "gemini-2.5-pro": 75,
        "qwen3-omni": 75,
    }
    assert sum(row["record_type"] == "judge_score" for row in journal) == 2
    assert journal[-1]["record_type"] == "judge_result"


def test_judge_phase_does_not_retry_nonretryable_4xx(
    monkeypatch, tmp_path: Path
) -> None:
    calls = 0

    async def fake_request(*_args, **kwargs) -> ChatResult:
        nonlocal calls
        calls += 1
        return ChatResult(
            request_id=kwargs["request_id"],
            text="",
            is_success=False,
            latency_s=0.001,
            status_code=400,
            error="HTTP 400: bad request",
            retryable=False,
        )

    monkeypatch.setattr(socialomni_tasks, "request_chat_completion", fake_request)
    sample = _level2(tmp_path / "video.mp4")
    judges = [
        JudgeSpec(name=name, model=name, base_url="http://example")
        for name in ("gpt-4o", "gemini-2.5-pro", "qwen3-omni")
    ]

    journal: list[dict] = []
    with pytest.raises(RuntimeError, match="judge request"):
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
                max_attempts=3,
                timeout_s=1,
                result_hook=journal.append,
            )
        )

    assert calls == 3
    assert [row["record_type"] for row in journal] == ["judge_failure"] * 3


def test_judge_phase_retries_invalid_2xx_scores(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, int] = {}

    async def fake_request(*_args, **kwargs) -> ChatResult:
        request_id = kwargs["request_id"]
        calls[request_id] = calls.get(request_id, 0) + 1
        first = calls[request_id] == 1
        return ChatResult(
            request_id=request_id,
            text="" if first else "75",
            is_success=not first,
            latency_s=0.001,
            status_code=200,
            error="empty response" if first else "",
            retryable=False,
        )

    async def no_sleep(*_args) -> None:
        return None

    monkeypatch.setattr(socialomni_tasks, "request_chat_completion", fake_request)
    monkeypatch.setattr(socialomni_tasks.asyncio, "sleep", no_sleep)
    sample = _level2(tmp_path / "video.mp4")
    judges = [
        JudgeSpec(name=name, model=name, base_url="http://example")
        for name in ("gpt-4o", "gemini-2.5-pro", "qwen3-omni")
    ]

    results = asyncio.run(
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
            max_attempts=2,
            timeout_s=1,
        )
    )

    assert set(results[0]["gold_judge_scores"].values()) == {75}
    assert set(calls.values()) == {2}


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


def test_upload_av_transport_reads_reference_server_answer(tmp_path: Path) -> None:
    video = tmp_path / "prefix.mp4"
    video.write_bytes(b"media")
    session = _Session([_Response(200, {"answer": "Answer: B"})])

    result = asyncio.run(
        request_upload_av_completion(
            session,  # type: ignore[arg-type]
            api_url="http://example/analyze",
            video_path=str(video),
            prompt="question",
            request_id="sample",
            phase="level1",
            max_attempts=1,
        )
    )

    assert result.is_success is True
    assert result.text == "Answer: B"


def test_attempt_offset_keeps_attempt_numbers_monotonic() -> None:
    attempts: list[dict] = []
    session = _Session([_Response(200, {"choices": [{"message": {"content": "OK"}}]})])

    asyncio.run(
        request_chat_completion(
            session,  # type: ignore[arg-type]
            api_url="http://example/v1/chat/completions",
            payload={},
            request_id="request",
            phase="test",
            max_attempts=1,
            attempt_offset=2,
            attempt_hook=attempts.append,
        )
    )

    assert attempts[0]["attempt"] == 3


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


def test_contract_fingerprint_covers_stable_runtime_identity(tmp_path: Path) -> None:
    dataset_info = SocialOmniDatasetInfo(
        root=str(tmp_path),
        version="revision",
        level1_file="dataset.json",
        level1_sha256="level1",
        level2_file="annotations.json",
        level2_sha256="level2",
        manifest_file=None,
        manifest_sha256=None,
    )
    config = SocialOmniEvalConfig(
        dataset_root=str(tmp_path),
        model="model",
        model_revision="model-revision",
        launch_command="serve --tp 8",
    )
    provenance = {
        "repository": {"commit": "abc123", "dirty": False},
        "host": {"platform": "Linux"},
        "gpu": {
            "nvidia_smi_csv": (
                "0, NVIDIA H20, GPU-uuid, 97871, 580.1, P0, 500, 1980, 2619, 9.0"
            )
        },
        "packages": {"sglang": "0.5.18"},
        "dependency_freeze_sha256": "freeze-one",
    }

    first = socialomni_benchmark._contract(
        config,
        dataset_info=dataset_info,
        level1_ids=["one"],
        level2_ids=[],
        judges=[],
        provenance=provenance,
    )
    dynamic_gpu_change = socialomni_benchmark._contract(
        config,
        dataset_info=dataset_info,
        level1_ids=["one"],
        level2_ids=[],
        judges=[],
        provenance={
            **provenance,
            "gpu": {
                "nvidia_smi_csv": (
                    "0, NVIDIA H20, GPU-uuid, 97871, 580.1, P2, 500, 1200, 1600, 9.0"
                )
            },
        },
    )
    changed_environment = socialomni_benchmark._contract(
        config,
        dataset_info=dataset_info,
        level1_ids=["one"],
        level2_ids=[],
        judges=[],
        provenance={**provenance, "dependency_freeze_sha256": "freeze-two"},
    )
    changed_media_transport = socialomni_benchmark._contract(
        replace(config, model_video_input="inline-frames"),
        dataset_info=dataset_info,
        level1_ids=["one"],
        level2_ids=[],
        judges=[],
        provenance=provenance,
    )

    assert RunArtifacts.fingerprint(first) == RunArtifacts.fingerprint(
        dynamic_gpu_change
    )
    assert RunArtifacts.fingerprint(first) != RunArtifacts.fingerprint(
        changed_environment
    )
    assert RunArtifacts.fingerprint(first) != RunArtifacts.fingerprint(
        changed_media_transport
    )


def test_resume_reuses_all_completed_model_decisions() -> None:
    assert socialomni_benchmark._level1_reusable(
        {"stage_complete": True, "is_success": False, "predicted_answer": ""}
    )
    assert socialomni_benchmark._level2_reusable(
        {
            "gold_when": "NO",
            "predicted_when": "",
            "when_success": False,
            "stage_complete": True,
        }
    )
    assert socialomni_benchmark._level2_reusable(
        {
            "gold_when": "YES",
            "stage_complete": True,
            "predicted_when": "",
            "gold_response_request_completed": True,
            "gold_response_success": False,
            "gold_response": "",
        }
    )


def test_formal_provenance_requires_exact_h20_and_sglang_environment() -> None:
    provenance = {
        "repository": {"commit": "abc123", "dirty": False},
        "artifacts": {"declared_model_revision": "model-revision"},
        "launch_command": "serve --tp 8",
        "gpu": {
            "nvidia_smi_csv": "\n".join(f"{index}, NVIDIA H20" for index in range(8))
        },
        "packages": {"sglang": "0.5.18"},
    }

    assert socialomni_benchmark._formal_provenance_complete(provenance)
    assert not socialomni_benchmark._formal_provenance_complete(
        {**provenance, "launch_command": None}
    )
    assert not socialomni_benchmark._formal_provenance_complete(
        {**provenance, "packages": {"sglang": "0.5.16.dev"}}
    )


@pytest.mark.parametrize("stage_complete", [True, False])
def test_level2_resume_only_reuses_complete_model_results(stage_complete: bool) -> None:
    record = {
        "gold_when": "YES",
        "predicted_when": "YES",
        "when_success": True,
        "gold_response": "candidate",
        "gold_response_success": True,
        "stage_complete": stage_complete,
    }

    assert socialomni_benchmark._level2_reusable(record) is stage_complete


def test_judge_resume_requires_scores_and_details_from_all_three_judges() -> None:
    judge_names = {"gpt-4o", "gemini-2.5-pro", "qwen3-omni"}
    complete = {
        "gold_judge_scores": {name: 75 for name in judge_names},
        "judge_details": {name: {"raw_response": "75"} for name in judge_names},
    }
    missing_detail = {
        **complete,
        "judge_details": {"gpt-4o": {"raw_response": "75"}},
    }
    invalid_score = {
        **complete,
        "gold_judge_scores": {**complete["gold_judge_scores"], "gpt-4o": 80},
    }

    assert socialomni_benchmark._judge_reusable(complete, judge_names) is True
    assert socialomni_benchmark._judge_reusable(missing_detail, judge_names) is False
    assert socialomni_benchmark._judge_reusable(invalid_score, judge_names) is False


def test_resume_recovers_individual_valid_judge_pairs() -> None:
    judge_names = {"gpt-4o", "gemini-2.5-pro", "qwen3-omni"}
    records = [
        {
            "record_type": "judge_score",
            "sample_id": "one",
            "judge": "gpt-4o",
            "score": 75,
            "detail": {"raw_response": "75"},
        },
        {
            "record_type": "judge_score",
            "sample_id": "one",
            "judge": "gemini-2.5-pro",
            "score": 80,
            "detail": {"raw_response": "80"},
        },
    ]

    assert socialomni_benchmark._reusable_judge_pairs(records, judge_names) == {
        ("one", "gpt-4o")
    }


def test_fresh_judge_request_count_uses_missing_pairs() -> None:
    judges = [
        JudgeSpec(name=name, model=name, base_url="http://example")
        for name in ("gpt-4o", "gemini-2.5-pro", "qwen3-omni")
    ]
    assert (
        socialomni_benchmark._fresh_judge_request_count(
            [{"sample_id": "one"}],
            judges,
            {("one", "gpt-4o"), ("one", "gemini-2.5-pro")},
        )
        == 1
    )


def test_failed_model_result_is_fixed_wrong_answer_and_preserves_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sample = _level1(tmp_path / "video.mp4")
    dataset_info = SocialOmniDatasetInfo(
        root=str(tmp_path),
        version="test-revision",
        level1_file="dataset.json",
        level1_sha256="dataset-sha",
        level2_file=None,
        level2_sha256=None,
        manifest_file=None,
        manifest_sha256=None,
    )
    monkeypatch.setattr(
        socialomni_benchmark, "inspect_socialomni_dataset", lambda _root: dataset_info
    )
    monkeypatch.setattr(
        socialomni_benchmark,
        "load_socialomni_level1_samples",
        lambda *_args, **_kwargs: [sample],
    )
    monkeypatch.setattr(
        socialomni_benchmark,
        "collect_benchmark_provenance",
        lambda **_kwargs: {
            "repository": {"commit": "abc123", "dirty": False},
        },
    )

    async def successful_preflight(*_args) -> None:
        return None

    async def failed_phase(_samples, **kwargs):
        record = {
            "record_type": "result",
            "phase": "level1",
            "sample_id": sample.sample_id,
            "gold_answer": sample.answer,
            "predicted_answer": "",
            "visibility": sample.visibility,
            "stage_complete": True,
            "is_success": False,
            "latency_s": 0.1,
            "error": "HTTP 500: internal error",
        }
        kwargs["result_hook"](record)
        return [record]

    monkeypatch.setattr(socialomni_benchmark, "_preflight_main", successful_preflight)
    monkeypatch.setattr(socialomni_benchmark, "run_level1_model_phase", failed_phase)
    config = SocialOmniEvalConfig(
        dataset_root=str(tmp_path),
        model="model",
        level="level1",
        mini=True,
        output_dir=str(tmp_path / "results"),
        run_id="failed-run",
        bootstrap_samples=10,
    )

    result = asyncio.run(socialomni_benchmark.run_socialomni(config))

    artifact_dir = tmp_path / "results" / "abc123" / "failed-run"
    assert result["status"] == "complete"
    assert result["formal_evaluation_complete"] is False
    assert result["completion"]["level1_model"] == {
        "expected": 1,
        "present": 1,
        "completed": 1,
        "missing_sample_ids": [],
        "incomplete_sample_ids": [],
        "complete": True,
    }
    assert json.loads((artifact_dir / "manifest.json").read_text())["status"] == (
        "complete"
    )
    assert json.loads((artifact_dir / "summary.json").read_text())["status"] == (
        "complete"
    )
    assert "HTTP 500" in (artifact_dir / "per_request.jsonl").read_text()
    assert "HTTP 500" in (artifact_dir / "failures.jsonl").read_text()


def test_missing_model_result_marks_artifacts_incomplete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sample = _level1(tmp_path / "video.mp4")
    dataset_info = SocialOmniDatasetInfo(
        root=str(tmp_path),
        version="test-revision",
        level1_file="dataset.json",
        level1_sha256="dataset-sha",
        level2_file=None,
        level2_sha256=None,
        manifest_file=None,
        manifest_sha256=None,
    )
    monkeypatch.setattr(
        socialomni_benchmark, "inspect_socialomni_dataset", lambda _root: dataset_info
    )
    monkeypatch.setattr(
        socialomni_benchmark,
        "load_socialomni_level1_samples",
        lambda *_args, **_kwargs: [sample],
    )
    monkeypatch.setattr(
        socialomni_benchmark,
        "collect_benchmark_provenance",
        lambda **_kwargs: {
            "repository": {"commit": "abc123", "dirty": False},
        },
    )

    async def successful_preflight(*_args) -> None:
        return None

    async def empty_phase(_samples, **_kwargs):
        return []

    monkeypatch.setattr(socialomni_benchmark, "_preflight_main", successful_preflight)
    monkeypatch.setattr(socialomni_benchmark, "run_level1_model_phase", empty_phase)
    config = SocialOmniEvalConfig(
        dataset_root=str(tmp_path),
        model="model",
        level="level1",
        mini=True,
        output_dir=str(tmp_path / "results"),
        run_id="missing-run",
        bootstrap_samples=10,
    )

    with pytest.raises(RuntimeError, match="missing result records"):
        asyncio.run(socialomni_benchmark.run_socialomni(config))

    artifact_dir = tmp_path / "results" / "abc123" / "missing-run"
    assert json.loads((artifact_dir / "manifest.json").read_text())["status"] == (
        "incomplete"
    )
    assert json.loads((artifact_dir / "summary.json").read_text())["status"] == (
        "incomplete"
    )


def test_main_exits_nonzero_for_incomplete_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SocialOmniEvalConfig(dataset_root="dataset", model="model")

    class Parser:
        def parse_args(self):
            return type("Args", (), {})()

    args = Parser().parse_args()
    args.__dict__.update(config.__dict__)
    monkeypatch.setattr(
        socialomni_benchmark,
        "_parser",
        lambda: type("Parser", (), {"parse_args": lambda self: args})(),
    )

    async def incomplete_run(_config: SocialOmniEvalConfig) -> dict:
        return {"status": "incomplete"}

    monkeypatch.setattr(socialomni_benchmark, "run_socialomni", incomplete_run)

    with pytest.raises(SystemExit, match="1"):
        socialomni_benchmark.main()


def test_ffmpeg_command_reencodes_instead_of_stream_copy(tmp_path: Path) -> None:
    command = build_ffmpeg_prefix_command(
        "ffmpeg", tmp_path / "input.mp4", 1.25, tmp_path / "output.mp4"
    )

    assert "copy" not in command
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-c:a") + 1] == "aac"
    assert command[command.index("-t") + 1] == "1.250000"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_inline_frame_extraction_matches_fixed_sampling(tmp_path: Path) -> None:
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
            "color=c=red:s=32x24:r=10:d=2.2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )

    frames = asyncio.run(
        extract_inline_video_frames(
            source,
            interval_s=1.0,
            max_count=None,
            width=128,
            height=128,
            jpeg_quality=50,
        )
    )

    assert len(frames) == 3
    assert all(frame.startswith("data:image/jpeg;base64,") for frame in frames)
    encoded = frames[0].split(",", 1)[1]
    assert base64.b64decode(encoded).startswith(b"\xff\xd8")


def test_ffmpeg_resolver_falls_back_to_project_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundled = tmp_path / "ffmpeg"
    bundled.write_text("binary")
    bundled.chmod(0o755)
    monkeypatch.setattr(socialomni_tasks.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        socialomni_tasks.imageio_ffmpeg,
        "get_ffmpeg_exe",
        lambda: str(bundled),
    )

    assert resolve_ffmpeg_executable() == str(bundled)


def test_socialomni_http_sessions_honor_proxy_environment(monkeypatch) -> None:
    observed = {}

    def fake_session(**kwargs):
        observed.update(kwargs)
        return object()

    monkeypatch.setattr(socialomni_tasks.aiohttp, "ClientSession", fake_session)
    timeout = socialomni_tasks.aiohttp.ClientTimeout(total=1)

    assert _client_session(timeout) is not None
    assert observed == {"timeout": timeout, "trust_env": True}


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
