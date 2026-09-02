# SPDX-License-Identifier: Apache-2.0
"""Run the SocialOmni fixed-prefix benchmark with auditable artifacts."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aiohttp

from benchmarks.dataset.socialomni import (
    SOCIALOMNI_PAPER_CORE_SIZE,
    SocialOmniLevel2Sample,
    inspect_socialomni_dataset,
    load_socialomni_level1_samples,
    load_socialomni_level2_samples,
)
from benchmarks.metrics.socialomni import (
    SOCIALOMNI_JUDGE_NAMES,
    compute_socialomni_level1_metrics,
    compute_socialomni_level2_metrics,
)
from benchmarks.runtime_metrics import collect_benchmark_provenance
from benchmarks.tasks.socialomni import (
    JudgeSpec,
    load_judge_config,
    merge_judge_records,
    preflight_endpoint,
    run_level1_model_phase,
    run_level2_judge_phase,
    run_level2_model_phase,
    speed_summary,
)

logger = logging.getLogger(__name__)
RESULT_SCHEMA_VERSION = 1
PROTOCOL_VERSION = "socialomni-paper-fixed-prefix-v1"


@dataclass(frozen=True)
class SocialOmniEvalConfig:
    dataset_root: str
    model: str
    base_url: str = "http://localhost:8000"
    level: str = "both"
    dataset_view: str = "all"
    judge_config: str | None = None
    output_dir: str = "results/socialomni"
    run_id: str | None = None
    prefix_cache_dir: str = "/local_nvme/xietianyu/tmp/socialomni-prefixes"
    mini: bool = False
    max_samples: int | None = None
    max_concurrency: int = 1
    judge_max_concurrency: int = 1
    max_attempts: int = 3
    timeout_s: int = 300
    level1_max_tokens: int = 32
    when_max_tokens: int = 8
    response_max_tokens: int = 256
    bootstrap_seed: int = 20260902
    bootstrap_samples: int = 10_000
    model_revision: str | None = None
    launch_command: str | None = None
    resume: bool = False


class RunArtifacts:
    """Append-only request journals plus atomic manifest and summary files."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest_path = root / "manifest.json"
        self.requests_path = root / "per_request.jsonl"
        self.judges_path = root / "judge_scores.jsonl"
        self.failures_path = root / "failures.jsonl"
        self.summary_path = root / "summary.json"

    @staticmethod
    def _canonical(payload: dict[str, Any]) -> str:
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )

    @classmethod
    def fingerprint(cls, contract: dict[str, Any]) -> str:
        return hashlib.sha256(cls._canonical(contract).encode()).hexdigest()

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise TypeError(
                    f"JSONL record at {path}:{line_number} must be an object"
                )
            rows.append(value)
        return rows

    def prepare(
        self, contract: dict[str, Any], provenance: dict[str, Any], *, resume: bool
    ) -> None:
        fingerprint = self.fingerprint(contract)
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if not resume:
                raise FileExistsError(
                    f"run manifest already exists at {self.manifest_path}; use --resume"
                )
            if existing.get("fingerprint") != fingerprint:
                raise ValueError(
                    "resume manifest fingerprint does not match this run contract"
                )
            return
        if resume and self.root.exists() and any(self.root.iterdir()):
            raise ValueError("cannot resume an artifact directory without a manifest")
        self._atomic_json(
            self.manifest_path,
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "fingerprint": fingerprint,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": "running",
                "contract": contract,
                "provenance": provenance,
            },
        )

    def finish(self, summary: dict[str, Any], *, status: str) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = status
        manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._atomic_json(self.manifest_path, manifest)
        self._atomic_json(self.summary_path, summary)

    def attempt_hook(self, payload: dict[str, Any]) -> None:
        self.append_jsonl(self.requests_path, payload)
        if payload.get("error"):
            self.append_jsonl(self.failures_path, payload)

    def model_result_hook(self, payload: dict[str, Any]) -> None:
        self.append_jsonl(self.requests_path, payload)
        if payload.get("error"):
            self.append_jsonl(self.failures_path, payload)

    def judge_result_hook(self, payload: dict[str, Any]) -> None:
        self.append_jsonl(self.judges_path, payload)


def _latest_by_sample(
    records: list[dict[str, Any]], record_type: str
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        if (
            record.get("record_type") == record_type
            and record.get("sample_id") is not None
        ):
            latest[str(record["sample_id"])] = record
    return latest


def _level1_reusable(record: dict[str, Any]) -> bool:
    return bool(record.get("is_success"))


def _level2_reusable(record: dict[str, Any]) -> bool:
    if not record.get("when_success"):
        return False
    return record.get("gold_when") != "YES" or bool(record.get("gold_response_success"))


def _ordered_records(
    samples: list[Any], records: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    missing = [
        sample.sample_id for sample in samples if sample.sample_id not in records
    ]
    if missing:
        raise RuntimeError(
            f"missing result records for {len(missing)} samples: {missing[:5]}"
        )
    return [records[sample.sample_id] for sample in samples]


def _judge_families(judges: list[JudgeSpec]) -> dict[str, str]:
    return {judge.name: judge.model_family or judge.name for judge in judges}


def _wall_speed(completed: int, elapsed_s: float) -> dict[str, Any]:
    return {
        "completed": completed,
        "wall_clock_s": round(elapsed_s, 6),
        "throughput_per_s": (
            round(completed / elapsed_s, 6) if completed and elapsed_s > 0 else None
        ),
    }


def _contract(
    config: SocialOmniEvalConfig,
    *,
    dataset_info: Any,
    level1_ids: list[str],
    level2_ids: list[str],
    judges: list[JudgeSpec],
    repository_commit: str | None,
) -> dict[str, Any]:
    return {
        "repository_commit": repository_commit,
        "protocol_version": PROTOCOL_VERSION,
        "model": config.model,
        "model_revision": config.model_revision,
        "base_url": config.base_url.rstrip("/"),
        "dataset": asdict(dataset_info),
        "level": config.level,
        "dataset_view": config.dataset_view,
        "mini": config.mini,
        "level1_sample_ids": level1_ids,
        "level2_sample_ids": level2_ids,
        "judges": [judge.public_dict() for judge in judges],
        "generation": {
            "temperature": 0.0,
            "top_p": None,
            "stream": False,
            "modalities": ["text"],
            "level1_max_tokens": config.level1_max_tokens,
            "when_max_tokens": config.when_max_tokens,
            "response_max_tokens": config.response_max_tokens,
        },
        "execution": {
            "max_concurrency": config.max_concurrency,
            "judge_max_concurrency": config.judge_max_concurrency,
            "max_attempts": config.max_attempts,
            "timeout_s": config.timeout_s,
            "prefix_cache_dir": str(Path(config.prefix_cache_dir)),
            "bootstrap_seed": config.bootstrap_seed,
            "bootstrap_samples": config.bootstrap_samples,
            "request_rate": None,
            "preflight_requests": {"main": 1, "per_judge": 1},
        },
    }


async def _preflight_main(config: SocialOmniEvalConfig) -> None:
    timeout = aiohttp.ClientTimeout(total=config.timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await preflight_endpoint(session, base_url=config.base_url, model=config.model)


async def _preflight_judges(
    config: SocialOmniEvalConfig, judges: list[JudgeSpec]
) -> None:
    timeout = aiohttp.ClientTimeout(total=config.timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for judge in judges:
            await preflight_endpoint(
                session,
                base_url=judge.base_url,
                model=judge.model,
                api_key_env=judge.api_key_env,
                judge_score=True,
                max_tokens=judge.max_tokens,
            )


async def run_socialomni(config: SocialOmniEvalConfig) -> dict[str, Any]:
    if config.level not in {"level1", "level2", "both"}:
        raise ValueError("level must be level1, level2, or both")
    if config.dataset_view not in {"all", "paper-core-200"}:
        raise ValueError("dataset_view must be all or paper-core-200")
    if config.max_concurrency < 1 or config.judge_max_concurrency < 1:
        raise ValueError("concurrency values must be >= 1")
    if config.max_attempts < 1 or config.bootstrap_samples < 1:
        raise ValueError("max_attempts and bootstrap_samples must be >= 1")

    dataset_info = inspect_socialomni_dataset(config.dataset_root)
    level1_samples = (
        load_socialomni_level1_samples(
            config.dataset_root, max_samples=config.max_samples, mini=config.mini
        )
        if config.level in {"level1", "both"}
        else []
    )
    level2_samples: list[SocialOmniLevel2Sample] = (
        load_socialomni_level2_samples(
            config.dataset_root,
            view=config.dataset_view,
            max_samples=config.max_samples,
            mini=config.mini,
        )
        if config.level in {"level2", "both"}
        else []
    )
    if config.level in {"level1", "both"} and not level1_samples:
        raise ValueError("SocialOmni Level 1 selected zero samples")
    if config.level in {"level2", "both"} and not level2_samples:
        raise ValueError("SocialOmni Level 2 selected zero samples")
    judges = load_judge_config(config.judge_config) if config.judge_config else []
    if level2_samples and len(judges) != 3:
        raise ValueError("Level 2 requires --judge-config with exactly three judges")
    if level2_samples and {judge.name for judge in judges} != set(
        SOCIALOMNI_JUDGE_NAMES
    ):
        raise ValueError(
            "Level 2 judge names must be exactly: " + ", ".join(SOCIALOMNI_JUDGE_NAMES)
        )
    judges = [
        JudgeSpec(
            name=judge.name,
            model=judge.model,
            base_url=judge.base_url,
            api_key_env=judge.api_key_env,
            max_concurrency=min(judge.max_concurrency, config.judge_max_concurrency),
            max_tokens=judge.max_tokens,
            model_family=judge.model_family,
            video_input=judge.video_input,
        )
        for judge in judges
    ]

    provenance = collect_benchmark_provenance(
        model_id=config.model,
        model_revision=config.model_revision,
        dataset_id="alexisty/SocialOmni",
        dataset_revision=dataset_info.version,
        evaluation_input_sha256=dataset_info.dataset_sha256,
        launch_command=config.launch_command,
        server_config={
            "base_url": config.base_url,
            "use_audio_in_video": True,
            "temperature": 0.0,
        },
    )
    contract = _contract(
        config,
        dataset_info=dataset_info,
        level1_ids=[sample.sample_id for sample in level1_samples],
        level2_ids=[sample.sample_id for sample in level2_samples],
        judges=judges,
        repository_commit=provenance["repository"]["commit"],
    )
    run_id = config.run_id or (
        f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}"
    )
    repository_commit = provenance["repository"]["commit"] or "unknown-commit"
    artifacts = RunArtifacts(Path(config.output_dir) / repository_commit[:12] / run_id)
    artifacts.prepare(contract, provenance, resume=config.resume)

    try:
        await _preflight_main(config)
        summary: dict[str, Any] = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "run_id": run_id,
            "status": "complete",
            "artifact_dir": str(artifacts.root),
        }

        if level1_samples:
            existing = _latest_by_sample(
                RunArtifacts.read_jsonl(artifacts.requests_path), "result"
            )
            reusable = {
                sample_id: record
                for sample_id, record in existing.items()
                if record.get("phase") == "level1" and _level1_reusable(record)
            }
            pending = [
                sample for sample in level1_samples if sample.sample_id not in reusable
            ]
            phase_started = time.perf_counter()
            fresh = await run_level1_model_phase(
                pending,
                base_url=config.base_url,
                model=config.model,
                max_tokens=config.level1_max_tokens,
                max_concurrency=config.max_concurrency,
                max_attempts=config.max_attempts,
                timeout_s=config.timeout_s,
                attempt_hook=artifacts.attempt_hook,
                result_hook=artifacts.model_result_hook,
            )
            phase_wall = time.perf_counter() - phase_started
            reusable.update({str(record["sample_id"]): record for record in fresh})
            level1_records = _ordered_records(level1_samples, reusable)
            summary["level1"] = {
                "metrics": compute_socialomni_level1_metrics(level1_records),
                "speed": {
                    "request_latency": speed_summary(level1_records, "latency_s"),
                    "fresh_phase": _wall_speed(len(fresh), phase_wall),
                },
            }

        if level2_samples:
            existing = _latest_by_sample(
                RunArtifacts.read_jsonl(artifacts.requests_path), "result"
            )
            reusable_model = {
                sample_id: record
                for sample_id, record in existing.items()
                if record.get("phase") == "level2_model" and _level2_reusable(record)
            }
            pending = [
                sample
                for sample in level2_samples
                if sample.sample_id not in reusable_model
            ]
            model_phase_started = time.perf_counter()
            fresh_model = await run_level2_model_phase(
                pending,
                base_url=config.base_url,
                model=config.model,
                prefix_cache_dir=config.prefix_cache_dir,
                when_max_tokens=config.when_max_tokens,
                response_max_tokens=config.response_max_tokens,
                max_concurrency=config.max_concurrency,
                max_attempts=config.max_attempts,
                timeout_s=config.timeout_s,
                attempt_hook=artifacts.attempt_hook,
                result_hook=artifacts.model_result_hook,
            )
            model_phase_wall = time.perf_counter() - model_phase_started
            reusable_model.update(
                {str(record["sample_id"]): record for record in fresh_model}
            )
            model_records = _ordered_records(level2_samples, reusable_model)

            existing_judges = _latest_by_sample(
                RunArtifacts.read_jsonl(artifacts.judges_path), "judge_result"
            )
            expected_names = {judge.name for judge in judges}
            reusable_judges = {
                sample_id: record
                for sample_id, record in existing_judges.items()
                if set(record.get("gold_judge_scores", {})) == expected_names
            }
            judge_pending = [
                record
                for record in model_records
                if record.get("gold_when") == "YES"
                and str(record.get("gold_response", "")).strip()
                and str(record["sample_id"]) not in reusable_judges
            ]
            if judge_pending:
                await _preflight_judges(config, judges)
            judge_phase_started = time.perf_counter()
            fresh_judges = await run_level2_judge_phase(
                {sample.sample_id: sample for sample in level2_samples},
                judge_pending,
                judges=judges,
                max_concurrency=config.judge_max_concurrency,
                max_attempts=config.max_attempts,
                timeout_s=config.timeout_s,
                attempt_hook=artifacts.attempt_hook,
                result_hook=artifacts.judge_result_hook,
            )
            judge_phase_wall = time.perf_counter() - judge_phase_started
            reusable_judges.update(
                {str(record["sample_id"]): record for record in fresh_judges}
            )
            merged = merge_judge_records(model_records, list(reusable_judges.values()))
            judge_names = [judge.name for judge in judges]
            all_metrics = compute_socialomni_level2_metrics(
                merged,
                judge_names=judge_names,
                judge_model_families=_judge_families(judges),
                bootstrap_seed=config.bootstrap_seed,
                bootstrap_samples=config.bootstrap_samples,
            )
            level2_summary: dict[str, Any] = {
                "selected_view": config.dataset_view,
                "selected": all_metrics,
                "speed": {
                    "when": speed_summary(merged, "when_latency_s"),
                    "gold_response": speed_summary(merged, "gold_response_latency_s"),
                    "fresh_model_phase": _wall_speed(
                        len(fresh_model), model_phase_wall
                    ),
                    "fresh_judge_phase": _wall_speed(
                        len(judge_pending) * len(judges), judge_phase_wall
                    ),
                },
            }
            if (
                config.dataset_view == "all"
                and len(merged) >= SOCIALOMNI_PAPER_CORE_SIZE
            ):
                level2_summary["paper_core_200"] = compute_socialomni_level2_metrics(
                    merged[:SOCIALOMNI_PAPER_CORE_SIZE],
                    judge_names=judge_names,
                    judge_model_families=_judge_families(judges),
                    bootstrap_seed=config.bootstrap_seed,
                    bootstrap_samples=config.bootstrap_samples,
                )
            summary["level2"] = level2_summary

        formal_level1 = config.level not in {"level1", "both"} or (
            config.max_samples is None and len(level1_samples) == 2000
        )
        formal_level2 = config.level not in {"level2", "both"} or (
            config.max_samples is None
            and (
                (config.dataset_view == "paper-core-200" and len(level2_samples) == 200)
                or (config.dataset_view == "all" and len(level2_samples) == 209)
            )
            and {judge.name for judge in judges} == set(SOCIALOMNI_JUDGE_NAMES)
        )
        clean_commit = (
            provenance["repository"]["commit"] is not None
            and provenance["repository"]["dirty"] is False
        )
        summary["clean_repository_commit"] = clean_commit
        summary["formal_evaluation_complete"] = (
            formal_level1 and formal_level2 and clean_commit
        )
        summary["failures"] = RunArtifacts.read_jsonl(artifacts.failures_path)
        artifacts.finish(summary, status="complete")
        return summary
    except Exception as exc:
        failure = {
            "record_type": "run_failure",
            "phase": "run",
            "error": f"{type(exc).__name__}: {exc}",
        }
        artifacts.append_jsonl(artifacts.failures_path, failure)
        artifacts.finish(
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "run_id": run_id,
                "status": "incomplete",
                "error": failure["error"],
                "artifact_dir": str(artifacts.root),
            },
            status="incomplete",
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--level", choices=("level1", "level2", "both"), default="both")
    parser.add_argument(
        "--dataset-view", choices=("all", "paper-core-200"), default="all"
    )
    parser.add_argument("--judge-config")
    parser.add_argument("--output-dir", default="results/socialomni")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--prefix-cache-dir",
        default="/local_nvme/xietianyu/tmp/socialomni-prefixes",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--judge-max-concurrency", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--level1-max-tokens", type=int, default=32)
    parser.add_argument("--when-max-tokens", type=int, default=8)
    parser.add_argument("--response-max-tokens", type=int, default=256)
    parser.add_argument("--bootstrap-seed", type=int, default=20260902)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--model-revision")
    parser.add_argument("--launch-command")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    config = SocialOmniEvalConfig(**vars(_parser().parse_args()))
    result = asyncio.run(run_socialomni(config))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
