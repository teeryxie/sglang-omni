# SPDX-License-Identifier: Apache-2.0
"""Strict local dataset loaders for the SocialOmni benchmark."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, TypeVar

SOCIALOMNI_DATASET_ID = "alexisty/SocialOmni"
SOCIALOMNI_DATASET_REVISION = "3b76009b45090eaa54007454c93a831f3cc8e1e6"
SOCIALOMNI_JFS_SNAPSHOT = "7e88c7e1afed65eb6fda17aac76bebbe2da57ec7"
SOCIALOMNI_LEVEL1_SHA256 = (
    "051f2e7ca0618de6e78843c64a3800606904be819f47c1e5c771f1c6a49faa23"
)
SOCIALOMNI_LEVEL2_SHA256 = (
    "87137aa2270f65e9c9e124e54dc5b73d9fe9c4bf9317219a4efe88feb036c058"
)
SOCIALOMNI_SUPPORTED_VERSIONS = frozenset(
    {SOCIALOMNI_DATASET_REVISION, SOCIALOMNI_JFS_SNAPSHOT}
)
SOCIALOMNI_LEVEL1_SIZE = 2_000
SOCIALOMNI_LEVEL2_SIZE = 209
SOCIALOMNI_PAPER_CORE_SIZE = 200

SocialOmniDatasetView = Literal["all", "paper-core-200"]


@dataclass(frozen=True)
class SocialOmniLevel1Sample:
    """One speaker-attribution (``who``) benchmark item."""

    sample_id: str
    video_path: str
    question: str
    options: list[str]
    answer: str
    visibility: str
    prompt: str
    asr_content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    consistency: str = "unknown"
    question_id: str = ""
    all_choices: list[str] = field(default_factory=lambda: ["A", "B", "C", "D"])
    index2ans: dict[str, str] = field(default_factory=dict)

    @property
    def gold_answer(self) -> str:
        return self.answer


# Retain the short name used by existing Video-MME-style benchmark adapters.
SocialOmniSample = SocialOmniLevel1Sample


@dataclass(frozen=True)
class SocialOmniLevel2Sample:
    """One fixed-prefix turn-entry and response-generation benchmark item."""

    sample_id: str
    video_path: str
    target_participant: str
    timestamp: str
    timestamp_s: float
    gold_when: str
    reference_response: str
    reference_context: str
    question_when: str
    question_how: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SocialOmniDatasetInfo:
    root: str
    version: str
    level1_file: str | None
    level1_sha256: str | None
    level2_file: str | None
    level2_sha256: str | None
    manifest_file: str | None
    manifest_sha256: str | None
    manifest_covers_media: bool = False

    @property
    def dataset_sha256(self) -> str:
        """Return one deterministic identity for the available metadata files."""

        hashes = [
            digest
            for digest in (
                self.level1_sha256,
                self.level2_sha256,
                self.manifest_sha256,
            )
            if digest is not None
        ]
        if len(hashes) == 1:
            return hashes[0]
        return hashlib.sha256("\n".join(hashes).encode()).hexdigest()

    @property
    def is_paper_snapshot(self) -> bool:
        """Whether metadata exactly matches the public paper dataset snapshot."""

        return (
            self.version in SOCIALOMNI_SUPPORTED_VERSIONS
            and self.level1_sha256 == SOCIALOMNI_LEVEL1_SHA256
            and self.level2_sha256 == SOCIALOMNI_LEVEL2_SHA256
            and self.manifest_covers_media
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_manifest_entries(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", line)
        if match is None:
            raise ValueError(f"Invalid SHA-256 manifest line at {path}:{line_number}")
        relative = PurePosixPath(match.group(2))
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise ValueError(f"Unsafe path in SHA-256 manifest at {path}:{line_number}")
        entries[relative.as_posix()] = match.group(1).lower()
    return entries


def socialomni_media_manifest_covers(
    dataset_info: SocialOmniDatasetInfo, media_paths: Sequence[str]
) -> bool:
    """Return whether every selected media file is named by the locked manifest."""

    if not dataset_info.manifest_file or not dataset_info.manifest_covers_media:
        return False
    manifest = Path(dataset_info.manifest_file).resolve()
    entries = _sha256_manifest_entries(manifest)
    manifest_root = manifest.parent
    covered = {(manifest_root / relative).resolve() for relative in entries}
    return all(Path(media_path).resolve() in covered for media_path in media_paths)


def _candidate_level_dirs(root: Path, level_name: str) -> list[Path]:
    candidates = [root / "data" / level_name, root / level_name]
    if root.name == level_name:
        candidates.insert(0, root)
    return candidates


def _find_level_dir(
    root: Path,
    level_name: str,
    metadata_name: str,
    *,
    required: bool,
) -> Path | None:
    for candidate in _candidate_level_dirs(root, level_name):
        metadata_path = candidate / metadata_name
        if metadata_path.is_file():
            resolved = candidate.resolve()
            if not resolved.is_relative_to(
                root
            ) or not metadata_path.resolve().is_relative_to(root):
                raise ValueError(
                    f"SocialOmni {level_name} directory escapes dataset root: "
                    f"{candidate}"
                )
            return resolved
    if not required:
        return None
    expected = root / "data" / level_name / metadata_name
    raise FileNotFoundError(
        f"SocialOmni {level_name} metadata is missing; expected {expected}. "
        "Pass the dataset snapshot directory, its data directory, or the level "
        "directory to --dataset-root."
    )


def inspect_socialomni_dataset(
    dataset_root: str | Path,
) -> SocialOmniDatasetInfo:
    """Return local metadata identities without reading any media files."""

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SocialOmni dataset root does not exist: {root}")

    level1_dir = _find_level_dir(root, "level_1", "dataset.json", required=False)
    level2_dir = _find_level_dir(root, "level_2", "annotations.json", required=False)
    if level1_dir is None and level2_dir is None:
        raise FileNotFoundError(
            f"No SocialOmni metadata found below dataset root: {root}"
        )

    level1_file = level1_dir / "dataset.json" if level1_dir else None
    level2_file = level2_dir / "annotations.json" if level2_dir else None
    telos_manifest = root / ".telos-manifest.sha256"
    revision_marker = root / ".socialomni-revision.json"
    public_manifest = root / ".socialomni-files.sha256"
    manifest = (
        telos_manifest
        if telos_manifest.is_file()
        else public_manifest if public_manifest.is_file() else revision_marker
    )
    version = root.name
    if revision_marker.is_file():
        marker = _read_json(revision_marker)
        if (
            not isinstance(marker, dict)
            or marker.get("dataset_id") != SOCIALOMNI_DATASET_ID
            or not isinstance(marker.get("revision"), str)
            or not marker["revision"].strip()
        ):
            raise ValueError(f"Invalid SocialOmni revision marker: {revision_marker}")
        version = marker["revision"].strip()
    manifest_entries = (
        _sha256_manifest_entries(manifest)
        if manifest.is_file() and manifest.suffix == ".sha256"
        else {}
    )
    manifest_media = [
        path
        for path in manifest_entries
        if "/videos/" in f"/{path}" and Path(path).suffix.lower() == ".mp4"
    ]
    return SocialOmniDatasetInfo(
        root=str(root),
        version=version,
        level1_file=str(level1_file) if level1_file else None,
        level1_sha256=_sha256(level1_file) if level1_file else None,
        level2_file=str(level2_file) if level2_file else None,
        level2_sha256=_sha256(level2_file) if level2_file else None,
        manifest_file=str(manifest) if manifest.is_file() else None,
        manifest_sha256=_sha256(manifest) if manifest.is_file() else None,
        manifest_covers_media=bool(manifest_media),
    )


def _require_object(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{description} must be a JSON object")
    return value


def _require_text(row: dict[str, Any], key: str, description: str) -> str:
    value = row.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"{description} has no non-empty {key!r}")
    return str(value).strip()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in SocialOmni metadata {path}: {exc}") from exc


def _relative_media_parts(
    raw_path: str,
    *,
    level_name: str,
    description: str,
) -> tuple[str, ...]:
    if "\\" in raw_path or "\x00" in raw_path:
        raise ValueError(f"{description} uses an unsafe media path: {raw_path!r}")
    relative = PurePosixPath(raw_path)
    if relative.is_absolute() or any(part == ".." for part in relative.parts):
        raise ValueError(f"{description} uses an unsafe media path: {raw_path!r}")

    parts = list(relative.parts)
    if parts[:2] == ["data", level_name]:
        parts = parts[2:]
    elif parts[:1] == [level_name]:
        parts = parts[1:]
    elif parts[:1] in (["data"], ["level_1"], ["level_2"]):
        raise ValueError(f"{description} uses a media path for the wrong level")
    if parts[:1] == ["videos"]:
        parts = parts[1:]
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{description} uses an unsafe media path: {raw_path!r}")
    return tuple(parts)


def _resolve_video_path(
    level_dir: Path,
    raw_path: str,
    *,
    level_name: str,
    description: str,
) -> Path:
    parts = _relative_media_parts(
        raw_path, level_name=level_name, description=description
    )
    videos_dir = (level_dir / "videos").resolve()
    if not videos_dir.is_relative_to(level_dir):
        raise ValueError(f"{description} videos directory escapes the dataset level")
    video_path = videos_dir.joinpath(*parts).resolve()
    if not video_path.is_relative_to(videos_dir):
        raise ValueError(f"{description} media path escapes videos/: {raw_path!r}")
    if not video_path.is_file():
        raise FileNotFoundError(f"{description} video is missing: {video_path}")
    return video_path


def _strip_option_prefix(option: str) -> str:
    return re.sub(r"^[A-D][.)]\s*", "", option.strip(), flags=re.IGNORECASE)


def _format_level1_prompt(question: str, options: Sequence[str]) -> str:
    option_lines = "\n".join(
        f"{letter}. {option}"
        for letter, option in zip(("A", "B", "C", "D"), options, strict=True)
    )
    return (
        f"{question.strip()}\n{option_lines}\n"
        "Answer the multiple-choice question using the video and its audio. "
        "The last line must use the format 'Answer: $LETTER'."
    )


_SampleT = TypeVar("_SampleT")


def _first_per_group(
    samples: Sequence[_SampleT],
    groups: Sequence[str],
    group_for_sample: Callable[[_SampleT], str],
    *,
    level_name: str,
) -> list[_SampleT]:
    selected: dict[str, _SampleT] = {}
    for sample in samples:
        group = group_for_sample(sample)
        if group in groups and group not in selected:
            selected[group] = sample
        if len(selected) == len(groups):
            break
    missing = [group for group in groups if group not in selected]
    if missing:
        raise ValueError(
            f"SocialOmni {level_name} cannot form its deterministic mini set; "
            f"missing groups: {missing}"
        )
    return [selected[group] for group in groups]


def select_socialomni_level1_mini(
    samples: Sequence[SocialOmniLevel1Sample],
) -> list[SocialOmniLevel1Sample]:
    """Select the first visible and first visibility-mismatch item."""

    return _first_per_group(
        samples,
        ("speaker_visible", "visibility_mismatch"),
        lambda sample: sample.visibility,
        level_name="Level 1",
    )


def select_socialomni_level2_mini(
    samples: Sequence[SocialOmniLevel2Sample],
) -> list[SocialOmniLevel2Sample]:
    """Select the first gold-YES and first gold-NO item."""

    return _first_per_group(
        samples,
        ("YES", "NO"),
        lambda sample: sample.gold_when,
        level_name="Level 2",
    )


def _validate_max_samples(max_samples: int | None) -> None:
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative")


def load_socialomni_level1_samples(
    dataset_root: str | Path,
    *,
    max_samples: int | None = None,
    mini: bool = False,
) -> list[SocialOmniLevel1Sample]:
    """Load Level 1 while keeping all nested paths beneath ``videos/``."""

    _validate_max_samples(max_samples)
    if max_samples == 0:
        return []

    root = Path(dataset_root).expanduser().resolve()
    level_dir = _find_level_dir(root, "level_1", "dataset.json", required=True)
    assert level_dir is not None
    payload = _read_json(level_dir / "dataset.json")
    if not isinstance(payload, list):
        raise TypeError("SocialOmni Level 1 dataset.json must contain a JSON array")

    samples: list[SocialOmniLevel1Sample] = []
    seen_ids: set[str] = set()
    for row_index, raw_row in enumerate(payload):
        description = f"SocialOmni Level 1 row {row_index}"
        row = _require_object(raw_row, description)
        sample_id = _require_text(row, "id", description)
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate SocialOmni Level 1 sample id: {sample_id}")
        seen_ids.add(sample_id)

        question = _require_text(row, "question", description)
        raw_options = row.get("options")
        if not isinstance(raw_options, list) or len(raw_options) != 4:
            raise ValueError(f"{description} must contain exactly four options")
        options = [_strip_option_prefix(str(option)) for option in raw_options]

        answer = _require_text(row, "correct_answer", description).upper()
        if answer not in {"A", "B", "C", "D"}:
            raise ValueError(f"{description} has invalid correct_answer {answer!r}")
        metadata = _require_object(row.get("metadata"), f"{description} metadata")
        consistency = _require_text(metadata, "consistency", description).lower()
        visibility_by_consistency = {
            "consistent": "speaker_visible",
            "inconsistent": "visibility_mismatch",
        }
        if consistency not in visibility_by_consistency:
            raise ValueError(f"{description} has invalid consistency {consistency!r}")

        video_path = _resolve_video_path(
            level_dir,
            _require_text(row, "video_path", description),
            level_name="level_1",
            description=description,
        )
        choices = ["A", "B", "C", "D"]
        samples.append(
            SocialOmniLevel1Sample(
                sample_id=sample_id,
                video_path=str(video_path),
                question=question,
                options=options,
                answer=answer,
                visibility=visibility_by_consistency[consistency],
                prompt=_format_level1_prompt(question, options),
                asr_content=str(row.get("asr_content") or "").strip(),
                metadata=dict(metadata),
                consistency=consistency,
                question_id=sample_id,
                all_choices=choices,
                index2ans=dict(zip(choices, options, strict=True)),
            )
        )
        if mini and {sample.visibility for sample in samples} == {
            "speaker_visible",
            "visibility_mismatch",
        }:
            break
        if not mini and max_samples is not None and len(samples) >= max_samples:
            break

    if mini:
        samples = select_socialomni_level1_mini(samples)
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples


def parse_socialomni_timestamp(value: Any) -> float:
    """Convert the dataset's seconds or ``MM:SS:centiseconds`` value to seconds."""

    if isinstance(value, bool):
        raise TypeError(f"Invalid SocialOmni timestamp type: {value!r}")
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value or "").strip()
        if not text:
            raise ValueError("SocialOmni timestamp must not be empty")
        if re.fullmatch(r"\d+(?:\.\d+)?", text):
            seconds = float(text)
        elif re.fullmatch(r"\d+:\d+(?:\.\d+)?", text):
            minutes, second_text = text.split(":", 1)
            seconds = int(minutes) * 60 + float(second_text)
        elif re.fullmatch(r"\d+:\d{2}:\d{2}", text):
            # The source benchmark records 00:17:00 for 17.00 seconds.
            minutes, second_text, centiseconds = text.split(":")
            seconds = int(minutes) * 60 + int(second_text) + int(centiseconds) / 100
        else:
            raise ValueError(f"Invalid SocialOmni timestamp: {value!r}")
    if seconds <= 0:
        raise ValueError(f"SocialOmni timestamp must be positive, got {value!r}")
    return seconds


def _normalize_gold_when(question: dict[str, Any], description: str) -> str:
    raw_answer = _require_text(question, "correct_answer", description).upper()
    if raw_answer in {"YES", "NO"}:
        return raw_answer
    if raw_answer not in {"A", "B"}:
        raise ValueError(f"{description} has invalid correct_answer {raw_answer!r}")
    option = _require_text(question, f"option_{raw_answer}", description).upper()
    if option not in {"YES", "NO"}:
        raise ValueError(
            f"{description} maps {raw_answer} to invalid option {option!r}"
        )
    return option


def _target_participant(question_how: str, question_when: str, description: str) -> str:
    match = re.fullmatch(r"What should (.+?) say\?", question_how, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.match(
        r"(?:Should|Did) (.+?) speak(?:\s|\?)", question_when, flags=re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    raise ValueError(f"{description} does not identify the target participant")


def load_socialomni_level2_samples(
    dataset_root: str | Path,
    *,
    view: SocialOmniDatasetView = "all",
    max_samples: int | None = None,
    mini: bool = False,
) -> list[SocialOmniLevel2Sample]:
    """Load Level 2 in source order using either all 209 or the paper core."""

    _validate_max_samples(max_samples)
    if view not in {"all", "paper-core-200"}:
        raise ValueError("view must be 'all' or 'paper-core-200'")
    if max_samples == 0:
        return []

    root = Path(dataset_root).expanduser().resolve()
    level_dir = _find_level_dir(root, "level_2", "annotations.json", required=True)
    assert level_dir is not None
    payload = _read_json(level_dir / "annotations.json")
    if isinstance(payload, dict):
        declared_total = payload.get("total_samples")
        payload = payload.get("data")
        if (
            declared_total is not None
            and isinstance(payload, list)
            and declared_total != len(payload)
        ):
            raise ValueError(
                "SocialOmni Level 2 total_samples does not match data length: "
                f"{declared_total} != {len(payload)}"
            )
    if not isinstance(payload, list):
        raise TypeError(
            "SocialOmni Level 2 annotations.json must be an array or contain a data array"
        )
    if view == "paper-core-200":
        payload = payload[:SOCIALOMNI_PAPER_CORE_SIZE]

    samples: list[SocialOmniLevel2Sample] = []
    seen_ids: set[str] = set()
    for row_index, raw_row in enumerate(payload):
        description = f"SocialOmni Level 2 row {row_index}"
        row = _require_object(raw_row, description)
        sample_id = _require_text(row, "video_id", description)
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate SocialOmni Level 2 sample id: {sample_id}")
        seen_ids.add(sample_id)

        question_when_data = _require_object(
            row.get("question_1"), f"{description} question_1"
        )
        question_how_data = _require_object(
            row.get("question_2"), f"{description} question_2"
        )
        question_when = _require_text(
            question_when_data, "question", f"{description} question_1"
        )
        question_how = _require_text(
            question_how_data, "question", f"{description} question_2"
        )
        timestamp = _require_text(
            question_when_data, "timestamp", f"{description} question_1"
        )
        video_path = _resolve_video_path(
            level_dir,
            _require_text(row, "video_file", description),
            level_name="level_2",
            description=description,
        )
        metadata = _require_object(row.get("metadata"), f"{description} metadata")
        gold_when = _normalize_gold_when(
            question_when_data, f"{description} question_1"
        )
        reference_response = str(question_how_data.get("answer") or "").strip()
        if gold_when == "YES" and not reference_response:
            raise ValueError(
                f"{description} question_2 has no non-empty 'answer' for a "
                "gold-positive item"
            )
        samples.append(
            SocialOmniLevel2Sample(
                sample_id=sample_id,
                video_path=str(video_path),
                target_participant=_target_participant(
                    question_how, question_when, description
                ),
                timestamp=timestamp,
                timestamp_s=parse_socialomni_timestamp(timestamp),
                gold_when=gold_when,
                reference_response=reference_response,
                reference_context=_require_text(row, "full_asr", description),
                question_when=question_when,
                question_how=question_how,
                metadata=dict(metadata),
            )
        )
        if mini and {sample.gold_when for sample in samples} == {"YES", "NO"}:
            break
        if not mini and max_samples is not None and len(samples) >= max_samples:
            break

    if mini:
        samples = select_socialomni_level2_mini(samples)
    if max_samples is not None:
        samples = samples[:max_samples]
    return samples
