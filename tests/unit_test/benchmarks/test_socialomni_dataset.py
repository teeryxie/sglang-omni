from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from benchmarks.dataset import prepare
from benchmarks.dataset.socialomni import (
    SOCIALOMNI_DATASET_REVISION,
    SOCIALOMNI_JFS_SNAPSHOT,
    SOCIALOMNI_LEVEL1_SHA256,
    SOCIALOMNI_LEVEL2_SHA256,
    SocialOmniDatasetInfo,
    inspect_socialomni_dataset,
    load_socialomni_level1_samples,
    load_socialomni_level2_samples,
    parse_socialomni_timestamp,
    socialomni_media_manifest_covers,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _level1_row(sample_id: int, video_path: str, consistency: str) -> dict[str, object]:
    return {
        "id": sample_id,
        "video_path": video_path,
        "question": "Who spoke?",
        "options": ["A. Person A", "B. Person B", "C. Person C", "D. Person D"],
        "correct_answer": "B",
        "asr_content": "judge-only reference",
        "metadata": {"consistency": consistency},
    }


def _level2_row(
    sample_id: int, video_file: str, answer: str, timestamp: str = "12"
) -> dict[str, object]:
    person = f"person {sample_id}"
    return {
        "video_id": f"video_{sample_id:04d}",
        "video_file": video_file,
        "question_1": {
            "question": f"Should {person} speak at the {timestamp}th second?",
            "timestamp": timestamp,
            "option_A": "YES",
            "option_B": "NO",
            "correct_answer": answer,
        },
        "question_2": {
            "question": f"What should {person} say?",
            "answer": f"reference {sample_id}",
        },
        "metadata": {"consistency": "consistent"},
        "full_asr": f"reference context {sample_id}",
    }


def test_level1_preserves_nested_paths_and_builds_paper_fields(tmp_path: Path) -> None:
    video = tmp_path / "data/level_1/videos/show/scene/video.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    _write_json(
        tmp_path / "data/level_1/dataset.json",
        [_level1_row(7, "data/level_1/videos/show/scene/video.mp4", "consistent")],
    )

    sample = load_socialomni_level1_samples(tmp_path)[0]

    assert sample.sample_id == "7"
    assert sample.video_path == str(video)
    assert sample.options == ["Person A", "Person B", "Person C", "Person D"]
    assert sample.answer == sample.gold_answer == "B"
    assert sample.visibility == "speaker_visible"
    assert sample.consistency == "consistent"
    assert "judge-only reference" not in sample.prompt


def test_level1_preserves_a_labeled_blank_option(tmp_path: Path) -> None:
    video = tmp_path / "data/level_1/videos/video.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    row = _level1_row(738, "video.mp4", "consistent")
    row["options"] = ["A. First", "B. ", "C. Third", "D. Fourth"]
    row["correct_answer"] = "B"
    _write_json(tmp_path / "data/level_1/dataset.json", [row])

    sample = load_socialomni_level1_samples(tmp_path)[0]

    assert sample.options == ["First", "", "Third", "Fourth"]
    assert "\nB. \n" in sample.prompt
    assert sample.answer == "B"


def test_level1_mini_is_deterministic_and_covers_both_visibility_groups(
    tmp_path: Path,
) -> None:
    rows = [
        _level1_row(1, "one.mp4", "consistent"),
        _level1_row(2, "two.mp4", "consistent"),
        _level1_row(3, "nested/three.mp4", "inconsistent"),
    ]
    for relative in ("one.mp4", "two.mp4", "nested/three.mp4"):
        video = tmp_path / "data/level_1/videos" / relative
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"video")
    _write_json(tmp_path / "data/level_1/dataset.json", rows)

    samples = load_socialomni_level1_samples(tmp_path, mini=True)

    assert [sample.sample_id for sample in samples] == ["1", "3"]
    assert [sample.visibility for sample in samples] == [
        "speaker_visible",
        "visibility_mismatch",
    ]


@pytest.mark.parametrize(
    "unsafe_path",
    ["../outside.mp4", "/tmp/outside.mp4", "data/level_2/videos/wrong.mp4"],
)
def test_level1_rejects_unsafe_or_wrong_level_paths(
    tmp_path: Path, unsafe_path: str
) -> None:
    _write_json(
        tmp_path / "data/level_1/dataset.json",
        [_level1_row(1, unsafe_path, "consistent")],
    )

    with pytest.raises(ValueError, match="unsafe|wrong level"):
        load_socialomni_level1_samples(tmp_path)


def test_level1_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"video")
    videos = tmp_path / "data/level_1/videos"
    videos.mkdir(parents=True)
    (videos / "link.mp4").symlink_to(outside)
    _write_json(
        tmp_path / "data/level_1/dataset.json",
        [_level1_row(1, "link.mp4", "consistent")],
    )

    with pytest.raises(ValueError, match="escapes videos"):
        load_socialomni_level1_samples(tmp_path)


def test_level1_rejects_symlinked_videos_directory_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "video.mp4").write_bytes(b"video")
    level_dir = tmp_path / "data/level_1"
    level_dir.mkdir(parents=True)
    (level_dir / "videos").symlink_to(outside, target_is_directory=True)
    _write_json(
        level_dir / "dataset.json",
        [_level1_row(1, "video.mp4", "consistent")],
    )

    with pytest.raises(ValueError, match="videos directory escapes"):
        load_socialomni_level1_samples(tmp_path)


def test_max_samples_does_not_require_unselected_media(tmp_path: Path) -> None:
    videos = tmp_path / "data/level_1/videos"
    videos.mkdir(parents=True)
    (videos / "available.mp4").write_bytes(b"video")
    _write_json(
        tmp_path / "data/level_1/dataset.json",
        [
            _level1_row(1, "available.mp4", "consistent"),
            _level1_row(2, "not-downloaded.mp4", "inconsistent"),
        ],
    )

    samples = load_socialomni_level1_samples(tmp_path, max_samples=1)

    assert [sample.sample_id for sample in samples] == ["1"]


def test_level2_views_preserve_source_order(tmp_path: Path) -> None:
    rows = []
    for index in range(1, 210):
        video_file = f"nested/video_{index:04d}.mp4"
        video = tmp_path / "data/level_2/videos" / video_file
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"video")
        rows.append(_level2_row(index, video_file, "A" if index % 2 else "B"))
    _write_json(
        tmp_path / "data/level_2/annotations.json",
        {"total_samples": 209, "data": rows},
    )

    all_samples = load_socialomni_level2_samples(tmp_path, view="all")
    paper_samples = load_socialomni_level2_samples(tmp_path, view="paper-core-200")

    assert len(all_samples) == 209
    assert len(paper_samples) == 200
    assert [sample.sample_id for sample in paper_samples] == [
        sample.sample_id for sample in all_samples[:200]
    ]
    assert all_samples[-1].sample_id == "video_0209"


def test_level2_fields_and_deterministic_mini(tmp_path: Path) -> None:
    videos = tmp_path / "data/level_2/videos"
    videos.mkdir(parents=True)
    for filename in ("yes.mp4", "yes-again.mp4", "no.mp4"):
        (videos / filename).write_bytes(b"video")
    rows = [
        _level2_row(1, "yes.mp4", "A", "00:17:00"),
        _level2_row(2, "yes-again.mp4", "A"),
        _level2_row(3, "no.mp4", "B"),
    ]
    _write_json(
        tmp_path / "data/level_2/annotations.json",
        {"total_samples": 3, "data": rows},
    )

    samples = load_socialomni_level2_samples(tmp_path, mini=True)

    assert [sample.sample_id for sample in samples] == ["video_0001", "video_0003"]
    assert [sample.gold_when for sample in samples] == ["YES", "NO"]
    assert samples[0].target_participant == "person 1"
    assert samples[0].timestamp == "00:17:00"
    assert samples[0].timestamp_s == 17.0
    assert samples[0].reference_response == "reference 1"
    assert samples[0].reference_context == "reference context 1"


def test_level2_allows_blank_reference_only_for_gold_negative(tmp_path: Path) -> None:
    videos = tmp_path / "data/level_2/videos"
    videos.mkdir(parents=True)
    for filename in ("no.mp4", "yes.mp4"):
        (videos / filename).write_bytes(b"video")
    no_row = _level2_row(1, "no.mp4", "B")
    no_row["question_2"]["answer"] = ""  # type: ignore[index]
    yes_row = _level2_row(2, "yes.mp4", "A")
    yes_row["question_2"]["answer"] = ""  # type: ignore[index]
    _write_json(
        tmp_path / "data/level_2/annotations.json",
        {"total_samples": 1, "data": [no_row]},
    )

    sample = load_socialomni_level2_samples(tmp_path)[0]
    assert sample.gold_when == "NO"
    assert sample.reference_response == ""

    _write_json(
        tmp_path / "data/level_2/annotations.json",
        {"total_samples": 1, "data": [yes_row]},
    )
    with pytest.raises(ValueError, match="gold-positive"):
        load_socialomni_level2_samples(tmp_path)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("27", 27.0), ("1:02", 62.0), ("00:17:50", 17.5), (3.25, 3.25)],
)
def test_parse_socialomni_timestamp(raw: object, expected: float) -> None:
    assert parse_socialomni_timestamp(raw) == expected


@pytest.mark.parametrize("raw", ["", "not-a-time", 0, -1])
def test_parse_socialomni_timestamp_rejects_invalid_values(raw: object) -> None:
    with pytest.raises(ValueError, match="timestamp"):
        parse_socialomni_timestamp(raw)


def test_parse_socialomni_timestamp_rejects_boolean() -> None:
    with pytest.raises(TypeError, match="timestamp type"):
        parse_socialomni_timestamp(True)


def test_inspection_supports_data_directory_as_root(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_json(data_root / "level_1/dataset.json", [])

    info = inspect_socialomni_dataset(data_root)

    assert info.root == str(data_root)
    assert info.level1_file == str(data_root / "level_1/dataset.json")
    assert info.level1_sha256 is not None
    assert info.dataset_sha256 == info.level1_sha256
    assert info.level2_file is None


def test_inspection_includes_snapshot_manifest_in_dataset_identity(
    tmp_path: Path,
) -> None:
    _write_json(tmp_path / "data/level_1/dataset.json", [])
    manifest = tmp_path / ".telos-manifest.sha256"
    manifest.write_text(f"{'1' * 64}  data/level_1/videos/one.mp4\n", encoding="utf-8")

    first = inspect_socialomni_dataset(tmp_path)
    manifest.write_text(f"{'2' * 64}  data/level_1/videos/one.mp4\n", encoding="utf-8")
    second = inspect_socialomni_dataset(tmp_path)

    assert first.level1_sha256 == second.level1_sha256
    assert first.dataset_sha256 != second.dataset_sha256


def test_selected_media_must_be_listed_in_snapshot_manifest(tmp_path: Path) -> None:
    _write_json(tmp_path / "data/level_1/dataset.json", [])
    video = tmp_path / "data/level_1/videos/one.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    manifest = tmp_path / ".telos-manifest.sha256"
    manifest.write_text(f"{'1' * 64}  data/level_1/videos/one.mp4\n", encoding="utf-8")

    info = inspect_socialomni_dataset(tmp_path)

    assert info.manifest_covers_media
    assert socialomni_media_manifest_covers(info, [str(video)])
    assert not socialomni_media_manifest_covers(
        info, [str(tmp_path / "data/level_1/videos/missing.mp4")]
    )


@pytest.mark.parametrize(
    "version", [SOCIALOMNI_DATASET_REVISION, SOCIALOMNI_JFS_SNAPSHOT]
)
def test_paper_snapshot_accepts_public_and_jfs_identities(version: str) -> None:
    info = SocialOmniDatasetInfo(
        root="/dataset",
        version=version,
        level1_file="dataset.json",
        level1_sha256=SOCIALOMNI_LEVEL1_SHA256,
        level2_file="annotations.json",
        level2_sha256=SOCIALOMNI_LEVEL2_SHA256,
        manifest_file=None,
        manifest_sha256=None,
        manifest_covers_media=True,
    )

    assert info.is_paper_snapshot
    assert not SocialOmniDatasetInfo(
        **{**info.__dict__, "level2_sha256": "different"}
    ).is_paper_snapshot


def test_prepare_socialomni_uses_pinned_snapshot_revision(
    monkeypatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}

    def fake_snapshot_download(**kwargs: object) -> str:
        observed.update(kwargs)
        media = Path(str(kwargs["local_dir"])) / "data/level_1/videos/one.mp4"
        media.parent.mkdir(parents=True)
        media.write_bytes(b"video")
        return "/cache/socialomni"

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            get_dataset_config_names=lambda *_args, **_kwargs: [],
            load_dataset=lambda *_args, **_kwargs: object(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            hf_hub_download=lambda *_args, **_kwargs: None,
            snapshot_download=fake_snapshot_download,
        ),
    )

    destination = tmp_path / "downloaded-socialomni"
    prepare.download_dataset(
        prepare.SOCIALOMNI_DATASET_ID, quiet=True, local_dir=str(destination)
    )

    assert observed["repo_id"] == prepare.SOCIALOMNI_DATASET_ID
    assert observed["repo_type"] == "dataset"
    assert observed["local_dir"] == str(destination)
    assert observed["revision"] == SOCIALOMNI_DATASET_REVISION
    assert observed["allow_patterns"] == [
        "README.md",
        "data/level_1/**",
        "data/level_2/**",
    ]
    marker = json.loads((destination / ".socialomni-revision.json").read_text())
    assert marker == {
        "dataset_id": prepare.SOCIALOMNI_DATASET_ID,
        "revision": SOCIALOMNI_DATASET_REVISION,
    }
    manifest = (destination / ".socialomni-files.sha256").read_text()
    assert "data/level_1/videos/one.mp4" in manifest
