# SPDX-License-Identifier: Apache-2.0
"""Tests for video audio extraction."""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sglang_omni.models.qwen3_omni.components.preprocessor import (
    _merge_extracted_video_audio,
)
from sglang_omni.preprocessing import video


def _write_stereo_wav(path: Path, *, sample_rate: int = 8_000) -> None:
    samples = []
    for index in range(sample_rate // 10):
        value = int(10_000 * math.sin(2 * math.pi * 440 * index / sample_rate))
        samples.append(struct.pack("<hh", value, -value))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"".join(samples))


def test_extract_audio_from_path_decodes_resamples_and_downmixes(
    tmp_path: Path,
) -> None:
    media = tmp_path / "stereo.wav"
    _write_stereo_wav(media)

    audio = video._extract_audio_from_path(media, 16_000)

    assert audio is not None
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert 1_500 <= audio.size <= 1_700


def test_extract_audio_from_path_returns_none_without_audio(monkeypatch) -> None:
    class Container:
        def __init__(self) -> None:
            self.streams = [SimpleNamespace(type="video")]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(video.av, "open", lambda _path: Container())

    assert video._extract_audio_from_path(Path("silent.mp4"), 16_000) is None


def test_extract_audio_from_path_surfaces_decode_failure(monkeypatch) -> None:
    class Container:
        def __init__(self) -> None:
            self.streams = [SimpleNamespace(type="audio", index=0)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def decode(self, **_kwargs):
            raise RuntimeError("broken stream")

    monkeypatch.setattr(video.av, "open", lambda _path: Container())

    with pytest.raises(video.VideoDecodeError, match="broken stream"):
        video._extract_audio_from_path(Path("broken.mp4"), 16_000)


def test_merge_extracted_video_audio_requires_audio_for_every_video() -> None:
    explicit = [np.ones(2, dtype=np.float32)]
    video_audio = [np.ones(3, dtype=np.float32), None]

    merged, enabled = _merge_extracted_video_audio(explicit, video_audio)

    assert merged is explicit
    assert enabled is False


def test_merge_extracted_video_audio_preserves_alignment() -> None:
    explicit = [np.ones(2, dtype=np.float32)]
    video_audio = [np.ones(3, dtype=np.float32), np.ones(4, dtype=np.float32)]

    merged, enabled = _merge_extracted_video_audio(explicit, video_audio)

    assert enabled is True
    expected = [*explicit, *video_audio]
    assert len(merged) == len(expected)
    assert all(actual is wanted for actual, wanted in zip(merged, expected))
