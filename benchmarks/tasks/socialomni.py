# SPDX-License-Identifier: Apache-2.0
"""Execution helpers for the SocialOmni paper protocol."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import math
import os
import re
import shutil
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

import aiohttp
import imageio_ffmpeg
from PIL import Image

from benchmarks.dataset.socialomni import SocialOmniLevel1Sample, SocialOmniLevel2Sample

JUDGE_SCORE_BUCKETS = frozenset({0, 25, 50, 75, 100})
RETRYABLE_HTTP_STATUS = frozenset({408, 429, 500, 502, 503, 504})
PREFIX_ENCODING_VERSION = "h264-aac-v1"
PREFIX_ENCODING_CONFIG = {
    "video_stream": "0:v:0",
    "audio_stream": "0:a?",
    "video_codec": "libx264",
    "preset": "fast",
    "crf": "18",
    "audio_codec": "aac",
    "audio_bitrate": "192k",
    "avoid_negative_ts": "make_zero",
    "movflags": "+faststart",
}
T = TypeVar("T")
R = TypeVar("R")
MODEL_VIDEO_INPUTS = frozenset({"server-path", "inline-frames", "upload-av"})


@dataclass(frozen=True)
class ModelSpec:
    """One tested-model endpoint and its paper-visible media transport."""

    model: str
    base_url: str
    api_key_env: str | None = None
    video_input: str = "server-path"
    use_audio_in_video: bool = True
    frame_interval_s: float = 1.0
    frame_max_count: int | None = None
    frame_width: int = 128
    frame_height: int = 128
    frame_jpeg_quality: int = 50

    @property
    def api_url(self) -> str:
        if self.video_input == "upload-av":
            return f"{self.base_url.rstrip('/')}/analyze"
        return chat_completions_url(self.base_url)

    def validate(self) -> None:
        if self.video_input not in MODEL_VIDEO_INPUTS:
            raise ValueError(
                f"model video_input must be one of {sorted(MODEL_VIDEO_INPUTS)}"
            )
        if self.frame_interval_s <= 0:
            raise ValueError("frame_interval_s must be > 0")
        if self.frame_max_count is not None and self.frame_max_count < 1:
            raise ValueError("frame_max_count must be >= 1")
        if self.frame_width < 1 or self.frame_height < 1:
            raise ValueError("frame dimensions must be >= 1")
        if not 1 <= self.frame_jpeg_quality <= 95:
            raise ValueError("frame_jpeg_quality must be between 1 and 95")

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JudgeSpec:
    """One fixed evaluator endpoint; the API key value is never serialized."""

    name: str
    model: str
    base_url: str
    api_key_env: str | None = None
    max_concurrency: int = 1
    max_tokens: int = 8
    model_family: str | None = None
    video_input: str = "none"

    @property
    def api_url(self) -> str:
        return chat_completions_url(self.base_url)

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChatResult:
    request_id: str
    text: str
    is_success: bool
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    status_code: int | None = None
    error: str = ""
    retryable: bool = False

    @property
    def request_completed(self) -> bool:
        """Whether the endpoint returned a successful HTTP response."""

        return self.status_code is not None and 200 <= self.status_code < 300


AttemptHook = Callable[[dict[str, Any]], None]
RecordHook = Callable[[dict[str, Any]], None]


def chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def load_judge_config(path: str | Path) -> list[JudgeSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("judges") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("judge config must contain exactly three judges")
    allowed = {
        "name",
        "model",
        "base_url",
        "api_key_env",
        "max_concurrency",
        "max_tokens",
        "model_family",
        "video_input",
    }
    specs: list[JudgeSpec] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"judges[{index}] must be an object")
        unknown = set(row) - allowed
        if unknown:
            raise ValueError(f"judges[{index}] has unknown keys: {sorted(unknown)}")
        for key in ("name", "model", "base_url"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"judges[{index}].{key} must be non-empty")
        concurrency = row.get("max_concurrency", 1)
        if not isinstance(concurrency, int) or concurrency < 1:
            raise ValueError(f"judges[{index}].max_concurrency must be >= 1")
        max_tokens = row.get("max_tokens", 8)
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError(f"judges[{index}].max_tokens must be >= 1")
        video_input = row.get("video_input", "none")
        if video_input not in {"none", "server-path"}:
            raise ValueError(
                f"judges[{index}].video_input must be 'none' or 'server-path'"
            )
        specs.append(
            JudgeSpec(
                name=row["name"].strip(),
                model=row["model"].strip(),
                base_url=row["base_url"].strip().rstrip("/"),
                api_key_env=(row.get("api_key_env") or None),
                max_concurrency=concurrency,
                max_tokens=max_tokens,
                model_family=(row.get("model_family") or None),
                video_input=video_input,
            )
        )
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("judge names must be unique")
    return specs


def _headers(api_key_env: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if not api_key_env:
        return headers
    token = os.environ.get(api_key_env)
    if not token:
        raise RuntimeError(
            f"judge API key environment variable is not set: {api_key_env}"
        )
    headers["Authorization"] = f"Bearer {token}"
    return headers


def _message_text(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")).strip()
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return ""


def _client_session(timeout: aiohttp.ClientTimeout) -> aiohttp.ClientSession:
    return aiohttp.ClientSession(timeout=timeout, trust_env=True)


async def request_chat_completion(
    session: aiohttp.ClientSession,
    *,
    api_url: str,
    payload: dict[str, Any],
    request_id: str,
    phase: str,
    api_key_env: str | None = None,
    max_attempts: int = 3,
    retry_backoff_s: float = 1.0,
    attempt_offset: int = 0,
    attempt_hook: AttemptHook | None = None,
) -> ChatResult:
    """Send one request with bounded retry and an audit record per attempt."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    last = ChatResult(request_id, "", False, 0.0, error="not attempted")
    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        status: int | None = None
        text = ""
        error = ""
        prompt_tokens = 0
        completion_tokens = 0
        retryable = False
        try:
            async with session.post(
                api_url,
                json=payload,
                headers=_headers(api_key_env),
            ) as response:
                status = response.status
                response_text = await response.text()
                if status >= 400:
                    error = f"HTTP {status}: {response_text[:2000]}"
                    retryable = status in RETRYABLE_HTTP_STATUS or status >= 500
                else:
                    body = json.loads(response_text)
                    text = _message_text(body)
                    usage = body.get("usage") or {}
                    prompt_tokens = int(usage.get("prompt_tokens") or 0)
                    completion_tokens = int(usage.get("completion_tokens") or 0)
                    if not text:
                        error = "empty response"
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            error = str(exc)
            retryable = True
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            error = f"invalid response: {exc}"
        latency = time.perf_counter() - started
        success = not error
        last = ChatResult(
            request_id=request_id,
            text=text,
            is_success=success,
            latency_s=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            status_code=status,
            error=error,
            retryable=retryable,
        )
        if attempt_hook:
            attempt_hook(
                {
                    "record_type": "attempt",
                    "phase": phase,
                    "request_id": request_id,
                    "attempt": attempt + attempt_offset,
                    "success": success,
                    "status_code": status,
                    "latency_s": round(latency, 6),
                    "raw_response": text,
                    "error": error,
                }
            )
        if success or not retryable or attempt == max_attempts:
            return last
        await asyncio.sleep(retry_backoff_s * (2 ** (attempt - 1)))
    return last


async def request_upload_av_completion(
    session: aiohttp.ClientSession,
    *,
    api_url: str,
    video_path: str,
    prompt: str,
    request_id: str,
    phase: str,
    max_attempts: int = 3,
    retry_backoff_s: float = 1.0,
    attempt_hook: AttemptHook | None = None,
) -> ChatResult:
    """Call the common SocialOmni native AV multipart `/analyze` endpoint."""

    source = Path(video_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"video not found: {source}")
    last = ChatResult(request_id, "", False, 0.0, error="not attempted")
    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        status: int | None = None
        text = ""
        error = ""
        retryable = False
        try:
            form = aiohttp.FormData()
            form.add_field("question", prompt)
            form.add_field("use_video", "true")
            form.add_field("use_audio", "true")
            form.add_field("visual_mask", "false")
            with source.open("rb") as handle:
                form.add_field(
                    "video",
                    handle,
                    filename=source.name,
                    content_type="video/mp4",
                )
                async with session.post(api_url, data=form) as response:
                    status = response.status
                    response_text = await response.text()
                    if status >= 400:
                        error = f"HTTP {status}: {response_text[:2000]}"
                        retryable = status in RETRYABLE_HTTP_STATUS or status >= 500
                    else:
                        body = json.loads(response_text)
                        text = str(body.get("answer") or "")
                        if not text.strip():
                            error = "empty response"
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            error = str(exc)
            retryable = True
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            error = f"invalid response: {exc}"
        latency = time.perf_counter() - started
        success = not error
        last = ChatResult(
            request_id=request_id,
            text=text,
            is_success=success,
            latency_s=latency,
            status_code=status,
            error=error,
            retryable=retryable,
        )
        if attempt_hook:
            attempt_hook(
                {
                    "record_type": "attempt",
                    "phase": phase,
                    "request_id": request_id,
                    "attempt": attempt,
                    "success": success,
                    "status_code": status,
                    "latency_s": round(latency, 6),
                    "raw_response": text,
                    "error": error,
                }
            )
        if success or not retryable or attempt == max_attempts:
            return last
        await asyncio.sleep(retry_backoff_s * (2 ** (attempt - 1)))
    return last


async def preflight_endpoint(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    model: str,
    api_key_env: str | None = None,
    judge_score: bool = False,
    max_tokens: int = 4,
    attempt_hook: AttemptHook | None = None,
) -> None:
    result = await request_chat_completion(
        session,
        api_url=chat_completions_url(base_url),
        payload={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Reply with exactly 75 and no other text."
                        if judge_score
                        else "Reply with OK."
                    ),
                }
            ],
            "modalities": ["text"],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        },
        request_id=f"preflight:{model}",
        phase="preflight",
        api_key_env=api_key_env,
        max_attempts=1,
        attempt_hook=attempt_hook,
    )
    if not result.is_success:
        raise RuntimeError(f"endpoint preflight failed for {model}: {result.error}")
    if judge_score and parse_judge_score(result.text) != 75:
        raise RuntimeError(
            f"judge response-format preflight failed for {model}: {result.text!r}"
        )


def parse_choice(text: str, choices: Sequence[str]) -> str:
    content = (text or "").strip().upper()
    if not content:
        return ""
    alphabet = "".join(re.escape(choice) for choice in choices)
    tagged = re.findall(rf"(?:ANSWER|CHOICE)\s*(?:IS|:)?\s*([{alphabet}])\b", content)
    boxed = re.findall(rf"\\?BOXED\s*\{{\s*([{alphabet}])\s*\}}", content)
    explicit = tagged + boxed
    if explicit:
        return explicit[0] if len(set(explicit)) == 1 else ""
    plain = re.fullmatch(rf"([{alphabet}])[.)]?", content)
    if plain:
        return plain.group(1)
    leading = re.match(rf"^([{alphabet}])(?:[.)]|\s*[:\-])\s+", content)
    return leading.group(1) if leading else ""


def parse_when(text: str) -> str:
    choice = parse_choice(text, ("A", "B"))
    if choice:
        return "YES" if choice == "A" else "NO"
    normalized = (text or "").strip().upper()
    if normalized == "YES":
        return "YES"
    if normalized == "NO":
        return "NO"
    return ""


def parse_judge_score(text: str) -> int | None:
    numbers = re.findall(r"(?<!\d)(?:100|75|50|25|0)(?!\d)", text or "")
    if len(numbers) != 1:
        return None
    score = int(numbers[0])
    return score if score in JUDGE_SCORE_BUCKETS else None


def build_level1_prompt(
    sample: SocialOmniLevel1Sample, *, visual_only: bool = False
) -> str:
    stripped_options = [
        re.sub(r"^[A-D][.)]\s*", "", option.strip()) for option in sample.options
    ]
    options = "\n".join(
        f"{letter}. {option}"
        for letter, option in zip(("A", "B", "C", "D"), stripped_options, strict=True)
    )
    evidence = "video frames" if visual_only else "video and its audio"
    return (
        f"{sample.question.strip()}\n{options}\nUse the {evidence}. "
        "Reply on the last line as Answer: X, where X is A, B, C, or D."
    )


def build_when_prompt(
    sample: SocialOmniLevel2Sample, *, visual_only: bool = False
) -> str:
    question = sample.question_when.strip() or (
        "Should the target participant begin a substantive turn now?"
    )
    evidence = "video-frame prefix" if visual_only else "audio-video prefix"
    return (
        f"Target participant: {sample.target_participant}\n"
        f"{question}\nA. YES\nB. NO\n"
        f"Use only the provided {evidence}. Reply exactly as Answer: A or Answer: B."
    )


def build_response_prompt(
    sample: SocialOmniLevel2Sample, *, visual_only: bool = False
) -> str:
    question = sample.question_how.strip() or (
        "What should the target participant say next?"
    )
    evidence = "video-frame prefix" if visual_only else "audio-video prefix"
    return (
        f"Target participant: {sample.target_participant}\n{question}\n"
        f"Use only the provided {evidence}. Return only the participant's next utterance."
    )


def build_judge_prompt(sample: SocialOmniLevel2Sample, candidate: str) -> str:
    return (
        "Evaluate the candidate continuation at the annotated conversation state.\n"
        "Use the exact scale {0, 25, 50, 75, 100}, considering contextual fit, "
        "target-role consistency, intent correctness, and key information completeness.\n"
        "Output exactly one score and no other text.\n\n"
        f"Target participant:\n{sample.target_participant}\n\n"
        f"Reference context or transcript:\n{sample.reference_context}\n\n"
        f"Reference continuation:\n{sample.reference_response}\n\n"
        f"Candidate continuation:\n{candidate}"
    )


def model_payload(
    model: str,
    prompt: str,
    video_path: str,
    *,
    max_tokens: int,
    use_audio_in_video: bool = True,
    frame_data_uris: Sequence[str] = (),
) -> dict[str, Any]:
    if frame_data_uris:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(
            {"type": "image_url", "image_url": {"url": uri}} for uri in frame_data_uris
        )
        return {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "videos": [video_path],
        "use_audio_in_video": use_audio_in_video,
        "modalities": ["text"],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }


async def extract_inline_video_frames(
    video_path: str | Path,
    *,
    interval_s: float,
    max_count: int | None,
    width: int,
    height: int,
    jpeg_quality: int,
) -> tuple[str, ...]:
    """Extract paper-compatible visual-only frames as JPEG data URIs."""

    source = Path(video_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"video not found: {source}")
    ffmpeg = resolve_ffmpeg_executable()
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-an",
        "-vf",
        f"fps=1/{interval_s:.9g},scale={width}:{height}",
        "-pix_fmt",
        "rgb24",
    ]
    if max_count is not None:
        command.extend(["-frames:v", str(max_count)])
    command.extend(["-f", "rawvideo", "pipe:1"])
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg frame extraction failed: {detail}")
    frame_size = width * height * 3
    if not stdout or len(stdout) % frame_size:
        raise RuntimeError(
            f"ffmpeg returned invalid raw frame bytes: {len(stdout)} for {frame_size}"
        )

    frames: list[str] = []
    for offset in range(0, len(stdout), frame_size):
        image = Image.frombytes(
            "RGB", (width, height), stdout[offset : offset + frame_size]
        )
        encoded = io.BytesIO()
        image.save(encoded, format="JPEG", quality=jpeg_quality)
        payload = base64.b64encode(encoded.getvalue()).decode("ascii")
        frames.append(f"data:image/jpeg;base64,{payload}")
    if not frames:
        raise RuntimeError(f"no frames extracted from {source}")
    return tuple(frames)


async def prepare_model_payload(
    spec: ModelSpec,
    prompt: str,
    video_path: str,
    *,
    max_tokens: int,
) -> dict[str, Any]:
    if spec.video_input == "inline-frames":
        frames = await extract_inline_video_frames(
            video_path,
            interval_s=spec.frame_interval_s,
            max_count=spec.frame_max_count,
            width=spec.frame_width,
            height=spec.frame_height,
            jpeg_quality=spec.frame_jpeg_quality,
        )
        return model_payload(
            spec.model,
            prompt,
            video_path,
            max_tokens=max_tokens,
            frame_data_uris=frames,
        )
    return model_payload(
        spec.model,
        prompt,
        video_path,
        max_tokens=max_tokens,
        use_audio_in_video=spec.use_audio_in_video,
    )


async def request_model_completion(
    session: aiohttp.ClientSession,
    *,
    spec: ModelSpec,
    prompt: str,
    video_path: str,
    max_tokens: int,
    request_id: str,
    phase: str,
    max_attempts: int,
    attempt_hook: AttemptHook | None = None,
) -> ChatResult:
    if spec.video_input == "upload-av":
        return await request_upload_av_completion(
            session,
            api_url=spec.api_url,
            video_path=video_path,
            prompt=prompt,
            request_id=request_id,
            phase=phase,
            max_attempts=max_attempts,
            attempt_hook=attempt_hook,
        )
    payload = await prepare_model_payload(
        spec, prompt, video_path, max_tokens=max_tokens
    )
    return await request_chat_completion(
        session,
        api_url=spec.api_url,
        payload=payload,
        request_id=request_id,
        phase=phase,
        api_key_env=spec.api_key_env,
        max_attempts=max_attempts,
        attempt_hook=attempt_hook,
    )


def judge_payload(
    judge: JudgeSpec,
    prompt: str,
    video_path: str,
    *,
    max_tokens: int,
) -> dict[str, Any]:
    """Build a judge request without exposing a host-local path by default."""
    if judge.video_input == "server-path":
        return model_payload(judge.model, prompt, video_path, max_tokens=max_tokens)
    return {
        "model": judge.model,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["text"],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_ffmpeg_prefix_command(
    ffmpeg: str,
    source: Path,
    timestamp_s: float,
    output: Path,
) -> list[str]:
    """Build the exact re-encoding command used for query-time prefixes."""
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-t",
        f"{timestamp_s:.6f}",
        "-map",
        PREFIX_ENCODING_CONFIG["video_stream"],
        "-map",
        PREFIX_ENCODING_CONFIG["audio_stream"],
        "-c:v",
        PREFIX_ENCODING_CONFIG["video_codec"],
        "-preset",
        PREFIX_ENCODING_CONFIG["preset"],
        "-crf",
        PREFIX_ENCODING_CONFIG["crf"],
        "-c:a",
        PREFIX_ENCODING_CONFIG["audio_codec"],
        "-b:a",
        PREFIX_ENCODING_CONFIG["audio_bitrate"],
        "-avoid_negative_ts",
        PREFIX_ENCODING_CONFIG["avoid_negative_ts"],
        "-movflags",
        PREFIX_ENCODING_CONFIG["movflags"],
        "-y",
        str(output),
    ]


def resolve_ffmpeg_executable() -> str | None:
    """Resolve system ffmpeg or the binary supplied by imageio-ffmpeg."""

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    bundled_ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
    if bundled_ffmpeg.is_file() and os.access(bundled_ffmpeg, os.X_OK):
        return str(bundled_ffmpeg)
    return None


async def create_video_prefix(
    input_path: str | Path,
    timestamp_s: float,
    cache_dir: str | Path,
) -> Path:
    """Re-encode a query-time-bounded prefix and atomically populate its cache."""
    if not math.isfinite(timestamp_s) or timestamp_s <= 0:
        raise ValueError("timestamp_s must be finite and positive")
    ffmpeg = resolve_ffmpeg_executable()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for SocialOmni Level 2")
    source = Path(input_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_sha = await asyncio.to_thread(_sha256_file, source)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key_payload = {
        "source_sha256": source_sha,
        "timestamp_s": f"{timestamp_s:.6f}",
        "encoding_version": PREFIX_ENCODING_VERSION,
        "encoding": PREFIX_ENCODING_CONFIG,
    }
    key = hashlib.sha256(
        json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output = cache / f"{key}.mp4"
    if output.is_file() and output.stat().st_size > 0:
        return output
    temporary = cache / f".{key}.{uuid.uuid4().hex}.tmp.mp4"
    command = build_ffmpeg_prefix_command(ffmpeg, source, timestamp_s, temporary)
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()
    if (
        process.returncode != 0
        or not temporary.is_file()
        or temporary.stat().st_size == 0
    ):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg prefix generation failed for {source}: {stderr.decode(errors='replace')[:2000]}"
        )
    try:
        temporary.replace(output)
    except FileExistsError:
        temporary.unlink(missing_ok=True)
    return output


async def bounded_map(
    items: Sequence[T], worker: Callable[[T], Awaitable[R]], max_concurrency: int
) -> list[R]:
    """Map with a fixed worker set, preserving input order without unbounded tasks."""
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")
    queue: asyncio.Queue[tuple[int, T] | None] = asyncio.Queue()
    for index, item in enumerate(items):
        queue.put_nowait((index, item))
    worker_count = min(max_concurrency, len(items))
    for _ in range(worker_count):
        queue.put_nowait(None)
    results: list[R | None] = [None] * len(items)

    async def consume() -> None:
        while True:
            queued = await queue.get()
            if queued is None:
                return
            index, item = queued
            results[index] = await worker(item)

    tasks = [asyncio.create_task(consume()) for _ in range(worker_count)]
    try:
        if tasks:
            await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return [result for result in results if result is not None]


async def run_level1_model_phase(
    samples: Sequence[SocialOmniLevel1Sample],
    *,
    model_spec: ModelSpec,
    max_tokens: int,
    max_concurrency: int,
    max_attempts: int,
    timeout_s: int,
    attempt_hook: AttemptHook | None = None,
    result_hook: RecordHook | None = None,
) -> list[dict[str, Any]]:
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with _client_session(timeout) as session:

        async def run_one(sample: SocialOmniLevel1Sample) -> dict[str, Any]:
            result = await request_model_completion(
                session,
                spec=model_spec,
                prompt=build_level1_prompt(
                    sample, visual_only=model_spec.video_input == "inline-frames"
                ),
                video_path=sample.video_path,
                max_tokens=max_tokens,
                request_id=f"{sample.sample_id}:level1",
                phase="level1",
                max_attempts=max_attempts,
                attempt_hook=attempt_hook,
            )
            prediction = (
                parse_choice(result.text, ("A", "B", "C", "D"))
                if result.is_success
                else ""
            )
            record = {
                "record_type": "result",
                "phase": "level1",
                "sample_id": sample.sample_id,
                "gold_answer": sample.answer,
                "predicted_answer": prediction,
                "visibility": sample.visibility,
                "stage_complete": True,
                "is_success": result.is_success,
                "request_completed": result.request_completed,
                "raw_response": result.text,
                "latency_s": round(result.latency_s, 6),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "error": result.error,
            }
            if result_hook:
                result_hook(record)
            return record

        return await bounded_map(samples, run_one, max_concurrency)


async def run_level2_model_phase(
    samples: Sequence[SocialOmniLevel2Sample],
    *,
    model_spec: ModelSpec,
    prefix_cache_dir: str | Path,
    when_max_tokens: int,
    response_max_tokens: int,
    max_concurrency: int,
    max_attempts: int,
    timeout_s: int,
    attempt_hook: AttemptHook | None = None,
    result_hook: RecordHook | None = None,
) -> list[dict[str, Any]]:
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with _client_session(timeout) as session:

        async def run_one(sample: SocialOmniLevel2Sample) -> dict[str, Any]:
            started = time.perf_counter()
            prefix = await create_video_prefix(
                sample.video_path, sample.timestamp_s, prefix_cache_dir
            )
            when_result = await request_model_completion(
                session,
                spec=model_spec,
                prompt=build_when_prompt(
                    sample, visual_only=model_spec.video_input == "inline-frames"
                ),
                video_path=str(prefix),
                max_tokens=when_max_tokens,
                request_id=f"{sample.sample_id}:when",
                phase="level2_when",
                max_attempts=max_attempts,
                attempt_hook=attempt_hook,
            )
            predicted = parse_when(when_result.text) if when_result.is_success else ""
            response_result: ChatResult | None = None
            if sample.gold_when == "YES":
                response_result = await request_model_completion(
                    session,
                    spec=model_spec,
                    prompt=build_response_prompt(
                        sample, visual_only=model_spec.video_input == "inline-frames"
                    ),
                    video_path=str(prefix),
                    max_tokens=response_max_tokens,
                    request_id=f"{sample.sample_id}:gold_response",
                    phase="level2_gold_response",
                    max_attempts=max_attempts,
                    attempt_hook=attempt_hook,
                )
            record = {
                "record_type": "result",
                "phase": "level2_model",
                "sample_id": sample.sample_id,
                "video_path": sample.video_path,
                "prefix_path": str(prefix),
                "timestamp_s": sample.timestamp_s,
                "gold_when": sample.gold_when,
                "stage_complete": True,
                "predicted_when": predicted,
                "when_raw_response": when_result.text,
                "when_success": when_result.is_success,
                "when_request_completed": when_result.request_completed,
                "when_latency_s": round(when_result.latency_s, 6),
                "gold_response": response_result.text if response_result else "",
                "gold_response_success": (
                    response_result.is_success if response_result else None
                ),
                "gold_response_request_completed": (
                    response_result.request_completed if response_result else None
                ),
                "gold_response_latency_s": (
                    round(response_result.latency_s, 6) if response_result else None
                ),
                "gold_judge_scores": {},
                "workflow_latency_s": round(time.perf_counter() - started, 6),
                "error": "; ".join(
                    error
                    for error in (
                        when_result.error,
                        response_result.error if response_result else "",
                    )
                    if error
                ),
            }
            if result_hook:
                result_hook(record)
            return record

        return await bounded_map(samples, run_one, max_concurrency)


async def run_level2_judge_phase(
    samples_by_id: dict[str, SocialOmniLevel2Sample],
    model_records: Sequence[dict[str, Any]],
    *,
    judges: Sequence[JudgeSpec],
    max_concurrency: int,
    max_attempts: int,
    timeout_s: int,
    existing_judge_records: Sequence[dict[str, Any]] = (),
    attempt_hook: AttemptHook | None = None,
    result_hook: RecordHook | None = None,
) -> list[dict[str, Any]]:
    if len(judges) != 3:
        raise ValueError("formal SocialOmni evaluation requires exactly three judges")
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    semaphores = {
        judge.name: asyncio.Semaphore(judge.max_concurrency) for judge in judges
    }
    eligible = [
        record
        for record in model_records
        if record.get("gold_when") == "YES"
        and str(record.get("gold_response", "")).strip()
    ]
    eligible_ids = {str(record["sample_id"]) for record in eligible}
    expected_names = {judge.name for judge in judges}
    scores_by_sample: dict[str, dict[str, int]] = {}
    details_by_sample: dict[str, dict[str, Any]] = {}
    for existing in existing_judge_records:
        sample_id = str(existing.get("sample_id", ""))
        if sample_id not in eligible_ids:
            continue
        if existing.get("record_type") == "judge_score":
            name = str(existing.get("judge", ""))
            score = existing.get("score")
            detail = existing.get("detail")
            if (
                name in expected_names
                and not isinstance(score, bool)
                and isinstance(score, (int, float))
                and math.isfinite(float(score))
                and score in JUDGE_SCORE_BUCKETS
                and isinstance(detail, dict)
            ):
                scores_by_sample.setdefault(sample_id, {})[name] = int(score)
                details_by_sample.setdefault(sample_id, {})[name] = detail
        elif existing.get("record_type") == "judge_result":
            scores = existing.get("gold_judge_scores")
            details = existing.get("judge_details")
            if isinstance(scores, dict) and isinstance(details, dict):
                for name, score in scores.items():
                    if (
                        name in expected_names
                        and not isinstance(score, bool)
                        and isinstance(score, (int, float))
                        and math.isfinite(float(score))
                        and score in JUDGE_SCORE_BUCKETS
                        and isinstance(details.get(name), dict)
                    ):
                        scores_by_sample.setdefault(sample_id, {})[name] = int(score)
                        details_by_sample.setdefault(sample_id, {})[name] = details[
                            name
                        ]
    jobs = [
        (record, judge)
        for record in eligible
        for judge in judges
        if judge.name not in scores_by_sample.get(str(record["sample_id"]), {})
    ]
    async with _client_session(timeout) as session:

        async def run_one(
            job: tuple[dict[str, Any], JudgeSpec],
        ) -> dict[str, Any]:
            record, judge = job
            sample_id = str(record["sample_id"])
            sample = samples_by_id[sample_id]
            payload = judge_payload(
                judge,
                build_judge_prompt(sample, str(record["gold_response"])),
                str(record["prefix_path"]),
                max_tokens=judge.max_tokens,
            )
            last_result: ChatResult | None = None
            attempts_used = 0
            for parse_attempt in range(1, max_attempts + 1):
                attempts_used = parse_attempt
                async with semaphores[judge.name]:
                    last_result = await request_chat_completion(
                        session,
                        api_url=judge.api_url,
                        payload=payload,
                        request_id=f"{sample_id}:judge:{judge.name}",
                        phase=f"judge:{judge.name}",
                        api_key_env=judge.api_key_env,
                        max_attempts=1,
                        attempt_offset=parse_attempt - 1,
                        attempt_hook=attempt_hook,
                    )
                score = (
                    parse_judge_score(last_result.text)
                    if last_result.is_success
                    else None
                )
                if score is not None:
                    detail = {
                        "raw_response": last_result.text,
                        "latency_s": round(last_result.latency_s, 6),
                        "attempt": parse_attempt,
                    }
                    result = {
                        "sample_id": sample_id,
                        "judge": judge.name,
                        "score": score,
                        "detail": detail,
                        "error": "",
                    }
                    if result_hook:
                        result_hook(
                            {
                                "record_type": "judge_score",
                                "phase": "level2_judge",
                                **result,
                            }
                        )
                    return result
                if (
                    not last_result.is_success
                    and last_result.status_code is not None
                    and last_result.status_code >= 400
                    and not last_result.retryable
                ):
                    break
                if parse_attempt < max_attempts:
                    await asyncio.sleep(2 ** (parse_attempt - 1))
            assert last_result is not None
            failure = {
                "record_type": "judge_failure",
                "phase": "level2_judge",
                "sample_id": sample_id,
                "judge": judge.name,
                "score": None,
                "detail": {
                    "raw_response": last_result.text,
                    "latency_s": round(last_result.latency_s, 6),
                    "attempt": attempts_used,
                },
                "error": last_result.error or "invalid score response",
            }
            if result_hook:
                result_hook(failure)
            return failure

        outcomes = await bounded_map(jobs, run_one, max_concurrency)
        failures: list[dict[str, Any]] = []
        for outcome in outcomes:
            sample_id = str(outcome["sample_id"])
            if outcome["error"]:
                failures.append(outcome)
                continue
            scores_by_sample.setdefault(sample_id, {})[str(outcome["judge"])] = int(
                outcome["score"]
            )
            details_by_sample.setdefault(sample_id, {})[str(outcome["judge"])] = (
                outcome["detail"]
            )

        results: list[dict[str, Any]] = []
        for record in eligible:
            sample_id = str(record["sample_id"])
            scores = scores_by_sample.get(sample_id, {})
            if set(scores) != expected_names:
                continue
            result = {
                "record_type": "judge_result",
                "phase": "level2_judge",
                "sample_id": sample_id,
                "gold_judge_scores": scores,
                "judge_details": details_by_sample[sample_id],
            }
            if result_hook:
                result_hook(result)
            results.append(result)

        if failures:
            preview = "; ".join(
                f"{failure['sample_id']}/{failure['judge']}: {failure['error']}"
                for failure in failures[:5]
            )
            raise RuntimeError(
                f"{len(failures)} SocialOmni judge request(s) failed: {preview}"
            )
        return results


def merge_judge_records(
    model_records: Sequence[dict[str, Any]],
    judge_records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {str(record["sample_id"]): record for record in judge_records}
    merged: list[dict[str, Any]] = []
    for record in model_records:
        row = dict(record)
        judge = by_id.get(str(record["sample_id"]))
        if judge:
            row["gold_judge_scores"] = dict(judge["gold_judge_scores"])
            row["judge_details"] = dict(judge["judge_details"])
        merged.append(row)
    return merged


def speed_summary(
    records: Sequence[dict[str, Any]], latency_key: str
) -> dict[str, Any]:
    latencies = [
        float(record[latency_key])
        for record in records
        if isinstance(record.get(latency_key), (int, float))
    ]
    if not latencies:
        return {"count": 0}
    ordered = sorted(latencies)

    def percentile(percent: float) -> float:
        index = (len(ordered) - 1) * percent / 100
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)

    total = sum(latencies)
    return {
        "count": len(latencies),
        "mean_s": round(total / len(latencies), 6),
        "p50_s": round(percentile(50), 6),
        "p95_s": round(percentile(95), 6),
        "p99_s": round(percentile(99), 6),
        "serial_throughput_qps": round(len(latencies) / total, 6) if total else None,
    }
