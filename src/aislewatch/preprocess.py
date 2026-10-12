"""Single source of truth for window frames, prompts and the caption schema.

Training, serving, teacher calls and evaluation must all build their inputs
through this module (see AGENTS.md). Changing anything here changes what every
model sees, so re-run the baseline afterwards.
"""

from __future__ import annotations

import base64
import io
import json
import re
import subprocess
from functools import lru_cache
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = REPO_ROOT / "prompts" / "caption_structured.md"
SCHEMA_PATH = REPO_ROOT / "schemas" / "caption.schema.json"

FRAMES_PER_WINDOW = 10
FRAME_SPACING_S = 0.5
WINDOW_S = FRAMES_PER_WINDOW * FRAME_SPACING_S
LONG_SIDE_PX = 768
JPEG_QUALITY = 85
IMAGE_PLACEHOLDER = "<image>"


@lru_cache(maxsize=1)
def _prompt_blocks() -> tuple[str, str]:
    """Return (system, user template) from the fenced blocks under '## System' and '## User'."""
    text = PROMPT_PATH.read_text()

    def block_after(heading: str) -> str:
        match = re.search(rf"^## {heading}\s*\n+```\n(.*?)\n```", text, re.S | re.M)
        if not match:
            raise ValueError(f"No fenced block under '## {heading}' in {PROMPT_PATH}")
        return match.group(1)

    return block_after("System"), block_after("User")


@lru_cache(maxsize=1)
def caption_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def caption_field_order() -> list[str]:
    return list(caption_schema()["required"])


def extract_window(video_path: Path, t_start: float) -> list[Image.Image]:
    """Return FRAMES_PER_WINDOW frames starting at t_start, FRAME_SPACING_S apart."""
    frames = []
    for i in range(FRAMES_PER_WINDOW):
        t = t_start + i * FRAME_SPACING_S
        png = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video_path),
             "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
            check=True, capture_output=True,
        ).stdout
        if not png:
            raise ValueError(f"No frame at t={t:.3f}s in {video_path}")
        frames.append(Image.open(io.BytesIO(png)).convert("RGB"))
    return frames


def resize(img: Image.Image) -> Image.Image:
    scale = LONG_SIDE_PX / max(img.size)
    if scale >= 1:
        return img
    return img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)


def encode_jpeg(img: Image.Image, quality: int = JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def to_data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


def build_messages(frames_jpeg: list[bytes], camera_id: str) -> list[dict]:
    """System + user messages in OpenAI multimodal format, images as base64 data URLs."""
    if len(frames_jpeg) != FRAMES_PER_WINDOW:
        raise ValueError(f"Expected {FRAMES_PER_WINDOW} frames, got {len(frames_jpeg)}")
    system, user_template = _prompt_blocks()
    pieces = user_template.replace("{camera_id}", camera_id).split(IMAGE_PLACEHOLDER)
    if len(pieces) != FRAMES_PER_WINDOW + 1:
        raise ValueError("User prompt must contain exactly one <image> placeholder per frame")
    content: list[dict] = []
    for i, piece in enumerate(pieces):
        if piece:
            content.append({"type": "text", "text": piece})
        if i < FRAMES_PER_WINDOW:
            content.append({"type": "image_url", "image_url": {"url": to_data_url(frames_jpeg[i])}})
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": content},
    ]


def caption_json(caption: dict) -> str:
    """Compact JSON with keys in schema order; the exact training target string."""
    missing = [k for k in caption_field_order() if k not in caption]
    if missing:
        raise ValueError(f"Caption missing fields: {missing}")
    return json.dumps({k: caption[k] for k in caption_field_order()}, separators=(",", ":"), ensure_ascii=False)


def assistant_message(caption: dict) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": caption_json(caption)}]}
