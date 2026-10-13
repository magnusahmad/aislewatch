"""CCTV-style degradation applied identically to generated stills and clip frames.

A window gets one set of camera conditions (lighting, blur, compression); each frame then gets its
own sensor noise and a tiny exposure flicker, so a static scene still varies frame to frame like
real footage. Lighting changes live here, not in generation prompts (generators do them badly).
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class Conditions:
    brightness: float
    contrast: float
    saturation: float
    blur: float
    noise: float
    jpeg_quality: int

    @classmethod
    def sample(cls, rng: random.Random, mild: bool = False) -> "Conditions":
        if mild:
            return cls(rng.uniform(0.9, 1.05), rng.uniform(0.95, 1.05), rng.uniform(0.9, 1.0),
                       rng.uniform(0.0, 0.5), rng.uniform(0.0, 2.0), rng.randint(70, 85))
        return cls(rng.uniform(0.55, 1.1), rng.uniform(0.8, 1.1), rng.uniform(0.7, 1.0),
                   rng.uniform(0.0, 1.2), rng.uniform(0.0, 6.0), rng.randint(35, 80))


def apply(img: Image.Image, c: Conditions, rng: random.Random) -> Image.Image:
    """Degrade one frame. Call per frame with a per-window Conditions and a shared rng."""
    flicker = rng.uniform(-0.02, 0.02)
    out = ImageEnhance.Brightness(img).enhance(c.brightness + flicker)
    out = ImageEnhance.Contrast(out).enhance(c.contrast)
    out = ImageEnhance.Color(out).enhance(c.saturation)
    if c.blur > 0.05:
        out = out.filter(ImageFilter.GaussianBlur(radius=c.blur))
    if c.noise > 0.05:
        arr = np.asarray(out, dtype=np.float32)
        arr += np.random.default_rng(rng.getrandbits(32)).normal(0, c.noise, arr.shape)
        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=max(20, c.jpeg_quality + rng.randint(-5, 5)))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
