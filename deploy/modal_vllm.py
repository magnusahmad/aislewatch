"""Serve base Qwen3.5-4B and the AisleWatch LoRA adapter (qwen4b-v2) on Modal, OpenAI-compatible, for the live demo.

Same vLLM version and flags as pod/serve.sh, so answers match the measured results. Scales to zero after 10 idle
minutes; pin one warm container during a demo with AW_MODAL_WARM=1 (billed while warm).

One-time setup:
    modal secret create aislewatch-vllm-key VLLM_API_KEY=<random key>      # also put AW_VLLM_KEY=<same> in .env
    modal volume create aw-adapters
    modal volume put aw-adapters models/adapters/v2-4b /v2-4b
Deploy / demo:
    modal deploy deploy/modal_vllm.py                       # prints the URL; the live demo uses <url>/v1
    AW_MODAL_WARM=1 modal deploy deploy/modal_vllm.py       # before going on stage; redeploy without it after
    uv run aw-showcase serve --vllm-url https://<url>/v1
"""

import os
import subprocess

import modal

VLLM_VERSION = "0.31.0"  # pinned: the version that passed S2 and served every eval in data/results
BASE_MODEL = "Qwen/Qwen3.5-4B"
GPU = "L40S"  # same card as the measured latencies (48 GB; a 24 GB card also fits a 4B model, more slowly)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_NO_USAGE_STATS": "1"})
)
hf_cache = modal.Volume.from_name("aw-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("aw-vllm-cache", create_if_missing=True)
adapters = modal.Volume.from_name("aw-adapters", create_if_missing=True)

app = modal.App("aislewatch-vllm")


@app.function(
    image=image,
    gpu=GPU,
    min_containers=1 if os.environ.get("AW_MODAL_WARM") == "1" else 0,
    scaledown_window=10 * 60,
    timeout=24 * 60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache, "/adapters": adapters},
    secrets=[modal.Secret.from_name("aislewatch-vllm-key")],
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=15 * 60)
def serve() -> None:
    cmd = [
        "vllm", "serve", BASE_MODEL,
        "--served-model-name", "qwen4b-base",
        "--host", "0.0.0.0", "--port", "8000",
        "--dtype", "bfloat16", "--max-model-len", "16384",
        "--limit-mm-per-prompt", '{"image": 10}',
        "--enable-lora", "--max-lora-rank", "16", "--max-loras", "2",
        "--lora-modules", "qwen4b-v2=/adapters/v2-4b",
        "--api-key", os.environ["VLLM_API_KEY"],
    ]
    subprocess.Popen(cmd)
