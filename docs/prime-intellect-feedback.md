# Prime Intellect feedback log

Ongoing log of issues, friction and suggestions from building AisleWatch on Prime Intellect (prime-rl, pods, CLI). Newest entries at the top of each section. Status: `draft` (not shared), `raised` (issue/DM/conversation, with link), `resolved`.

Nothing here mentions the client. Keep it that way: entries may be pasted into public issues.

---

## prime-rl

### 1. LoRA SFT doesn't produce a vLLM-loadable adapter unless online evals are configured

- **Logged:** 2026-10-13 · **Status:** `draft`, issue text ready, awaiting go-ahead to post · **Severity for us:** blocking (patched locally)
- **Repo/commit:** `PrimeIntellect-ai/prime-rl` @ `7c9d474` (Oct 12, 2026)
- **Why not a PR yet:** the LoRA code is being reworked. Open PR #3855 ("single-adapter LoRA layers", updated Oct 11) edits the same file (`trainer/sft/train.py`). #3813 ("remove LoRA support") was recently closed and #3688 proposed making the filesystem broadcast debug-only, so their preferred direction for adapter export is unclear. Their README asks for issues first.
- **Our workaround:** `pod/patch_export_adapter.py` (also `smoke/`), which inserts an adapter export after the final checkpoint.

Draft issue:

> **Title:** SFT with LoRA doesn't produce a vLLM-loadable adapter unless online evals are configured
>
> **What happened:** Running LoRA SFT on `Qwen/Qwen3.5-9B` with `[model.vlm]` (commit `7c9d474`), the run finished, but there was no PEFT adapter to serve. Checkpoints are DCP, and `tools/convert_dcp_to_bf16.py` rejects LoRA checkpoints. The only code path that writes `adapter_model.safetensors` + `adapter_config.json` is the filesystem weight broadcast, which SFT only sets up when online evals are configured, and which waits for an inference receiver.
>
> **Workaround:** after the final checkpoint in `trainer/sft/train.py`, call the same functions the filesystem broadcaster uses (`get_lora_state().adapter_state_dict()` → `save_state_dict(..., adapter=True)` + `save_lora_config(...)`) to write `<run_dir>/adapter`. vLLM 0.31.0 then loads it, both with `--lora-modules` and at runtime via `/v1/load_lora_adapter`.
>
> **Suggestion:** export the final adapter for LoRA SFT runs (behind a flag such as `[ckpt] export_adapter = true`), or add a DCP → PEFT conversion tool for LoRA checkpoints. Happy to open a PR once #3855 lands, if either fits your direction.
>
> **Related, smaller:** the default `target_modules` skip Qwen3.5's 24 linear-attention layers (only the 8 full-attention layers and the MLPs get adapters). That may be intended, but it isn't documented.

### 2. Default LoRA targets skip Qwen3.5's linear-attention layers

- **Logged:** 2026-10-13 · **Status:** `draft` (folded into #1 as a side note)
- Default `target_modules` (q/k/v/o/gate/up/down) adapts the MLPs in all 32 layers and attention in the 8 full-attention layers. The 24 Gated DeltaNet (linear-attention) layers' projections get no adapter, and unknown names are silently ignored.
- **Ask:** document it, or log a warning listing layers with linear modules that matched no target.

### 3. Image dataset format for VLM SFT isn't documented

- **Logged:** 2026-10-13 · **Status:** `draft`
- `docs/training.md` describes the `messages` / prompt-completion layouts, but not how images go in. We found in `trainer/multimodal.py` that images must be **base64 `data:image/...` URLs in `image_url` content parts**; file paths or http URLs fail with "Multimodal training requires base64 data image URLs".
- **Ask:** one short example row in the VLM SFT docs (`examples/extra/vlm/README.md` points to "your multimodal chat dataset" with no format).

### 4. 768 px × 10 images runs out of memory on a 48 GB card in the fla Triton autotune

- **Logged:** 2026-10-13 · **Status:** `draft` · **Severity:** worked around (512 px)
- Qwen3.5-9B LoRA SFT, 10 images per sample at 768×432 (about 3.4k image tokens, about 4k tokens total), full activation checkpointing and optimizer offload, on an L40S 48 GB. Failed with `Triton Error [CUDA]: out of memory` inside `fla/ops/gated_delta_rule` backward, during autotune benchmarking. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` didn't help. At 512 px it fits at 41.5/44.4 GiB peak.
- **Question for them:** is the autotune memory spike known, and can it be capped or cached ahead of time?

### 5. `prime-envs` submodule uses an SSH URL

- **Logged:** 2026-10-12 · **Status:** `draft` · **Severity:** minor
- `git submodule update --init` fails on a fresh pod without GitHub SSH keys: `deps/prime-envs` is `git@github.com:...`, while the other submodules use HTTPS. The failed attempt then left `deps/pydantic-config` empty, which broke `uv sync` with a confusing "not a Python project" error until a forced re-checkout.
- **Workaround:** `git config --global url."https://github.com/".insteadOf git@github.com:`.
- **Ask:** switch `.gitmodules` to an HTTPS URL.

---

## Pods and CLI

### 6. vLLM's FlashInfer sampler needs `nvcc`, absent on the `ubuntu_22_cuda_12` image

- **Logged:** 2026-10-12 · **Status:** `draft`
- vLLM 0.31.0 engine startup failed at warm-up: FlashInfer JIT-compiles its top-k/top-p sampler and found no `nvcc` / `/usr/local/cuda`.
- **Workaround:** `VLLM_USE_FLASHINFER_SAMPLER=0`.
- **Ask:** an image variant with the CUDA toolkit, or a note in the pod image docs.

### 7. uv installer exits non-zero on the pod image

- **Logged:** 2026-10-12 · **Status:** `draft` · **Severity:** minor
- `curl -LsSf https://astral.sh/uv/install.sh | sh` installed uv, then failed on `mkdir -p /home/ubuntu/.config/fish/conf.d` (`~/.config` is owned by root on the image), returning non-zero and stopping `set -e` scripts.
- **Ask:** the `ubuntu` user should own `~/.config`.

### 8. CLI defaults the SSH key to `~/.ssh/id_rsa`

- **Logged:** 2026-10-12 · **Status:** `draft` · **Severity:** minor
- `prime config` had `ssh_key_path: ~/.ssh/id_rsa` on a machine with only `id_ed25519`. We fixed it with `prime config set-ssh-key-path`.
- **Ask:** detect existing keys, or mention it during `prime login`.

### 9. Pod user is `ubuntu`, not `root`

- **Logged:** 2026-10-12 · **Status:** `draft` · **Severity:** note
- Scripts assuming `/root` broke. Fine once known; worth stating in the pod docs next to the SSH instructions.

### 10. No H100 single-GPU availability at the time; A100 80 GB stock "Low"

- **Logged:** 2026-10-12 · **Status:** `draft` · **Severity:** note
- `prime availability list --gpu-type H100_80GB --gpu-count 1` returned nothing. L40S 48 GB at $0.82/hr (Massed Compute) was available and is what we use.

---

## Hosted Training

### 11. Hosted Training (beta) doesn't yet fit VLM LoRA SFT on our own data

- **Logged:** 2026-10-13 · **Status:** `draft` · **Use for:** Wednesday conversation
- The dashboard says on-demand training is in beta and needs access. `prime train` help: SFT on our own data needs **volumes** (closed beta); otherwise SFT only works with fake datasets. Shared-deployment LoRA runs look RL/environment-oriented.
- **Ask:** what's needed to run Qwen3.5 VLM LoRA SFT on Hosted Training, and can we get volumes and Hosted Training access? We'd move the per-site training loop there.

---

## What worked well (worth saying too)

- **prime-rl's VLM SFT example** (`examples/extra/vlm/sft.toml`) got Qwen3.5-9B with image input training on the first real attempt, once the format and memory were sorted.
- **`--dry-run`** writing the resolved config made it easy to check overrides took effect.
- **Pod creation through the CLI** (`prime pods create --id … -y`) took about 2 minutes to an active pod. Total smoke-test cost was $0.82.
