# Smoke tests S1–S3

S1 and S2 share one rented A100 80GB pod (S1 trains, then S2 serves from the same machine). S3 runs on the laptop.

## 0. Laptop prep (done once)

```bash
uv sync
uv run python smoke/make_smoke_data.py ~/Downloads/gemini_generated_video_83bd1aee.mp4
```

Writes `data/smoke/sft/train.parquet` (8 rows), `data/smoke/requests/w0.json`, `w1.json` and the frames.

## 1. Pod

Add your SSH public key (`~/.ssh/id_ed25519.pub`) in the Prime Intellect dashboard, then create the pod (A100 80GB, ≥200 GB disk). Copy the kit and data:

```bash
POD=root@<pod-ip>   # from: prime pods status <pod-id>
ssh $POD 'mkdir -p /root/smoke'
scp -r smoke/* $POD:/root/smoke/
scp -r data/smoke/sft $POD:/root/smoke/sft
```

## S1: prime-rl VLM LoRA SFT

```bash
ssh $POD 'bash /root/smoke/s1_pod.sh'
```

Pass when:
- the dry run prints a resolved config with `lr = 2e-4`, renderer `qwen3.5`, thinking off
- loss falls towards zero over 40 steps (8 rows: it should memorise them)
- `/root/runs/s1/adapter/` has `adapter_model.safetensors` + `adapter_config.json`
- the target-module report says whether linear-attention layers were adapted

prime-rl at the pinned commit only writes a PEFT adapter through the online-eval broadcast, so `patch_export_adapter.py` adds an export at the end of training. If S1 fails for a reason that isn't quickly fixable, switch to Unsloth (SPEC §8.2).

## S2: vLLM serving + runtime LoRA + structured output

On the pod (leave running):

```bash
ssh $POD 'bash /root/smoke/s2_pod.sh'          # add "nightly" if the release can't load Qwen3.5
```

On the laptop, in another terminal:

```bash
ssh -N -L 8000:127.0.0.1:8000 $POD            # tunnel; vLLM never listens publicly
uv run python smoke/s2_client.py
```

Pass when: base, `smoke-s1` and runtime-loaded `smoke-rt` all return schema-valid JSON with no `<think>`; the adapter gets both windows right (it was trained on them); p95 latency at 3 cameras × 2 models is reported. Record the vLLM version.

## S3: Hermes + Codex subscription + stub MCP

1. Install Hermes and log in to the Codex subscription:
   ```bash
   curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
   hermes model            # choose "ChatGPT or Codex Subscription"
   ```
2. Add to `~/.hermes/config.yaml`:
   ```yaml
   mcp_servers:
     aislewatch:
       command: uv
       args: ["run", "--project", "/path/to/aislewatch", "aw-smoke-stub-mcp"]
   ```
3. `hermes`, then ask: "Any alerts today? Use the aislewatch tools."

Pass when Hermes calls `get_alerts` and reports the aisle B temperature alert.

## Results

| Test | Result | Notes (versions, numbers, surprises) |
|---|---|---|
| S1 | **Pass** (Oct 12, L40S 48GB) | prime-rl `7c9d474`. Loss 0.80 → 0.0000 by step 25 (memorised 8 rows), ~8.5 s/step. **768 px OOMed twice** in the fla gated-delta-rule Triton autotune (`expandable_segments` didn't help); **512 px fits** at 41.5/44.4 GiB peak (~144 image tokens/frame, ~2.0k tokens/sample). Adapter exported by the patch (58 MB, PEFT). LoRA covers MLPs in all 32 layers + q/k/v/o in the 8 full-attention layers; the 24 linear-attention layers' projections are not adapted. Vision tower untouched. |
| S2 | **Pass** (same pod) | vLLM 0.31.0. Base, static adapter and runtime-loaded adapter all return schema-valid JSON, no `<think>`. Needs `VLLM_USE_FLASHINFER_SAMPLER=0` (FlashInfer sampler JIT needs nvcc, absent on the image). Latency at 3 cameras × 2 models, 512 px: p50 2.54 s, p95 3.02 s. Base Qwen called the propped window "ajar" (same error as Flash-Lite in the Oct 10 test); adapter correct (trained on it, so not evidence of learning). |
| S3 | Pending | Needs Hermes installed and logged in to the Codex subscription. |

Pod setup fixes found on the way: GitHub SSH URLs for the `prime-envs` submodule need `url.https://github.com/.insteadOf`; the uv installer exits non-zero on the pod image (fish config permission), so the script tolerates it; the pod user is `ubuntu`, not `root`.

Cost: one L40S pod, ~62 min, $0.82.
