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

## Results (fill in)

| Test | Result | Notes (versions, numbers, surprises) |
|---|---|---|
| S1 | | |
| S2 | | |
| S3 | | |
