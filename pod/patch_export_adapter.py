"""Patch prime-rl's SFT trainer to write the final LoRA adapter as a PEFT directory.

At commit 7c9d474, SFT only writes a vLLM-loadable adapter (adapter_model.safetensors +
adapter_config.json) through the online-eval weight broadcast, which waits for an
inference receiver. Checkpoints are DCP and tools/convert_dcp_to_bf16.py rejects LoRA.
This inserts the same save calls the filesystem broadcaster uses, writing to
<run_dir>/adapter after training. Idempotent.

Usage (on the pod, from the prime-rl checkout): python patch_export_adapter.py
"""

from pathlib import Path

TARGET = Path("src/prime_rl/trainer/sft/train.py")
ANCHOR = "    # Broadcast the final weights so the evals process can run its forced final epoch\n"
MARKER = "# aislewatch: export final LoRA adapter"
PATCH = f"""    {MARKER}
    if config.model.lora is not None:
        from torch.distributed.tensor import DTensor as _DTensor

        from prime_rl.trainer.lora import save_lora_config as _save_lora_config
        from prime_rl.utils.weights import save_state_dict as _save_state_dict

        _state = get_lora_state().adapter_state_dict()
        for _k, _v in list(_state.items()):
            if isinstance(_v, _DTensor):
                _v = _v.full_tensor()
            _state[_k] = _v.to("cpu", non_blocking=False)
        if world.is_master:
            _out = config.run_dir / "adapter"
            _out.mkdir(parents=True, exist_ok=True)
            _save_state_dict(_state, _out, save_sharded=False, adapter=True)
            _save_lora_config(
                model, _out, rank=config.model.lora.rank, alpha=config.model.lora.alpha,
                dropout=config.model.lora.dropout,
            )
            logger.info(f"Exported LoRA adapter to {{_out}}")

"""

src = TARGET.read_text()
if MARKER in src:
    print("Already patched")
elif ANCHOR not in src:
    raise SystemExit(f"Anchor not found in {TARGET}; prime-rl changed, inspect the end of the SFT train loop")
else:
    TARGET.write_text(src.replace(ANCHOR, PATCH + ANCHOR, 1))
    print(f"Patched {TARGET}")
