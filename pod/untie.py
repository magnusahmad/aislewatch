"""Make an untied copy of a tied-embedding HF checkpoint (e.g. Qwen3.5-4B) for prime-rl LoRA SFT.

Why: prime-rl's VLM trainer pops lm_head.weight when tie_word_embeddings=True, but after FSDP sharding the
head is no longer the same parameter as embed_tokens, so it stays zero: every logit is 0, loss = ln(vocab)
(12.4225 for Qwen3.5) and grad norm = 0. Untying with lm_head = embed_tokens is numerically the same model,
so a LoRA adapter trained on the copy serves unchanged on the original (tied) model in vLLM.

Usage: python pod/untie.py <hf_model_id> <out_dir>
Prints the model path to train on as its last line: out_dir if the model is tied, else the model id unchanged.
"""

import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

EMBED = "model.language_model.embed_tokens.weight"


def main(model_id: str, out_dir: str) -> None:
    src = Path(snapshot_download(model_id))
    out = Path(out_dir)
    cfg = json.loads((src / "config.json").read_text())
    tied = cfg.get("tie_word_embeddings") or (cfg.get("text_config") or {}).get("tie_word_embeddings")
    if not tied:
        print(f"untie: {model_id} has untied embeddings, using it directly")
        print(model_id)
        return
    if (out / "lm_head.safetensors").exists():
        print(f"untie: {out} already prepared")
        print(out)
        return
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():  # symlink everything; config + index are rewritten below
        if f.name not in ("config.json", "model.safetensors.index.json") and not (out / f.name).exists():
            os.symlink(f.resolve(), out / f.name)

    cfg["tie_word_embeddings"] = False
    if isinstance(cfg.get("text_config"), dict):
        cfg["text_config"]["tie_word_embeddings"] = False
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    index = json.loads((src / "model.safetensors.index.json").read_text())
    assert "lm_head.weight" not in index["weight_map"], "checkpoint already has lm_head.weight"
    with safe_open(src / index["weight_map"][EMBED], "pt") as f:
        embed = f.get_tensor(EMBED)
    save_file({"lm_head.weight": embed.clone()}, out / "lm_head.safetensors", metadata={"format": "pt"})
    index["weight_map"]["lm_head.weight"] = "lm_head.safetensors"
    index.setdefault("metadata", {})["total_size"] = index.get("metadata", {}).get("total_size", 0) + embed.numel() * embed.element_size()
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    print(f"untie: wrote {out} (lm_head.weight {tuple(embed.shape)} {embed.dtype} copied from {EMBED})")
    print(out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
