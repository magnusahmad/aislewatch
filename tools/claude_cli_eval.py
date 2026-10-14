"""Evaluate a Claude model on a dataset split through the Claude Code subscription (claude -p), not an API.

Same structured prompt as aw-eval; the ten frames are given as image files the model reads in order.
Writes data/results/<dataset>/<split>/<name>-structured.jsonl + .summary.json in aw-eval's format.
Cost is ESTIMATED from config/models.json list prices (OpenRouter) with Anthropic's image-token formula
(w*h/750 per image) plus the prompt text; latency includes CLI/agent overhead, so it is not comparable.

Usage: uv run python tools/claude_cli_eval.py --dataset v2 --split test --name claude-opus-5.5 --model claude-opus-5-5 [--limit N] [--concurrency 4]
"""
import argparse, asyncio, json, shutil, tempfile, time
from pathlib import Path
from PIL import Image
from aislewatch import evaluate as ev, preprocess as pp

def prompt_parts(camera_id: str) -> tuple[str, str]:
    system, user = pp._prompt_blocks()
    pieces = user.replace("{camera_id}", camera_id).split(pp.IMAGE_PLACEHOLDER)
    text = "".join(p + (f"[frame {i}: image file f{i:02d}.jpg]" if i < pp.FRAMES_PER_WINDOW else "") for i, p in enumerate(pieces))
    return system, ("The ten frames are image files in the current directory. Use the Read tool to view f00.jpg to f09.jpg, "
                    "in order, before answering; each one replaces its [frame N] marker below. Then reply with only the JSON.\n\n" + text)

async def one(sem, item, root, model, system_extra):
    async with sem:
        d = Path(tempfile.mkdtemp(prefix="aw-claude-"))
        for i in range(pp.FRAMES_PER_WINDOW):
            shutil.copy(root / item["frames_dir"] / f"f{i:02d}.jpg", d / f"f{i:02d}.jpg")
        system, user = prompt_parts(item["camera_id"])
        t = time.perf_counter()
        proc = await asyncio.create_subprocess_exec(
            "claude", "-p", user, "--model", model, "--output-format", "json", "--system-prompt", system + system_extra,
            "--tools", "Read", "--allowedTools", "Read", "--strict-mcp-config", "--max-turns", "15",
            cwd=d, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, errb = await proc.communicate()
        latency = time.perf_counter() - t
        shutil.rmtree(d, ignore_errors=True)
        text, err, usage = None, None, {}
        try:
            res = json.loads(out); text = res.get("result"); usage = res.get("usage", {})
            if res.get("is_error"): err = f"cli error: {str(text)[:200]}"
        except Exception as e:
            err = f"{type(e).__name__}: {out[:200]!r} {errb[:200]!r}"
        return {"window_id": item["window_id"], "camera_id": item["camera_id"], "source": item["source"], "label": item["label"],
                "text": text, "pred": ev.parse(text), "latency_s": round(latency, 3), "usage": usage, "error": err}

def est_cost_per_window(root, item, completion_tokens, price) -> float:
    w, h = Image.open(root / item["frames_dir"] / "f00.jpg").size
    image_tokens = pp.FRAMES_PER_WINDOW * w * h / 750
    system, user = prompt_parts(item["camera_id"])
    text_tokens = (len(system) + len(user)) / 4
    return (image_tokens + text_tokens) * price["usd_per_m_input"] / 1e6 + completion_tokens * price["usd_per_m_output"] / 1e6

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--name", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int); ap.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args()
    root = pp.REPO_ROOT / "data" / "datasets" / a.dataset
    items = [json.loads(l) for l in (root / "test" / "test.jsonl").read_text().splitlines()][: a.limit]
    sem = asyncio.Semaphore(a.concurrency); started = time.perf_counter()
    rows = await asyncio.gather(*(one(sem, i, root, a.model, "") for i in items))
    out = pp.REPO_ROOT / "data" / "results" / a.dataset / a.split; out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.name}-structured"
    with (out / f"{tag}.jsonl").open("w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    price = ev.models_config()[a.name]["price"]
    comp = [len(r["text"] or "") / 4 for r in rows]
    cost = sum(est_cost_per_window(root, i, c, price) for i, c in zip(items, comp)) / len(items)
    s = ev.metrics(rows)
    s.update(model=a.name, prompt="structured", shuffled=False, errors=sum(1 for r in rows if r["error"]),
             wall_s=round(time.perf_counter() - started, 1), concurrency=a.concurrency, via="claude -p (subscription)",
             est_usd_per_window=round(cost, 5), est_usd_per_10k=round(cost * 1e4, 2), latency_comparable=False)
    (out / f"{tag}.summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({k: s.get(k) for k in ("n", "macro_f1", "valid_json_rate", "errors", "est_usd_per_10k", "wall_s")}))

if __name__ == "__main__":
    asyncio.run(main())
