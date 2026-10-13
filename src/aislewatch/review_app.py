"""Local review page for generated stills and keyframe sequences.

Shows every item with its intended state, Claude's suggested state and QA note, and writes your
decision straight into the sidecar JSON (label, human_state, review_note). Localhost only.

Usage: uv run aw-review [--port 8765]   then open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from aislewatch.preprocess import REPO_ROOT

GEN = REPO_ROOT / "data" / "gen"
STATES = ["closed", "passing", "propped", "ajar"]


def _item(meta_path: Path) -> dict:
    m = json.loads(meta_path.read_text())
    return {"meta": str(meta_path.relative_to(GEN)), "img": str(meta_path.with_suffix(".png").relative_to(GEN)),
            "id": m.get("id", meta_path.stem), "camera": m.get("camera_id", ""), "intended": m.get("intended_state"),
            "suggested": m.get("suggested_state") or m.get("intended_state"), "qa_note": m.get("qa_note", ""),
            "note": m.get("note", ""), "label": m.get("label", "unverified"), "human_state": m.get("human_state"),
            "review_note": m.get("review_note", ""), "order": m.get("order")}


def inventory() -> dict:
    stills = [_item(p) for p in sorted(GEN.glob("bases/*.json")) + sorted(GEN.glob("stills/cam-*/*.json"))
              if p.with_suffix(".png").exists()]
    seqs = []
    for d in sorted(GEN.glob("seq/cam-*/*")):
        keys = sorted(d.glob("k*.json"), key=lambda p: int(p.stem[1:]))
        if keys:
            seqs.append({"id": d.name, "camera": d.parent.name, "frames": [_item(k) for k in keys]})
    return {"stills": stills, "sequences": seqs}


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>AisleWatch review</title>
<style>
body{font:14px system-ui;margin:0;background:#111;color:#eee}
header{position:sticky;top:0;background:#222;padding:8px 16px;display:flex;gap:16px;align-items:center;z-index:2}
main{padding:16px}
.card{background:#1b1b1b;border:1px solid #333;border-radius:8px;margin:0 0 16px;padding:10px}
.card.accepted{border-color:#2e7d32}.card.rejected{border-color:#b71c1c;opacity:.6}
.row{display:flex;gap:8px;overflow-x:auto}
.frame{flex:0 0 auto;width:420px}
img{width:100%;border-radius:4px;cursor:zoom-in}
.meta{color:#aaa;font-size:12px;margin:4px 0}
.qa{color:#ffcc80}
button,select{background:#333;color:#eee;border:1px solid #555;border-radius:4px;padding:3px 8px;margin:2px;cursor:pointer}
button.on{background:#2e7d32;border-color:#4caf50}
.rej{background:#5d1a1a}
input.note{width:60%;background:#222;color:#eee;border:1px solid #444;padding:3px}
#zoom{position:fixed;inset:0;background:#000d;display:none;align-items:center;justify-content:center;z-index:5}
#zoom img{max-width:96vw;max-height:94vh;width:auto}
</style></head><body>
<header><b>AisleWatch review</b>
<label><input type=checkbox id=only> unreviewed only</label>
<span id=count></span>
<span class=meta>Accept = suggested state. Click a state to accept with that state. Click an image to zoom.</span></header>
<main id=main></main><div id=zoom onclick="this.style.display='none'"><img></div>
<script>
const STATES=%STATES%;
let data;
async function load(){data=await (await fetch('/api/items')).json();render();}
function stateButtons(it){return STATES.map(s=>`<button class="${it.label==='accepted'&&it.human_state===s?'on':''}" onclick="decide('${it.meta}','accepted','${s}')">${s}</button>`).join('')}
function frameHtml(it,showButtons){
 return `<div class=frame><img src="/img/${it.img}" onclick="zoom(this.src)">
 <div class=meta>${it.order?('k'+it.order+' · '):''}${it.id} · intended <b>${it.intended}</b> · suggested <b>${it.suggested}</b>
 ${it.label!=='unverified'?` · <b>${it.label}${it.human_state?(': '+it.human_state):''}</b>`:''}</div>
 ${it.qa_note?`<div class="meta qa">QA: ${it.qa_note}</div>`:''}${it.note?`<div class=meta>${it.note}</div>`:''}
 ${showButtons?`<div>${stateButtons(it)}<button class=rej onclick="decide('${it.meta}','rejected',null)">reject</button></div>`:''}</div>`}
function render(){
 const only=document.getElementById('only').checked; let html='', todo=0;
 html+='<h2>Sequences</h2>';
 for(const s of data.sequences){
  const status=s.frames.every(f=>f.label==='accepted')?'accepted':(s.frames.some(f=>f.label==='rejected')?'rejected':'');
  if(!status) todo++; if(only&&status) continue;
  html+=`<div class="card ${status}"><b>${s.id}</b> <span class=meta>${s.camera}</span>
  <button onclick="acceptSeq('${s.id}')">accept all (suggested)</button>
  <button class=rej onclick="rejectSeq('${s.id}')">reject sequence</button>
  <div class=row>${s.frames.map(f=>frameHtml(f,true)).join('')}</div></div>`}
 html+='<h2>Stills</h2>';
 for(const it of data.stills){ if(it.label==='unverified') todo++; if(only&&it.label!=='unverified') continue;
  html+=`<div class="card ${it.label}"><div class=row>${frameHtml(it,true)}<div style="flex:1">
  <button onclick="decide('${it.meta}','accepted','${it.suggested}')">accept as ${it.suggested}</button><br>
  <input class=note placeholder="note" value="${it.review_note||''}" onchange="note('${it.meta}',this.value)"></div></div></div>`}
 document.getElementById('main').innerHTML=html; document.getElementById('count').textContent=todo+' to review';
}
async function post(body){await fetch('/api/label',{method:'POST',body:JSON.stringify(body)});}
async function decide(meta,label,state){await post({meta,label,human_state:state});await load();}
async function note(meta,text){await post({meta,review_note:text});}
async function acceptSeq(id){const s=data.sequences.find(x=>x.id===id);for(const f of s.frames)await post({meta:f.meta,label:'accepted',human_state:f.suggested});await load();}
async function rejectSeq(id){const s=data.sequences.find(x=>x.id===id);for(const f of s.frames)await post({meta:f.meta,label:'rejected',human_state:null});await load();}
function zoom(src){const z=document.getElementById('zoom');z.querySelector('img').src=src;z.style.display='flex';}
document.getElementById('only').onchange=render; load();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _safe(self, rel: str) -> Path | None:
        p = (GEN / unquote(rel)).resolve()
        return p if GEN.resolve() in p.parents and p.exists() else None

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, PAGE.replace("%STATES%", json.dumps(STATES)).encode(), "text/html; charset=utf-8")
        elif path == "/api/items":
            self._send(200, json.dumps(inventory()).encode(), "application/json")
        elif path.startswith("/img/") and (p := self._safe(path[5:])) and p.suffix == ".png":
            self._send(200, p.read_bytes(), "image/png")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/label":
            return self._send(404, b"not found", "text/plain")
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        p = self._safe(body["meta"])
        if not p or p.suffix != ".json":
            return self._send(400, b"bad path", "text/plain")
        m = json.loads(p.read_text())
        if "label" in body:
            m.update(label=body["label"], human_state=body.get("human_state"), reviewed_by="Magnus (review page)",
                     reviewed_at=date.today().isoformat())
        if "review_note" in body:
            m["review_note"] = body["review_note"]
        p.write_text(json.dumps(m, indent=2))
        self._send(200, b"ok", "text/plain")

    def log_message(self, *args) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print(f"Review page: http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
