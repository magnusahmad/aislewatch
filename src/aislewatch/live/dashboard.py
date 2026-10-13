"""Live dashboard (SPEC §5.8): camera tiles with base vs live-model door state, door-state timelines,
inlet temperature, alerts, training loop and scoreboard. Polls the store every 2 s. Localhost only.

Usage: uv run aw-dashboard [--port 8780] [--dataset v1]
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from aislewatch.live.store import LIVE_DIR, connect, current_model, rows
from aislewatch.preprocess import REPO_ROOT

TIMELINE_S = 600
ARGS = argparse.Namespace(dataset=None)


def state() -> dict:
    con = connect()
    live = current_model()
    cams = []
    t_now = (con.execute("SELECT max(t_end) FROM windows").fetchone()[0]) or 0
    for cam in [r["camera_id"] for r in rows(con, "SELECT DISTINCT camera_id FROM windows ORDER BY camera_id")]:
        ws = rows(con, "SELECT * FROM windows WHERE camera_id=? AND t_end >= ? ORDER BY t_end", cam, t_now - TIMELINE_S)
        caps = {}
        for m in {"qwen-base", live}:
            caps[m] = {r["window_id"]: r for r in rows(con, f"""SELECT c.window_id, c.door_state, c.caption, c.latency_ms FROM captions c
                       JOIN windows w ON w.window_id = c.window_id WHERE w.camera_id=? AND w.t_end >= ? AND c.model_id=?""",
                                                                     cam, t_now - TIMELINE_S, m)}
        # latest window that has captions from every shown model (falls back to the newest window)
        latest = next((w for w in reversed(ws) if all(w["window_id"] in caps[m] for m in caps)), ws[-1] if ws else None)
        tile = {"camera_id": cam, "aisle": ws[-1]["aisle"] if ws else "", "live_model": live}
        if latest:
            tile.update(window_id=latest["window_id"], t=latest["t_end"], gt=latest["gt_state"],
                        frame=f"{latest['frames_dir']}/f09.jpg",
                        base=caps["qwen-base"].get(latest["window_id"], {}),
                        current=caps[live].get(latest["window_id"], {}))
        tile["timeline"] = {
            "truth": [[w["t_start"], w["t_end"], w["gt_state"]] for w in ws],
            "qwen-base": [[w["t_start"], w["t_end"], caps["qwen-base"].get(w["window_id"], {}).get("door_state")] for w in ws],
        }
        if live != "qwen-base":
            tile["timeline"][live] = [[w["t_start"], w["t_end"], caps[live].get(w["window_id"], {}).get("door_state")] for w in ws]
        cams.append(tile)
    temps = {a["aisle"]: [[r["t"], r["temp_c"]] for r in rows(con, "SELECT t, temp_c FROM sensor WHERE aisle=? AND t >= ? ORDER BY t", a["aisle"], t_now - TIMELINE_S)]
             for a in rows(con, "SELECT DISTINCT aisle FROM sensor")}
    alerts = rows(con, "SELECT * FROM alerts ORDER BY detected_at DESC LIMIT 20")
    loop = {"corrections": {r["status"]: r["n"] for r in rows(con, "SELECT status, count(*) n FROM corrections GROUP BY status")},
            "latest": rows(con, "SELECT correction_id, status, corrected_fact FROM corrections ORDER BY created_at DESC LIMIT 4")}
    return {"t_now": t_now, "window_s": TIMELINE_S, "live_model": live, "cameras": cams, "temps": temps,
            "alerts": alerts, "loop": loop, "scoreboard": scoreboard(), "models": available_models()}


def available_models() -> list[str]:
    cfg = json.loads((REPO_ROOT / "config" / "models.json").read_text())
    return [k for k, v in cfg.items() if isinstance(v, dict) and v.get("kind") == "vllm"]


def scoreboard() -> str:
    base = REPO_ROOT / "data" / "results"
    if ARGS.dataset:
        p = base / ARGS.dataset / "test" / "SCOREBOARD.md"
    else:
        found = sorted(base.glob("*/test/SCOREBOARD.md"), key=lambda q: q.stat().st_mtime)
        p = found[-1] if found else None
    return p.read_text() if p and p.exists() else ""


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>AisleWatch</title>
<style>
:root{--closed:#607d8b;--passing:#1e88e5;--propped:#e53935;--ajar:#fb8c00;--none:#2a2a2a}
body{margin:0;font:14px system-ui;background:#0e0f11;color:#e8e8e8}
header{display:flex;gap:16px;align-items:center;padding:10px 16px;background:#16181c;border-bottom:1px solid #26282d}
header b{font-size:16px}.pill{padding:2px 8px;border-radius:10px;background:#26282d}
main{display:grid;grid-template-columns:2fr 1fr;gap:12px;padding:12px}
.cams{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px}
.tile{background:#16181c;border:1px solid #26282d;border-radius:8px;padding:8px}
.tile img{width:100%;border-radius:4px;display:block}
.states{display:flex;gap:6px;margin:6px 0}
.st{flex:1;padding:4px 6px;border-radius:4px;font-size:12px}
.st b{display:block;font-size:15px}
.diff{outline:2px solid #ffd54f}
.tl{display:flex;height:10px;margin:3px 0;border-radius:2px;overflow:hidden;background:#222}
.tl span{height:100%}
.lbl{font-size:11px;color:#9aa0a6}
.panel{background:#16181c;border:1px solid #26282d;border-radius:8px;padding:8px;margin-bottom:12px}
.alert{padding:4px 6px;border-left:3px solid #e53935;margin:4px 0;background:#1d1f23;font-size:12px}
.alert.temp{border-color:#fb8c00}.alert.shadow{opacity:.45}
table{border-collapse:collapse;font-size:12px;width:100%}td,th{border-bottom:1px solid #26282d;padding:3px 4px;text-align:left}
select,button{background:#26282d;color:#eee;border:1px solid #3a3d44;border-radius:4px;padding:3px 8px}
.legend span{display:inline-block;width:10px;height:10px;margin:0 3px 0 10px;border-radius:2px}
</style></head><body>
<header><b>AisleWatch</b><span class=pill id=clock></span>
<span>Live captioner: <select id=model></select></span>
<span class=legend><span style="background:var(--closed)"></span>closed<span style="background:var(--passing)"></span>passing<span style="background:var(--propped)"></span>propped<span style="background:var(--ajar)"></span>ajar</span></header>
<main><div><div class=cams id=cams></div></div>
<div><div class=panel><b>Alerts</b><div id=alerts></div></div>
<div class=panel><b>Inlet temperature</b><div id=temps></div></div>
<div class=panel><b>Training loop</b><div id=loop></div></div>
<div class=panel><b>Scoreboard (held-out camera B)</b><div id=score></div></div></div></main>
<script>
const color=s=>`var(--${s||'none'})`, fmt=t=>new Date(t*1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'});
function tl(segs,t0,span){return `<div class=tl>${segs.map(([a,b,s])=>`<span style="width:${(b-a)/span*100}%;background:${color(s)}" title="${fmt(a)} ${s||'—'}"></span>`).join('')}</div>`}
function stateBox(name,c,diff){return `<div class="st ${diff?'diff':''}" style="background:${color(c.door_state)}33;border:1px solid ${color(c.door_state)}"><span class=lbl>${name}</span><b>${c.door_state||'…'}</b></div>`}
function spark(pts){if(pts.length<2)return'';const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(21.5,...ys),y1=Math.max(26.5,...ys);
 const P=pts.map(([x,y])=>`${(x-x0)/(x1-x0||1)*300},${60-(y-y0)/(y1-y0)*60}`).join(' ');const ya=60-(24-y0)/(y1-y0)*60;
 return `<svg viewBox="0 0 300 60" width=100% height=60><line x1=0 x2=300 y1=${ya} y2=${ya} stroke="#fb8c0066" stroke-dasharray=3 /><polyline fill=none stroke="#4fc3f7" stroke-width=2 points="${P}"/></svg><div class=lbl>${ys[ys.length-1].toFixed(1)} °C now · alarm at +2 °C (dashed)</div>`}
function mdTable(md){const L=md.trim().split('\n').filter(l=>l.startsWith('|'));if(!L.length)return'<span class=lbl>no results yet</span>';
 const cells=l=>l.split('|').slice(1,-1).map(c=>c.trim());return '<table>'+L.filter((l,i)=>i!==1).map((l,i)=>'<tr>'+cells(l).map(c=>i?`<td>${c}</td>`:`<th>${c}</th>`).join('')+'</tr>').join('')+'</table>'}
let models=[];
async function tick(){
 const d=await (await fetch('/api/state')).json();
 document.getElementById('clock').textContent=d.t_now?fmt(d.t_now):'waiting for feeds';
 const sel=document.getElementById('model');
 if(JSON.stringify(models)!==JSON.stringify(d.models)){models=d.models;sel.innerHTML=models.map(m=>`<option>${m}</option>`).join('')}
 if(document.activeElement!==sel) sel.value=d.live_model;
 const t0=d.t_now-d.window_s;
 document.getElementById('cams').innerHTML=d.cameras.map(c=>{
  const diff=c.base&&c.current&&c.base.door_state!==c.current.door_state&&d.live_model!=='qwen-base';
  return `<div class=tile><div class=lbl>${c.camera_id} · ${c.aisle} · ${c.t?fmt(c.t):''}</div>
  ${c.frame?`<img src="/live/${c.frame}">`:''}
  <div class=states>${stateBox('base Qwen',c.base||{},diff)}${d.live_model!=='qwen-base'?stateBox(d.live_model+' (fine-tuned)',c.current||{},diff):''}</div>
  <div class=lbl>${(c.current&&c.current.caption)||(c.base&&c.base.caption)||''}</div>
  ${Object.entries(c.timeline).map(([k,v])=>`<div class=lbl>${k}</div>${tl(v,t0,d.window_s)}`).join('')}</div>`}).join('');
 document.getElementById('alerts').innerHTML=d.alerts.map(a=>`<div class="alert ${a.type.startsWith('temp')?'temp':''} ${a.model_id&&a.model_id!==d.live_model?'shadow':''}">
  <b>${fmt(a.detected_at)}</b> ${a.type.replace('_',' ')} · ${a.camera_id||a.aisle}<br>${a.detail}${a.model_id?` · by ${a.model_id}`:''}</div>`).join('')||'<span class=lbl>none</span>';
 document.getElementById('temps').innerHTML=Object.entries(d.temps).map(([a,p])=>`<div class=lbl>${a}</div>${spark(p)}`).join('');
 const l=d.loop;document.getElementById('loop').innerHTML=`<div class=lbl>corrections: ${JSON.stringify(l.corrections)}</div>`+l.latest.map(c=>`<div class=lbl>${c.status} · ${c.corrected_fact}</div>`).join('');
 document.getElementById('score').innerHTML=mdTable(d.scoreboard);
}
document.getElementById('model').onchange=async e=>{await fetch('/api/current',{method:'POST',body:JSON.stringify({model:e.target.value})});tick()};
tick();setInterval(tick,2000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if path == "/api/state":
            return self._send(200, json.dumps(state()).encode(), "application/json")
        if path.startswith("/live/"):
            p = (LIVE_DIR / unquote(path[6:])).resolve()
            if LIVE_DIR.resolve() in p.parents and p.suffix == ".jpg" and p.exists():
                return self._send(200, p.read_bytes(), "image/jpeg")
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/current":
            return self._send(404, b"not found", "text/plain")
        model = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["model"]
        if model not in available_models():
            return self._send(400, b"unknown model", "text/plain")
        p = REPO_ROOT / "models" / "current.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"model": model}))
        self._send(200, b"ok", "text/plain")

    def log_message(self, *args) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--dataset")
    args = ap.parse_args()
    ARGS.dataset = args.dataset
    print(f"Dashboard: http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
