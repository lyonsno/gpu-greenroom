"""Authenticated localhost operator console for GPU Greenroom."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import secrets
import signal
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .queue import GPUQueue


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GPU Greenroom</title>
<style>
:root{color-scheme:dark;--bg:#111315;--panel:#191c1f;--line:#34393e;--text:#f1f3f4;--muted:#9fa7ad;--green:#4dd58b;--amber:#f2bd57;--red:#ff6b6b;--blue:#70b8ff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:0;overflow-x:hidden}header{height:56px;display:flex;align-items:center;gap:14px;padding:0 18px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:2}h1{font:600 17px/1 system-ui;margin:0}.spacer{flex:1;min-width:0}.state{display:flex;align-items:center;gap:8px;color:var(--muted)}.dot{width:9px;height:9px;border-radius:50%;background:var(--green)}.dot.paused{background:var(--amber)}button{border:1px solid var(--line);background:#25292d;color:var(--text);height:34px;padding:0 11px;border-radius:5px;font:inherit;cursor:pointer}button:hover{border-color:#596169}button.danger{color:#ffd8d8;border-color:#6a3538}button:disabled{opacity:.4;cursor:default}.icon{width:36px;padding:0;font-size:17px}main{width:100%;max-width:100vw;padding:14px 18px 40px;overflow-x:auto}.toolbar{display:flex;gap:8px;align-items:center;margin-bottom:12px;overflow:auto}.filter.active{border-color:var(--blue);color:var(--blue)}table{width:100%;min-width:680px;border-collapse:collapse;table-layout:fixed}th{text-align:left;color:var(--muted);font-weight:500;border-bottom:1px solid var(--line);padding:9px 8px}td{border-bottom:1px solid #292d31;padding:9px 8px;vertical-align:top;overflow-wrap:anywhere}.id{width:150px;white-space:nowrap}.id .meta{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.status{width:94px}.age{width:94px}.actions{width:128px;text-align:right}.route{color:#d9e7f7}.meta{color:var(--muted);font-size:12px;margin-top:3px}.pill{display:inline-block;padding:2px 6px;border:1px solid var(--line);border-radius:4px}.running{color:var(--green)}.pending{color:var(--amber)}.failed{color:var(--red)}.empty{padding:40px 8px;color:var(--muted);text-align:center}.error{position:fixed;left:18px;right:18px;bottom:16px;background:#391f22;border:1px solid #7a3d43;padding:10px 12px;border-radius:5px;display:none}@media(max-width:760px){header{width:100vw;padding:0 10px;gap:4px}h1{font-size:14px}.state{gap:4px;font-size:12px}.spacer{display:none}.icon{width:32px}main{padding:10px}.age,.ageCell{display:none}.routeCol{width:44%}.actions{width:88px}button.text{font-size:0;width:36px}.text span{font-size:17px}}
</style>
</head>
<body><header><h1>GPU Greenroom</h1><div class="state"><span id="dot" class="dot"></span><span id="queueState">loading</span></div><div class="spacer"></div><button id="pause" title="Pause queue" class="icon">Ⅱ</button><button id="resume" title="Resume queue" class="icon">▶</button><button id="refresh" title="Refresh" class="icon">↻</button></header>
<main><div id="pauseMeta" class="meta" hidden></div><div id="lease" class="meta" hidden></div><div class="toolbar" id="filters"></div><table><thead><tr><th class="id">Job</th><th class="status">State</th><th class="routeCol">Route</th><th class="age">Elapsed</th><th class="actions">Actions</th></tr></thead><tbody id="jobs"></tbody></table><div id="empty" class="empty" hidden>No jobs in this view.</div></main><div id="error" class="error"></div>
<script>
const hash=new URLSearchParams(location.hash.slice(1));if(hash.get('token')){sessionStorage.setItem('greenroom-token',hash.get('token'));history.replaceState(null,'',location.pathname)}const token=sessionStorage.getItem('greenroom-token')||'';let filter='active',last=null;const statuses=['active','all','pending','running','done','failed','cancelled'];
function auth(){return {'Authorization':'Bearer '+token,'Content-Type':'application/json'}}function err(e){const n=document.querySelector('#error');n.textContent=e;n.style.display='block';setTimeout(()=>n.style.display='none',6000)}function elapsed(j){const end=j.finished_at||Date.now()/1000,start=j.started_at||j.submitted_at;if(!start)return '—';const s=Math.max(0,end-start);if(s<60)return Math.round(s)+'s';if(s<3600)return Math.floor(s/60)+'m '+Math.round(s%60)+'s';return Math.floor(s/3600)+'h '+Math.floor((s%3600)/60)+'m'}function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function render(s){last=s;document.querySelector('#queueState').textContent=s.paused?'paused':'admitting';document.querySelector('#dot').className='dot'+(s.paused?' paused':'');document.querySelector('#pause').disabled=s.paused;document.querySelector('#resume').disabled=!s.paused;const pm=document.querySelector('#pauseMeta'),p=s.pause_state;pm.hidden=!p;pm.textContent=p?`Pause owner: ${p.owner} · epoch ${p.epoch} · ${p.contention_class} · effective ${new Date(p.effective_at*1000).toLocaleString()}`:'';const lease=document.querySelector('#lease');lease.hidden=!s.lease;lease.textContent=s.lease?`External owner: ${s.lease.owner} · ${s.lease.effective_route} · ${s.lease.lifecycle_state}`:'';document.querySelector('#filters').innerHTML=statuses.map(x=>`<button class="filter ${filter===x?'active':''}" data-filter="${x}">${x}</button>`).join('');document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{filter=b.dataset.filter;load()});const rows=s.jobs;document.querySelector('#empty').hidden=rows.length>0;document.querySelector('#jobs').innerHTML=rows.map(j=>{let a='';if(j.consistent&&j.status==='pending')a=`<button class="danger text" title="Cancel pending job" onclick="act('cancel','${j.job_id}')"><span>×</span> Cancel</button>`;if(j.consistent&&j.status==='running')a=`<button class="danger text" title="Stop running job" onclick="act('terminate','${j.job_id}')"><span>■</span> Stop</button> <button class="danger icon" title="Force stop running job" onclick="act('kill','${j.job_id}')">!</button>`;const route=j.requested_route||j.effective_route||j.job_type;return `<tr><td class="id">${esc(j.job_id)}<div class="meta">${esc(j.job_type)}</div></td><td class="status"><span class="pill ${esc(j.status)}">${esc(j.status)}</span></td><td><div class="route">${esc(route)}</div><div class="meta">${esc(j.repo_root||j.output_dir||j.record_error||'')}</div></td><td class="ageCell">${elapsed(j)}</td><td class="actions">${a}</td></tr>`}).join('')}
async function load(){try{const r=await fetch('/api/state?view='+encodeURIComponent(filter),{headers:auth()});if(!r.ok)throw new Error(await r.text());render(await r.json())}catch(e){err(e.message)}}async function post(path,body={}){const r=await fetch(path,{method:'POST',headers:auth(),body:JSON.stringify(body)});if(!r.ok)throw new Error(await r.text());await load()}async function act(kind,id){try{if(kind==='cancel')await post('/api/cancel',{job_id:id,requested_by:'operator-console',reason:'operator cancelled pending job'});else await post('/api/terminate',{job_id:id,requested_by:'operator-console',reason:kind==='kill'?'operator force-stopped running job':'operator stopped running job',force:kind==='kill'})}catch(e){err(e.message)}}document.querySelector('#pause').onclick=()=>post('/api/pause',{requested_by:'operator-console'}).catch(e=>err(e.message));document.querySelector('#resume').onclick=()=>post('/api/resume',{requested_by:'operator-console',epoch:last?.pause_state?.epoch}).catch(e=>err(e.message));document.querySelector('#refresh').onclick=load;load();setInterval(()=>{if(filter==='active')load()},2000);
</script></body></html>"""


def queue_snapshot(queue: GPUQueue, view: str = "all") -> dict:
    statuses = ("pending", "running", "done", "failed", "cancelled")
    if view == "active":
        selected = ("pending", "running")
    elif view == "all":
        selected = statuses
    elif view in statuses:
        selected = (view,)
    else:
        raise ValueError(f"invalid view: {view}")

    jobs = []
    for containment_status in selected:
        status_dir = queue.queue_dir / containment_status
        for job_dir in sorted(status_dir.iterdir()):
            status_path = job_dir / "status.json"
            if not status_path.is_file():
                continue
            try:
                row = json.loads(status_path.read_text())
                if not isinstance(row, dict):
                    raise ValueError("status record is not an object")
            except (OSError, ValueError, json.JSONDecodeError):
                try:
                    row = json.loads(status_path.read_text())
                    if not isinstance(row, dict):
                        raise ValueError("status record is not an object")
                except (OSError, ValueError, json.JSONDecodeError) as second_error:
                    jobs.append({
                        "job_id": job_dir.name,
                        "job_type": "unknown",
                        "status": "inconsistent",
                        "declared_status": None,
                        "containment_status": containment_status,
                        "consistent": False,
                        "record_error": f"unreadable status: {second_error}",
                    })
                    continue
            declared_status = row.get("status")
            consistent = declared_status == containment_status
            row["job_id"] = row.get("job_id") or job_dir.name
            row["declared_status"] = declared_status
            row["containment_status"] = containment_status
            row["consistent"] = consistent
            row["status"] = containment_status if consistent else "inconsistent"
            request_path = job_dir / "request.json"
            if request_path.is_file():
                try:
                    request = json.loads(request_path.read_text())
                except (OSError, json.JSONDecodeError):
                    request = {}
                row["requested_route"] = request.get("route_identity")
                row["repo_root"] = request.get("repo_root")
            jobs.append(row)
    jobs.sort(key=lambda item: (item.get("submitted_at") or 0), reverse=True)
    lease = queue.lease_status()
    lease_payload = None
    if lease is not None:
        lease_payload = json.loads(lease.to_json())
    return {
        "schema": "gpu-greenroom.operator-snapshot.v1",
        "view": view,
        "queue_dir": str(queue.queue_dir.resolve()),
        "paused": queue.is_paused(),
        "pause_state": queue.pause_state(),
        "lease": lease_payload,
        "jobs": jobs,
    }


def make_handler(queue: GPUQueue, token: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format, *_args):
            return

        def _authorized(self) -> bool:
            return secrets.compare_digest(
                self.headers.get("Authorization", ""), f"Bearer {token}"
            )

        def _json(self, status: int, payload: dict):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                body = PAGE.encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if not self._authorized():
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            if path == "/api/state":
                query = parse_qs(urlparse(self.path).query)
                view = query.get("view", ["active"])[0]
                try:
                    snapshot = queue_snapshot(queue, view)
                except ValueError:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_view"})
                    return
                self._json(HTTPStatus.OK, snapshot)
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self):
            if not self._authorized():
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            path = urlparse(self.path).path
            try:
                body = self._body()
                actor = str(body.get("requested_by") or "operator-console")
                if path == "/api/pause":
                    result = queue.pause(owner=actor, contention_class="operator-console")
                elif path == "/api/resume":
                    result = queue.resume(owner=actor, epoch=body.get("epoch"))
                elif path == "/api/cancel":
                    job_id = str(body["job_id"])
                    if not queue.cancel(job_id):
                        raise RuntimeError(f"pending job {job_id} not found")
                    result = {"status": "cancelled", "job_id": job_id}
                elif path == "/api/terminate":
                    result = queue.terminate_running(
                        str(body["job_id"]),
                        requested_by=actor,
                        reason=str(body.get("reason") or "operator console stop"),
                        signal_number=(signal.SIGKILL if body.get("force") else signal.SIGTERM),
                    )
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                    return
            except (KeyError, ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
                return
            self._json(HTTPStatus.OK, result)

    return Handler


def serve(queue_dir: str | Path, *, port: int = 8765, token: str | None = None):
    token = token or secrets.token_urlsafe(32)
    queue = GPUQueue(queue_dir)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(queue, token))
    effective_port = server.server_address[1]
    print(f"http://127.0.0.1:{effective_port}/#token={token}", flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="GPU Greenroom operator console")
    parser.add_argument("--queue-dir", default="~/.local/state/gpu-greenroom")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(Path(args.queue_dir).expanduser(), port=args.port)


if __name__ == "__main__":
    main()
