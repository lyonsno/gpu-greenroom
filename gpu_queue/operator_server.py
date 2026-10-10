"""Authenticated localhost operator console for GPU Greenroom."""

from __future__ import annotations

import argparse
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import html
import json
import math
import secrets
import shutil
import signal
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .queue import GPUQueue
from .admission_control import transition
from .smoke_requests import SmokeRequestConflict, SmokeRequests
from . import dispatch


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GPU Greenroom</title>
<style>
:root{color-scheme:dark;--bg:#111315;--panel:#191c1f;--line:#34393e;--text:#f1f3f4;--muted:#9fa7ad;--green:#4dd58b;--amber:#f2bd57;--red:#ff6b6b;--blue:#70b8ff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:0;overflow-x:hidden}header{height:56px;display:flex;align-items:center;gap:14px;padding:0 18px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:2}h1{font:600 17px/1 system-ui;margin:0}.spacer{flex:1;min-width:0}.state{display:flex;align-items:center;gap:8px;color:var(--muted)}.dot{width:9px;height:9px;border-radius:50%;background:var(--green)}.dot.paused{background:var(--amber)}button{border:1px solid var(--line);background:#25292d;color:var(--text);height:34px;padding:0 11px;border-radius:5px;font:inherit;cursor:pointer}button:hover{border-color:#596169}button.danger{color:#ffd8d8;border-color:#6a3538}button:disabled{opacity:.4;cursor:default}.icon{width:36px;padding:0;font-size:17px}main{width:100%;max-width:100vw;padding:14px 18px 40px;overflow-x:auto}.toolbar{display:flex;gap:8px;align-items:center;margin-bottom:12px;overflow:auto}.filter.active{border-color:var(--blue);color:var(--blue)}table{width:100%;min-width:680px;border-collapse:collapse;table-layout:fixed}th{text-align:left;color:var(--muted);font-weight:500;border-bottom:1px solid var(--line);padding:9px 8px}td{border-bottom:1px solid #292d31;padding:9px 8px;vertical-align:top;overflow-wrap:anywhere}.id{width:150px;white-space:nowrap}.id .meta{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.status{width:94px}.age{width:94px}.actions{width:128px;text-align:right}.route{color:#d9e7f7}.meta{color:var(--muted);font-size:12px;margin-top:3px}.pill{display:inline-block;padding:2px 6px;border:1px solid var(--line);border-radius:4px}.running{color:var(--green)}.pending{color:var(--amber)}.failed{color:var(--red)}.empty{padding:40px 8px;color:var(--muted);text-align:center}.error{position:fixed;left:18px;right:18px;bottom:16px;background:#391f22;border:1px solid #7a3d43;padding:10px 12px;border-radius:5px;display:none}@media(max-width:760px){header{width:100vw;padding:0 10px;gap:4px}h1{font-size:14px}.state{gap:4px;font-size:12px}.spacer{display:none}.icon{width:32px}main{padding:10px}.age,.ageCell{display:none}.routeCol{width:44%}.actions{width:88px}button.text{font-size:0;width:36px}.text span{font-size:17px}}
.smoke-card fieldset{border:0;padding:0;margin:12px 0;display:flex;flex-wrap:wrap;gap:10px}.smoke-card legend{margin-bottom:6px}.smoke-card fieldset label{display:flex;align-items:center;gap:5px}.smokeRun{margin:12px 0}.smokeRun summary{cursor:pointer;color:var(--blue);padding:6px 0}.smokeRun pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.smoke-audio audio{width:100%;max-width:480px}.smoke-actions{flex-wrap:wrap;gap:8px}
</style>
</head>
<body><header><h1>GPU Greenroom</h1><div class="state"><span id="dot" class="dot"></span><span id="queueState">loading</span></div><div class="spacer"></div><button id="pause" title="Pause queue" class="icon">Ⅱ</button><button id="resume" title="Resume queue" class="icon">▶</button><button id="refresh" title="Refresh" class="icon">↻</button></header>
<main><section id="smokeRequests" class="smoke-panel" aria-labelledby="smokeHeading"><div class="smoke-heading"><h2 id="smokeHeading">Smoke requests</h2><button id="refreshSmoke" type="button">Refresh</button><span id="smokeStatus" class="meta" aria-live="polite">Loading</span></div><div id="smokeRequestList"><div class="empty">Loading requests…</div></div></section><div id="pauseMeta" class="meta" hidden></div><div id="lease" class="meta" hidden></div><div class="toolbar" id="filters"></div><table><thead><tr><th class="id">Job</th><th class="status">State</th><th class="routeCol">Route</th><th class="age">Elapsed</th><th class="actions">Actions</th></tr></thead><tbody id="jobs"></tbody></table><div id="empty" class="empty" hidden>No jobs in this view.</div></main><div id="error" class="error"></div>
<script>
const hash=new URLSearchParams(location.hash.slice(1)),bootstrap=globalThis.__GREENROOM_BOOTSTRAP_TOKEN__||'';if(bootstrap){sessionStorage.setItem('greenroom-token',bootstrap)}else if(hash.get('token')){sessionStorage.setItem('greenroom-token',hash.get('token'));history.replaceState(null,'',location.pathname)}const token=sessionStorage.getItem('greenroom-token')||'';let filter='active',last=null;const statuses=['active','all','pending','running','done','failed','cancelled'];
const smokeDrafts=new Map(),smokeParticipation=new Map(),smokeRuns=new Map(),smokeRunOpen=new Set(),smokeAudio=new Map(),smokeSubmitting=new Set(),smokeAccepted=new Set();let smokeLoadGeneration=0,smokeHasLoaded=false,smokeRefreshUnavailable=false,smokeArrivalHandled=false,smokeFilter='active',lastSmoke=null;
const smokePhases={'awaiting-start':'Awaiting start','waiting-gpu':'Queued',running:'Executing; participation unverified',loading:'Loading',interactive:'Interactive (reported)',preparing:'Preparing',blocked:'Blocked','operator-needed':'Waiting for you','awaiting-response':'Waiting for response',responded:'Response returned',failed:'Failed',cancelled:'Cancelled',unknown:'Unverified'};
function smokeProgress(item){const d=item.display||{},p=d.progress;return `<p class="smoke-progress"><strong>${esc(smokePhases[d.phase]||'Unverified')}</strong>${d.queue_position?' · Queue position '+esc(d.queue_position):''}${d.label?' · '+esc(d.label):''}</p>${p?.total?`<div class="smoke-progress"><progress value="${esc(p.completed)}" max="${esc(p.total)}"></progress> ${esc(p.completed)} / ${esc(p.total)} ${esc(p.unit)}</div>`:''}${d.failure_phase?`<p class="meta">Failure phase: ${esc(d.failure_phase)}${d.exit_code!=null?' · Exit '+esc(d.exit_code):''}</p>`:''}${d.error?`<p class="meta">${esc(d.error)}</p>`:''}`}
function smokeRunView(item){
  const id=item.request.id,d=item.display||{},run=smokeRuns.get(id);
  if(!d.run_available)return '';
  const result=d.job_state==='done'?'Process completed':d.job_state==='failed'?'Process failed':'Job cancelled';
  return `<p>${esc(result)}${d.exit_code!=null?' · Exit '+esc(d.exit_code):''} · Ended ${d.finished_at?esc(new Date(d.finished_at*1000).toLocaleString()):'time not recorded'}</p><details class="smokeRun" data-id="${esc(id)}" ${smokeRunOpen.has(id)?'open':''}><summary>View last run</summary>${run?`<p class="meta">Job ${esc(run.job_id)} · Participation: ${esc(run.participation)}</p>${run.errors.map(e=>'<p class="meta">'+esc(e)+'</p>').join('')}${run.artifacts.map(a=>`<div class="smoke-audio"><p>${esc(a.label)}</p>${smokeAudio.has(id+':'+a.index)?`<div data-smoke-player="${esc(id+':'+a.index)}"></div>`:`<button type="button" class="smokeAudio" data-id="${esc(id)}" data-index="${a.index}">Load recording</button>`}</div>`).join('')}<details><summary>Process output</summary><pre>${esc(run.stdout??'Output unavailable')}</pre></details><details><summary>Error output</summary><pre>${esc(run.stderr??'Error output unavailable')}</pre></details>`:'<p class="meta">Run details not loaded.</p>'}</details>`;
}
function smokeNext(item){
  const id=item.request.id,d=item.display||{},next=d.next_session;
  if(!d.run_available)return '';
  if(next?.phase==='awaiting-start')return document.body.dataset.readOnly==='true'?'<p class="meta">New session prepared; this view is read-only.</p>':`<div class="smoke-actions"><button type="button" class="smokeStart" data-id="${esc(next.id)}" data-digest="${esc(next.request_digest)}">▶ Start new session</button></div>`;
  if(next)return `<p><a class="smokeNext" href="#smoke-${esc(next.id)}">View new session</a></p>`;
  return `${d.next_session_error?'<p class="meta">New session unavailable: '+esc(d.next_session_error)+'</p>':''}${item.repeat_request?'<p class="meta">Another session requested; waiting for the owner to prepare it.</p>':document.body.dataset.readOnly==='true'?'':`<div class="smoke-actions"><button type="button" class="smokeRepeat" data-id="${esc(id)}" data-digest="${esc(item.request_digest)}">Request another session</button></div>`}`;
}
function smokeReply(item){
  const id=item.request.id,ended=item.display?.run_available,choice=smokeParticipation.get(id)||'';
  return `<form class="smokeReply" data-id="${esc(id)}">${ended?`<fieldset><legend>Did you get to try this session?</legend><label><input type="radio" name="participation" value="tried" ${choice==='tried'?'checked':''} required> I tried it</label><label><input type="radio" name="participation" value="not-tried" ${choice==='not-tried'?'checked':''} required> I didn't get to try it</label></fieldset>`:''}<label for="reply-${esc(id)}">${ended?'What happened?':'Your response'}</label><textarea class="smoke-response" id="reply-${esc(id)}" ${choice==='not-tried'?'':'required'}></textarea><div class="smoke-actions"><button type="submit">Send response</button></div></form>`;
}
function auth(){return {'Authorization':'Bearer '+token,'Content-Type':'application/json'}}async function request(path,options={}){const r=await fetch(path,options);if(r.status===401&&globalThis.__GREENROOM_LOCAL_OPERATOR__){if(!sessionStorage.getItem('greenroom-auth-reload')){sessionStorage.setItem('greenroom-auth-reload','1');location.reload();return new Promise(()=>{})}}else if(r.ok){sessionStorage.removeItem('greenroom-auth-reload')}return r}function err(e){const n=document.querySelector('#error');n.textContent=e;n.style.display='block';setTimeout(()=>n.style.display='none',6000)}function elapsed(j){const end=j.finished_at||Date.now()/1000,start=j.started_at||j.submitted_at;if(!start)return '—';const s=Math.max(0,end-start);if(s<60)return Math.round(s)+'s';if(s<3600)return Math.floor(s/60)+'m '+Math.round(s%60)+'s';return Math.floor(s/3600)+'h '+Math.floor((s%3600)/60)+'m'}function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
let busy=false;
function duration(seconds){if(seconds==null)return 'not recorded';const s=Math.floor(seconds);return s>=3600?`${Math.floor(s/3600)}h ${Math.floor(s%3600/60)}m`:s>=60?`${Math.floor(s/60)}m ${s%60}s`:`${s}s`}
function render(s){
  last=s;
  const state=s.admission_state==='paused_running'?'Paused; work still running':s.paused?'Paused; no queued job running':'Queue starts enabled';
  document.querySelector('#queueState').textContent=state;
  if(s.dispatch?.error)document.querySelector('#queueState').textContent+=' · Dispatch blocked: '+s.dispatch.error;
  document.querySelector('#dot').className='dot'+(s.paused?' paused':'');
  document.querySelector('#pause').disabled=busy||s.paused;
  document.querySelector('#resume').disabled=busy||!s.paused||(s.admission_control&&!s.pause_state?.epoch);
  const pm=document.querySelector('#pauseMeta'),p=s.pause_state;
  pm.hidden=false;pm.textContent=`Observed: ${new Date(s.observed_at*1000).toLocaleString()}`+(p?` · Pause owner: ${p.owner||'not recorded'} · epoch ${p.epoch||'not recorded'}`:'');
  if(s.paused&&!p?.epoch)pm.textContent+=' · Legacy pause: epoch not recorded.'+(s.admission_control?' Verify pause ownership and recover through the native queue CLI; browser resume cannot bind this marker.':'');
  const lease=document.querySelector('#lease');lease.hidden=!s.lease;
  lease.textContent=s.lease?`Recorded external owner: ${s.lease.owner} · ${s.lease.effective_route} · ${s.lease.lifecycle_state}`:'';
  document.querySelector('#filters').innerHTML=statuses.map(x=>`<button class="filter ${filter===x?'active':''}" data-filter="${x}">${x}</button>`).join('');
  document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{filter=b.dataset.filter;load()});
  const rows=s.jobs;document.querySelector('#empty').hidden=rows.length>0;
  document.querySelector('#jobs').innerHTML=rows.map(j=>{
    let a='';
    if(j.consistent&&j.status==='pending')a=`<button class="danger text" title="Cancel pending job" onclick="act('cancel','${j.job_id}')"><span>×</span> Cancel</button>`;
    if(j.consistent&&j.status==='running')a=`<button class="danger text" title="Stop running job" onclick="act('terminate','${j.job_id}')"><span>■</span> Stop</button> <button class="danger icon" title="Force stop running job" onclick="act('kill','${j.job_id}')">!</button>`;
    const route=j.requested_route||j.effective_route||j.job_type;
    const conflict=j.consistent?'':`<div class="meta">containment: ${esc(j.containment_status)} · declared: ${esc(j.declared_status)}</div>`;
    return `<tr><td class="id">${esc(j.job_id)}<div class="meta">${esc(j.job_type)}</div></td><td class="status"><span class="pill ${esc(j.status)}">${esc(j.status)}</span>${conflict}</td><td><div class="route">${esc(route)}</div><div class="meta">${esc(document.body.dataset.identityLabel||'Agent')}: ${esc(j.agent_id||'not recorded')}</div><div class="meta">${esc(j.repo_root||j.output_dir||j.record_error||'')}</div></td><td class="ageCell"><div>Submitted: ${j.submitted_at?esc(new Date(j.submitted_at*1000).toLocaleString()):'not recorded'}</div><div>Queue wait: ${duration(j.queue_wait_seconds)}</div><div>Execution wall time: ${duration(j.execution_wall_seconds)}</div><div class="meta">GPU hold time: not recorded</div></td><td class="actions">${a}</td></tr>`;
  }).join('');
}
async function load(){try{const r=await request('/api/state?view='+encodeURIComponent(filter),{headers:auth()});if(!r.ok)throw new Error(await r.text());render(await r.json())}catch(e){document.querySelector('#queueState').textContent='State unavailable';document.querySelector('#pause').disabled=true;document.querySelector('#resume').disabled=true;err(e.message)}}
function renderSmoke(payload){
  if(!payload||payload.schema!=='gpu-greenroom.smoke-request-list.v1'||!Array.isArray(payload.items)||!Array.isArray(payload.errors))throw new Error('Invalid smoke request list response');
  lastSmoke=payload;
  if(!smokeArrivalHandled&&location.hash.startsWith('#smoke-'))smokeFilter='all';
  const host=document.querySelector('#smokeRequestList'),items=payload.items.filter(item=>smokeFilter==='all'||(item.display?.section||'active')===smokeFilter),focused=document.activeElement,
    focusedForm=focused?.closest('form.smokeReply'),focusId=focusedForm?.dataset.id,
    focusControl=focused?.matches('textarea')?'textarea':focused?.matches('button[type="submit"]')?'button':null,
    selectionStart=focusControl==='textarea'?focused.selectionStart:null,
    selectionEnd=focusControl==='textarea'?focused.selectionEnd:null,
    selectionDirection=focusControl==='textarea'?focused.selectionDirection:null;
  host.querySelectorAll('form.smokeReply').forEach(form=>smokeDrafts.set(form.dataset.id,form.querySelector('textarea').value));
  const missingReply=[...host.querySelectorAll('form.smokeReply')].find(form=>!payload.items.some(item=>item.request?.id===form.dataset.id));
  if(missingReply){keepLastSmokeView('Request missing from refresh; showing last loaded requests');return}
  if(payload.errors.length&&!payload.items.length){
    err(payload.errors.join('\n'));
    if(smokeHasLoaded){keepLastSmokeView('Requests unreadable; showing last loaded requests');return}
    smokeRefreshUnavailable=true;
    document.querySelector('#smokeStatus').textContent=`${payload.errors.length} unreadable record(s)`;
    host.innerHTML='<div class="empty">Smoke requests could not be read.</div>';
    return;
  }
  smokeRefreshUnavailable=false;smokeHasLoaded=true;
  document.querySelector('#smokeStatus').textContent=payload.errors?.length?`${payload.errors.length} unreadable record(s)`:items.length?`${items.length} request(s)`:'No requests';
  let views=document.querySelector('#smokeViews');if(!views){views=document.createElement('div');views.id='smokeViews';views.className='toolbar';host.before(views)}
  const active=payload.items.filter(item=>(item.display?.section||'active')==='active').length;
  views.innerHTML=[['active',`Needs attention (${active})`],['history',`History (${payload.items.length-active})`]].map(([key,label])=>`<button type="button" class="filter ${smokeFilter===key?'active':''}" data-smoke-filter="${key}">${label}</button>`).join('');
  views.querySelectorAll('button').forEach(button=>button.onclick=()=>{smokeFilter=button.dataset.smokeFilter;renderSmoke(lastSmoke)});
  if(payload.errors?.length)err(payload.errors.join('\n'));
  if(!items.length){host.innerHTML='<div class="empty">No smoke requests.</div>';return}
  host.innerHTML=items.map(item=>{
    const q=item.request||{},response=item.response||{};
    if(item.status!=='operator-needed'){smokeDrafts.delete(q.id);smokeAccepted.delete(q.id)}
    const action=item.status==='operator-needed'?(document.body.dataset.readOnly==='true'?'<p class="meta">Read-only view; responses are unavailable.</p>':item.display?.phase==='awaiting-start'?`<div class="smoke-actions"><button type="button" class="smokeStart" data-id="${esc(q.id)}" data-digest="${esc(item.request_digest)}">▶ Start session</button></div>`:['operator-needed','interactive','awaiting-response'].includes(item.display?.phase)?smokeReply(item):''):`<p class="smoke-response-copy">${esc(response.text||'Response text not recorded.')}</p><p class="meta">${response.participation?'Participation: '+esc(response.participation)+' · ':''}Accepted through Greenroom; caller identity unverified.</p>`;
    const target=item.destination||{kind:'context',url:q.url};
    let destination='';
    if(target.kind==='browser')destination=`<a href="${esc(target.url)}" target="_blank" rel="noopener noreferrer">Open application</a>`;
    else if(target.kind==='terminal'&&!item.display?.run_available)destination=`<button type="button" class="smokeFocus" data-id="${esc(q.id)}" data-digest="${esc(item.request_digest)}" data-destination="${esc(target.identity_digest||'')}" ${item.display?.terminal_verified&&document.body.dataset.readOnly!=='true'?'':'disabled'}>Open terminal${target.pane_id!=null?' · pane '+esc(target.pane_id):''}</button>`;
    else if(target.url&&new URL(target.url,location.href).href!==location.origin+location.pathname)destination=`<a href="${esc(target.url)}" target="_blank" rel="noopener noreferrer">View linked page</a>`;
    const phase=item.display?.phase,instructions=item.display?.run_available?'':phase==='waiting-gpu'?'<p>Waiting for GPU dispatch; the session is not running yet.</p>':item.display?.job_state==='running'?`<p>${item.display.terminal_verified?'Open the terminal for the current session.':'Execution is underway; the application destination has not been verified.'}</p>`:`<p>${esc(q.prompt||'No request prompt recorded.')}</p>`;
    return `<article class="smoke-card" id="smoke-${esc(q.id)}"><h3>${esc(q.title||'Untitled smoke request')}</h3><p class="meta">Reported by (unverified): ${esc(q.source?.agent_id||'not recorded')}</p>${smokeProgress(item)}${item.display?.phase_reported_at?'<p class="meta">Phase reported: '+esc(new Date(item.display.phase_reported_at).toLocaleString())+'</p>':''}${instructions}${destination?'<p>'+destination+'</p>':''}${smokeRunView(item)}${smokeNext(item)}${action}</article>`
  }).join('');
  host.querySelectorAll('button.smokeStart').forEach(button=>{
    button.disabled=smokeRefreshUnavailable||smokeSubmitting.has(button.dataset.id);
    button.onclick=()=>startSmoke(button);
  });
  host.querySelectorAll('button.smokeFocus').forEach(button=>{button.onclick=()=>focusSmoke(button)});
  host.querySelectorAll('button.smokeRepeat').forEach(button=>{button.disabled=smokeSubmitting.has(button.dataset.id);button.onclick=()=>repeatSmoke(button)});
  host.querySelectorAll('details.smokeRun').forEach(details=>details.ontoggle=()=>{const id=details.dataset.id;if(details.open){smokeRunOpen.add(id);if(!smokeRuns.has(id))loadSmokeRun(id)}else smokeRunOpen.delete(id)});
  host.querySelectorAll('button.smokeAudio').forEach(button=>button.onclick=()=>loadSmokeAudio(button));
  host.querySelectorAll('[data-smoke-player]').forEach(holder=>holder.append(smokeAudio.get(holder.dataset.smokePlayer)));
  host.querySelectorAll('a.smokeNext').forEach(link=>link.onclick=()=>{smokeFilter='all';smokeArrivalHandled=false;setTimeout(()=>renderSmoke(lastSmoke),0)});
  host.querySelectorAll('form.smokeReply').forEach(form=>{
    const id=form.dataset.id,textarea=form.querySelector('textarea'),button=form.querySelector('button[type="submit"]');
    textarea.value=smokeDrafts.get(id)||'';
    textarea.addEventListener('input',()=>smokeDrafts.set(id,textarea.value));
    form.querySelectorAll('input[name="participation"]').forEach(input=>input.onchange=()=>{smokeParticipation.set(id,input.value);textarea.required=input.value!=='not-tried'});
    button.disabled=smokeRefreshUnavailable||smokeSubmitting.has(id)||smokeAccepted.has(id);
    form.addEventListener('submit',sendSmokeResponse);
    if(id===focusId&&focusControl){
      const target=focusControl==='textarea'?textarea:button;
      if(!target.disabled){target.focus({preventScroll:true});if(focusControl==='textarea'&&selectionStart!==null)target.setSelectionRange(selectionStart,selectionEnd,selectionDirection)}
    }
  });
  if(!smokeArrivalHandled&&location.hash.startsWith('#smoke-')){const target=document.getElementById(location.hash.slice(1));if(target){if(!focusControl){target.style.scrollMarginTop=(document.querySelector('header').getBoundingClientRect().height+12)+'px';target.scrollIntoView({block:'start'});target.tabIndex=-1;target.focus({preventScroll:true})}smokeArrivalHandled=true}else{document.querySelector('#smokeStatus').textContent='Smoke request in link is not available'}}
}
function keepLastSmokeView(message){
  smokeRefreshUnavailable=true;
  document.querySelector('#smokeStatus').textContent=message;
    document.querySelectorAll('#smokeRequestList button').forEach(button=>button.disabled=true);
}
async function loadSmoke(){
  const generation=++smokeLoadGeneration;
  try{
    const r=await request('/api/smoke-requests',{headers:auth()});
    if(!r.ok)throw new Error(await r.text());
    const payload=await r.json();
    if(generation!==smokeLoadGeneration)return;
    renderSmoke(payload);
  }catch(e){
    if(generation!==smokeLoadGeneration)return;
    if(smokeHasLoaded){keepLastSmokeView('Refresh failed; showing last loaded requests');err(e.message);return}
    document.querySelector('#smokeStatus').textContent='State unavailable';
    document.querySelector('#smokeRequestList').innerHTML='<div class="empty">Smoke requests unavailable.</div>';
    err(e.message);
  }
}
async function startSmoke(button){const id=button.dataset.id;if(smokeSubmitting.has(id))return;smokeSubmitting.add(id);button.disabled=true;try{const r=await request(`/api/smoke-requests/${encodeURIComponent(id)}/start`,{method:'POST',headers:auth(),body:JSON.stringify({request_digest:button.dataset.digest,request_id:crypto.randomUUID()})});if(!r.ok)throw new Error(await r.text());await loadSmoke();await load()}catch(e){err(e.message)}finally{smokeSubmitting.delete(id);const current=document.querySelector(`button.smokeStart[data-id="${CSS.escape(id)}"]`);if(current)current.disabled=smokeRefreshUnavailable}}
async function focusSmoke(button){button.disabled=true;try{const r=await request(`/api/smoke-requests/${encodeURIComponent(button.dataset.id)}/focus`,{method:'POST',headers:auth(),body:JSON.stringify({request_digest:button.dataset.digest,destination_digest:button.dataset.destination})});if(!r.ok)throw new Error(await r.text())}catch(e){err(e.message)}finally{await loadSmoke()}}
async function repeatSmoke(button){const id=button.dataset.id;if(smokeSubmitting.has(id))return;smokeSubmitting.add(id);button.disabled=true;try{const r=await request(`/api/smoke-requests/${encodeURIComponent(id)}/repeat`,{method:'POST',headers:auth(),body:JSON.stringify({request_digest:button.dataset.digest})});if(!r.ok)throw new Error(await r.text());await loadSmoke()}catch(e){err(e.message)}finally{smokeSubmitting.delete(id)}}
async function loadSmokeRun(id){try{const r=await request(`/api/smoke-requests/${encodeURIComponent(id)}/run`,{headers:auth()});if(!r.ok)throw new Error(await r.text());const run=await r.json(),item=lastSmoke?.items.find(i=>i.request.id===id);if(run.schema!=='gpu-greenroom.smoke-run.v1'||run.request_digest!==item?.request_digest||run.job_id!==item?.display?.job_id)throw new Error('Last-run identity changed');smokeRuns.set(id,run);renderSmoke(lastSmoke)}catch(e){smokeRunOpen.delete(id);err(e.message)}}
async function loadSmokeAudio(button){const id=button.dataset.id,index=button.dataset.index;button.disabled=true;try{const r=await request(`/api/smoke-requests/${encodeURIComponent(id)}/audio/${index}`,{headers:auth()});if(!r.ok)throw new Error(await r.text());const player=document.createElement('audio');player.controls=true;player.preload='metadata';player.src=URL.createObjectURL(await r.blob());smokeAudio.set(id+':'+index,player);renderSmoke(lastSmoke)}catch(e){err(e.message);button.disabled=false}}
async function sendSmokeResponse(event){event.preventDefault();const form=event.currentTarget,id=form.dataset.id,button=form.querySelector('button[type="submit"]'),participation=form.querySelector('input[name="participation"]:checked')?.value;let text=form.querySelector('textarea').value;if(participation==='not-tried'&&!text.trim())text="I didn't get to try this session.";if(smokeSubmitting.has(id)||smokeAccepted.has(id))return;smokeSubmitting.add(id);button.disabled=true;try{const r=await request(`/api/smoke-requests/${encodeURIComponent(id)}/response`,{method:'POST',headers:auth(),body:JSON.stringify({text,...(participation?{participation}:{})})});if(!r.ok)throw new Error(await r.text());smokeAccepted.add(id);smokeDrafts.delete(id);await loadSmoke()}catch(e){err(e.message)}finally{smokeSubmitting.delete(id);const current=document.querySelector(`form.smokeReply[data-id="${CSS.escape(id)}"] button[type="submit"]`);if(current)current.disabled=smokeRefreshUnavailable||smokeAccepted.has(id)}}
async function post(path,body={}){if(busy)return;busy=true;if(last)render(last);try{const r=await request(path,{method:'POST',headers:auth(),body:JSON.stringify({...body,request_id:crypto.randomUUID()})});if(!r.ok)throw new Error(await r.text())}finally{busy=false;await load()}}
async function act(kind,id){try{if(kind==='cancel')await post('/api/cancel',{job_id:id,requested_by:'operator-console',reason:'operator cancelled pending job'});else await post('/api/terminate',{job_id:id,requested_by:'operator-console',reason:'operator stopped running job',force:kind==='kill'})}catch(e){err(e.message)}}
document.querySelector('#pause').onclick=()=>post('/api/pause',{requested_by:'operator-console'}).catch(e=>err(e.message));document.querySelector('#resume').onclick=()=>post('/api/resume',{requested_by:'operator-console',epoch:last?.pause_state?.epoch}).catch(e=>err(e.message));document.querySelector('#refresh').onclick=load;document.querySelector('#refreshSmoke').onclick=loadSmoke;load();loadSmoke();setInterval(()=>{if(filter==='active')load();loadSmoke()},5000);
</script></body></html>"""


def queue_snapshot(queue: GPUQueue | Path, view: str = "all", *, read_only: bool = False) -> dict:
    queue_dir = queue if isinstance(queue, Path) else queue.queue_dir
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
        status_dir = queue_dir / containment_status
        for job_dir in sorted(status_dir.iterdir()) if status_dir.is_dir() else ():
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
                if not isinstance(request, dict):
                    request = {}
                row["agent_id"] = request.get("agent_id")
                row["requested_route"] = request.get("route_identity")
                row["repo_root"] = request.get("repo_root")
            now = time.time()
            submitted, started, finished = (row.get(key) for key in ('submitted_at', 'started_at', 'finished_at'))
            def difference(end, start):
                if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in (end, start)):
                    return None
                return end - start if end >= start else None
            wait_end = started if started is not None else (now if declared_status == 'pending' else None)
            row['queue_wait_seconds'] = difference(wait_end, submitted) if consistent else None
            row['execution_wall_seconds'] = difference(finished or (now if declared_status == 'running' else None), started) if consistent else None
            jobs.append(row)
    jobs.sort(key=lambda item: (item.get("submitted_at") or 0), reverse=True)
    lease_payload = None
    pause_path = queue_dir / "paused"
    if read_only:
        lease_path = queue_dir / "leases" / "current.json"
        if lease_path.exists():
            lease_payload = json.loads(lease_path.read_text())
        try:
            text = pause_path.read_text()
        except FileNotFoundError:
            paused, pause = False, None
        else:
            paused, pause = True, json.loads(text) if text.strip() else None
    else:
        lease = queue.lease_status()
        if lease is not None:
            lease_payload = json.loads(lease.to_json())
        paused, pause = queue.is_paused(), queue.pause_state()
    running_dir = queue_dir / 'running'
    running_ids = sorted(path.name for path in running_dir.iterdir()) if running_dir.is_dir() else []
    try:
        dispatch_state={'policy':dispatch.policy(queue_dir),'last_class':dispatch.last_class(queue_dir),'error':None}
    except (ValueError,OSError) as error:
        dispatch_state={'policy':None,'error':str(error)}
    prepared_root=queue_dir/'continuation-staging'
    dispatch_state['unpublished_continuations']=sorted(path.name for path in prepared_root.iterdir()) if prepared_root.is_dir() else []
    if paused:
        admission_state = 'paused_running' if running_ids else 'paused'
    else:
        admission_state = 'not_paused'
    return {
        "schema": "gpu-greenroom.operator-snapshot.v1",
        "view": view,
        "read_only": read_only,
        "observed_at": time.time(),
        "queue_dir": str(queue_dir.resolve()),
        "paused": paused,
        "admission_state": admission_state,
        "running_job_ids": running_ids,
        "pause_state": pause,
        "lease": lease_payload,
        "dispatch": dispatch_state,
        "jobs": jobs,
    }


def make_handler(queue: GPUQueue | Path, token: str, *, read_only: bool = False, admission_control: bool = False, identity_label: str = 'Agent', local_operator: bool = False):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format, *_args):
            return

        def parse_request(self) -> bool:
            # BaseHTTPRequestHandler collapses a leading // in self.path. Preserve
            # the request-target first so authority validation sees the wire form.
            words = self.raw_requestline.rstrip(b"\r\n").split()
            self._raw_request_target = (
                words[1].decode("iso-8859-1") if 2 <= len(words) <= 3 else ""
            )
            return super().parse_request()

        def _authorized(self) -> bool:
            return secrets.compare_digest(
                self.headers.get("Authorization", ""), f"Bearer {token}"
            )

        def _canonical_authority(self) -> bool:
            hosts = self.headers.get_all("Host", [])
            raw_target = self._raw_request_target
            target = urlparse(raw_target)
            return (
                hosts == [f"127.0.0.1:{self.server.server_port}"]
                and raw_target.startswith("/")
                and not raw_target.startswith("//")
                and not target.scheme
                and not target.netloc
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

        def _smoke_requests(self) -> SmokeRequests:
            root = queue if isinstance(queue, Path) else queue.queue_dir
            return SmokeRequests(root / "smoke-requests")

        def _smoke_post(self, path: str, body: dict) -> bool:
            if path == "/api/smoke-requests":
                target = body.get('target')
                if 'operator_command' in body or (isinstance(target, dict) and target.get('kind') == 'terminal'):
                    raise ValueError('operator commands and terminal destinations must be published through the local CLI')
                record, created = self._smoke_requests().submit(body)
                self._json(HTTPStatus.CREATED if created else HTTPStatus.OK, self._smoke_requests().public_record(record))
                return True
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[0:2] == ["api", "smoke-requests"] and parts[3] == "response":
                if "responded_by" in body:
                    raise ValueError("response actor identity is not authenticated; omit responded_by")
                record = self._smoke_requests().respond(parts[2], body.get("text"), participation=body.get('participation'))
                self._json(HTTPStatus.OK, self._smoke_requests().public_record(record))
                return True
            if len(parts) == 4 and parts[0:2] == ['api', 'smoke-requests'] and parts[3] == 'start':
                if set(body) - {'request_digest', 'request_id'}:
                    raise ValueError('Start accepts only the published request identity')
                record = self._smoke_requests().start(parts[2], body.get('request_digest'))
                self._json(HTTPStatus.OK, self._smoke_requests().public_record(record))
                return True
            if len(parts) == 4 and parts[0:2] == ['api', 'smoke-requests'] and parts[3] == 'focus':
                if set(body) != {'request_digest', 'destination_digest'}:
                    raise ValueError('Focus accepts only the published request and destination digests')
                result = self._smoke_requests().focus(parts[2], body['request_digest'], body['destination_digest'])
                self._json(HTTPStatus.OK, result)
                return True
            if len(parts) == 4 and parts[0:2] == ['api', 'smoke-requests'] and parts[3] == 'repeat':
                if set(body) != {'request_digest'}:
                    raise ValueError('Repeat accepts only the published request identity')
                record = self._smoke_requests().request_repeat(parts[2], body['request_digest'])
                self._json(HTTPStatus.OK, self._smoke_requests().public_record(record))
                return True
            return False

        @staticmethod
        def _is_smoke_post_path(path: str) -> bool:
            parts = path.strip("/").split("/")
            return path == "/api/smoke-requests" or (
                len(parts) == 4
                and parts[0:2] == ["api", "smoke-requests"]
                and parts[3] in {"response", "start", "focus", "repeat"}
            )

        def do_GET(self):
            if not self._canonical_authority():
                self._json(HTTPStatus.MISDIRECTED_REQUEST, {"error": "invalid_host"})
                return
            path = urlparse(self.path).path
            if path == "/":
                page = PAGE
                root = queue if isinstance(queue, Path) else queue.queue_dir
                source = {'schema':'gpu-greenroom.smoke-source.v1', 'queue_dir':str(root.resolve()),
                          'reader_sha256':hashlib.sha256(Path(__file__).with_name('smoke_requests.py').read_bytes()).hexdigest()}
                page = page.replace('<title>', '<meta name="greenroom-smoke-source" content="'+html.escape(json.dumps(source),quote=True)+'"><title>', 1)
                if local_operator:
                    bootstrap = (
                        f"<script>globalThis.__GREENROOM_LOCAL_OPERATOR__=true;"
                        f"globalThis.__GREENROOM_BOOTSTRAP_TOKEN__={json.dumps(token)};</script>"
                    )
                    page = page.replace("<script>", bootstrap + "<script>", 1)
                page = page.replace('<body>', f'<body data-read-only="{str(read_only).lower()}" data-identity-label="{html.escape(identity_label, quote=True)}">')
                page = page.replace('<th class="age">Elapsed</th>', '<th class="age">Timing</th>')
                page = page.replace('</style>', 'header{min-height:56px;height:auto;flex-wrap:wrap;padding-top:8px;padding-bottom:8px}.age,.ageCell{width:260px}.ageCell{font-size:12px}table{min-width:1000px}.smoke-panel{max-width:1100px;margin:0 0 14px;padding:12px;border:1px solid var(--line);border-radius:6px;background:var(--panel)}.smoke-heading{display:flex;align-items:center;gap:10px;margin:0 0 8px}.smoke-heading h2{font:600 15px/1.2 system-ui;margin:0}.smoke-card{border-top:1px solid var(--line);padding:10px 0}.smoke-card h3{font:600 14px/1.3 system-ui;margin:0 0 4px}.smoke-card p{margin:5px 0;white-space:pre-wrap;overflow-wrap:anywhere}.smoke-card a{color:var(--blue)}.smoke-response{width:100%;min-height:76px;padding:8px;border:1px solid var(--line);border-radius:4px;background:#111315;color:var(--text);font:inherit;resize:vertical}.smoke-actions{display:flex;justify-content:flex-end;margin-top:6px}.smoke-response-copy{border-left:2px solid var(--green);padding-left:9px}@media(max-width:760px){.age,.ageCell{display:table-cell}.state{flex:1;min-width:170px}}\n</style>')
                if admission_control:
                    page = page.replace('<style>', '<style>.actions{display:none!important}')
                elif read_only:
                    page = page.replace('<h1>GPU Greenroom</h1>', '<h1>GPU Greenroom · Read only</h1>')
                    page = page.replace('<style>', '<style>#pause,#resume,.actions{display:none!important}')
                    page = page.replace("s.paused?'paused':'admitting'", "s.paused?'paused':'not paused'")
                    page = page.replace('External owner:', 'Recorded external owner:')
                body = page.encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                    "connect-src 'self'; img-src 'self'; media-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'",
                )
                self.end_headers()
                self.wfile.write(body)
                return
            if not self._authorized():
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            if path == "/api/smoke-requests":
                self._json(HTTPStatus.OK, self._smoke_requests().snapshot())
                return
            parts = path.strip("/").split("/")
            if parts[:2] == ['api', 'smoke-requests'] and (len(parts) == 4 and parts[3] == 'run' or len(parts) == 5 and parts[3] == 'audio'):
                try:
                    if len(parts) == 4:
                        self._json(HTTPStatus.OK, self._smoke_requests().run_details(parts[2]))
                    else:
                        source, content_type = self._smoke_requests().review_artifact(parts[2], int(parts[4]))
                        with source.open('rb') as stream:
                            self.send_response(HTTPStatus.OK)
                            self.send_header('Content-Type', content_type)
                            self.send_header('Content-Length', str(source.stat().st_size))
                            self.send_header('Cache-Control', 'no-store')
                            self.end_headers()
                            shutil.copyfileobj(stream, self.wfile)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    self._json(HTTPStatus.BAD_REQUEST, {'error': str(error)})
                return
            if len(parts) == 3 and parts[0:2] == ["api", "smoke-requests"]:
                try:
                    record = self._smoke_requests().get(parts[2])
                except FileNotFoundError:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "smoke_request_not_found"})
                    return
                except ValueError as exc:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                self._json(HTTPStatus.OK, self._smoke_requests().public_record(record))
                return
            if path == "/api/state":
                query = parse_qs(urlparse(self.path).query)
                view = query.get("view", ["active"])[0]
                try:
                    snapshot = queue_snapshot(queue, view, read_only=read_only or admission_control)
                    snapshot['admission_control'] = admission_control
                except ValueError:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_view"})
                    return
                self._json(HTTPStatus.OK, snapshot)
                return
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self):
            if not self._canonical_authority():
                self._json(HTTPStatus.MISDIRECTED_REQUEST, {"error": "invalid_host"})
                return
            if not self._authorized():
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return
            path = urlparse(self.path).path
            if read_only:
                self._json(HTTPStatus.FORBIDDEN, {"error": "read_only_monitor"})
                return
            smoke_route = self._is_smoke_post_path(path)
            if admission_control:
                if path not in {'/api/pause', '/api/resume'} and not smoke_route:
                    self._json(HTTPStatus.FORBIDDEN, {'error': 'admission_controls_only'})
                    return
                try:
                    body = self._body()
                    if self._smoke_post(path, body):
                        return
                    root = queue if isinstance(queue, Path) else queue.queue_dir
                    result = transition(GPUQueue(root), action=path.removeprefix('/api/'),
                        request_id=body.get('request_id'), owner=body.get('requested_by'), epoch=body.get('epoch'))
                except SmokeRequestConflict as exc:
                    self._json(HTTPStatus.CONFLICT, {'error': str(exc)})
                    return
                except FileNotFoundError:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "smoke_request_not_found"})
                    return
                except ValueError as exc:
                    self._json(HTTPStatus.BAD_REQUEST if smoke_route else HTTPStatus.CONFLICT, {'error': str(exc)})
                    return
                except (RuntimeError, OSError) as exc:
                    self._json(HTTPStatus.CONFLICT, {'error': str(exc)})
                    return
                self._json(HTTPStatus.OK, result)
                return
            path = urlparse(self.path).path
            try:
                body = self._body()
                actor = str(body.get("requested_by") or "operator-console")
                if self._smoke_post(path, body):
                    return
                elif path == "/api/pause":
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
            except SmokeRequestConflict as exc:
                self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
                return
            except FileNotFoundError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "smoke_request_not_found"})
                return
            except ValueError as exc:
                self._json(HTTPStatus.BAD_REQUEST if smoke_route else HTTPStatus.CONFLICT, {"error": str(exc)})
                return
            except (KeyError, RuntimeError, OSError) as exc:
                self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
                return
            self._json(HTTPStatus.OK, result)

    return Handler


def operator_url(port: int, token: str, *, local_operator: bool = False) -> str:
    base = f"http://127.0.0.1:{port}/"
    return base if local_operator else f"{base}#token={token}"


def serve(queue_dir: str | Path, *, port: int = 8765, token: str | None = None, read_only: bool = False, admission_control: bool = False, identity_label: str = 'Agent', local_operator: bool = False):
    token = token or secrets.token_urlsafe(32)
    queue = Path(queue_dir) if read_only or admission_control else GPUQueue(queue_dir)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(queue, token, read_only=read_only, admission_control=admission_control, identity_label=identity_label, local_operator=local_operator))
    effective_port = server.server_address[1]
    print(operator_url(effective_port, token, local_operator=local_operator), flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="GPU Greenroom operator console")
    parser.add_argument("--queue-dir", default="~/.local/state/gpu-greenroom")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--identity-label", default="Agent")
    parser.add_argument(
        "--local-operator",
        action="store_true",
        help="seat the rotating process credential in pages served on the loopback-only console",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--read-only", action="store_true")
    modes.add_argument("--admission-control", action="store_true")
    args = parser.parse_args()
    serve(Path(args.queue_dir).expanduser(), port=args.port, read_only=args.read_only, admission_control=args.admission_control, identity_label=args.identity_label, local_operator=args.local_operator)


if __name__ == "__main__":
    main()
