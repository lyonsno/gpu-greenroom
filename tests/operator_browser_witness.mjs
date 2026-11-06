import assert from 'node:assert/strict';
import {mkdtempSync, readFileSync, rmSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {spawn} from 'node:child_process';
import {setTimeout as delay} from 'node:timers/promises';

const [chromePath, base, scenario, requestId, sourceRevision, artifactDirectory] = process.argv.slice(2);
if (!chromePath || !base || !scenario || !requestId || !sourceRevision || !artifactDirectory) {
  throw new Error('usage: witness.mjs CHROME BASE SCENARIO REQUEST_ID SOURCE_ID ARTIFACT_DIRECTORY');
}

let profile, browser, socket;
let phase = 'profile-creation';
let browserIdentity = null;
let screenshotArtifact = null;
function markPhase(value) {
  phase = value;
  writeFileSync(join(artifactDirectory, `${scenario}-phase.json`), JSON.stringify({phase, pid: browser.pid, profile}));
}

async function eventually(read, predicate, label) {
  const until = Date.now() + 8000;
  let value;
  while (Date.now() < until) {
    value = await read();
    if (predicate(value)) return value;
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  throw new Error(`${label} timed out; last value=${JSON.stringify(value)}`);
}

try {
  profile = mkdtempSync(join(tmpdir(), 'greenroom-browser-witness-'));
  phase = 'chrome-startup';
  browser = spawn(chromePath, [
    '--headless=new', '--no-first-run', '--no-default-browser-check',
    '--disable-gpu',
    '--use-mock-keychain', '--password-store=basic',
    '--disable-background-networking', '--remote-allow-origins=*',
    '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank',
  ], {detached: true, stdio: 'ignore'});
  const spawned = new Promise((resolve, reject) => {
    browser.once('spawn', resolve);
    browser.once('error', reject);
  });
  writeFileSync(join(artifactDirectory, `${scenario}-browser-owner.json`), JSON.stringify({
    pid: browser.pid ?? null, profile, executable: chromePath, sourceRevision,
  }));
  await spawned;
  const activePort = join(profile, 'DevToolsActivePort');
  await eventually(() => {
    try { return readFileSync(activePort, 'utf8').trim().split('\n'); }
    catch { return null; }
  }, value => Array.isArray(value) && value.length >= 2, 'Chrome DevTools startup');
  const [port] = readFileSync(activePort, 'utf8').trim().split('\n');
  const browserVersion = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
  browserIdentity = browserVersion.Browser;
  markPhase('target-creation');
  const targetResponse = await fetch(`http://127.0.0.1:${port}/json/new?${base}/`, {method: 'PUT'});
  assert.equal(targetResponse.status, 200, 'Chrome must create an isolated page target');
  const target = await targetResponse.json();
  writeFileSync(join(artifactDirectory, `${scenario}-target.json`), JSON.stringify(target));
  socket = new WebSocket(target.webSocketDebuggerUrl);
  markPhase('devtools-connection');
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, {once: true});
    socket.addEventListener('error', reject, {once: true});
  });
  let nextId = 0;
  const pending = new Map();
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (!message.id) return;
    const callbacks = pending.get(message.id);
    if (!callbacks) return;
    pending.delete(message.id);
    if (message.error) callbacks.reject(new Error(message.error.message));
    else callbacks.resolve(message.result);
  });
  const command = (method, params = {}, sessionId = undefined) => new Promise((resolve, reject) => {
    writeFileSync(join(artifactDirectory, `${scenario}-last-command.json`), JSON.stringify({phase, method}));
    const id = ++nextId;
    pending.set(id, {resolve, reject});
    socket.send(JSON.stringify({id, method, params, sessionId}));
  });
  const evaluate = async (expression, sessionId) => {
    const result = await command('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true}, sessionId);
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
    return result.result.value;
  };

  await command('Page.enable');
  await command('Runtime.enable');
  markPhase('initial-render');
  await command('Page.navigate', {url: `${base}/#${scenario==='attention'?'job':'smoke'}-${requestId}`});
  try {
    await eventually(() => evaluate(scenario==='attention'?`Boolean(document.querySelector('[data-performance="defer"]:not(:disabled)'))`:`Boolean(document.querySelector('#smokeRequestList .smoke-card'))`), Boolean, 'initial Greenroom request render');
  } catch (error) {
    writeFileSync(join(artifactDirectory, `${scenario}-document.json`), await evaluate(`JSON.stringify({href:location.href,body:document.body?.innerText})`));
    throw error;
  }
  markPhase('initial-screenshot');
  await command('Page.bringToFront');
  const initialFrame = await command('Page.captureScreenshot', {format: 'png', fromSurface: true});
  const artifact = join(artifactDirectory, `${scenario}.png`);
  writeFileSync(artifact, Buffer.from(initialFrame.data, 'base64'));
  screenshotArtifact = artifact;

  if (scenario === 'attention') {
    for(const [width,height] of [[1440,1000],[390,844]]) {
      await command('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:width===390});
      await delay(250);
      const layout=await evaluate(`({text:document.querySelector('#operatorAttention').innerText,overflow:document.documentElement.scrollWidth>innerWidth,buttons:[...document.querySelectorAll('#operatorAttention button')].map(n=>{const r=n.getBoundingClientRect();return {left:r.left,right:r.right,width:r.width}})})`);
      assert.ok(layout.text.includes('Needs you'));
      assert.ok(layout.text.includes('Quit Chrome'));
      assert.equal(layout.overflow,false);
      assert.ok(layout.buttons.every(b=>b.left>=0&&b.right<=width&&b.width>=30));
      const shot=await command('Page.captureScreenshot',{format:'png',fromSurface:true});
      writeFileSync(join(artifactDirectory,`attention-${width}.png`),Buffer.from(shot.data,'base64'));
    }
    await evaluate(`window.__decisions=[];const realFetch=window.fetch.bind(window);window.fetch=async(i,o)=>{if(String(i).endsWith('/api/performance-decision'))window.__decisions.push(JSON.parse(o.body));return realFetch(i,o)};const n=document.querySelector('[data-delay]');n.value='7';n.dispatchEvent(new Event('input',{bubbles:true}));n.focus()`);
    await delay(5200);
    assert.equal(await evaluate(`document.querySelector('[data-delay]').value`),'7');
    await evaluate(`document.querySelector('[data-performance="defer"]').click()`);
    await eventually(()=>evaluate(`last.jobs.find(j=>j.job_id===${JSON.stringify(requestId)})?.status`),v=>v==='deferred','actual worker deferral');
    await eventually(()=>evaluate(`!attention.items.some(i=>i.job_id===${JSON.stringify(requestId)})`),Boolean,'cleared durable attention');
    const decisions=await evaluate('window.__decisions');
    assert.equal(decisions.length,1);
    assert.equal(decisions[0].defer_minutes,7);
    console.log(JSON.stringify({scenario,sourceRevision,effectiveRoute:base,browser:browserVersion.Browser,
      artifact,decisions,assertions:['persistent top attention','desktop/mobile fit','delay draft survives refresh','one authenticated exact-attempt decision','worker defers without execution','attention clears']}));
  } else if (scenario === 'lease-recovery') {
    await eventually(()=>evaluate(`document.querySelector('#queueState').textContent`),text=>text.includes('External workload blocks starts'),'effective external blocker');
    await evaluate(`window.__recoverCalls=[];window.__realFetch=window.fetch.bind(window);window.fetch=async(input,options)=>{if(String(input).endsWith('/api/lease/recover')){if(!window.__recoverCalls.length){window.__recoverCalls.push({status:'injected-network-failure',body:JSON.parse(options.body)});throw new Error('injected recovery network failure')}const r=await window.__realFetch(input,options);window.__recoverCalls.push({status:r.status,body:JSON.parse(options.body)});return r}return window.__realFetch(input,options)}`);
    for(const [width,height] of [[1440,1000],[390,844]]) {
      await command('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:width===390});
      const banner=await evaluate(`({text:document.querySelector('#lease').innerText,overflow:document.documentElement.scrollWidth>innerWidth})`);
      assert.ok(banner.text.includes('page-owner')&&banner.text.includes('fixture/browser-work')&&banner.text.includes('legacy-page'),JSON.stringify(banner));
      assert.equal(banner.overflow,false);
      const frame=await command('Page.captureScreenshot',{format:'png',fromSurface:true});
      writeFileSync(join(artifactDirectory,`${scenario}-blocker-${width}.png`),Buffer.from(frame.data,'base64'));
      await evaluate(`document.querySelector('#reviewLease').click()`);
      const modal=await evaluate(`(()=>{const d=document.querySelector('#leaseRecoveryDialog'),r=d.getBoundingClientRect();return {open:d.open,text:d.innerText,disabled:document.querySelector('#confirmLeaseRecovery').disabled,left:r.left,right:r.right,top:r.top,bottom:r.bottom,focus:document.activeElement?.textContent}})()`);
      assert.equal(modal.open,true);
      assert.ok(modal.text.includes('browser')&&modal.text.includes('fixture/browser-work'));
      assert.equal(modal.disabled,true);
      assert.equal(modal.focus,'Keep blocked');
      assert.ok(modal.left>=0&&modal.right<=width&&modal.top>=0&&modal.bottom<=height,JSON.stringify(modal));
      const dialogFrame=await command('Page.captureScreenshot',{format:'png',fromSurface:true});
      writeFileSync(join(artifactDirectory,`${scenario}-modal-${width}.png`),Buffer.from(dialogFrame.data,'base64'));
      await evaluate(`document.querySelector('#leaseRecoveryDialog button[value="back"]').click()`);
      assert.equal(await evaluate(`window.__recoverCalls.length`),0);
    }
    await evaluate(`document.querySelector('#reviewLease').click();document.querySelector('#leaseStopped').click();document.querySelector('#confirmLeaseRecovery').click()`);
    await eventually(()=>evaluate(`window.__recoverCalls.length`),n=>n===1,'one exact recovery request');
    await eventually(()=>evaluate(`!busy`),Boolean,'failed recovery request settles');
    assert.equal(await evaluate(`document.querySelector('#lease').hidden`),false);
    await evaluate(`document.querySelector('#reviewLease').click();document.querySelector('#leaseStopped').click();document.querySelector('#confirmLeaseRecovery').click()`);
    await eventually(()=>evaluate(`window.__recoverCalls.length`),n=>n===2,'explicit recovery retry');
    assert.equal(await evaluate(`window.__recoverCalls[1].status`),200);
    assert.deepEqual(await evaluate(`window.__recoverCalls[0].body`),await evaluate(`window.__recoverCalls[1].body`),'retry must preserve the exact action identity');
    assert.equal(await evaluate(`window.__recoverCalls[0].body.quiescence_confirmed`),true);
    await eventually(()=>evaluate(`document.querySelector('#lease').hidden`),Boolean,'released blocker disappears');
    assert.ok(await evaluate(`document.querySelector('#queueState').textContent.includes('Queue starts enabled')`));
    markPhase('owned-browser-shutdown');
    await command('Browser.close');
    console.log(JSON.stringify({scenario,sourceRevision,effectiveRoute:base,effectiveMode:'admission-control with disposable request fixture',apiBehavior:'real authenticated API; injected first recovery transport failure only',browser:browserVersion.Browser,artifact,assertions:['actual expired lease visible','effective header','desktop/mobile fit','explicit browser settlement confirmation','keep blocked has no mutation','retry preserves exact recovery action','one effective recovery','released blocker disappears']}));
  } else if (scenario === 'job-cancel') {
    await eventually(() => evaluate(`document.querySelectorAll('.job-item').length`), n=>n===2, 'actual queue jobs');
    await evaluate(`window.__cancelCalls=[];window.__realFetch=window.fetch.bind(window);window.fetch=async(input,options)=>{const r=await window.__realFetch(input,options);if(String(input).endsWith('/api/job-cancel'))window.__cancelCalls.push({status:r.status,body:JSON.parse(options.body)});return r}`);
    for (const [width,height] of [[1440,1000],[390,844]]) {
      await command('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:width===390});
      await evaluate(`document.querySelector('[data-job-id="browser-pending"] button[title="Cancel job"]').click()`);
      const modal=await evaluate(`(()=>{const d=document.querySelector('#cancelDialog'),r=d.getBoundingClientRect();return {open:d.open,text:d.innerText,left:r.left,right:r.right,bottom:r.bottom,top:r.top,width:innerWidth,height:innerHeight,focus:document.activeElement?.textContent}})()`);
      assert.equal(modal.open,true);
      assert.ok(modal.text.includes('fixture-queued-owner')&&modal.text.includes('fixture/browser-cancel-pending'));
      assert.equal(modal.focus,'Keep job');
      assert.ok(modal.left>=0&&modal.right<=width&&modal.top>=0&&modal.bottom<=height,JSON.stringify(modal));
      const frame=await command('Page.captureScreenshot',{format:'png',fromSurface:true});
      writeFileSync(join(artifactDirectory,`${scenario}-${width}.png`),Buffer.from(frame.data,'base64'));
      await evaluate(`document.querySelector('#cancelDialog button[value="back"]').click()`);
      assert.equal(await evaluate(`window.__cancelCalls.length`),0,'Keep job must not cancel');
    }
    await evaluate(`document.querySelector('[data-job-id="browser-pending"] button[title="Cancel job"]').click();document.querySelector('#confirmCancel').click()`);
    await eventually(()=>evaluate(`window.__cancelCalls.length`),n=>n===1,'queued cancellation');
    assert.equal(await evaluate(`window.__cancelCalls[0].status`),200);
    await eventually(()=>evaluate(`!document.querySelector('[data-job-id="browser-pending"]')`),Boolean,'pending removed');
    await evaluate(`document.querySelector('[data-job-id="browser-running"] button[title="Cancel job"]').click();document.querySelector('#confirmCancel').click()`);
    await eventually(()=>evaluate(`document.querySelector('[data-job-id="browser-running"]')?.innerText`),text=>text?.includes('Cancelling; waiting for process exit'),'running cancelling state');
    await evaluate(`document.querySelector('[data-job-id="browser-running"] button[title="Force stop job"]').click()`);
    assert.ok(await evaluate(`document.querySelector('#cancelDialog').innerText.includes('Unsaved work may be lost')`));
    await evaluate(`document.querySelector('#confirmCancel').click()`);
    await eventually(()=>evaluate(`window.__cancelCalls.length`),n=>n===3,'force request');
    await eventually(()=>evaluate(`load().then(()=>!document.querySelector('[data-job-id="browser-running"]'))`),Boolean,'running terminal cancellation');
    assert.equal(await evaluate(`window.__cancelCalls.every(c=>c.status===200)`),true);
    markPhase('owned-browser-shutdown');
    await command('Browser.close');
    console.log(JSON.stringify({scenario,sourceRevision,effectiveRoute:base,effectiveMode:'admission-control with disposable request fixture',apiBehavior:'real authenticated API',browser:browserVersion.Browser,artifact,assertions:['Keep job makes no request','modal identifies exact job and owner','desktop/mobile fit','queued cancellation','running cancellation waits','separate force confirmation','worker terminal cancellation']}));
  } else if (scenario === 'performance-deferred') {
    await eventually(() => evaluate(`Boolean(document.querySelector('.job-item .pill.deferred'))`), Boolean, 'deferred queue render');
    for (const [width, height] of [[1440, 1000], [390, 844]]) {
      await command('Emulation.setDeviceMetricsOverride', {width, height, deviceScaleFactor: 1, mobile: width === 390});
      await evaluate(`document.querySelector('.job-item .pill.deferred').closest('article').scrollIntoView({block:'start'})`);
      await evaluate(`scrollBy(0,-document.querySelector('header').getBoundingClientRect().height-8)`);
      const state = await evaluate(`(()=>{const node=document.querySelector('.job-item .pill.deferred').closest('article'),button=node.querySelector('button');return {text:node.innerText,cancel:button?.title,enabled:button&&!button.disabled,overflow:document.documentElement.scrollWidth>innerWidth}})()`);
      assert.ok(state.text.includes('Reminder after'), JSON.stringify(state));
      assert.ok(state.text.includes('Other work may proceed'), JSON.stringify(state));
      assert.equal(state.cancel, undefined, 'admission-only monitor must keep its existing control boundary');
      assert.equal(state.overflow, false, 'page must fit the viewport');
      const frame = await command('Page.captureScreenshot', {format:'png', fromSurface:true});
      writeFileSync(join(artifactDirectory, `${scenario}-${width}.png`), Buffer.from(frame.data, 'base64'));
    }
    markPhase('owned-browser-shutdown');
    await command('Browser.close');
    console.log(JSON.stringify({scenario, sourceRevision, effectiveRoute:base, effectiveMode:'admission-control with disposable request fixture', apiBehavior:'real authenticated API', browser:browserVersion.Browser, artifact, assertions:['validated deferral visible', 'reminder time visible', 'existing control boundary preserved', 'desktop and mobile fit']}));
  } else if (scenario === 'browser-entry') {
    const selector=`#smoke-${requestId}`;
    assert.equal(await evaluate(`document.querySelector('${selector} a')===null`),true,'request must not link to itself or open a second application');
    assert.equal(await evaluate(`document.querySelector('${selector} .smokeFocus').disabled`),false);
    const initialTargets=await command('Target.getTargets');
    await evaluate(`window.__focusCalls=[];window.__realFetch=window.fetch.bind(window);window.fetch=async(input,options)=>{const r=await window.__realFetch(input,options);if(String(input).endsWith('/focus'))window.__focusCalls.push({status:r.status,body:await r.clone().json()});return r}`);
    for(const [width,height] of [[1440,1000],[390,844]]){
      markPhase(`browser-entry-${width}`);
      await command('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:width===390});
      await evaluate(`document.querySelector('${selector}').scrollIntoView({block:'start'})`);
      const position=await evaluate(`(()=>{const r=document.querySelector('${selector} .smokeFocus').getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()`);
      const before=await evaluate(`window.__focusCalls.length`);
      await command('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',clickCount:1,...position});
      await command('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',clickCount:1,...position});
      await eventually(()=>evaluate(`window.__focusCalls.length`),n=>n>before,'new click response');
      const receipt=await evaluate(`window.__focusCalls[${before}]`);
      assert.equal(receipt.status,200,JSON.stringify(receipt));
      assert.equal(receipt.body.request_id,requestId);
      const action=await eventually(()=>evaluate(`(async()=>{await loadSmoke();return lastSmoke.items.find(i=>i.request.id==='${requestId}').display.focus_action})()`),
        value=>value?.action_id===receipt.body.action_id&&value.status!=='pending','that click owner result');
      assert.equal(await evaluate(`location.pathname`),'/');
      assert.deepEqual((await command('Target.getTargets')).targetInfos.map(t=>t.targetId).sort(),initialTargets.targetInfos.map(t=>t.targetId).sort(),'entry must not create any browser page');
      assert.equal(await evaluate(`document.documentElement.scrollWidth<=innerWidth`),true);
      const frame=await command('Page.captureScreenshot',{format:'png',fromSurface:true});
      writeFileSync(join(artifactDirectory,`browser-entry-${width}.png`),Buffer.from(frame.data,'base64'));
      screenshotArtifact=join(artifactDirectory,`browser-entry-${width}.png`);
      assert.equal(action.status,'complete',JSON.stringify(action));
      assert.equal(action.receipt.foreground,true);
      await eventually(()=>evaluate(`document.querySelector('${selector} .smokeFocus').disabled`),v=>v===false,'focus control restored');
    }
    console.log(JSON.stringify({scenario,sourceRevision,effectiveRoute:base,browser:browserVersion.Browser,artifact,
      assertions:['real monitor button and authenticated focus API','owner-bound reveal transport','no URL navigation or new page','desktop/mobile fit'],
      nativeForeground:'synthetic owner receipt; not claimed',simulation:'no GPU workload'}));
  } else if (scenario === 'finished-session' || scenario === 'installed-finished') {
    const selector=`#smoke-${requestId}`;
    const card=await evaluate(`document.querySelector('${selector}').innerText`);
    assert.ok(card.includes('Process completed')&&card.includes('Ended '),card);
    assert.ok(!card.includes('Start session.')&&!card.includes('Open terminal'),card);
    assert.ok(card.includes('Did you get to try this session?'),card);
    await evaluate(`document.querySelector('${selector} details.smokeRun').open=true`);
    await eventually(()=>evaluate(`document.querySelector('${selector} .smokeRun').innerText`),s=>s.includes('Process output')&&s.includes('Participation:'),'actual retained run');
    const audioButton=await evaluate(`Boolean(document.querySelector('${selector} .smokeAudio'))`);
    assert.equal(audioButton,true,'producer audio must be present for this witness');
    await evaluate(`document.querySelector('${selector} .smokeAudio').click()`);
    await eventually(()=>evaluate(`document.querySelector('${selector} audio')?.readyState`),s=>s>=1,'retained audio metadata');
    await evaluate(`window.__retainedPlayer=document.querySelector('${selector} audio');window.__retainedPlayer.currentTime=0.02;loadSmoke()`);
    assert.equal(await evaluate(`document.querySelector('${selector} audio')===window.__retainedPlayer`),true,'refresh must preserve the actual audio element');
    for(const [width,height] of [[1440,1000],[390,844]]){
      markPhase(`finished-${width}`);
      await command('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:width===390});
      await evaluate(`(()=>{const card=document.querySelector('${selector}');card.style.scrollMarginTop=(document.querySelector('header').getBoundingClientRect().height+12)+'px';card.scrollIntoView({block:'start'})})()`);
      assert.equal(await evaluate(`document.documentElement.scrollWidth<=innerWidth`),true);
      const frame=await command('Page.captureScreenshot',{format:'png',fromSurface:true});
      writeFileSync(join(artifactDirectory,`finished-${width}.png`),Buffer.from(frame.data,'base64'));
    }
    if(scenario==='finished-session'){
      markPhase('repeat-without-launch');
      await evaluate(`document.querySelector('${selector} .smokeRepeat').click()`);
      await eventually(()=>evaluate(`document.querySelector('${selector}').innerText`),s=>s.includes('Another session requested'),'repeat request receipt');
      await evaluate(`(()=>{const radio=document.querySelector('${selector} input[value="not-tried"]');radio.click();const t=document.querySelector('${selector} textarea');t.value='Synthetic missed-session fixture';t.dispatchEvent(new Event('input'));return loadSmoke()})()`);
      assert.equal(await evaluate(`document.querySelector('${selector} input[value="not-tried"]').checked`),true);
      assert.equal(await evaluate(`document.querySelector('${selector} textarea').value`),'Synthetic missed-session fixture');
      await evaluate(`document.querySelector('${selector} form').requestSubmit()`);
      await eventually(()=>evaluate(`document.querySelector('${selector}').innerText`),s=>s.includes('Feedback saved')&&s.includes('Participation: not-tried'),'missed-session response');
    }
    console.log(JSON.stringify({scenario,sourceRevision,effectiveRoute:base,browser:browserVersion.Browser,artifact,
      assertions:['finished not live','actual retained run and audio metadata','audio element preserved through refresh','desktop and mobile fit',
        ...(scenario==='finished-session'?['repeat request without launch','missed-session response and draft continuity']:[])],operatorAudiblePlayback:'not claimed'}));
  } else if (scenario === 'quick-drag') {
    markPhase('quick-drag-native-input');
    await command('Emulation.setDeviceMetricsOverride',{width:1440,height:1000,deviceScaleFactor:1,mobile:false});
    await evaluate(`location.hash='';smokeFilter='active';renderSmoke(lastSmoke);load()`);
    await eventually(()=>evaluate(`document.querySelectorAll('#jobs [draggable="true"]').length`),n=>n===2,'two draggable pending jobs');
    assert.equal(await evaluate(`document.querySelector('#quickQueue').hidden`),false,'empty Quick must remain a drop target');
    assert.equal(await evaluate(`document.querySelectorAll('#quickJobs .job-item').length`),0);
    await evaluate(`window.__moves=[];window.__realFetch=window.fetch.bind(window);window.fetch=async(input,options)=>{const response=await window.__realFetch(input,options);if(String(input)==='/api/dispatch-class')window.__moves.push({status:response.status,body:await response.clone().json(),input:JSON.parse(options.body)});return response}`);
    const positions=await evaluate(`(()=>{scrollTo(0,0);const item=document.querySelector('#jobs .job-item'),a=item.querySelector('strong').getBoundingClientRect(),b=document.querySelector('#quickQueue').getBoundingClientRect();window.__dragNode=item;return {id:item.dataset.jobId,a:{x:a.x+a.width/2,y:a.y+a.height/2},b:{x:b.x+b.width/2,y:b.y+100}}})()`);
    await command('Input.dispatchMouseEvent',{type:'mouseMoved',...positions.a});
    await command('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',buttons:1,clickCount:1,...positions.a});
    await command('Input.dispatchMouseEvent',{type:'mouseMoved',button:'left',buttons:1,x:positions.a.x+15,y:positions.a.y+15});
    await command('Input.dispatchMouseEvent',{type:'mouseMoved',button:'left',buttons:1,x:positions.a.x+25,y:positions.a.y+20});
    await eventually(()=>evaluate(`draggedJob?.job_id`),id=>id===positions.id,'native dragstart');
    await evaluate(`load()`);
    assert.equal(await evaluate(`window.__dragNode===document.querySelector('[data-job-id="${positions.id}"]')`),true,'refresh must not replace a dragged row');
    await command('Input.dispatchMouseEvent',{type:'mouseMoved',button:'left',buttons:1,...positions.b});
    await command('Input.dispatchMouseEvent',{type:'mouseMoved',button:'left',buttons:1,...positions.b});
    await command('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',clickCount:1,...positions.b});
    await eventually(()=>evaluate(`document.querySelector('#quickJobs [data-job-id="${positions.id}"]')?.innerText`),s=>s?.includes('Manual Quick'),'native drop confirmed by API and render');
    const moved=await evaluate(`window.__moves[0]`);
    assert.equal(moved.status,200);
    assert.equal(moved.body.job_id,positions.id);
    assert.equal(moved.body.service_class,'quick');
    const replay=await evaluate(`(async()=>{const r=await window.__realFetch('/api/dispatch-class',{method:'POST',headers:auth(),body:JSON.stringify(window.__moves[0].input)});return {status:r.status,body:await r.json()}})()`);
    assert.equal(replay.status,200);
    assert.deepEqual(replay.body,moved.body,'replayed native drop is idempotent');
    const frame=await command('Page.captureScreenshot',{format:'png',fromSurface:true});
    writeFileSync(join(artifactDirectory,'quick-drag-1440.png'),Buffer.from(frame.data,'base64'));
    await command('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
    assert.equal(await evaluate(`document.documentElement.scrollWidth<=innerWidth`),true);
    await evaluate(`(()=>{const select=document.querySelector('#quickJobs .dispatchClass');select.scrollIntoView({block:'center'});select.focus();select.value='normal';select.dispatchEvent(new Event('change',{bubbles:true}))})()`);
    await eventually(()=>evaluate(`document.querySelectorAll('#quickJobs .job-item').length`),n=>n===0,'mobile queue selector confirmation');
    assert.equal(await evaluate(`document.querySelector('#quickQueue').hidden`),false,'empty Quick after move-back');
    await evaluate(`document.querySelector('#automaticQueue').scrollIntoView({block:'start'})`);
    const phone=await command('Page.captureScreenshot',{format:'png',fromSurface:true});
    writeFileSync(join(artifactDirectory,'quick-drag-390.png'),Buffer.from(phone.data,'base64'));
    markPhase('quick-drag-stale-request');
    const rejected=await evaluate(`(async()=>{const j=last.jobs[0];const r=await window.__realFetch('/api/dispatch-class',{method:'POST',headers:auth(),body:JSON.stringify({job_id:j.job_id,service_class:'quick',request_digest:'0'.repeat(64),request_id:crypto.randomUUID()})});return {status:r.status,body:await r.json()}})()`);
    assert.equal(rejected.status,409);
    assert.ok(rejected.body.error.includes('digest'));
    // A fresh normal override also demonstrates another action remains independent.
    await evaluate(`moveJob(last.jobs.find(j=>j.job_id==='${positions.id}'),'quick')`);
    await evaluate(`moveJob(last.jobs.find(j=>j.job_id==='${positions.id}'),'normal')`);
    markPhase('quick-drag-visible-rejection');
    await evaluate(`(()=>{last.jobs.find(j=>j.job_id==='${positions.id}').request_digest='0'.repeat(64);render(last);const select=document.querySelector('[data-job-id="${positions.id}"] .dispatchClass');select.value='quick';select.dispatchEvent(new Event('change',{bubbles:true}))})()`);
    await eventually(()=>evaluate(`window.__moves.length`),n=>n===5,'real stale UI move response');
    assert.equal(await evaluate(`window.__moves[4].status`),409);
    assert.equal(await evaluate(`document.querySelector('#error').style.display==='block'&&document.querySelector('#error').textContent.includes('digest')`),true,'rejected UI move must display its actual error');
    assert.equal(await evaluate(`document.querySelector('[data-job-id="${positions.id}"] .dispatchClass').value`),'normal','failed move retains confirmed Automatic class');
    markPhase('quick-drag-refresh-outage');
    await evaluate(`window.fetch=(input,options)=>String(input).startsWith('/api/state')?Promise.reject(new Error('synthetic queue read outage')):window.__realFetch(input,options);load()`);
    assert.equal(await evaluate(`document.querySelectorAll('.job-item[draggable="true"]').length`),0);
    assert.equal(await evaluate(`[...document.querySelectorAll('.dispatchClass')].every(n=>n.disabled)`),true);
    console.log(JSON.stringify({scenario,sourceRevision,effectiveRoute:base,effectiveMode:'admission-control with disposable request fixture',apiBehavior:'real authenticated API; injected refresh outage only',browser:browserVersion.Browser,artifact,
      assertions:['native mouse drag into empty Quick','real authenticated override','idempotent replay','refresh preserves active drag','mobile selector move-back','stale UI move displays error','stale digest rejected','failed refresh disables moves','desktop/mobile fit']}));
  } else if (scenario === 'columns' || scenario === 'installed-columns') {
    await evaluate(`location.hash='';smokeFilter='active';renderSmoke(lastSmoke);load()`);
    if(scenario==='columns'){
      await eventually(()=>evaluate(`document.querySelectorAll('#quickJobs .job-item').length`),n=>n===1,'one quick automatic job; smoke excluded');
      assert.equal(await evaluate(`document.querySelectorAll('#quickJobs .quick').length`),1,'quick class comes from policy');
      assert.equal(await evaluate(`document.querySelectorAll('#jobs .quick').length`),0,'quick automatic work is separate');
    }else{
      await evaluate(`load()`);
      assert.equal(await evaluate(`(()=>{const linked=new Set(lastSmoke.items.map(item=>item.display?.job_id).filter(Boolean));return [...document.querySelectorAll('#jobs .job-item')].some(row=>linked.has(row.dataset.jobId))})()`),false,'installed smoke jobs not duplicated');
    }
    for(const [width,height] of [[1440,1000],[390,844]]){
      await command('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:width===390});
      const bounds=await evaluate(`(()=>{scrollTo(0,0);const a=document.querySelector('#automaticQueue').getBoundingClientRect(),b=document.querySelector('#operatorQueue').getBoundingClientRect(),q=document.querySelector('#quickQueue'),c=q.getBoundingClientRect();return {a:{x:a.x,y:a.y,right:a.right,bottom:a.bottom},b:{x:b.x,y:b.y,right:b.right,bottom:b.bottom},quick:!q.hidden,c:{x:c.x,y:c.y}}})()`);
      if(width===1440){assert.ok(bounds.a.right<bounds.b.x);assert.ok(Math.abs(bounds.a.y-bounds.b.y)<2)}
      else assert.ok(bounds.b.y>=bounds.a.bottom);
      if(bounds.quick){if(width===1440){assert.ok(bounds.b.right<bounds.c.x);assert.ok(Math.abs(bounds.b.y-bounds.c.y)<2)}else assert.ok(bounds.c.y>=bounds.b.bottom)}
      assert.equal(await evaluate(`document.documentElement.scrollWidth<=innerWidth`),true);
      const frame=await command('Page.captureScreenshot',{format:'png',fromSurface:true});
      writeFileSync(join(artifactDirectory,`columns-${width}.png`),Buffer.from(frame.data,'base64'));
    }
    console.log(JSON.stringify({scenario,sourceRevision,effectiveRoute:base,browser:browserVersion.Browser,artifact,
      assertions:['separate automatic/operator columns','conditional quick column','native quick badge','no duplicate smoke job','desktop/mobile fit']}));
  } else if (scenario === 'destinations') {
    for (const [width, height] of [[1440,1000],[390,844]]) {
      markPhase(`destinations-${width}`);
      await command('Emulation.setDeviceMetricsOverride', {width,height,deviceScaleFactor:1,mobile:width===390});
      await evaluate(`location.hash='';smokeFilter='active';renderSmoke(lastSmoke)`);
      assert.equal(await evaluate(`document.querySelectorAll('.smoke-card').length`),2);
      assert.equal(await evaluate(`document.documentElement.scrollWidth<=innerWidth`),true);
      const frame=await command('Page.captureScreenshot',{format:'png',fromSurface:true});
      writeFileSync(join(artifactDirectory,`active-${width}.png`),Buffer.from(frame.data,'base64'));
      const expectedPane=await evaluate(`Number(document.querySelector('.smokeFocus').textContent.split('pane ')[1])`);
      await evaluate(`(()=>{const realFetch=window.fetch.bind(window);window.fetch=async (...args)=>{const r=await realFetch(...args);if(String(args[0]).endsWith('/focus'))window.__focusReceipt={status:r.status,body:await r.clone().json()};return r};window.__focusReceipt=null;document.querySelector('.smokeFocus').click()})()`);
      const focused=await eventually(()=>evaluate(`window.__focusReceipt`),Boolean,'real terminal focus receipt');
      assert.equal(focused.status,200,JSON.stringify(focused));
      assert.equal(focused.body.pane_id,expectedPane);
      const bounds=await evaluate(`(()=>{const link=document.querySelector('a');link.scrollIntoView({block:'center'});const r=link.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()`);
      await command('Input.dispatchMouseEvent',{type:'mousePressed',button:'left',clickCount:1,...bounds});
      await command('Input.dispatchMouseEvent',{type:'mouseReleased',button:'left',clickCount:1,...bounds});
      const targets=await eventually(()=>command('Target.getTargets'),value=>value.targetInfos.some(x=>x.url===base+'/actual-smoke'),'application popup');
      const app=targets.targetInfos.find(x=>x.url===base+'/actual-smoke');
      const {sessionId}=await command('Target.attachToTarget',{targetId:app.targetId,flatten:true});
      const heading=await eventually(()=>evaluate(`document.querySelector('h1')?.textContent`,sessionId),x=>x==='Actual smoke application','actual application content');
      assert.equal(heading,'Actual smoke application');
      const appFrame=await command('Page.captureScreenshot',{format:'png'},sessionId);
      writeFileSync(join(artifactDirectory,`application-${width}.png`),Buffer.from(appFrame.data,'base64'));
      await command('Target.closeTarget',{targetId:app.targetId});
      await evaluate(`document.querySelector('[data-smoke-filter="history"]').click()`);
      assert.equal(await evaluate(`document.querySelectorAll('.smoke-card').length`),1);
      assert.equal(await evaluate(`document.querySelector('.smoke-card h3').textContent`),'Synthetic diagnostic');
    }
    console.log(JSON.stringify({scenario,sourceRevision,effectiveRoute:base,browser:browserVersion.Browser,artifact,assertions:['actual browser destination opened','real observed terminal focused','active/history filters','desktop and mobile fit']}));
  } else if (scenario === 'stale-submit') {
    markPhase('draft-and-stale-submit');
    const typing = await evaluate(`(()=>{const t=document.querySelector('textarea.smoke-response');t.value='the draft survives';t.focus();t.setSelectionRange(4,10);return true})()`);
    assert.equal(typing, true);
    await evaluate('loadSmoke()');
    const continuity = await evaluate(`JSON.stringify({
      focused:document.activeElement?.matches('textarea.smoke-response'),
      value:document.querySelector('textarea.smoke-response')?.value,
      start:document.querySelector('textarea.smoke-response')?.selectionStart,
      end:document.querySelector('textarea.smoke-response')?.selectionEnd
    })`);
    assert.deepEqual(JSON.parse(continuity), {focused: true, value: 'the draft survives', start: 4, end: 10}, 'polling must preserve the operator draft and caret');

    await evaluate(`(()=>{const realFetch=window.fetch.bind(window);window.fetch=(input,options={})=>{
      const url=new URL(typeof input==='string'?input:input.url,location.href);
      if(url.pathname.endsWith('/response'))return Promise.resolve(new Response('{}',{status:200,headers:{'Content-Type':'application/json'}}));
      if(url.pathname==='/api/smoke-requests')return Promise.reject(new Error('witness list outage'));
      return realFetch(input,options)
    }})()`);
    await evaluate(`document.querySelector('form.smokeReply').dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}))`);
    const button = await eventually(
      () => evaluate(`JSON.stringify({disabled:document.querySelector('form.smokeReply button[type="submit"]')?.disabled,status:document.querySelector('#smokeStatus').textContent})`),
      value => JSON.parse(value).status.includes('Refresh failed'),
      'accepted response followed by list outage',
    );
    const observed = JSON.parse(button);
    assert.equal(observed.disabled, true, 'an accepted response must not re-enable a stale reply form');
    console.log(JSON.stringify({scenario, sourceRevision, effectiveRoute: base, effectiveMode: 'admission-control with disposable request fixture', apiBehavior: 'scenario-specific browser fetch interception', browser: browserVersion.Browser, artifact, assertions: ['draft/caret survives refresh', 'accepted response remains non-actionable after failed list refresh']}));
  } else if (scenario === 'malformed-list' || scenario === 'unreadable-list') {
    markPhase('malformed-list-refresh');
    await evaluate(`(()=>{const realFetch=window.fetch.bind(window);window.fetch=(input,options={})=>{
      const url=new URL(typeof input==='string'?input:input.url,location.href);
      if(url.pathname==='/api/smoke-requests')return Promise.resolve(new Response(JSON.stringify(${scenario === 'malformed-list' ? "{schema:'gpu-greenroom.smoke-request-list.v1'}" : "{schema:'gpu-greenroom.smoke-request-list.v1',items:[],errors:['request record unreadable']}"}),{status:200,headers:{'Content-Type':'application/json'}}));
      return realFetch(input,options)
    }})()`);
    await evaluate(`document.querySelector('#refreshSmoke').click()`);
    const view = await eventually(
      () => evaluate(`JSON.stringify({status:document.querySelector('#smokeStatus').textContent,empty:!!document.querySelector('#smokeRequestList .empty'),cards:document.querySelectorAll('#smokeRequestList .smoke-card').length})`),
      value => JSON.parse(value).status !== '1 request(s)',
      'incomplete successful list refresh',
    );
    const observed = JSON.parse(view);
    assert.ok(!(observed.status === 'No requests' && observed.cards === 0), 'incomplete list data must not impersonate a healthy empty queue');
    assert.equal(observed.cards, 1, 'incomplete refresh should preserve the last loaded request');
    console.log(JSON.stringify({scenario, sourceRevision, effectiveRoute: base, effectiveMode: 'admission-control with disposable request fixture', apiBehavior: 'scenario-specific browser fetch interception', browser: browserVersion.Browser, artifact, assertions: ['incomplete successful list preserves last loaded request', 'incomplete data is not reported as empty']}));
  } else {
    throw new Error(`unknown browser witness scenario: ${scenario}`);
  }
} catch (error) {
  const failure = {scenario, sourceRevision, effectiveRoute: base, effectiveMode: 'admission-control with disposable request fixture', browser: browserIdentity, failurePhase: phase, lastTrustworthyEvidence: {screenshotArtifact}, error: String(error.stack || error)};
  writeFileSync(join(artifactDirectory, `${scenario}-failure.json`), JSON.stringify(failure));
  console.error(JSON.stringify(failure));
  process.exitCode = 1;
} finally {
  try { socket?.close(); } catch {}
  if (browser?.pid) {
    try { process.kill(-browser.pid, 'SIGTERM'); } catch {}
    if (browser.exitCode === null) await Promise.race([new Promise(resolve => browser.once('exit', resolve)), delay(3000)]);
  }
  if (profile) rmSync(profile, {recursive: true, force: true, maxRetries: 5, retryDelay: 100});
}
