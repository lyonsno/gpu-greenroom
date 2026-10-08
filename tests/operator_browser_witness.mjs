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

const profile = mkdtempSync(join(tmpdir(), 'greenroom-browser-witness-'));
const browser = spawn(chromePath, [
  '--headless=new', '--no-first-run', '--no-default-browser-check',
  '--disable-gpu',
  '--disable-background-networking', '--remote-allow-origins=*',
  '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank',
], {detached: true, stdio: 'ignore'});
let socket;
let phase = 'chrome-startup';
let browserIdentity = null;
let screenshotArtifact = null;
writeFileSync(join(artifactDirectory, `${scenario}-browser-owner.json`), JSON.stringify({
  pid: browser.pid, profile, executable: chromePath, sourceRevision,
}));
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
  const activePort = join(profile, 'DevToolsActivePort');
  await eventually(() => {
    try { return readFileSync(activePort, 'utf8').trim().split('\n'); }
    catch { return null; }
  }, value => Array.isArray(value) && value.length >= 2, 'Chrome DevTools startup');
  const [port] = readFileSync(activePort, 'utf8').trim().split('\n');
  const browserVersion = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
  browserIdentity = browserVersion.Browser;
  markPhase('target-creation');
  const targetResponse = await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, {method: 'PUT'});
  assert.equal(targetResponse.status, 200, 'Chrome must create an isolated page target');
  const target = await targetResponse.json();
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
  const command = (method, params = {}) => new Promise((resolve, reject) => {
    writeFileSync(join(artifactDirectory, `${scenario}-last-command.json`), JSON.stringify({phase, method}));
    const id = ++nextId;
    pending.set(id, {resolve, reject});
    socket.send(JSON.stringify({id, method, params}));
  });
  const evaluate = async expression => {
    const result = await command('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true});
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
    return result.result.value;
  };

  await command('Page.enable');
  await command('Runtime.enable');
  markPhase('initial-render');
  await command('Page.navigate', {url: `${base}/#token=secret`});
  await eventually(() => evaluate(`Boolean(document.querySelector('#smokeRequestList .smoke-card'))`), Boolean, 'initial Greenroom request render');
  markPhase('initial-screenshot');
  await command('Page.bringToFront');
  const initialFrame = await command('Page.captureScreenshot', {format: 'png', fromSurface: true});
  const artifact = join(artifactDirectory, `${scenario}.png`);
  writeFileSync(artifact, Buffer.from(initialFrame.data, 'base64'));
  screenshotArtifact = artifact;

  if (scenario === 'stale-submit') {
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
  try { process.kill(-browser.pid, 'SIGTERM'); } catch {}
  if (browser.exitCode === null) await Promise.race([new Promise(resolve => browser.once('exit', resolve)), delay(3000)]);
  rmSync(profile, {recursive: true, force: true, maxRetries: 5, retryDelay: 100});
}
