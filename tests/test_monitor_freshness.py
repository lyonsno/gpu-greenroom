"""Native loopback snapshots in an isolated browser, with injected transport faults."""
import json
import os
from pathlib import Path
import threading
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer

import pytest

from gpu_queue.operator_server import make_handler
from gpu_queue.queue import GPUQueue
from tests.test_job_controls import job


def test_monitor_does_not_present_old_queue_as_live(tmp_path):
    chromium = os.environ.get('GREENROOM_TEST_CHROMIUM')
    if not chromium:
        pytest.skip('set independent Chromium executable')
    evidence = Path(os.environ.get('GREENROOM_TEST_EVIDENCE', str(tmp_path/'evidence')))
    evidence.mkdir(parents=True, exist_ok=True)
    report = dict(status='failed', phase='fixture', source=str(Path(__file__).resolve().parents[1]),
                  executable=chromium, evidence_kind='native API with synthetic CPU jobs and transport faults')
    server = thread = None
    try:
        from playwright.sync_api import sync_playwright
        queue = GPUQueue(tmp_path/'queue')
        path = job(queue)
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(queue, 'fixture-token', local_operator=True, job_control=True))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(str(tmp_path/'profile'), executable_path=chromium,
                headless=True, args=['--use-mock-keychain', '--password-store=basic', '--disable-gpu'])
            try:
                page = context.new_page()
                page.clock.install()
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{server.server_address[1]}', wait_until='networkidle')
                report['phase'] = 'failed-queue-with-successful-smoke-refresh'
                page.route('**/api/state?*', lambda route: route.abort())
                page.evaluate('load()')
                page.evaluate('loadSmoke()')
                assert 'unavailable' in page.locator('#queueState').inner_text().lower()
                assert page.locator('#pause').is_disabled()
                assert page.locator('.job-actions button').first.is_disabled()
                page.unroute('**/api/state?*')
                page.evaluate('load()')
                assert 'enabled' in page.locator('#queueState').inner_text().lower()
                report['phase'] = 'non-active-filter-refresh'
                page.get_by_role('button', name='pending', exact=True).click()
                page.wait_for_function("()=>last.view==='pending'")
                queue.cancel(path.name)
                page.wait_for_function('()=>last.jobs.length===0', timeout=15000)
                report['phase'] = 'hung-fetch-age'
                # Keep promises pending without changing native snapshot bytes or invoking jobs.
                page.evaluate("""()=>{window.realFetch=window.fetch;window.fetch=(...args)=>
                    String(args[0]).startsWith('/api/')?new Promise(()=>{}):window.realFetch(...args)}""")
                page.clock.fast_forward(16000)
                assert 'stale' in page.locator('#queueFreshness').inner_text().lower()
                assert 'stale' in page.locator('#smokeFreshness').inner_text().lower()
                assert page.locator('#pause').is_disabled()
                for width, height in [(1440,1000),(390,844)]:
                    page.set_viewport_size({'width':width,'height':height})
                    page.screenshot(path=str(evidence/f'stale-{width}.png'), full_page=True)
                    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                report['phase'] = 'recovery'
                page.clock.set_system_time(datetime.now(timezone.utc))
                page.evaluate('window.fetch=window.realFetch')
                page.evaluate('load()')
                page.evaluate('loadSmoke()')
                assert 'stale' not in page.locator('#queueFreshness').inner_text().lower()
                assert not page.locator('#pause').is_disabled()
                report['phase'] = 'old-response-cannot-replace-new-read'
                page.evaluate("""()=>{window.oldSnapshot=last;window.fetch=(...args)=>{
                    if(String(args[0]).startsWith('/api/state')){window.fetch=window.realFetch;
                        return new Promise(resolve=>window.releaseOld=()=>resolve(new Response(JSON.stringify({...window.oldSnapshot,paused:true}))));}
                    return window.realFetch(...args)};window.oldLoad=load()}""")
                page.evaluate('load()')
                page.evaluate('async()=>{window.releaseOld();await window.oldLoad}')
                assert 'enabled' in page.locator('#queueState').inner_text().lower()
                report['phase'] = 'old-source-observation'
                page.route('**/api/state?*', lambda route: route.fulfill(json={
                    **page.evaluate('last'), 'observed_at': 1}))
                page.evaluate('load()')
                assert 'stale' in page.locator('#queueFreshness').inner_text().lower()
                page.unroute('**/api/state?*')
                page.evaluate('load()')
                report['phase'] = 'missing-observation'
                page.route('**/api/state?*', lambda route: route.fulfill(json={
                    **page.evaluate('last'), 'observed_at': None}))
                page.evaluate('load()')
                assert 'unavailable' in page.locator('#queueState').inner_text().lower()
                assert not errors, errors
                report.update(status='passed', phase='complete', errors=errors)
            finally:
                context.close()
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        if thread is not None:
            server.shutdown()
            thread.join()
        if server is not None:
            server.server_close()
        (evidence/'report.json').write_text(json.dumps(report, indent=2))
