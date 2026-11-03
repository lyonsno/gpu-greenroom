"""Historical views remain explicit snapshots; working views keep polling."""
import os
from pathlib import Path
import threading
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
import pytest
from gpu_queue.operator_server import make_handler
from gpu_queue.queue import GPUQueue


def test_history_is_dated_snapshot_and_running_view_refreshes(tmp_path):
    chromium = os.environ.get('GREENROOM_TEST_CHROMIUM')
    if not chromium:
        pytest.skip('set independent Chromium executable')
    assert 'Google Chrome.app' not in Path(chromium).resolve().parts, 'Use an independent automation browser'
    from playwright.sync_api import sync_playwright
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(GPUQueue(tmp_path/'queue'), 'fixture', local_operator=True))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(str(tmp_path/'profile'), executable_path=chromium,
                headless=True, args=['--use-mock-keychain','--password-store=basic','--disable-gpu'])
            try:
                page = context.new_page()
                page.clock.install()
                reads = []
                page.on('request', lambda req: reads.append(req.url) if '/api/state?' in req.url else None)
                page.goto(f'http://127.0.0.1:{server.server_address[1]}', wait_until='networkidle')
                page.get_by_role('button', name='done', exact=True).click()
                page.wait_for_function("()=>last.view==='done'&&!queueLoading")
                reads.clear()
                page.clock.fast_forward(16000)
                assert not reads, 'Historical archive was polled automatically'
                assert 'snapshot' in page.locator('#queueFreshness').inner_text().lower()
                assert 'stale' in page.locator('#queueFreshness').inner_text().lower()
                assert page.locator('#pause').is_disabled()
                page.clock.set_system_time(datetime.now(timezone.utc))
                page.locator('#refresh').click()
                page.wait_for_function("()=>!queueLoading&&queueAvailable")
                assert not page.locator('#pause').is_disabled()
                page.get_by_role('button', name='running', exact=True).click()
                page.wait_for_function("()=>last.view==='running'&&!queueLoading")
                reads.clear()
                page.clock.fast_forward(5000)
                page.wait_for_function('()=>!queueLoading')
                assert any('view=running' in url for url in reads)
            finally:
                context.close()
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
