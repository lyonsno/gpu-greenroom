"""Real loopback API and disposable-browser exercise; never touches the live queue."""
import json
import os
from pathlib import Path
import threading
from http.server import ThreadingHTTPServer

import pytest

from gpu_queue.operator_server import make_handler
from gpu_queue.queue import GPUQueue
from tests.test_job_controls import job, cancelled


def test_pause_estimate_visible_and_honest_on_desktop_and_mobile(tmp_path):
    import hashlib
    import time
    from gpu_queue.models import JobState, JobStatus
    from gpu_queue.dispatch import atomic_write
    chromium = os.environ.get('GREENROOM_TEST_CHROMIUM')
    if not chromium:
        pytest.skip('set independent Chromium executable')
    from playwright.sync_api import sync_playwright
    q = GPUQueue(tmp_path/'queue')
    path = job(q, cooperative_checkpoint=True)
    state = JobState.from_json((path/'status.json').read_text())
    state.status, state.started_at = JobStatus.RUNNING, time.time()-60
    state.child_pid, state.child_start_identity = 123, 'synthetic-browser-attempt'
    (path/'status.json').write_text(state.to_json())
    path = path.rename(q.queue_dir/'running'/path.name)
    record = dict(schema='gpu-greenroom.pause-estimate.v1', job_id=path.name,
        request_sha256=hashlib.sha256((path/'request.json').read_bytes()).hexdigest(),
        started_at=state.started_at, child_pid=123, child_start_identity=state.child_start_identity,
        reported_at=time.time(), seconds=90, checkpoint='End of current tile')
    atomic_write(path/'pause-estimate.json', record)
    evidence = Path(os.environ.get('GREENROOM_TEST_EVIDENCE', str(tmp_path/'evidence')))
    evidence.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(('127.0.0.1',0), make_handler(q,'fixture-secret',local_operator=True,job_control=True))
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    report = {'status':'failed','phase':'launch','executable':chromium,'queue':str(q.queue_dir),
              'source':str(Path(__file__).resolve().parents[1]),'evidence_kind':'synthetic CPU display fixture'}
    try:
        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(str(tmp_path/'profile'),executable_path=chromium,
                headless=True,args=['--use-mock-keychain','--password-store=basic','--disable-gpu'])
            try:
                page = context.new_page()
                errors = []
                page.on('pageerror',lambda error: errors.append(str(error)))
                report['phase'] = 'render'
                page.goto(f'http://127.0.0.1:{server.server_address[1]}',wait_until='networkidle')
                label = page.locator('.pause-estimate')
                assert 'producer estimate' in label.inner_text()
                assert 'End of current tile' in label.inner_text()
                for width,height in [(1440,1000),(390,844)]:
                    page.set_viewport_size({'width':width,'height':height})
                    page.screenshot(path=str(evidence/f'pause-estimate-{width}.png'),full_page=True)
                    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                report['phase'] = 'degradation'
                atomic_write(path/'pause-estimate.json',{**record,'reported_at':time.time()-10,'seconds':1})
                page.evaluate('load()')
                page.wait_for_function("()=>document.querySelector('.pause-estimate').textContent.includes('overdue')")
                assert 'paused' not in label.inner_text().lower()
                atomic_write(path/'pause-estimate.json',{**record,'child_start_identity':'old'})
                page.evaluate('load()')
                page.wait_for_function("()=>document.querySelector('.pause-estimate').textContent.includes('Unknown')")
                assert not errors
                report.update(status='passed',phase='complete',errors=errors)
            finally:
                context.close()
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        server.shutdown(); server.server_close(); thread.join()
        (evidence/'pause-estimate-browser.json').write_text(json.dumps(report,indent=2))


def test_process_ownership_warning_does_not_claim_lane_unresolved(tmp_path):
    chromium=os.environ.get('GREENROOM_TEST_CHROMIUM')
    if not chromium:
        pytest.skip('set independent Chromium executable')
    from playwright.sync_api import sync_playwright
    from gpu_queue.models import JobState, JobStatus
    q=GPUQueue(tmp_path/'queue')
    path=job(q)
    state=JobState.from_json((path/'status.json').read_text())
    state.status=JobStatus.RUNNING
    state.started_at=1
    state.warnings=['ownership_unknown:process_group_live']
    (path/'status.json').write_text(state.to_json())
    path.rename(q.queue_dir/'running'/path.name)
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(q,'fixture-secret',local_operator=True,job_control=True))
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        with sync_playwright() as p:
            context=p.chromium.launch_persistent_context(str(tmp_path/'profile'),executable_path=chromium,
                headless=True,args=['--use-mock-keychain','--password-store=basic','--disable-gpu'])
            try:
                page=context.new_page()
                page.goto(f'http://127.0.0.1:{server.server_address[1]}',wait_until='networkidle')
                row=page.locator(f'[data-job-id="{path.name}"].job-item')
                assert 'Process ownership unresolved; cancellation unavailable' in row.inner_text()
                assert row.get_by_role('button',name='Cancel',exact=True).count()==0
                row.get_by_role('button',name='Ask lane to implement pause').click()
                page.wait_for_function("()=>last.jobs.some(j=>j.pause_support_request==='saved') && !busy")
                assert 'Request saved locally; delivery pending' in row.inner_text()
                assert 'Request submitted to lane' not in row.inner_text()
                event=next((q.queue_dir/'job-feedback').glob('*.json'))
                record=json.loads(event.read_text())
                assert record['target_diaulos']=='fixture-owner'
                assert record['status']=='saved'
                for status,label in [('queued','Request queued in lane inbox'),('submitted','Request submitted to lane'),('uncertain','Request delivery unconfirmed')]:
                    record['status']=status
                    event.write_text(json.dumps(record))
                    page.reload(wait_until='networkidle')
                    assert label in row.inner_text()
                evidence=Path(os.environ.get('GREENROOM_TEST_EVIDENCE',str(tmp_path/'evidence')))
                evidence.mkdir(parents=True,exist_ok=True)
                page.screenshot(path=str(evidence/'process-ownership.png'))
                assert not page.locator('#error').is_visible()
            finally:
                context.close()
    finally:
        server.shutdown();server.server_close();thread.join()


def test_controls_through_browser(tmp_path):
    chromium=os.environ.get('GREENROOM_TEST_CHROMIUM')
    if not chromium:
        pytest.skip('set independent Chromium executable')
    from playwright.sync_api import sync_playwright
    evidence=Path(os.environ.get('GREENROOM_TEST_EVIDENCE',str(tmp_path/'evidence')))
    evidence.mkdir(parents=True,exist_ok=True)
    q=GPUQueue(tmp_path/'queue')
    support=job(q,1)
    pausable=job(q,2,cooperative_checkpoint=True)
    old=cancelled(q,0)
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(q,'fixture-secret',local_operator=True,job_control=True))
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    report={'phase':'launch','status':'failed','executable':chromium}
    try:
        with sync_playwright() as p:
            context=p.chromium.launch_persistent_context(str(tmp_path/'profile'),executable_path=chromium,
                headless=True,args=['--use-mock-keychain','--password-store=basic','--disable-gpu'])
            try:
                page=context.new_page()
                errors=[]
                page.on('pageerror',lambda error:errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{server.server_address[1]}',wait_until='networkidle')
                report['phase']='pause-support'
                row=page.locator(f'[data-job-id="{support.name}"].job-item')
                row.get_by_role('button',name='Ask lane to implement pause').click()
                page.wait_for_function("()=>last.jobs.some(j=>j.pause_support_request==='saved')")
                page.wait_for_function("()=>!busy")
                assert not page.locator('#error').is_visible(),page.locator('#error').inner_text()
                assert len(list((q.queue_dir/'job-feedback').glob('*.json')))==1
                report['phase']='pause-resume'
                row=page.locator(f'[data-job-id="{pausable.name}"].job-item')
                row.get_by_title('Pause job',exact=True).click()
                page.wait_for_function("()=>last.jobs.some(j=>j.pause_control?.phase==='paused')")
                row.get_by_role('checkbox').uncheck()
                page.wait_for_function("()=>last.jobs.some(j=>j.pause_control?.allow_bypass===false)")
                assert q._next_pending() is None
                row.get_by_title('Resume job',exact=True).click()
                page.wait_for_function("()=>last.jobs.every(j=>!j.pause_control?.phase)")
                report['phase']='restore'
                page.get_by_role('button',name='Restore cancelled',exact=True).click()
                page.get_by_title('Restore cancelled performance job').click()
                for width,height in [(1440,1000),(390,844)]:
                    page.set_viewport_size({'width':width,'height':height})
                    page.screenshot(path=str(evidence/f'requeue-dialog-{width}.png'))
                page.locator('#requeuePlacement').select_option('second')
                page.locator('#requeueConfirm').click()
                page.wait_for_function("()=>!busy && last.view==='recoverable' && last.jobs.length===0")
                assert len(list((q.queue_dir/'pending').iterdir()))==3
                page.get_by_role('button',name='active',exact=True).click()
                page.wait_for_function("()=>last.view==='active' && last.jobs.length===3")
                report['phase']='visual'
                for width,height in [(1440,1000),(390,844)]:
                    page.set_viewport_size({'width':width,'height':height})
                    page.screenshot(path=str(evidence/f'controls-{width}.png'),full_page=True)
                    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
                    assert page.locator('.job-actions button').evaluate_all('(buttons)=>buttons.every(b=>b.scrollHeight<=b.clientHeight && b.scrollWidth<=b.clientWidth)')
                assert not errors,errors
                report.update(status='passed',phase='complete',errors=errors,
                    jobs=page.evaluate('last.jobs.map(j=>({id:j.job_id,position:j.queue_position}))'))
            finally:
                context.close()
    except Exception as error:
        report['error']=str(error)
        raise
    finally:
        server.shutdown();server.server_close();thread.join()
        (evidence/'browser-report.json').write_text(json.dumps(report,indent=2))
