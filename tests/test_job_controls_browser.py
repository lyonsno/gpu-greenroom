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
