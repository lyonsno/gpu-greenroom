"""Isolated CPU queue for shared native/web decisions and sidebar conformance.

Run as a module with explicit --output and --port. No live queue or agent mail.
The script writes its effective source/queue/server coordinates before serving.
"""
import argparse
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from gpu_queue import performance_window
from gpu_queue.models import JobRequest
from gpu_queue.operator_attention import start
from gpu_queue.operator_server import make_handler
from gpu_queue.queue import GPUQueue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--native',action='store_true')
    parser.add_argument('--auto-web-defer',action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    report_path = args.output/'witness.json'
    report = {'phase':'setup','source':str(Path(__file__).resolve().parents[1]),
              'queue_dir':str((args.output/'queue').resolve()),'native':args.native,
              'claims':'Isolated CPU control path, not production queue or GPU performance'}
    server = None
    stop = None
    worker = None
    try:
        report['source_revision'] = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        report['source_diff'] = subprocess.check_output(['git','diff'],text=True)
        queue = GPUQueue(args.output/'queue')
        request = JobRequest(job_type='command',input_path='',agent_id='handy-handy-man',
            command_argv=[sys.executable,'-c','import time;time.sleep(.5);print("isolated CPU command completed")'],
            performance_window='Synthetic CPU: shared native/web decision conformance',
            runtime_hint={'expected_seconds':1,'reason':'synthetic CPU fixture'},
            route_identity='synthetic/cpu/attention-witness')
        queue.submit(request)
        server = ThreadingHTTPServer(('127.0.0.1',args.port),
            make_handler(queue,'isolated-attention-test',local_operator=True,job_control=True))
        url = f'http://127.0.0.1:{server.server_port}/'
        report.update(job_id=request.job_id,url=url,snapshot_path=str(queue.queue_dir/'operator-attention.json'))
        if not args.native:
            original = performance_window.runner_command
            def runner(*a):
                command = original(*a)
                return [sys.executable,'-c',
                    'import json,sys,time;from pathlib import Path;from gpu_queue.performance_window import run,decision,answer_from_decision;'
                    'spec=json.loads(Path(sys.argv[1]).read_text());'
                    '\ndef prompt(s):\n while not (d:=decision(Path(s["receipt_path"]).parent,s["attempt_id"])): time.sleep(.05)\n return answer_from_decision(d)\n'
                    'sys.exit(run(spec,prompt=prompt))',command[-1]]
            performance_window.runner_command = runner
        stop = start(queue.queue_dir,url,notices=False)
        worker = threading.Thread(target=queue.run_one,args=({},),daemon=True)
        worker.start()
        if args.auto_web_defer:
            def answer_web():
                from urllib.request import Request,urlopen
                from gpu_queue import dispatch
                job = queue.queue_dir/'running'/request.job_id
                try:
                    while worker.is_alive():
                        paths = list(job.glob('performance-native-*.json'))
                        if paths and json.loads(paths[0].read_text()).get('status')=='created':
                            break
                        time.sleep(.02)
                    perf = json.loads((job/'performance-window.json').read_text())
                    state = json.loads((job/'status.json').read_text())
                    body = {'job_id':request.job_id,'request_digest':dispatch.read_request(job/'request.json')[1],
                            'expected_started_at':state['started_at'],'attempt_id':perf['attempt_id'],
                            'request_id':'automatic-native-conformance','action':'defer','defer_minutes':7}
                    with urlopen(Request(url+'api/performance-decision',data=json.dumps(body).encode(),
                        headers={'Authorization':'Bearer isolated-attention-test','Content-Type':'application/json'})) as response:
                        answer = json.load(response)
                    worker.join()
                    final = queue.queue_dir/'deferred'/request.job_id
                    native = json.loads(next(final.glob('performance-native-*.json')).read_text())
                    receipt = json.loads((final/'performance-window.json').read_text())
                    assert native['status']=='dismissed-after-shared-decision' and native['cancel_result']==0
                    assert receipt['workload_started'] is False and receipt['defer_seconds']==420
                    result={'phase':'complete','decision':answer,'native':native,'performance':receipt}
                except BaseException as error:
                    result={'phase':'failed','error':str(error)}
                performance_window.atomic_write(args.output/'native-conformance.json',result)
                threading.Thread(target=server.shutdown,daemon=True).start()
            threading.Thread(target=answer_web,daemon=True).start()
        report['phase']='serving'
        performance_window.atomic_write(report_path,report)
        print(json.dumps({key:value for key,value in report.items() if key!='source_diff'}),flush=True)
        def shutdown(*_):
            threading.Thread(target=server.shutdown,daemon=True).start()
        signal.signal(signal.SIGTERM,shutdown)
        signal.signal(signal.SIGINT,shutdown)
        server.serve_forever()
    except BaseException as error:
        report.update(phase='failed',error=str(error))
        raise
    finally:
        if stop:
            stop.set()
        if worker and worker.is_alive() and queue._owned_child and queue._owned_child.poll() is None:
            # Exact owned group only; this fixture never shares a job with the live queue.
            import os
            os.killpg(queue._owned_child.pid,signal.SIGTERM)
            worker.join()
        if server:
            server.server_close()
        report['finished_at']=time.time()
        performance_window.atomic_write(report_path,report)


if __name__=='__main__':
    main()
