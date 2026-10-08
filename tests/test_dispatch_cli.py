import json
from pathlib import Path
import subprocess
import sys


ROOT=Path(__file__).resolve().parent.parent


def cli(queue,*args):
    return subprocess.run([sys.executable,'-m','gpu_queue.cli','--queue-dir',str(queue),*args],cwd=ROOT,capture_output=True,text=True)


def test_class_options_are_round_tripped_in_v2_and_bad_admission_has_a_report(tmp_path):
    root=tmp_path/'repo'
    root.mkdir()
    queue=tmp_path/'queue'
    manifest={'schema':'gpu-greenroom.command.v2','repo_root':str(root),'cwd':str(root),'route_identity':'fixture/route',
              'argv':[sys.executable,'-c','pass'],'agent_id':'fixture-agent','service_class':'quick'}
    path=tmp_path/'request.json'
    path.write_text(json.dumps(manifest))
    failed=cli(queue,'submit-command','--manifest',str(path))
    assert failed.returncode==2
    report=json.loads(failed.stderr)
    assert report['failure_phase']=='submission-validation'
    assert Path(report['report_path']).is_file()
    assert not list((queue/'pending').iterdir())
    config={'schema':'gpu-greenroom.dispatch-policy.v1','mode':'two-class','quick_job_types':{},
            'quick_routes':{'fixture/route':'observed CPU fixture timings'}}
    cfg=tmp_path/'policy.json'
    cfg.write_text(json.dumps(config))
    assert cli(queue,'dispatch-policy','--config',str(cfg)).returncode==0
    result=cli(queue,'submit-command','--manifest',str(path))
    assert result.returncode==0,result.stderr
    request=json.loads(Path(json.loads(result.stdout)['request_path']).read_text())
    assert request['service_class']=='quick'
    assert {'structured-command.v2','two-class-dispatch.v1'}.issubset(request['required_worker_capabilities'])
    manifest['schema']='gpu-greenroom.command.v1'
    path.write_text(json.dumps(manifest))
    assert cli(queue,'submit-command','--manifest',str(path)).returncode==2
