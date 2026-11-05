"""Containment and telemetry authority regressions from owner reassessment."""
import json
import sys

import pytest

from gpu_queue.models import JobRequest, JobStatus
from gpu_queue.queue import GPUQueue
from gpu_queue.telemetry import render_telemetry_text, summarize_telemetry


SIGNATURE = 'kIOGPUCommandBufferCallbackErrorTimeout'


@pytest.mark.parametrize('stream', ['stdout', 'stderr'])
@pytest.mark.parametrize('exit_code', [0, 1])
def test_timeout_before_large_cleanup_output_pauses_independently_of_exit(tmp_path, stream, exit_code):
    queue = GPUQueue(tmp_path / 'queue')
    request = JobRequest(
        job_type='command', input_path='', command_cwd=str(tmp_path),
        command_argv=[sys.executable, '-c',
                      f'import sys; sys.{stream}.write({SIGNATURE!r} + "\\n" + "x" * 70000); '
                      f'sys.exit({exit_code})'],
    )
    later = JobRequest(job_type='command', input_path='', command_cwd=str(tmp_path),
                       command_argv=[sys.executable, '-c', 'pass'])
    queue.submit(request)
    queue.submit(later)
    assert queue.run_one({})
    assert queue.is_paused()
    assert queue.get_job(request.job_id).failure_phase == 'metal_gpu_timeout'
    assert queue.get_job(request.job_id).exit_code == exit_code
    assert not queue.run_one({})
    assert queue.get_job(later.job_id).status == JobStatus.PENDING


def test_timeout_signature_crosses_scan_chunk_boundary(tmp_path):
    log = tmp_path / 'stderr.log'
    log.write_bytes(b'x' * (65536 - 10) + SIGNATURE.encode() + b'y' * 70000)
    assert SIGNATURE in (GPUQueue._detect_metal_gpu_timeout(log) or '')
    log.write_text('ordinary output without a GPU error\n')
    assert GPUQueue._detect_metal_gpu_timeout(log) is None


def _record(root, job_id, service_class, submitted, started, finished):
    job = root / 'done' / job_id
    job.mkdir(parents=True)
    status = dict(job_id=job_id, status='done', job_type='command',
                  submitted_at=submitted, started_at=started, finished_at=finished,
                  dispatch={'mode': 'two-class', 'service_class': service_class})
    (job / 'status.json').write_text(json.dumps(status))
    (job / 'request.json').write_text(json.dumps(dict(
        job_id=job_id, job_type='command', route_identity='fixture/' + job_id,
        submitted_at=submitted)))
    (job / 'receipt.json').write_text(json.dumps(status))


def test_unknown_wait_is_not_attributed_to_pause_or_lease(tmp_path):
    _record(tmp_path, 'quick001', 'quick', 100, 200, 201)
    report = summarize_telemetry(tmp_path)
    attribution = report['quick_efficiency']['wait_attribution']
    assert attribution.get('paused_or_external_lease_block', 0) == 0
    assert attribution.get('unattributed', 0) == 1
    assert 'unattributed=1' in render_telemetry_text(report)


def test_fast_normal_wait_is_observation_not_counterfactual_savings(tmp_path):
    _record(tmp_path, 'normal01', 'normal', 100, 200, 201)
    report = summarize_telemetry(tmp_path)
    candidates = report['unlabeled_quick_candidates']
    assert 'total_avoidable_wait_seconds' not in candidates
    assert candidates.get('observed_wait_gt_60s_seconds') == 100
    assert candidates.get('wait_not_accounted_for_by_those_intervals_seconds') == 100
    text = render_telemetry_text(report)
    assert 'avoidable' not in text
    assert 'saving ~' not in text


def test_json_retains_all_candidate_routes(tmp_path):
    for i in range(23):
        for j in range(2):
            job_id = f'route{i:02d}-{j}'
            _record(tmp_path, job_id, 'normal', 100, 200, 201)
            request = tmp_path / 'done' / job_id / 'request.json'
            body = json.loads(request.read_text())
            body['route_identity'] = f'fixture/route{i:02d}'
            request.write_text(json.dumps(body))
    report = summarize_telemetry(tmp_path)
    assert len(report['unlabeled_quick_candidates']['candidate_routes']) == 23
