import fcntl
import json
from pathlib import Path

import pytest

from gpu_queue import cooperative, dispatch
from gpu_queue.models import JobRequest, JobState, JobStatus
from gpu_queue.queue import GPUQueue


@pytest.mark.parametrize('mode', ['fifo', 'two-class'])
def test_checkpoint_decision_holds_coordination_lock_through_head_read(tmp_path, monkeypatch, mode):
    queue = GPUQueue(tmp_path / 'queue')
    queue.configure_dispatch({'schema': 'gpu-greenroom.dispatch-policy.v1', 'mode': mode,
                              'quick_job_types': {}, 'quick_routes': {}}, owner='fixture')
    request = JobRequest(job_type='command', input_path='', agent_id='fixture', cooperative_checkpoint=True,
                         command_argv=['python', '-c', 'pass'])
    pending = JobRequest(job_type='command', input_path='', command_argv=['python', '-c', 'pass'])
    queue.submit(pending)
    state = JobState(job_id=request.job_id, status=JobStatus.RUNNING, job_type='command',
                     input_path='', output_dir='', params={}, submitted_at=1)
    state_data = json.loads(state.to_json())
    state_data['dispatch'] = {'service_class': 'normal'}
    monkeypatch.setattr(cooperative, '_context', lambda: (queue, tmp_path / 'job', request, state_data))
    monkeypatch.setattr(cooperative, 'read_cancellation', lambda *args: None)
    original = dispatch.read_request
    checked = []
    def guarded(path):
        # An independent fd can acquire the lock only if the reader failed to hold it.
        with queue.coordination_lock_path.open('w') as fd:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                checked.append(str(path))
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
                pytest.fail('pending request read can race cancellation/rename')
        return original(path)
    monkeypatch.setattr(dispatch, 'read_request', guarded)
    assert cooperative.yield_if_requested(save_checkpoint=lambda _: pytest.fail('unrequested save'),
                                         quiesce=lambda: pytest.fail('unrequested release'),
                                         continuation=lambda _: None) is False
    assert str(queue.queue_dir / 'pending' / pending.job_id / 'request.json') in checked
