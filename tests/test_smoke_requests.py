from copy import deepcopy

import pytest

from gpu_queue.smoke_requests import SmokeRequestConflict, SmokeRequests


@pytest.fixture
def request_payload(tmp_path):
    return {
        "schema": "gpu-greenroom.interactive-smoke.v1",
        "id": "9c0a03f6-b2d8-43f4-b7b7-5e848e733661",
        "kind": "interactive-smoke",
        "source": {"agent_id": "greenroom-floor-manager", "repo_root": str(tmp_path)},
        "title": "Inspect Greenroom",
        "prompt": "Open the page and report whether the controls are legible.",
        "url": "http://127.0.0.1:8766/",
        "availability": "prepared",
        "availability_note": "The monitor is already running.",
    }


def test_smoke_request_submit_is_idempotent_but_identity_cannot_change(tmp_path, request_payload):
    store = SmokeRequests(tmp_path / "smoke-requests")
    first, created = store.submit(request_payload)
    replay, replay_created = store.submit(deepcopy(request_payload))

    assert created is True
    assert first["status"] == "operator-needed"
    assert replay == first
    assert replay_created is False

    changed = {**request_payload, "prompt": "A different request"}
    with pytest.raises(SmokeRequestConflict, match="different request"):
        store.submit(changed)
    assert store.get(request_payload["id"]) == first


def test_smoke_request_response_is_exact_and_bound_to_original_request(tmp_path, request_payload):
    store = SmokeRequests(tmp_path / "smoke-requests")
    submitted, _ = store.submit(request_payload)
    response_text = "Keep the same view.\nThe route label wraps at narrow widths.\n"

    answered = store.respond(request_payload["id"], response_text, responded_by="operator")
    replay = store.respond(request_payload["id"], response_text, responded_by="another-client")

    assert answered["status"] == "responded"
    assert answered["response"]["request_digest"] == submitted["request_digest"]
    assert answered["response"]["text"] == response_text
    assert replay == answered
    with pytest.raises(SmokeRequestConflict, match="different response"):
        store.respond(request_payload["id"], "A conflicting second answer")


def test_smoke_request_rejects_missing_owner_and_credential_bearing_url(tmp_path, request_payload):
    store = SmokeRequests(tmp_path / "smoke-requests")
    no_owner = deepcopy(request_payload)
    no_owner["source"].pop("agent_id")
    with pytest.raises(ValueError, match="source agent_id"):
        store.submit(no_owner)

    secret_url = {**request_payload, "url": "https://user:secret@example.test/smoke"}
    with pytest.raises(ValueError, match="without credentials"):
        store.submit(secret_url)


def test_smoke_request_scan_keeps_valid_records_visible_beside_corruption(tmp_path, request_payload):
    store = SmokeRequests(tmp_path / "smoke-requests")
    store.submit(request_payload)
    broken_id = "9c0a03f6-b2d8-43f4-b7b7-5e848e733662"
    store.path(broken_id).write_text("[]", encoding="utf-8")

    records, errors = store.scan()

    assert [record["request"]["id"] for record in records] == [request_payload["id"]]
    assert len(errors) == 1
    assert broken_id in errors[0]
