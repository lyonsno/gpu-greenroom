"""Stage B control-plane benchmark evidence contracts."""

from benchmarks.bench_control_plane import _measure


def test_benchmark_records_exact_uncapped_samples_and_effective_source():
    report = _measure(iterations=3, samples=2)

    assert report["schema"] == "gpu-greenroom.control-plane-benchmark.v1"
    assert report["requested"] == {
        "iterations_per_sample": 3,
        "samples": 2,
    }
    assert report["effective"] == {
        "iterations_per_sample": 3,
        "samples": 2,
        "operations_per_measurement": 6,
    }
    assert report["source"]["repo_root"]
    assert report["source"]["commit"]
    assert isinstance(report["source"]["git_dirty"], bool)
    assert report["source"]["python_executable"]

    for measurement in (
        "known_job_submit",
        "empty_dequeue_check",
        "paused_dequeue_check",
    ):
        result = report[measurement]
        assert result["operation_count"] == 6
        assert len(result["sample_seconds"]) == 2
        assert len(result["sample_microseconds_per_operation"]) == 2
        assert result["microseconds_per_operation"]["min"] >= 0
        assert result["microseconds_per_operation"]["median"] >= 0
        assert result["microseconds_per_operation"]["max"] >= 0
        assert (
            result["microseconds_per_operation"]["min"]
            <= result["microseconds_per_operation"]["median"]
            <= result["microseconds_per_operation"]["max"]
        )
