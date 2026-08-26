"""Durability of the Demo Run store across a redeploy.

Run records used to live in the app container's ephemeral filesystem, so every
`wendy run` wiped them. They now sit on a mounted persist volume. These tests
pin the behaviour that makes that mount actually useful: a fresh store instance
(the new container) reading a root written by a previous one, and a first boot
against a volume that is mounted but still empty.
"""

from __future__ import annotations

from border_collie_demo.evidence import EvidenceArtifact
from border_collie_demo.run_results import RunResultStore


def _start(store: RunResultStore) -> str:
    run = store.start_run(target_fruit="mango", activation_source="voice")
    return str(run["run_id"])


def test_construction_does_not_touch_a_missing_run_root(tmp_path) -> None:
    """The volume must not be written just by building the app."""
    root = tmp_path / "run-store" / "runs"

    store = RunResultStore(root)

    # Nothing is created at construction time, so importing/creating the app
    # never needs write access to the mount.
    assert not root.exists()
    # An empty or absent volume reads as "no runs yet" rather than crashing.
    assert store.list_results() == []
    assert store.seal_interrupted_runs() == []


def test_first_run_creates_the_run_tree_on_an_empty_volume(tmp_path) -> None:
    root = tmp_path / "run-store" / "runs"
    store = RunResultStore(root)

    run_id = _start(store)

    assert (root / run_id / "result.json").is_file()
    assert (root / run_id / "snapshots").is_dir()


def test_sealed_runs_survive_a_fresh_store_on_the_same_root(tmp_path) -> None:
    root = tmp_path / "run-store" / "runs"
    first = RunResultStore(root)
    run_id = _start(first)
    first.seal(
        run_id,
        phase="failed",
        outcome="FAILED",
        reason="TARGET_RECOGNITION_FAILURE",
        message="mango was not found in the bounded search sweep",
        final_safety_state="DISARMED_CONFIRMED",
        failed_phase="turn_to_fruit",
        failure_details={"searched_rad": 6.283185307179586},
    )

    # A redeploy replaces the container but keeps the mounted volume.
    second = RunResultStore(root)

    listed = second.list_results()
    assert [result["run_id"] for result in listed] == [run_id]
    assert listed[0]["outcome"] == "FAILED"
    assert listed[0]["failed_phase"] == "turn_to_fruit"
    # failure_details is the field the operator debugs from, so it has to
    # round-trip through the volume intact.
    assert listed[0]["failure_details"] == {"searched_rad": 6.283185307179586}
    assert second.get(run_id)["reason"] == "TARGET_RECOGNITION_FAILURE"


def test_evidence_artifacts_survive_inside_the_run_directory(tmp_path) -> None:
    """Evidence lives under the run dir, so it travels with the same volume."""
    root = tmp_path / "run-store" / "runs"
    first = RunResultStore(root)
    run_id = _start(first)
    first.record_artifacts(
        run_id,
        [
            EvidenceArtifact(
                filename="terminal.jpg",
                content_type="image/jpeg",
                content=b"terminal-frame-bytes",
            )
        ],
    )
    first.seal(
        run_id,
        phase="completed",
        outcome="COMPLETED",
        reason="SUCCESS",
        message="mango reached and Home restored",
        final_safety_state="DISARMED_CONFIRMED",
    )

    second = RunResultStore(root)

    path, content_type = second.artifact_path(run_id, "terminal.jpg")
    assert path.read_bytes() == b"terminal-frame-bytes"
    assert content_type == "image/jpeg"
    # The frame sits beside the record it belongs to rather than in a separate
    # tree, so a persisted run never carries a dangling evidence reference.
    assert path.parent == (root / run_id / "snapshots").resolve()


def test_a_run_interrupted_by_a_redeploy_is_sealed_on_the_next_boot(
    tmp_path,
) -> None:
    """Now that the volume survives, an unsealed run is visible next boot."""
    root = tmp_path / "run-store" / "runs"
    first = RunResultStore(root)
    run_id = _start(first)  # never sealed: the redeploy killed the container

    second = RunResultStore(root)
    sealed = second.seal_interrupted_runs()

    assert [result["run_id"] for result in sealed] == [run_id]
    assert sealed[0]["outcome"] == "FAILED"
    assert sealed[0]["reason"] == "PROCESS_INTERRUPTED"
    assert sealed[0]["final_safety_state"] == "UNKNOWN"
    # Sealing is idempotent, so a later boot does not re-seal or duplicate it.
    assert RunResultStore(root).seal_interrupted_runs() == []


def test_runs_accumulate_across_several_redeploys(tmp_path) -> None:
    """The operator wants more runs, not a store that resets each deploy."""
    root = tmp_path / "run-store" / "runs"
    expected: list[str] = []

    for _ in range(3):
        store = RunResultStore(root)
        run_id = _start(store)
        store.seal(
            run_id,
            phase="completed",
            outcome="COMPLETED",
            reason="SUCCESS",
            message="mango reached and Home restored",
            final_safety_state="DISARMED_CONFIRMED",
        )
        expected.append(run_id)

    assert sorted(
        result["run_id"] for result in RunResultStore(root).list_results()
    ) == sorted(expected)
