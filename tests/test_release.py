import pytest

from border_collie_demo.release import (
    AtomicReleaseStore,
    ReleaseCohort,
    ReleaseManifest,
    ServiceArtifact,
    evaluate_peer_release,
)


def artifact(name: str, digit: str) -> ServiceArtifact:
    return ServiceArtifact(f"registry/{name}@sha256:{digit * 64}", f"sha256:{digit * 64}")


def manifest(release_id: str, digit: str) -> ReleaseManifest:
    return ReleaseManifest(
        release_id=release_id,
        source_revision=f"revision-{release_id}",
        config_schema=1,
        services={"app": artifact("app", digit), "media": artifact("media", digit)},
    )


def evidence(release_id: str) -> dict[str, dict[str, object]]:
    return {
        name: ReleaseCohort(release_id, 1, name).to_dict()
        for name in ("app", "media")
    }


def test_release_cohort_rejects_missing_or_mixed_peer() -> None:
    local = ReleaseCohort("release-7", 3, "app")

    assert evaluate_peer_release(local, None, peer_service="media")["ready"] is False
    assert (
        evaluate_peer_release(
            local,
            {"release_id": "release-6", "config_schema": 3, "service": "media"},
            peer_service="media",
        )["ready"]
        is False
    )
    assert (
        evaluate_peer_release(
            local,
            {"release_id": "release-7", "config_schema": 3, "service": "media"},
            peer_service="media",
        )["ready"]
        is True
    )


def test_manifest_requires_exactly_two_digest_pinned_services() -> None:
    with pytest.raises(ValueError, match="app and media"):
        ReleaseManifest(
            release_id="release-1",
            source_revision="abc123",
            config_schema=1,
            services={"app": artifact("app", "a")},
        )
    with pytest.raises(ValueError, match="digest"):
        ServiceArtifact("registry/app:latest", "latest")


def test_atomic_release_store_promotes_only_verified_pair_and_rolls_back(tmp_path) -> None:
    store = AtomicReleaseStore(tmp_path)
    first = manifest("release-1", "a")
    second = manifest("release-2", "b")

    store.stage(first)
    assert store.promote(evidence("release-1")) == first
    store.stage(second)
    with pytest.raises(ValueError, match="media"):
        store.promote(
            {
                "app": evidence("release-2")["app"],
                "media": evidence("release-1")["media"],
            }
        )
    assert store.current() == first

    assert store.promote(evidence("release-2")) == second
    assert store.current() == second
    assert store.rollback() == first
    assert store.current() == first
