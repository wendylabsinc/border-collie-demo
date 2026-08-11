"""Fail-closed release-cohort identity and durable release manifests."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_RELEASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class ReleaseCohort:
    release_id: str
    config_schema: int
    service: str

    def __post_init__(self) -> None:
        if not _RELEASE_ID.fullmatch(self.release_id):
            raise ValueError("release_id must be a safe non-empty identifier")
        if self.config_schema < 1:
            raise ValueError("config_schema must be positive")
        if self.service not in {"app", "media"}:
            raise ValueError("release service must be app or media")

    @classmethod
    def from_env(cls, service: str) -> ReleaseCohort | None:
        release_id = os.environ.get("BORDER_COLLIE_RELEASE_ID", "").strip()
        if not release_id:
            return None
        return cls(
            release_id=release_id,
            config_schema=int(os.environ.get("BORDER_COLLIE_CONFIG_SCHEMA", "1")),
            service=service,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "release_id": self.release_id,
            "config_schema": self.config_schema,
            "service": self.service,
        }


def evaluate_peer_release(
    local: ReleaseCohort | None,
    peer: object,
    *,
    peer_service: str,
) -> dict[str, object]:
    """Return an explicit gate; an enabled local cohort never trusts absence."""
    if local is None:
        return {
            "enabled": False,
            "ready": True,
            "detail": "release cohort checking is not configured",
            "local": None,
            "peer": peer if isinstance(peer, dict) else None,
        }
    expected = {
        "release_id": local.release_id,
        "config_schema": local.config_schema,
        "service": peer_service,
    }
    ready = isinstance(peer, dict) and all(
        peer.get(key) == value for key, value in expected.items()
    )
    return {
        "enabled": True,
        "ready": ready,
        "detail": (
            f"{peer_service} belongs to release cohort {local.release_id}"
            if ready
            else f"{peer_service} release cohort is missing or mismatched"
        ),
        "local": local.to_dict(),
        "peer": dict(peer) if isinstance(peer, dict) else None,
    }


@dataclass(frozen=True)
class ServiceArtifact:
    image: str
    digest: str

    def __post_init__(self) -> None:
        if not self.image.strip():
            raise ValueError("release image reference must be non-empty")
        if not _DIGEST.fullmatch(self.digest):
            raise ValueError("release artifact digest must be sha256:<64 lowercase hex>")

    def to_dict(self) -> dict[str, str]:
        return {"image": self.image, "digest": self.digest}


@dataclass(frozen=True)
class ReleaseManifest:
    release_id: str
    source_revision: str
    config_schema: int
    services: Mapping[str, ServiceArtifact]
    schema_version: int = 1

    def __post_init__(self) -> None:
        ReleaseCohort(self.release_id, self.config_schema, "app")
        if self.schema_version != 1:
            raise ValueError("unsupported release manifest schema")
        if not self.source_revision.strip():
            raise ValueError("source_revision must be non-empty")
        if set(self.services) != {"app", "media"}:
            raise ValueError("release manifest must contain app and media artifacts")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ReleaseManifest:
        raw_services = payload.get("services")
        if not isinstance(raw_services, dict):
            raise TypeError("release manifest services are missing")
        services: dict[str, ServiceArtifact] = {}
        for name, raw in raw_services.items():
            if not isinstance(name, str) or not isinstance(raw, dict):
                raise TypeError("release manifest service entry is invalid")
            services[name] = ServiceArtifact(
                image=str(raw.get("image") or ""),
                digest=str(raw.get("digest") or ""),
            )
        return cls(
            schema_version=int(payload.get("schema_version", 0)),
            release_id=str(payload.get("release_id") or ""),
            source_revision=str(payload.get("source_revision") or ""),
            config_schema=int(payload.get("config_schema", 0)),
            services=services,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "release_id": self.release_id,
            "source_revision": self.source_revision,
            "config_schema": self.config_schema,
            "services": {
                name: artifact.to_dict()
                for name, artifact in sorted(self.services.items())
            },
        }

    def content_digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


class AtomicReleaseStore:
    """Stage, promote, and roll back a verified two-service release manifest."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._fsync_directory()

    def stage(self, manifest: ReleaseManifest) -> None:
        self._write_atomic(self.root / "staged.json", manifest.to_dict())

    def promote(self, service_evidence: Mapping[str, object]) -> ReleaseManifest:
        staged = self._read("staged.json")
        for service in ("app", "media"):
            expected = ReleaseCohort(
                staged.release_id, staged.config_schema, service
            ).to_dict()
            evidence = service_evidence.get(service)
            if not isinstance(evidence, dict) or any(
                evidence.get(key) != value for key, value in expected.items()
            ):
                raise ValueError(f"{service} did not verify the staged release cohort")
        current_path = self.root / "current.json"
        if current_path.exists():
            current = self._read("current.json")
            self._write_atomic(self.root / "previous.json", current.to_dict())
        self._write_atomic(current_path, staged.to_dict())
        (self.root / "staged.json").unlink(missing_ok=True)
        self._fsync_directory()
        return staged

    def rollback(self) -> ReleaseManifest:
        previous = self._read("previous.json")
        current_path = self.root / "current.json"
        if current_path.exists():
            rejected = self._read("current.json")
            self._write_atomic(self.root / "rejected.json", rejected.to_dict())
        self._write_atomic(current_path, previous.to_dict())
        return previous

    def current(self) -> ReleaseManifest | None:
        try:
            return self._read("current.json")
        except FileNotFoundError:
            return None

    def _read(self, name: str) -> ReleaseManifest:
        payload = json.loads((self.root / name).read_text())
        if not isinstance(payload, dict):
            raise TypeError("release manifest must be a JSON object")
        return ReleaseManifest.from_dict(payload)

    def _write_atomic(self, path: Path, payload: Mapping[str, object]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        self._fsync_directory()

    def _fsync_directory(self) -> None:
        descriptor = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
