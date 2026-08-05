from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urljoin
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class EvidenceArtifact:
    filename: str
    content_type: str
    content: bytes

    def __post_init__(self) -> None:
        if not self.filename or "/" in self.filename or "\\" in self.filename:
            raise ValueError("evidence filename must be a plain filename")
        if not self.content_type:
            raise ValueError("evidence content type is required")


class TerminalEvidenceClient:
    """Fetch the media sidecar's bounded terminal evidence artifacts."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8111/",
        *,
        timeout_s: float = 3.0,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout_s = timeout_s

    @classmethod
    def from_env(cls) -> TerminalEvidenceClient:
        return cls(
            os.environ.get(
                "BORDER_COLLIE_EVIDENCE_BASE_URL",
                "http://127.0.0.1:8111/",
            ).strip(),
            timeout_s=float(
                os.environ.get("BORDER_COLLIE_EVIDENCE_TIMEOUT_S", "3.0")
            ),
        )

    def capture(self) -> list[EvidenceArtifact]:
        return [
            EvidenceArtifact(
                filename="evidence.zip",
                content_type="application/zip",
                content=self._fetch("api/evidence/clip.zip"),
            ),
            EvidenceArtifact(
                filename="terminal.jpg",
                content_type="image/jpeg",
                content=self._fetch("api/camera/frame.jpg"),
            ),
        ]

    def _fetch(self, path: str) -> bytes:
        request = Request(
            urljoin(self.base_url, path),
            headers={"Accept": "application/octet-stream"},
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            content = response.read(20 * 1024 * 1024 + 1)
        if len(content) > 20 * 1024 * 1024:
            raise ValueError("terminal evidence exceeds the 20 MiB limit")
        return content
