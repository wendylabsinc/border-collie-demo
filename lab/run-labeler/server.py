from __future__ import annotations

import argparse
import json
import mimetypes
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse
from zipfile import BadZipFile, ZipFile

ROOT = Path(__file__).resolve().parent
CLASS_NAME_PATTERN = re.compile(r"[a-z0-9]+(?:[ _-][a-z0-9]+)*")


def load_manifest(archive_path: Path) -> dict[str, object]:
    try:
        with ZipFile(archive_path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
    except (BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid evidence archive: {exc}") from exc
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("evidence archive contains no frames")
    for frame in frames:
        if not isinstance(frame, dict):
            raise TypeError("evidence manifest contains an invalid frame")
        filename = frame.get("filename")
        if not isinstance(filename, str) or not safe_frame_name(filename):
            raise ValueError("evidence manifest contains an unsafe frame filename")
    return manifest


def safe_frame_name(filename: str) -> bool:
    path = PurePosixPath(filename)
    return (
        len(path.parts) == 2
        and path.parts[0] == "frames"
        and path.suffix.casefold() in {".jpg", ".jpeg", ".png"}
        and ".." not in path.parts
    )


def configure_manifest(
    manifest: dict[str, object], class_name: str
) -> dict[str, object]:
    normalized = class_name.casefold().strip()
    if not normalized or CLASS_NAME_PATTERN.fullmatch(normalized) is None:
        raise ValueError(
            "class name must contain only words, spaces, hyphens, or underscores"
        )
    return {**manifest, "labeling_class": normalized}


def make_handler(archive_path: Path, manifest: dict[str, object]):
    frames = manifest["frames"]

    class RunLabelerHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send_file(ROOT / "index.html", "text/html; charset=utf-8")
                return
            if parsed.path == "/api/manifest":
                self._send_json(manifest)
                return
            if parsed.path.startswith("/api/frames/"):
                frame_name = unquote(parsed.path.removeprefix("/api/frames/"))
                matching = next(
                    (
                        frame
                        for frame in frames
                        if isinstance(frame, dict)
                        and frame.get("filename") == frame_name
                    ),
                    None,
                )
                if matching is None or not safe_frame_name(frame_name):
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                with ZipFile(archive_path) as archive:
                    payload = archive.read(frame_name)
                content_type = mimetypes.guess_type(frame_name)[0] or "image/jpeg"
                self._send_bytes(payload, content_type)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send_json(self, value: object) -> None:
            payload = json.dumps(value).encode("utf-8")
            self._send_bytes(payload, "application/json; charset=utf-8")

        def _send_file(self, path: Path, content_type: str) -> None:
            self._send_bytes(path.read_bytes(), content_type)

        def _send_bytes(self, payload: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

    return RunLabelerHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Label a Border Collie fruit archive")
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument(
        "--class-name",
        default="pear",
        help="fruit class shown and exported by the labeler (default: pear)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8120, type=int)
    args = parser.parse_args()
    archive_path = args.archive.expanduser().resolve()
    if not archive_path.is_file():
        parser.error(f"archive does not exist: {archive_path}")
    try:
        manifest = configure_manifest(load_manifest(archive_path), args.class_name)
    except ValueError as exc:
        parser.error(str(exc))
    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(archive_path, manifest),
    )
    print(f"Run labeler ready at http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
