from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent


def load_review(
    manifest_path: Path,
    ranking_path: Path,
    class_name: str,
    limit: int,
) -> tuple[dict[str, object], dict[str, Path]]:
    manifest = json.loads(manifest_path.read_text())
    ranking = json.loads(ranking_path.read_text())
    normalized_class = class_name.casefold().strip()
    records = {str(record["image_id"]): record for record in manifest["records"]}
    images: dict[str, Path] = {}
    candidates = []
    for candidate in ranking["candidates"]:
        if str(candidate["class_name"]).casefold() != normalized_class:
            continue
        image_id = str(candidate["image_id"])
        record = records[image_id]
        image_path = (manifest_path.parent / str(record["local_image"])).resolve()
        if not image_path.is_file():
            raise ValueError(f"candidate image is missing: {image_id}")
        images[image_id] = image_path
        candidates.append(
            {
                **candidate,
                "image_url": f"/api/images/{image_id}",
                "license_url": record["license_url"],
                "author": record["author"],
            }
        )
        if len(candidates) >= limit:
            break
    if not candidates:
        raise ValueError(f"no candidates found for class: {class_name}")
    return (
        {
            "schema_version": 1,
            "review_class": normalized_class,
            "source_manifest_sha256": ranking["source_manifest_sha256"],
            "candidate_count": len(candidates),
            "candidates": candidates,
        },
        images,
    )


def make_handler(review: dict[str, object], images: dict[str, Path]):
    class WholeFruitReviewHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send(ROOT / "index.html", "text/html; charset=utf-8")
                return
            if parsed.path == "/api/review":
                self._send_bytes(
                    json.dumps(review).encode(), "application/json; charset=utf-8"
                )
                return
            if parsed.path.startswith("/api/images/"):
                image_id = unquote(parsed.path.removeprefix("/api/images/"))
                image_path = images.get(image_id)
                if image_path is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self._send(
                    image_path,
                    mimetypes.guess_type(image_path.name)[0] or "image/jpeg",
                )
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def _send(self, path: Path, content_type: str) -> None:
            self._send_bytes(path.read_bytes(), content_type)

        def _send_bytes(self, payload: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    return WholeFruitReviewHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Review whole-fruit candidates")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ranking", required=True, type=Path)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--limit", default=200, type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8122, type=int)
    args = parser.parse_args()
    try:
        review, images = load_review(
            args.manifest.resolve(),
            args.ranking.resolve(),
            args.class_name,
            args.limit,
        )
    except ValueError as exc:
        parser.error(str(exc))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(review, images))
    print(f"Whole-fruit review ready at http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
