from __future__ import annotations

import os

import uvicorn

from .api import create_app


def main() -> None:
    port = int(os.environ.get("BORDER_COLLIE_PORT", "8096"))
    uvicorn.run(create_app(), host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
