"""Run the RobotKit command website."""

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "robotkit.perception.website.app:app",
        host=os.getenv("WEB_HOST", "0.0.0.0"),
        port=int(os.getenv("WEB_PORT", "8090")),
    )


if __name__ == "__main__":
    main()
