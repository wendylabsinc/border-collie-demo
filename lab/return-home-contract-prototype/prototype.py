"""Throwaway TUI for the return-to-Home draft state model."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from return_model import (  # noqa: E402
    DraftConfig,
    ReturnState,
    confirm_disarm,
    observe,
    replan,
    start,
)


def render(state: ReturnState, config: DraftConfig) -> None:
    print("\033[2J\033[H", end="")
    print("\033[1mRETURN-TO-HOME CONTRACT PROTOTYPE\033[0m")
    print("\033[2mNo hardware, motion client, or persistence.\033[0m\n")
    print(json.dumps(asdict(state), indent=2, default=str))
    print("\n\033[2mDraft configuration\033[0m")
    print(json.dumps(asdict(config), indent=2))
    print(
        "\n[s] start  [a] align course  [p] progress  [h] reach Home  "
        "[o] obstacle  [r] replan  [t] stall  [x] stale pose  "
        "[d] confirm disarm  [q] quit"
    )


def main() -> None:
    config = DraftConfig()
    state = ReturnState()
    while True:
        render(state, config)
        command = input("> ").strip().lower()
        if command == "q":
            return
        if command == "s":
            state = start(state, config)
        elif command == "a":
            state = observe(state, config, course_error_deg=2.0)
        elif command == "p":
            state = observe(
                state,
                config,
                home_distance_m=max(0.0, state.home_distance_m - 0.08),
                course_error_deg=2.0,
            )
        elif command == "h":
            state = observe(
                state,
                config,
                home_distance_m=0.08,
                home_heading_error_deg=3.0,
            )
        elif command == "o":
            state = observe(state, config, obstacle_blocked=True)
        elif command == "r":
            state = replan(state, config, available=True)
        elif command == "t":
            state = observe(state, config, elapsed_s=3.1)
        elif command == "x":
            state = observe(state, config, pose_age_s=0.75)
        elif command == "d":
            state = confirm_disarm(state)


if __name__ == "__main__":
    main()
