from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_supported_deploy_command_requests_boot_and_exit_restart_policy(tmp_path) -> None:
    invocation = tmp_path / "invocation.txt"
    fake_wendy = tmp_path / "wendy"
    fake_wendy.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$WENDY_INVOCATION\"\n",
        encoding="utf-8",
    )
    fake_wendy.chmod(0o755)
    env = {
        **os.environ,
        "WENDY_BIN": str(fake_wendy),
        "WENDY_INVOCATION": str(invocation),
    }

    subprocess.run(
        [str(ROOT / "scripts/deploy-stage-default"), "--device", "woof.local"],
        cwd=ROOT,
        env=env,
        check=True,
    )

    assert invocation.read_text(encoding="utf-8").splitlines() == [
        "run",
        "--detach",
        "--restart-unless-stopped",
        "--device",
        "woof.local",
    ]
