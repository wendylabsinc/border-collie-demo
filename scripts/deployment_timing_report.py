"""Validate and summarize the checked-in Border Collie deployment ledger.

This tool is deliberately offline. It reads JSONL evidence, validates every
row, and emits the calculations used by ``benchmarks/deployment-timing-report.md``.

Usage:
    python3 scripts/deployment_timing_report.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = REPOSITORY_ROOT / "benchmarks/results/deployment-timings.jsonl"
DEFAULT_HISTORICAL = (
    REPOSITORY_ROOT / "benchmarks/results/historical-deployment-comparison.json"
)

REQUIRED_KEYS = {
    "schema_version",
    "date_utc",
    "commit",
    "branch",
    "target",
    "scope",
    "command",
    "stagefiles",
    "cli_source",
    "command_elapsed_s",
    "service_build_s",
    "readiness_s",
    "outcome",
    "note",
}

OUTCOMES = {
    "success",
    "configuration_not_applied",
    "deployed_readiness_timeout",
    "failed-before-build",
    "failed-before-device-replacement",
    "stagefile_compile_failure",
}

COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")


class LedgerValidationError(ValueError):
    """A row in the deployment ledger violates its checked-in contract."""


def _error(row_number: int, message: str) -> LedgerValidationError:
    return LedgerValidationError(f"row {row_number}: {message}")


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_optional_seconds(
    value: object, *, row_number: int, field: str
) -> None:
    if value is None:
        return
    if not _is_number(value) or not math.isfinite(value) or value < 0:
        raise _error(row_number, f"{field} must be finite, non-negative, or null")


def validate_row(row: object, row_number: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise _error(row_number, "must be a JSON object")
    missing = REQUIRED_KEYS - row.keys()
    extra = row.keys() - REQUIRED_KEYS
    if missing or extra:
        raise _error(
            row_number,
            f"schema keys differ; missing={sorted(missing)}, extra={sorted(extra)}",
        )
    if row["schema_version"] != 1:
        raise _error(row_number, "schema_version must be 1")
    try:
        parsed_date = date.fromisoformat(row["date_utc"])
    except (TypeError, ValueError) as exc:
        raise _error(row_number, "date_utc must be an ISO calendar date") from exc
    if parsed_date.isoformat() != row["date_utc"]:
        raise _error(row_number, "date_utc must use YYYY-MM-DD")
    if not isinstance(row["commit"], str) or not COMMIT_RE.fullmatch(row["commit"]):
        raise _error(row_number, "commit must be a 7-40 character lowercase SHA")
    for field in ("branch", "target", "scope", "cli_source", "note"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise _error(row_number, f"{field} must be a non-empty string")
    native_command = isinstance(row["command"], str) and (
        "wendy" in row["command"].lower()
        or row["command"].startswith("scripts/deploy-stage-default ")
    )
    if not native_command:
        raise _error(row_number, "command must identify the native Wendy command")
    if "dlo" in row["command"].lower():
        raise _error(row_number, "command must not invoke DLO")
    if not isinstance(row["stagefiles"], list) or not all(
        isinstance(item, str) and item for item in row["stagefiles"]
    ):
        raise _error(row_number, "stagefiles must be a list of non-empty strings")
    if len(row["stagefiles"]) != len(set(row["stagefiles"])):
        raise _error(row_number, "stagefiles must not contain duplicates")
    if row["outcome"] not in OUTCOMES:
        raise _error(row_number, f"unknown outcome {row['outcome']!r}")

    _validate_optional_seconds(
        row["command_elapsed_s"], row_number=row_number, field="command_elapsed_s"
    )
    _validate_optional_seconds(
        row["readiness_s"], row_number=row_number, field="readiness_s"
    )
    service_build = row["service_build_s"]
    if service_build is not None:
        if not isinstance(service_build, dict) or not service_build:
            raise _error(row_number, "service_build_s must be a non-empty map or null")
        for service, seconds in service_build.items():
            if not isinstance(service, str) or not service:
                raise _error(row_number, "service_build_s keys must be non-empty")
            _validate_optional_seconds(
                seconds,
                row_number=row_number,
                field=f"service_build_s.{service}",
            )

    before_build = {"stagefile_compile_failure", "failed-before-build"}
    if row["outcome"] in before_build:
        observed_builds = [
            value for value in (service_build or {}).values() if value is not None
        ]
        if observed_builds:
            raise _error(row_number, "pre-build failure cannot report a build duration")
        if row["readiness_s"] is not None:
            raise _error(row_number, "pre-build failure cannot report readiness")
    return row


def load_ledger(path: Path = DEFAULT_LEDGER) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            raise _error(row_number, "blank JSONL lines are not allowed")
        try:
            raw_row = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise _error(row_number, f"invalid JSON: {exc.msg}") from exc
        rows.append(validate_row(raw_row, row_number))
    if not rows:
        raise LedgerValidationError("ledger must contain at least one row")
    return rows


def _validate_historical_seconds(value: object, field: str) -> float:
    if not _is_number(value) or not math.isfinite(value) or value <= 0:
        raise LedgerValidationError(f"historical {field} must be finite and positive")
    return float(value)


def _validate_sha256(value: object, field: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise LedgerValidationError(f"historical {field} must be a lowercase SHA-256")


def load_historical(path: Path = DEFAULT_HISTORICAL) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LedgerValidationError(f"historical evidence is invalid JSON: {exc.msg}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise LedgerValidationError("historical schema_version must be 1")

    warm = data.get("equal_warm_deployments")
    if not isinstance(warm, list) or len(warm) != 4:
        raise LedgerValidationError("historical equal_warm_deployments must have 4 rows")
    allowed_cohorts = {"old_warm_path", "repaired_warm_path"}
    for index, row in enumerate(warm, start=1):
        if not isinstance(row, dict) or row.get("cohort") not in allowed_cohorts:
            raise LedgerValidationError(f"historical warm row {index} has unknown cohort")
        if row.get("services") != ["app", "media"] or row.get("status") != "success":
            raise LedgerValidationError(
                f"historical warm row {index} must be a successful app+media deploy"
            )
        _validate_historical_seconds(
            row.get("command_elapsed_s"), f"warm row {index} command_elapsed_s"
        )
        if row.get("cached_buildkit_steps") != 15:
            raise LedgerValidationError(
                f"historical warm row {index} must report 15 cached steps"
            )
        if row.get("rebuilt_buildkit_steps") != 2:
            raise LedgerValidationError(
                f"historical warm row {index} must report 2 rebuilt steps"
            )
        _validate_sha256(row.get("source_sha256"), f"warm row {index} source_sha256")
    if Counter(row["cohort"] for row in warm) != Counter(
        {"old_warm_path": 2, "repaired_warm_path": 2}
    ):
        raise LedgerValidationError("historical warm comparison requires two rows per cohort")

    migration = data.get("cache_migration_deployment")
    if not isinstance(migration, dict) or migration.get("comparison_eligible") is not False:
        raise LedgerValidationError("historical cache migration must be comparison-ineligible")
    _validate_historical_seconds(
        migration.get("command_elapsed_s"), "cache migration command_elapsed_s"
    )
    _validate_sha256(migration.get("source_sha256"), "cache migration source_sha256")

    local = data.get("local_media_build_pair")
    if not isinstance(local, dict) or local.get("complete_deployment") is not False:
        raise LedgerValidationError("historical local media pair must be build-only")
    for name in ("cold_restore", "immediate_noop"):
        observation = local.get(name)
        if not isinstance(observation, dict):
            raise LedgerValidationError(f"historical local media {name} is missing")
        _validate_historical_seconds(observation.get("duration_s"), f"local {name}")
    _validate_sha256(local.get("source_sha256"), "local media source_sha256")
    return data


def summarize_historical(data: dict[str, Any]) -> dict[str, Any]:
    warm = data["equal_warm_deployments"]
    old_values = [
        float(row["command_elapsed_s"])
        for row in warm
        if row["cohort"] == "old_warm_path"
    ]
    repaired_values = [
        float(row["command_elapsed_s"])
        for row in warm
        if row["cohort"] == "repaired_warm_path"
    ]
    old_median = statistics.median(old_values)
    repaired_median = statistics.median(repaired_values)
    reduction = old_median - repaired_median
    return {
        "old_warm": _series(old_values),
        "repaired_warm": _series(repaired_values),
        "same_cache_topology": {"cached_steps": 15, "rebuilt_steps": 2},
        "median_reduction_s": round(reduction, 4),
        "median_reduction_percent": round(100 * reduction / old_median, 4),
        "cache_migration_deploy_s": float(
            data["cache_migration_deployment"]["command_elapsed_s"]
        ),
        "local_media_build_only": {
            "cold_restore_s": float(
                data["local_media_build_pair"]["cold_restore"]["duration_s"]
            ),
            "immediate_noop_s": float(
                data["local_media_build_pair"]["immediate_noop"]["duration_s"]
            ),
        },
    }


def _scope_class(row: dict[str, Any]) -> str:
    scope = row["scope"]
    if scope.startswith("whole-project"):
        return "whole-project"
    if scope.startswith("service:"):
        return "service-scoped"
    return "external"


def _cli_group(row: dict[str, Any]) -> str:
    source = row["cli_source"]
    if "PR 1698" in source:
        return "PR 1698 development builds"
    if "2026.08.13-010134" in source:
        return "released 2026.08.13-010134"
    if "2026.08.15-001455" in source:
        return "released 2026.08.15-001455"
    return source


def _series(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median_s": None, "min_s": None, "max_s": None}
    return {
        "count": len(values),
        "median_s": round(statistics.median(values), 4),
        "min_s": round(min(values), 4),
        "max_s": round(max(values), 4),
    }


def _timings(
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    extractor: Callable[[dict[str, Any]], float | None],
) -> dict[str, float | int | None]:
    values = [
        value
        for row in rows
        if predicate(row)
        if (value := extractor(row)) is not None
    ]
    return _series(values)


def _cache_state(row: dict[str, Any], row_number: int) -> str:
    note = row["note"].lower()
    if "reused its cached image" in note:
        return "warm-cache-reuse-explicit"
    if "stagefile source layers rebuilt" in note:
        return "source-layer-rebuild-explicit"
    if "cold-rebuilt cyclonedds" in note:
        return "cold-dependency-rebuild-explicit"
    if row_number == 1 and row["commit"] == "532fc35f931e62db477a75e4657aab3ef93e3257":
        return "cache-migration-candidate-inferred"
    return "unreported"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status = lambda row: "success" if row["outcome"] == "success" else "non-success"
    scope_status = Counter((_scope_class(row), status(row)) for row in rows)

    app_success = lambda row: (
        row["scope"] == "service:app" and row["outcome"] == "success"
    )

    def app_build(row: dict[str, Any]) -> float | None:
        build = row["service_build_s"] or {}
        return build.get("app")

    low_build_app = lambda row: (
        app_success(row)
        and row["command_elapsed_s"] is not None
        and app_build(row) is not None
        and app_build(row) <= 0.6
    )

    cli_groups: dict[str, dict[str, Any]] = {}
    for group in sorted({_cli_group(row) for row in rows}):
        group_rows = [row for row in rows if _cli_group(row) == group]
        cli_groups[group] = {
            "attempts": len(group_rows),
            "successes": sum(row["outcome"] == "success" for row in group_rows),
            "non_successes": sum(row["outcome"] != "success" for row in group_rows),
            "outcomes": dict(sorted(Counter(row["outcome"] for row in group_rows).items())),
        }

    failures = [
        {
            "row": row_number,
            "outcome": row["outcome"],
            "scope": row["scope"],
            "command_elapsed_s": row["command_elapsed_s"],
            "cli_source": row["cli_source"],
        }
        for row_number, row in enumerate(rows, start=1)
        if row["outcome"] != "success"
    ]
    cache_states = Counter(
        _cache_state(row, row_number)
        for row_number, row in enumerate(rows, start=1)
    )

    return {
        "rows": len(rows),
        "successes": sum(row["outcome"] == "success" for row in rows),
        "non_successes": sum(row["outcome"] != "success" for row in rows),
        "outcomes": dict(sorted(Counter(row["outcome"] for row in rows).items())),
        "scope_status": {
            f"{scope}:{row_status}": count
            for (scope, row_status), count in sorted(scope_status.items())
        },
        "phase_coverage": {
            "command_elapsed_observed": sum(
                row["command_elapsed_s"] is not None for row in rows
            ),
            "service_build_map_present": sum(
                row["service_build_s"] is not None for row in rows
            ),
            "service_build_any_observed": sum(
                any(value is not None for value in (row["service_build_s"] or {}).values())
                for row in rows
            ),
            "readiness_observed": sum(row["readiness_s"] is not None for row in rows),
        },
        "app_only_success_command": _timings(
            rows, app_success, lambda row: row["command_elapsed_s"]
        ),
        "app_only_low_build_time_cohort": {
            "criteria": (
                "outcome=success, scope=service:app, command observed, "
                "reported app build <=0.6s; this is not proof of cache warmth"
            ),
            "command": _timings(rows, low_build_app, lambda row: row["command_elapsed_s"]),
            "build": _timings(rows, low_build_app, app_build),
        },
        "whole_project_success_command": _timings(
            rows,
            lambda row: _scope_class(row) == "whole-project"
            and row["outcome"] == "success",
            lambda row: row["command_elapsed_s"],
        ),
        "cli_groups": cli_groups,
        "cache_states": dict(sorted(cache_states.items())),
        "non_success_attempts": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", nargs="?", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--historical", type=Path, default=DEFAULT_HISTORICAL)
    args = parser.parse_args()
    rows = load_ledger(args.ledger)
    summary = summarize(rows)
    summary["historical_comparison"] = summarize_historical(
        load_historical(args.historical)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
