import json

import pytest

from scripts.deployment_timing_report import (
    DEFAULT_HISTORICAL,
    DEFAULT_LEDGER,
    LedgerValidationError,
    load_historical,
    load_ledger,
    summarize,
    summarize_historical,
)


def test_checked_in_ledger_and_report_calculations():
    rows = load_ledger()
    summary = summarize(rows)

    assert summary["rows"] == 43
    assert summary["successes"] == 36
    assert summary["non_successes"] == 7
    assert summary["outcomes"] == {
        "configuration_not_applied": 1,
        "deployed_readiness_timeout": 1,
        "failed-before-build": 1,
        "failed-before-device-replacement": 1,
        "stagefile_compile_failure": 3,
        "success": 36,
    }
    assert summary["scope_status"] == {
        "external:success": 1,
        "service-scoped:non-success": 3,
        "service-scoped:success": 29,
        "whole-project:non-success": 4,
        "whole-project:success": 6,
    }
    assert summary["phase_coverage"] == {
        "command_elapsed_observed": 40,
        "service_build_map_present": 40,
        "service_build_any_observed": 33,
        "readiness_observed": 15,
    }
    assert summary["app_only_success_command"] == {
        "count": 22,
        "median_s": 2.1065,
        "min_s": 1.11,
        "max_s": 64.24,
    }
    assert summary["app_only_low_build_time_cohort"]["command"] == {
        "count": 15,
        "median_s": 2.06,
        "min_s": 1.11,
        "max_s": 2.634,
    }
    assert summary["app_only_low_build_time_cohort"]["build"] == {
        "count": 15,
        "median_s": 0.373,
        "min_s": 0.088,
        "max_s": 0.479,
    }
    assert summary["whole_project_success_command"] == {
        "count": 6,
        "median_s": 7.678,
        "min_s": 3.13,
        "max_s": 17.25,
    }
    assert summary["cache_states"] == {
        "cache-migration-candidate-inferred": 1,
        "cold-dependency-rebuild-explicit": 1,
        "source-layer-rebuild-explicit": 1,
        "unreported": 39,
        "warm-cache-reuse-explicit": 1,
    }
    assert summary["cli_groups"] == {
        "PR 1698 development builds": {
            "attempts": 31,
            "successes": 29,
            "non_successes": 2,
            "outcomes": {
                "configuration_not_applied": 1,
                "deployed_readiness_timeout": 1,
                "success": 29,
            },
        },
        "WendyOS development binary": {
            "attempts": 2,
            "successes": 2,
            "non_successes": 0,
            "outcomes": {"success": 2},
        },
        "released 2026.08.13-010134": {
            "attempts": 5,
            "successes": 1,
            "non_successes": 4,
            "outcomes": {
                "failed-before-build": 1,
                "stagefile_compile_failure": 3,
                "success": 1,
            },
        },
        "released 2026.08.15-001455": {
            "attempts": 5,
            "successes": 4,
            "non_successes": 1,
            "outcomes": {
                "failed-before-device-replacement": 1,
                "success": 4,
            },
        },
    }
    assert [item["row"] for item in summary["non_success_attempts"]] == [
        1,
        22,
        24,
        30,
        31,
        37,
        42,
    ]
    assert rows[-1] == {
        "schema_version": 1,
        "date_utc": "2026-08-15",
        "commit": "940c197",
        "branch": "codex/controller-start-pear-autostart",
        "target": "woof.local",
        "scope": "whole-project",
        "command": "scripts/deploy-stage-default --device woof.local -y",
        "stagefiles": ["app", "home-recorder", "media", "voice"],
        "cli_source": "released Wendy CLI 2026.08.15-001455",
        "command_elapsed_s": 8.46,
        "service_build_s": {
            "app": None,
            "home-recorder": 0.741,
            "media": 0.205,
            "voice": 0.206,
        },
        "readiness_s": None,
        "outcome": "success",
        "note": rows[-1]["note"],
    }


def test_every_checked_in_jsonl_line_is_validated():
    line_count = len(DEFAULT_LEDGER.read_text(encoding="utf-8").splitlines())
    assert len(load_ledger()) == line_count


def test_invalid_timing_is_rejected(tmp_path):
    row = json.loads(DEFAULT_LEDGER.read_text(encoding="utf-8").splitlines()[0])
    row["command_elapsed_s"] = -1
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(LedgerValidationError, match="row 1: command_elapsed_s"):
        load_ledger(path)


def test_historical_comparison_keeps_warm_and_cold_evidence_separate():
    historical = load_historical()
    summary = summarize_historical(historical)

    assert summary == {
        "old_warm": {
            "count": 2,
            "median_s": 143.7481,
            "min_s": 140.6178,
            "max_s": 146.8784,
        },
        "repaired_warm": {
            "count": 2,
            "median_s": 8.8972,
            "min_s": 6.4644,
            "max_s": 11.33,
        },
        "same_cache_topology": {"cached_steps": 15, "rebuilt_steps": 2},
        "median_reduction_s": 134.8509,
        "median_reduction_percent": 93.8106,
        "cache_migration_deploy_s": 239.806019,
        "local_media_build_only": {
            "cold_restore_s": 182.72,
            "immediate_noop_s": 0.323,
        },
    }

    assert historical["cache_migration_deployment"]["comparison_eligible"] is False
    assert historical["local_media_build_pair"]["complete_deployment"] is False
    assert all(
        item["reason"] for item in historical["excluded_mixed_evidence"]
    )


def test_invalid_historical_warm_topology_is_rejected(tmp_path):
    historical = json.loads(DEFAULT_HISTORICAL.read_text(encoding="utf-8"))
    historical["equal_warm_deployments"][0]["cached_buildkit_steps"] = 14
    path = tmp_path / "historical.json"
    path.write_text(json.dumps(historical), encoding="utf-8")

    with pytest.raises(LedgerValidationError, match="must report 15 cached steps"):
        load_historical(path)
