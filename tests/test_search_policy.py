from __future__ import annotations

import pytest

from border_collie_demo.search_policy import SearchPolicy


def test_cli_search_policy_overrides_environment() -> None:
    policy = SearchPolicy.configured(
        cli_name="double-back",
        environ={"BORDER_COLLIE_SEARCH_POLICY": "slow-sweep"},
    )

    assert policy.name == "double-back"


def test_search_policy_uses_environment_when_cli_is_absent() -> None:
    policy = SearchPolicy.configured(
        environ={"BORDER_COLLIE_SEARCH_POLICY": "fast-lock"},
    )

    assert policy.name == "fast-lock"


def test_search_policy_defaults_to_slow_sweep_for_production_safety() -> None:
    assert SearchPolicy.configured(environ={}).name == "slow-sweep"


@pytest.mark.parametrize("name", ["", "baseline", "FAST", "double_back"])
def test_search_policy_rejects_unknown_or_noncanonical_names(name: str) -> None:
    with pytest.raises(ValueError, match="fast-lock, slow-sweep, double-back"):
        SearchPolicy.configured(
            cli_name=name,
            environ={"BORDER_COLLIE_SEARCH_POLICY": "slow-sweep"},
        )
