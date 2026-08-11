from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

SEARCH_POLICY_ENV = "BORDER_COLLIE_SEARCH_POLICY"
SEARCH_POLICY_NAMES = ("fast-lock", "slow-sweep", "double-back")
DEFAULT_SEARCH_POLICY = "slow-sweep"


@dataclass(frozen=True)
class SearchPolicy:
    """Select one attributable Target Fruit search experiment."""

    name: str

    @classmethod
    def named(cls, name: str) -> SearchPolicy:
        normalized = name.strip()
        if normalized not in SEARCH_POLICY_NAMES:
            choices = ", ".join(SEARCH_POLICY_NAMES)
            raise ValueError(f"search policy must be one of: {choices}")
        return cls(normalized)

    @classmethod
    def configured(
        cls,
        *,
        cli_name: str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> SearchPolicy:
        environment = os.environ if environ is None else environ
        selected = (
            cli_name
            if cli_name is not None
            else environment.get(SEARCH_POLICY_ENV, DEFAULT_SEARCH_POLICY)
        )
        return cls.named(selected)
