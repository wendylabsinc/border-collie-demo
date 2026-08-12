from __future__ import annotations

from border_collie_demo import main as app_main


def test_search_policy_cli_argument_overrides_environment(monkeypatch) -> None:
    selected: dict[str, object] = {}
    sentinel_app = object()

    monkeypatch.setenv("BORDER_COLLIE_SEARCH_POLICY", "slow-sweep")

    def build_app_from_env(*, search_policy):
        selected["policy"] = search_policy.name
        return sentinel_app

    def run(app, **options):
        selected["app"] = app
        selected["options"] = options

    monkeypatch.setattr(app_main, "build_app_from_env", build_app_from_env)
    monkeypatch.setattr(app_main.uvicorn, "run", run)

    app_main.main(["--search-policy", "double-back"])

    assert selected["policy"] == "double-back"
    assert selected["app"] is sentinel_app
    assert selected["options"] == {"host": "0.0.0.0", "port": 8110}


def test_search_policy_cli_uses_environment_without_argument(monkeypatch) -> None:
    selected: dict[str, object] = {}

    monkeypatch.setenv("BORDER_COLLIE_SEARCH_POLICY", "fast-lock")
    monkeypatch.setattr(
        app_main,
        "build_app_from_env",
        lambda *, search_policy: selected.setdefault("policy", search_policy.name),
    )
    monkeypatch.setattr(app_main.uvicorn, "run", lambda *_args, **_kwargs: None)

    app_main.main([])

    assert selected["policy"] == "fast-lock"
