import ast
from pathlib import Path


LEGACY_ROOTS = {"collie_demo", "voice", "nav2"}


def test_production_package_has_no_legacy_imports() -> None:
    source_root = Path(__file__).parents[1] / "src" / "border_collie_demo"
    violations = []

    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".", 1)[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module.split(".", 1)[0]}
            else:
                continue
            blocked = names & LEGACY_ROOTS
            if blocked:
                violations.append((path.name, node.lineno, sorted(blocked)))

    assert violations == []
