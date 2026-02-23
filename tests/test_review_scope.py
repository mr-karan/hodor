from pathlib import Path

import pytest

from hodor.review_scope import ReviewScope, filter_files, load_review_scope_config, resolve_review_scope


def test_filter_files_applies_include_then_exclude() -> None:
    paths = ["main.go", "ui/app.vue", "vendor/lib.go", "i18n/en.json", "pkg/generated/generated.go"]

    filtered = filter_files(
        paths,
        include=("*.go", "*.vue"),
        exclude=("vendor/**", "pkg/generated/**"),
    )

    assert filtered == ["main.go", "ui/app.vue"]


def test_load_review_scope_config_reads_hodor_toml(tmp_path: Path) -> None:
    config_dir = tmp_path / ".hodor"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(
        """
include = ["*.py", "*.sql"]
exclude = ["vendor/**", "*.lock"]
""".strip(),
        encoding="utf-8",
    )

    scope = load_review_scope_config(tmp_path)

    assert scope == ReviewScope(include=("*.py", "*.sql"), exclude=("vendor/**", "*.lock"))


def test_resolve_review_scope_cli_overrides_config(tmp_path: Path) -> None:
    config_dir = tmp_path / ".hodor"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(
        """
include = ["*.go"]
exclude = ["i18n/**"]
""".strip(),
        encoding="utf-8",
    )

    scope = resolve_review_scope(
        workspace=tmp_path,
        cli_include=("*.py",),
        cli_exclude=("vendor/**",),
    )

    assert scope.include == ("*.py",)
    assert scope.exclude == ("vendor/**",)


def test_load_review_scope_config_rejects_invalid_format(tmp_path: Path) -> None:
    config_dir = tmp_path / ".hodor"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text("exclude = \"*.lock\"", encoding="utf-8")

    with pytest.raises(RuntimeError, match="expected array of strings"):
        load_review_scope_config(tmp_path)
