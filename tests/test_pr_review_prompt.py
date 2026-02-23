from hodor.prompts.pr_review_prompt import _build_mr_sections, build_pr_review_prompt


def test_build_mr_sections_handles_string_labels() -> None:
    metadata = {
        "title": "Add string labels support",
        "labels": ["bug", "gitlab"],
    }

    context_section, _, _ = _build_mr_sections(metadata)

    assert "- Labels: bug, gitlab" in context_section


def test_build_mr_sections_prefers_label_details_when_available() -> None:
    metadata = {
        "title": "Prefer detailed labels",
        "labels": ["fallback"],
        "label_details": [
            {"name": "frontend"},
            {"name": "regression"},
        ],
    }

    context_section, _, _ = _build_mr_sections(metadata)

    assert "- Labels: frontend, regression" in context_section


def test_prompt_uses_scoped_file_list_command() -> None:
    prompt = build_pr_review_prompt(
        pr_url="https://github.com/acme/repo/pull/1",
        owner="acme",
        repo="repo",
        pr_number="1",
        platform="github",
        target_branch="main",
        include_patterns=("*.py",),
        exclude_patterns=("vendor/**",),
        scoped_files=["app/main.py", "tests/test_app.py"],
    )

    assert "printf '%s\\n' app/main.py tests/test_app.py" in prompt
    assert "## File Scope Filters" in prompt
    assert "- Include patterns: *.py" in prompt
    assert "- Exclude patterns: vendor/**" in prompt
    assert "- `app/main.py`" in prompt


def test_prompt_handles_empty_scoped_files() -> None:
    prompt = build_pr_review_prompt(
        pr_url="https://github.com/acme/repo/pull/1",
        owner="acme",
        repo="repo",
        pr_number="1",
        platform="github",
        target_branch="main",
        scoped_files=[],
    )

    assert "printf ''" in prompt
    assert "No files matched the configured scope. Return no findings." in prompt
