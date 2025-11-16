from hodor.prompts.pr_review_prompt import _build_mr_sections


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
