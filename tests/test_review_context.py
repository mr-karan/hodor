from hodor.agent import ReviewContext


def test_review_context_tracks_and_marks_files():
    ctx = ReviewContext(owner="foo", repo="bar", pr_number=1)
    ctx.update_files_from_tool(
        {
            "files": [
                {"filename": "src/app.py", "patch": "+print('hi')\n"},
                {"filename": "docs/readme.md", "patch": None},  # ignored
            ]
        }
    )
    assert ctx.is_known_file("src/app.py")
    assert not ctx.is_known_file("docs/readme.md")
    assert ctx.missing_files() == {"src/app.py"}

    ctx.mark_diffed("src/app.py")
    assert ctx.missing_files() == set()
    assert "src/app.py" in ctx.diffed_files
