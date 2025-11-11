import pytest

from hodor.agent import detect_platform, parse_pr_url


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://github.com/foo/bar/pull/42", "github"),
        ("https://gitlab.com/foo/bar/-/merge_requests/3", "gitlab"),
        ("https://gitlab.example.dev/group/repo/-/merge_requests/19", "gitlab"),
    ],
)
def test_detect_platform(url, expected):
    assert detect_platform(url) == expected


def test_parse_pr_url_github():
    owner, repo, number = parse_pr_url("https://github.com/octo/hoDor/pull/123")
    assert owner == "octo"
    assert repo == "hoDor"
    assert number == 123


def test_parse_pr_url_gitlab_subgroups():
    url = "https://gitlab.example.com/org/security/tools/hodor/-/merge_requests/99"
    owner, repo, number = parse_pr_url(url)
    assert owner == "org/security/tools"
    assert repo == "hodor"
    assert number == 99


def test_parse_pr_url_invalid():
    with pytest.raises(ValueError):
        parse_pr_url("https://gitlab.com/foo/bar/issues/1")
