"""
Resilient JSON parser for code review output.

This module implements a 3-tier parsing fallback strategy inspired by Codex:
1. Try to parse the full text as JSON
2. Extract {...} block and parse that
3. Fallback: wrap text in overall_explanation field

This handles cases where the LLM wraps JSON in markdown fences or adds prose.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ReviewLineRange:
    """Line range for a code location."""

    start: int
    end: int

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewLineRange":
        return cls(start=data["start"], end=data["end"])

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end}


@dataclass
class ReviewCodeLocation:
    """Code location with absolute file path and line range."""

    absolute_file_path: Path
    line_range: ReviewLineRange

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewCodeLocation":
        return cls(
            absolute_file_path=Path(data["absolute_file_path"]),
            line_range=ReviewLineRange.from_dict(data["line_range"]),
        )

    def to_dict(self) -> dict:
        return {
            "absolute_file_path": str(self.absolute_file_path),
            "line_range": self.line_range.to_dict(),
        }


@dataclass
class ReviewFinding:
    """A single review finding with location, priority, and confidence."""

    title: str
    body: str
    confidence_score: float
    code_location: ReviewCodeLocation
    priority: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewFinding":
        return cls(
            title=data["title"],
            body=data["body"],
            confidence_score=data["confidence_score"],
            code_location=ReviewCodeLocation.from_dict(data["code_location"]),
            priority=data.get("priority"),
        )

    def to_dict(self) -> dict:
        result = {
            "title": self.title,
            "body": self.body,
            "confidence_score": self.confidence_score,
            "code_location": self.code_location.to_dict(),
        }
        if self.priority is not None:
            result["priority"] = self.priority
        return result


@dataclass
class ReviewOutputEvent:
    """Complete review output with findings and overall verdict."""

    findings: list[ReviewFinding] = field(default_factory=list)
    overall_correctness: str = ""
    overall_explanation: str = ""
    overall_confidence_score: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "ReviewOutputEvent":
        return cls(
            findings=[ReviewFinding.from_dict(f) for f in data.get("findings", [])],
            overall_correctness=data.get("overall_correctness", ""),
            overall_explanation=data.get("overall_explanation", ""),
            overall_confidence_score=data.get("overall_confidence_score", 0.0),
        )

    def to_dict(self) -> dict:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "overall_correctness": self.overall_correctness,
            "overall_explanation": self.overall_explanation,
            "overall_confidence_score": self.overall_confidence_score,
        }


def parse_review_output(text: str) -> ReviewOutputEvent:
    """
    Parse review output with 3-tier fallback strategy.

    1. Try to parse the full text as JSON
    2. If that fails, extract the first {...} block and parse that
    3. If that fails, wrap the text in overall_explanation

    Args:
        text: Raw output from the review agent

    Returns:
        Parsed ReviewOutputEvent (never fails, always returns something)
    """
    # Tier 1: Try to parse full text as JSON
    try:
        data = json.loads(text)
        return ReviewOutputEvent.from_dict(data)
    except json.JSONDecodeError:
        pass

    # Tier 2: Extract {...} block and parse that
    # Find the outermost JSON object
    first_brace = text.find("{")
    last_brace = text.rfind("}")

    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        json_slice = text[first_brace : last_brace + 1]
        try:
            data = json.loads(json_slice)
            return ReviewOutputEvent.from_dict(data)
        except json.JSONDecodeError:
            pass

    # Tier 3: Fallback - wrap text in overall_explanation
    return ReviewOutputEvent(
        findings=[],
        overall_correctness="",
        overall_explanation=text,
        overall_confidence_score=0.0,
    )


def format_review_markdown(review: ReviewOutputEvent) -> str:
    """
    Convert structured review JSON to human-readable markdown.

    This is used when posting reviews to PRs - we never post raw JSON.

    Args:
        review: Parsed review output

    Returns:
        Formatted markdown string
    """
    lines = []

    # Add overall explanation if present
    if review.overall_explanation.strip():
        lines.append(review.overall_explanation.strip())
        lines.append("")

    # Group findings by priority
    if review.findings:
        p0_findings = [f for f in review.findings if f.priority == 0]
        p1_findings = [f for f in review.findings if f.priority == 1]
        p2_findings = [f for f in review.findings if f.priority == 2]
        p3_findings = [f for f in review.findings if f.priority == 3]
        untagged = [f for f in review.findings if f.priority is None]

        lines.append("### Issues Found")
        lines.append("")

        # Critical findings
        if p0_findings or p1_findings:
            lines.append("**Critical (P0/P1)**")
            for finding in p0_findings + p1_findings:
                lines.append(_format_finding(finding))
            lines.append("")

        # Important findings
        if p2_findings:
            lines.append("**Important (P2)**")
            for finding in p2_findings:
                lines.append(_format_finding(finding))
            lines.append("")

        # Minor findings
        if p3_findings:
            lines.append("**Minor (P3)**")
            for finding in p3_findings:
                lines.append(_format_finding(finding))
            lines.append("")

        # Untagged findings
        if untagged:
            lines.append("**Other Issues**")
            for finding in untagged:
                lines.append(_format_finding(finding))
            lines.append("")

        # Summary
        lines.append("### Summary")
        total_critical = len(p0_findings) + len(p1_findings)
        total_important = len(p2_findings)
        total_minor = len(p3_findings)
        lines.append(
            f"Total issues: {total_critical} critical, {total_important} important, {total_minor} minor."
        )
        lines.append("")

    # Overall verdict
    if review.overall_correctness:
        lines.append("### Overall Verdict")
        status = (
            "Patch is correct"
            if review.overall_correctness == "patch is correct"
            else "Patch has blocking issues"
        )
        lines.append(f"**Status**: {status}")
        if review.overall_explanation.strip():
            lines.append("")
            lines.append(f"**Explanation**: {review.overall_explanation.strip()}")

    return "\n".join(lines)


def _format_finding(finding: ReviewFinding) -> str:
    """Format a single finding as markdown."""
    location = finding.code_location
    file_path = location.absolute_file_path
    line_range = location.line_range

    # Format location as "path:start-end" or "path:line" if single line
    if line_range.start == line_range.end:
        loc_str = f"{file_path}:{line_range.start}"
    else:
        loc_str = f"{file_path}:{line_range.start}-{line_range.end}"

    # Extract priority tag from title if present, or use numeric priority
    title = finding.title
    if not title.startswith("[P"):
        if finding.priority is not None:
            title = f"[P{finding.priority}] {title}"

    lines = [f"- **{title}** (`{loc_str}`)"]

    # Add body with indentation
    for body_line in finding.body.split("\n"):
        if body_line.strip():
            lines.append(f"  {body_line}")

    return "\n".join(lines)
