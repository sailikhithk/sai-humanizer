"""Two-pass humanization pipeline.

Pass 1: Deterministic pattern removal (regex-based)
Pass 2: Statistical rhythm and cadence inversion
"""

import re
from dataclasses import dataclass, field

from sai_humanizer.patterns import (
    BANNED_WORDS,
    BANNED_PHRASES,
    DASH_REPLACEMENTS,
    SYCOPHANTIC_OPENERS,
    TRIADIC_PATTERN,
)
from sai_humanizer.stats import BurstinessScorer


@dataclass
class HumanizerReport:
    original: str
    humanized: str
    pass1_changes: list[str] = field(default_factory=list)
    pass2_changes: list[str] = field(default_factory=list)
    total_changes: int = 0
    burstiness_before: float = 0.0
    burstiness_after: float = 0.0


class Humanizer:
    """Two-pass de-AI text humanizer.

    Modes:
      - technical: preserve technical terms, conservative replacements
      - marketing: aggressive pattern removal, punchier output
      - resume: preserve action verbs, strip filler, keep metrics
    """

    def __init__(self, mode: str = "technical"):
        if mode not in ("technical", "marketing", "resume"):
            raise ValueError(f"Unknown mode: {mode}. Use technical, marketing, or resume.")
        self.mode = mode
        self._burstiness = BurstinessScorer()

    def humanize(self, text: str) -> HumanizerReport:
        """Run the full two-pass pipeline on text."""
        report = HumanizerReport(original=text, humanized="")

        # Score before
        before = self._burstiness.score(text)
        report.burstiness_before = before.burstiness

        # Pass 1: deterministic
        pass1_result, pass1_changes = self._pass1_deterministic(text)
        report.pass1_changes = pass1_changes

        # Pass 2: statistical
        pass2_result, pass2_changes = self._pass2_statistical(pass1_result)
        report.pass2_changes = pass2_changes

        report.humanized = pass2_result
        report.total_changes = len(pass1_changes) + len(pass2_changes)

        # Score after
        after = self._burstiness.score(pass2_result)
        report.burstiness_after = after.burstiness

        return report

    def _pass1_deterministic(self, text: str) -> tuple[str, list[str]]:
        """Pass 1: regex-based pattern removal."""
        changes: list[str] = []
        result = text

        # 1. Replace em-dashes and en-dashes
        for char, replacement in DASH_REPLACEMENTS.items():
            if char in result:
                count = result.count(char)
                result = result.replace(char, replacement)
                changes.append(f"Replaced {count} em/en-dash(es) with hyphens")

        # 2. Replace curly quotes with straight quotes
        curly_map = {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"}
        for curly, straight in curly_map.items():
            if curly in result:
                count = result.count(curly)
                result = result.replace(curly, straight)
                changes.append(f"Replaced {count} curly quote(s) with straight quotes")

        # 3. Remove sycophantic openers
        lower = result.lower()
        for opener in SYCOPHANTIC_OPENERS:
            if lower.startswith(opener):
                # Remove the opener and any trailing punctuation
                idx = len(opener)
                while idx < len(result) and result[idx] in ".!?, ":
                    idx += 1
                result = result[idx:]
                changes.append(f"Removed sycophantic opener: '{opener}'")
                break

        # 4. Replace banned AI phrases
        for pattern, suggestion in BANNED_PHRASES:
            matches = list(re.finditer(pattern, result, re.IGNORECASE))
            if matches:
                for match in reversed(matches):
                    result = result[:match.start()] + suggestion + result[match.end():]
                changes.append(f"Replaced {len(matches)}x AI phrase matching /{pattern}/")

        # 5. Replace banned AI words
        for word, suggestion in BANNED_WORDS.items():
            pattern = rf"\b{re.escape(word)}\b"
            matches = list(re.finditer(pattern, result, re.IGNORECASE))
            if matches:
                for match in reversed(matches):
                    # Preserve capitalization
                    replacement = suggestion
                    if match.group(0)[0].isupper():
                        replacement = suggestion[0].upper() + suggestion[1:]
                    result = result[:match.start()] + replacement + result[match.end():]
                changes.append(f"Replaced {len(matches)}x AI word '{word}' -> '{suggestion}'")

        # 6. Fix copula avoidance: "serves as" / "stands as" -> "is"
        copula_patterns = [
            (r"\bserves as\b", "is"),
            (r"\bstands as\b", "is"),
            (r"\bfunctions as\b", "is"),
            (r"\bacts as\b", "is"),
        ]
        for pattern, replacement in copula_patterns:
            matches = list(re.finditer(pattern, result, re.IGNORECASE))
            if matches:
                for match in reversed(matches):
                    r = replacement
                    if match.group(0)[0].isupper():
                        r = replacement[0].upper() + replacement[1:]
                    result = result[:match.start()] + r + result[match.end():]
                changes.append(f"Replaced {len(matches)}x copula avoidance '{pattern}' -> '{replacement}'")

        # 7. Fix filler phrases
        filler_map = {
            r"\bin order to\b": "to",
            r"\bdue to the fact that\b": "because",
            r"\bat this point in time\b": "now",
            r"\bin the event that\b": "if",
            r"\bhas the ability to\b": "can",
            r"\bit is important to note that\b": "",
        }
        for pattern, replacement in filler_map.items():
            matches = list(re.finditer(pattern, result, re.IGNORECASE))
            if matches:
                for match in reversed(matches):
                    r = replacement
                    if match.group(0)[0].isupper() and r:
                        r = r[0].upper() + r[1:]
                    result = result[:match.start()] + r + result[match.end():]
                changes.append(f"Replaced {len(matches)}x filler '{pattern}' -> '{replacement}'")

        # 8. Clean up double spaces from replacements
        result = re.sub(r'  +', ' ', result)
        result = re.sub(r'\s+\.', '.', result)

        return result, changes

    def _pass2_statistical(self, text: str) -> tuple[str, list[str]]:
        """Pass 2: structural rhythm and cadence fixes."""
        changes: list[str] = []
        result = text

        # 1. Detect and flag triadic parallelism
        triadic_matches = list(TRIADIC_PATTERN.finditer(result))
        if triadic_matches and self.mode != "resume":
            changes.append(
                f"Found {len(triadic_matches)} triadic parallelism pattern(s) (rule of three). "
                f"Consider breaking 'X, Y, and Z' into two items or using a different structure."
            )

        # 2. Check burstiness and suggest variance
        burst = self._burstiness.score(result)
        if burst.is_ai_like and burst.suggestion:
            changes.append(burst.suggestion)
            # Apply variance enforcement
            result = self._burstiness.enforce_variance(result)
            if result != text:
                changes.append("Split long sentences to increase burstiness")

        # 3. Remove bold-lead monotony in bullet lists
        bold_lead_pattern = re.compile(r'^\s*[-*]\s+\*\*([^*]+):\*\*\s+', re.MULTILINE)
        bold_matches = bold_lead_pattern.findall(result)
        if bold_matches:
            changes.append(
                f"Found {len(bold_matches)} bold-lead bullet(s). "
                f"Consider removing bold-colon formatting for natural flow."
            )

        # 4. Detect title case in headings
        heading_pattern = re.compile(r'^#+\s+(.+)$', re.MULTILINE)
        for match in heading_pattern.finditer(result):
            heading = match.group(1)
            words = heading.split()
            if len(words) > 2 and sum(1 for w in words if w[0].isupper()) > len(words) * 0.6:
                changes.append(
                    f"Title case heading detected: '{heading}'. "
                    f"Consider sentence case for natural tone."
                )

        return result, changes
