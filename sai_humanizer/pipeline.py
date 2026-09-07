"""Three-layer humanization pipeline.

Layer 1: Stylistic pattern removal (deterministic regex)
Layer 2: Unicode watermark detection and removal (deterministic)
Layer 3: Statistical AI text detection (heuristic or model-based)

Layers run sequentially. Each layer receives the output of the previous
layer. The report aggregates findings across all layers.
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
from sai_humanizer.unicode_watermarks import UnicodeWatermarkScanner
from sai_humanizer.statistical import StatisticalDetector
from sai_humanizer.report import HumanizationReport, LayerReport


class Humanizer:
    """Three-layer de-AI text humanizer.

    Modes:
      - technical: preserve technical terms, conservative replacements
      - marketing: aggressive pattern removal, punchier output
      - resume: preserve action verbs, strip filler, keep metrics

    Layers:
      - stylistic: regex-based removal of 57+ AI patterns
      - unicode: detect and strip invisible Unicode watermarks
      - statistical: perplexity, burstiness, green-list scoring
    """

    def __init__(
        self,
        mode: str = "technical",
        layers: list[str] | None = None,
        use_model: bool = False,
    ):
        if mode not in ("technical", "marketing", "resume"):
            raise ValueError(f"Unknown mode: {mode}. Use technical, marketing, or resume.")
        self.mode = mode
        self.layers = layers or ["stylistic", "unicode", "statistical"]
        self._burstiness = BurstinessScorer()
        self._unicode_scanner = UnicodeWatermarkScanner()
        self._stat_detector = StatisticalDetector(use_model=use_model)

    def humanize(self, text: str) -> HumanizationReport:
        """Run the full three-layer pipeline on text."""
        report = HumanizationReport(original=text, mode=self.mode)

        # Score before
        before = self._burstiness.score(text)
        report.burstiness_before = before.burstiness

        current = text

        # Layer 1: Stylistic
        if "stylistic" in self.layers:
            current, layer1_report = self._layer1_stylistic(current)
            report.layers["stylistic"] = layer1_report
            report.stylistic_changes = layer1_report.details

        # Layer 2: Unicode watermarks
        if "unicode" in self.layers:
            current, layer2_report = self._layer2_unicode(current)
            report.layers["unicode"] = layer2_report
            report.unicode_findings = self._unicode_scanner.scan(text).findings

        # Layer 3: Statistical
        if "statistical" in self.layers:
            stat_result = self._stat_detector.scan(current)
            layer3_report = LayerReport(
                name="statistical",
                detected=sum(1 for f in stat_result.findings if f.is_ai_like),
                fixed=0,  # Statistical layer detects, doesn't auto-fix
                details=[f.description for f in stat_result.findings if f.is_ai_like],
                is_clean=not stat_result.is_ai_like,
            )
            report.layers["statistical"] = layer3_report
            report.statistical_findings = stat_result.findings
            report.ai_probability = stat_result.ai_probability

        report.humanized = current

        # Score after
        after = self._burstiness.score(current)
        report.burstiness_after = after.burstiness

        # Aggregate totals
        report.total_detected = sum(l.detected for l in report.layers.values())
        report.total_fixed = sum(l.fixed for l in report.layers.values())

        return report

    def detect(self, text: str) -> HumanizationReport:
        """Detection only. No modifications to text."""
        report = HumanizationReport(original=text, humanized=text, mode=self.mode)

        before = self._burstiness.score(text)
        report.burstiness_before = before.burstiness
        report.burstiness_after = before.burstiness

        # Layer 1: Stylistic (detect only)
        if "stylistic" in self.layers:
            _, layer1_report = self._layer1_stylistic(text, fix=False)
            report.layers["stylistic"] = layer1_report
            report.stylistic_changes = layer1_report.details

        # Layer 2: Unicode (detect only)
        if "unicode" in self.layers:
            scan = self._unicode_scanner.scan(text)
            report.layers["unicode"] = LayerReport(
                name="unicode",
                detected=scan.total_found,
                fixed=0,
                details=[
                    f"{f.codepoint} {f.name} at pos {f.position} ({f.category}, {f.vendor_hint})"
                    for f in scan.findings
                ],
                is_clean=scan.is_clean,
            )
            report.unicode_findings = scan.findings

        # Layer 3: Statistical
        if "statistical" in self.layers:
            stat_result = self._stat_detector.scan(text)
            report.layers["statistical"] = LayerReport(
                name="statistical",
                detected=sum(1 for f in stat_result.findings if f.is_ai_like),
                fixed=0,
                details=[f.description for f in stat_result.findings if f.is_ai_like],
                is_clean=not stat_result.is_ai_like,
            )
            report.statistical_findings = stat_result.findings
            report.ai_probability = stat_result.ai_probability

        report.total_detected = sum(l.detected for l in report.layers.values())
        report.total_fixed = sum(l.fixed for l in report.layers.values())
        return report

    def _layer1_stylistic(self, text: str, fix: bool = True) -> tuple[str, LayerReport]:
        """Layer 1: deterministic regex-based pattern removal."""
        changes: list[str] = []
        result = text

        # 1. Replace em-dashes and en-dashes
        for char, replacement in DASH_REPLACEMENTS.items():
            if char in result:
                count = result.count(char)
                if fix:
                    result = result.replace(char, replacement)
                changes.append(f"Replaced {count} em/en-dash(es) with hyphens")

        # 2. Replace curly quotes with straight quotes
        curly_map = {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"}
        for curly, straight in curly_map.items():
            if curly in result:
                count = result.count(curly)
                if fix:
                    result = result.replace(curly, straight)
                changes.append(f"Replaced {count} curly quote(s) with straight quotes")

        # 3. Remove sycophantic openers
        if fix:
            lower = result.lower()
            for opener in SYCOPHANTIC_OPENERS:
                if lower.startswith(opener):
                    idx = len(opener)
                    while idx < len(result) and result[idx] in ".!?, ":
                        idx += 1
                    result = result[idx:]
                    changes.append(f"Removed sycophantic opener: '{opener}'")
                    break
        else:
            lower = text.lower()
            for opener in SYCOPHANTIC_OPENERS:
                if lower.startswith(opener):
                    changes.append(f"Found sycophantic opener: '{opener}'")
                    break

        # 4. Replace banned AI phrases
        for pattern, suggestion in BANNED_PHRASES:
            matches = list(re.finditer(pattern, result, re.IGNORECASE))
            if matches:
                if fix:
                    for match in reversed(matches):
                        result = result[:match.start()] + suggestion + result[match.end():]
                changes.append(f"Found {len(matches)}x AI phrase matching /{pattern}/")

        # 5. Replace banned AI words
        for word, suggestion in BANNED_WORDS.items():
            pattern = rf"\b{re.escape(word)}\b"
            matches = list(re.finditer(pattern, result, re.IGNORECASE))
            if matches:
                if fix:
                    for match in reversed(matches):
                        replacement = suggestion
                        if match.group(0)[0].isupper():
                            replacement = suggestion[0].upper() + suggestion[1:]
                        result = result[:match.start()] + replacement + result[match.end():]
                changes.append(f"Found {len(matches)}x AI word '{word}'")

        # 6. Fix copula avoidance
        copula_patterns = [
            (r"\bserves as\b", "is"),
            (r"\bstands as\b", "is"),
            (r"\bfunctions as\b", "is"),
            (r"\bacts as\b", "is"),
        ]
        for pattern, replacement in copula_patterns:
            matches = list(re.finditer(pattern, result, re.IGNORECASE))
            if matches:
                if fix:
                    for match in reversed(matches):
                        r = replacement
                        if match.group(0)[0].isupper():
                            r = replacement[0].upper() + replacement[1:]
                        result = result[:match.start()] + r + result[match.end():]
                changes.append(f"Found {len(matches)}x copula avoidance '{pattern}'")

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
                if fix:
                    for match in reversed(matches):
                        r = replacement
                        if match.group(0)[0].isupper() and r:
                            r = r[0].upper() + r[1:]
                        result = result[:match.start()] + r + result[match.end():]
                changes.append(f"Found {len(matches)}x filler '{pattern}'")

        # 8. Clean up double spaces
        if fix:
            result = re.sub(r'  +', ' ', result)
            result = re.sub(r'\s+\.', '.', result)

        # 9. Detect triadic parallelism
        triadic_matches = list(TRIADIC_PATTERN.finditer(result))
        if triadic_matches and self.mode != "resume":
            changes.append(
                f"Found {len(triadic_matches)} triadic parallelism pattern(s) (rule of three)"
            )

        # 10. Check burstiness
        burst = self._burstiness.score(result)
        if burst.is_ai_like and burst.suggestion:
            changes.append(burst.suggestion)
            if fix:
                result = self._burstiness.enforce_variance(result)

        # 11. Detect bold-lead monotony
        bold_lead_pattern = re.compile(r'^\s*[-*]\s+\*\*([^*]+):\*\*\s+', re.MULTILINE)
        bold_matches = bold_lead_pattern.findall(result)
        if bold_matches:
            changes.append(f"Found {len(bold_matches)} bold-lead bullet(s)")

        # 12. Detect title case headings
        heading_pattern = re.compile(r'^#+\s+(.+)$', re.MULTILINE)
        for match in heading_pattern.finditer(result):
            heading = match.group(1)
            words = heading.split()
            if len(words) > 2 and sum(1 for w in words if w[0].isupper()) > len(words) * 0.6:
                changes.append(f"Title case heading: '{heading}'")

        fixed = len(changes) if fix else 0
        return result, LayerReport(
            name="stylistic",
            detected=len(changes),
            fixed=fixed,
            details=changes,
            is_clean=len(changes) == 0,
        )

    def _layer2_unicode(self, text: str) -> tuple[str, LayerReport]:
        """Layer 2: Unicode watermark detection and removal."""
        result = self._unicode_scanner.clean(text, remove_homoglyphs=True)

        details = []
        by_category: dict[str, int] = {}
        for f in result.findings:
            by_category[f.category] = by_category.get(f.category, 0) + 1

        for cat, count in by_category.items():
            details.append(f"Removed {count} {cat} character(s)")

        if result.findings:
            # Show first 5 specific findings
            for f in result.findings[:5]:
                details.append(f"  {f.codepoint} {f.name} at pos {f.position}")
            if len(result.findings) > 5:
                details.append(f"  ... and {len(result.findings) - 5} more")

        return result.cleaned, LayerReport(
            name="unicode",
            detected=result.total_found,
            fixed=result.total_found,  # All detected Unicode is fixed
            details=details,
            is_clean=result.is_clean,
        )
