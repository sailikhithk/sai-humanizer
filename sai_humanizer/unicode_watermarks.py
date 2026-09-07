"""Layer 2: Unicode watermark detection and removal.

AI vendors embed invisible Unicode characters into generated text as
provenance marks. This module detects and removes them deterministically.

Covered artifacts:
  - Zero-width characters (ZWSP, ZWNJ, ZWJ)
  - Exotic Unicode spaces (thin, hair, narrow no-break, etc.)
  - Bidirectional control characters (LRE, RLE, PDF, LRO, RLO, LRI, RLI, FSI, PDI)
  - Tag characters (U+E0000-E007F, used by some AI systems)
  - Mixed-script homoglyphs (Cyrillic a, Greek o, etc.)
  - Variation selectors (VS1-VS16)
  - Interlinear annotation characters

Sources: guillaumemeyer/watermarks-remover, cyzanfar/text-watermark-remover,
PyModel/watermark-remover, virajshoor/ADAFAI
"""

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass
class UnicodeFinding:
    """A single Unicode artifact found in text."""
    char: str
    codepoint: str
    name: str
    position: int
    category: str  # "zero-width", "exotic-space", "bidi", "tag", "homoglyph", "variation-selector"
    vendor_hint: str  # which AI vendor is known to use this


@dataclass
class UnicodeScanResult:
    """Result of scanning text for Unicode watermarks."""
    text: str
    findings: list[UnicodeFinding] = field(default_factory=list)
    cleaned: str = ""
    total_found: int = 0
    categories_found: set[str] = field(default_factory=set)

    @property
    def is_clean(self) -> bool:
        return self.total_found == 0


# ---------------------------------------------------------------------------
# Character categories
# ---------------------------------------------------------------------------

ZERO_WIDTH_CHARS = {
    "\u200b": ("ZERO WIDTH SPACE", "zero-width", "Various AI systems"),
    "\u200c": ("ZERO WIDTH NON-JOINER", "zero-width", "Various AI systems"),
    "\u200d": ("ZERO WIDTH JOINER", "zero-width", "Various AI systems"),
    "\u200e": ("LEFT-TO-RIGHT MARK", "zero-width", "Various AI systems"),
    "\u200f": ("RIGHT-TO-LEFT MARK", "zero-width", "Various AI systems"),
    "\u2060": ("WORD JOINER", "zero-width", "Claude"),
    "\u2061": ("FUNCTION APPLICATION", "zero-width", "Rare"),
    "\u2062": ("INVISIBLE TIMES", "zero-width", "Rare"),
    "\u2063": ("INVISIBLE SEPARATOR", "zero-width", "Rare"),
    "\u2064": ("INVISIBLE PLUS", "zero-width", "Rare"),
    "\ufeff": ("ZERO WIDTH NO-BREAK SPACE / BOM", "zero-width", "Various AI systems"),
}

EXOTIC_SPACES = {
    "\u00a0": ("NO-BREAK SPACE", "exotic-space", "Various"),
    "\u2000": ("EN QUAD", "exotic-space", "Rare"),
    "\u2001": ("EM QUAD", "exotic-space", "Rare"),
    "\u2002": ("EN SPACE", "exotic-space", "Various"),
    "\u2003": ("EM SPACE", "exotic-space", "Various"),
    "\u2004": ("THREE-PER-EM SPACE", "exotic-space", "Rare"),
    "\u2005": ("FOUR-PER-EM SPACE", "exotic-space", "Rare"),
    "\u2006": ("SIX-PER-EM SPACE", "exotic-space", "Rare"),
    "\u2007": ("FIGURE SPACE", "exotic-space", "Rare"),
    "\u2008": ("PUNCTUATION SPACE", "exotic-space", "Rare"),
    "\u2009": ("THIN SPACE", "exotic-space", "Various AI systems"),
    "\u200a": ("HAIR SPACE", "exotic-space", "Various AI systems"),
    "\u202f": ("NARROW NO-BREAK SPACE", "exotic-space", "Gemini"),
    "\u205f": ("MEDIUM MATHEMATICAL SPACE", "exotic-space", "Rare"),
}

BIDI_CONTROLS = {
    "\u202a": ("LEFT-TO-RIGHT EMBEDDING", "bidi", "Various"),
    "\u202b": ("RIGHT-TO-LEFT EMBEDDING", "bidi", "Various"),
    "\u202c": ("POP DIRECTIONAL FORMATTING", "bidi", "Various"),
    "\u202d": ("LEFT-TO-RIGHT OVERRIDE", "bidi", "Various"),
    "\u202e": ("RIGHT-TO-LEFT OVERRIDE", "bidi", "Various"),
    "\u2066": ("LEFT-TO-RIGHT ISOLATE", "bidi", "Various"),
    "\u2067": ("RIGHT-TO-LEFT ISOLATE", "bidi", "Various"),
    "\u2068": ("FIRST STRONG ISOLATE", "bidi", "Various"),
    "\u2069": ("POP DIRECTIONAL ISOLATE", "bidi", "Various"),
}

VARIATION_SELECTORS = {
    "\ufe00": ("VARIATION SELECTOR-1", "variation-selector", "Various"),
    "\ufe01": ("VARIATION SELECTOR-2", "variation-selector", "Various"),
    "\ufe02": ("VARIATION SELECTOR-3", "variation-selector", "Various"),
    "\ufe03": ("VARIATION SELECTOR-4", "variation-selector", "Various"),
    "\ufe04": ("VARIATION SELECTOR-5", "variation-selector", "Various"),
    "\ufe05": ("VARIATION SELECTOR-6", "variation-selector", "Various"),
    "\ufe06": ("VARIATION SELECTOR-7", "variation-selector", "Various"),
    "\ufe07": ("VARIATION SELECTOR-8", "variation-selector", "Various"),
    "\ufe08": ("VARIATION SELECTOR-9", "variation-selector", "Various"),
    "\ufe09": ("VARIATION SELECTOR-10", "variation-selector", "Various"),
    "\ufe0a": ("VARIATION SELECTOR-11", "variation-selector", "Various"),
    "\ufe0b": ("VARIATION SELECTOR-12", "variation-selector", "Various"),
    "\ufe0c": ("VARIATION SELECTOR-13", "variation-selector", "Various"),
    "\ufe0d": ("VARIATION SELECTOR-14", "variation-selector", "Various"),
    "\ufe0e": ("VARIATION SELECTOR-15", "variation-selector", "Various"),
    "\ufe0f": ("VARIATION SELECTOR-16", "variation-selector", "Various"),
}

# Tag characters U+E0000-E007F (used by some AI provenance systems)
TAG_CHAR_RANGE = (0xE0000, 0xE007F)

# Interlinear annotation
INTERLINEAR_CHARS = {
    "\ufff9": ("INTERLINEAR ANNOTATION ANCHOR", "tag", "Rare"),
    "\ufffa": ("INTERLINEAR ANNOTATION SEPARATOR", "tag", "Rare"),
    "\ufffb": ("INTERLINEAR ANNOTATION TERMINATOR", "tag", "Rare"),
}

# Common homoglyphs: Latin chars that look identical to ASCII
HOMOGLYPHS = {
    "\u0430": ("a", "CYRILLIC SMALL LETTER A", "homoglyph"),
    "\u0435": ("e", "CYRILLIC SMALL LETTER IE", "homoglyph"),
    "\u043e": ("o", "CYRILLIC SMALL LETTER O", "homoglyph"),
    "\u0440": ("p", "CYRILLIC SMALL LETTER ER", "homoglyph"),
    "\u0441": ("c", "CYRILLIC SMALL LETTER ES", "homoglyph"),
    "\u0445": ("x", "CYRILLIC SMALL LETTER HA", "homoglyph"),
    "\u0443": ("y", "CYRILLIC SMALL LETTER U", "homoglyph"),
    "\u03bf": ("o", "GREEK SMALL LETTER OMICRON", "homoglyph"),
    "\u03c1": ("p", "GREEK SMALL LETTER RHO", "homoglyph"),
    "\u03b1": ("a", "GREEK SMALL LETTER ALPHA", "homoglyph"),
    "\u03b5": ("e", "GREEK SMALL LETTER EPSILON", "homoglyph"),
}

# All chars to strip (zero-width, exotic spaces, bidi, variation selectors, interlinear)
STRIP_MAP: dict[str, tuple[str, str, str]] = {}
STRIP_MAP.update(ZERO_WIDTH_CHARS)
STRIP_MAP.update(EXOTIC_SPACES)
STRIP_MAP.update(BIDI_CONTROLS)
STRIP_MAP.update(VARIATION_SELECTORS)
STRIP_MAP.update(INTERLINEAR_CHARS)


class UnicodeWatermarkScanner:
    """Detect and remove invisible Unicode watermarks from text."""

    def scan(self, text: str) -> UnicodeScanResult:
        """Scan text for Unicode artifacts. Does not modify text."""
        result = UnicodeScanResult(text=text)
        for i, char in enumerate(text):
            finding = self._check_char(char, i)
            if finding:
                result.findings.append(finding)
                result.categories_found.add(finding.category)

        # Check for tag characters in range
        for i, char in enumerate(text):
            cp = ord(char)
            if TAG_CHAR_RANGE[0] <= cp <= TAG_CHAR_RANGE[1]:
                result.findings.append(UnicodeFinding(
                    char=char,
                    codepoint=f"U+{cp:04X}",
                    name=unicodedata.name(char, f"TAG CHARACTER U+{cp:04X}"),
                    position=i,
                    category="tag",
                    vendor_hint="AI provenance tag characters",
                ))
                result.categories_found.add("tag")

        # Check for homoglyphs
        for i, char in enumerate(text):
            if char in HOMOGLYPHS:
                ascii_equiv, name, cat = HOMOGLYPHS[char]
                result.findings.append(UnicodeFinding(
                    char=char,
                    codepoint=f"U+{ord(char):04X}",
                    name=name,
                    position=i,
                    category=cat,
                    vendor_hint=f"Looks like ASCII '{ascii_equiv}' but is {name.split()[0]}",
                ))
                result.categories_found.add(cat)

        result.total_found = len(result.findings)
        return result

    def clean(self, text: str, remove_homoglyphs: bool = True) -> UnicodeScanResult:
        """Scan and remove Unicode artifacts from text."""
        result = self.scan(text)

        cleaned = list(text)
        # Mark positions to remove (zero-width, bidi, variation selectors, exotic spaces, tags)
        positions_to_remove = set()
        for finding in result.findings:
            if finding.category in ("zero-width", "bidi", "variation-selector", "exotic-space", "tag"):
                positions_to_remove.add(finding.position)
            elif finding.category == "homoglyph" and remove_homoglyphs:
                # Replace homoglyph with ASCII equivalent
                ascii_equiv = HOMOGLYPHS[finding.char][0]
                cleaned[finding.position] = ascii_equiv

        # Remove marked positions (reverse order to preserve indices)
        for pos in sorted(positions_to_remove, reverse=True):
            del cleaned[pos]

        result.cleaned = "".join(cleaned)
        return result

    def _check_char(self, char: str, position: int) -> UnicodeFinding | None:
        """Check a single character against known watermark chars."""
        if char in STRIP_MAP:
            name, category, vendor = STRIP_MAP[char]
            return UnicodeFinding(
                char=char,
                codepoint=f"U+{ord(char):04X}",
                name=name,
                position=position,
                category=category,
                vendor_hint=vendor,
            )
        return None

    def inspect(self, text: str) -> str:
        """Return a human-readable inspection report."""
        result = self.scan(text)
        if result.is_clean:
            return "No Unicode watermarks detected."

        lines = [f"Unicode Watermark Scan: {result.total_found} artifact(s) found\n"]
        by_category: dict[str, list[UnicodeFinding]] = {}
        for f in result.findings:
            by_category.setdefault(f.category, []).append(f)

        for cat, findings in by_category.items():
            lines.append(f"  {cat} ({len(findings)}):")
            for f in findings[:10]:  # Show first 10 per category
                lines.append(
                    f"    pos {f.position:6d} | {f.codepoint} | {f.name} | {f.vendor_hint}"
                )
            if len(findings) > 10:
                lines.append(f"    ... and {len(findings) - 10} more")

        return "\n".join(lines)
