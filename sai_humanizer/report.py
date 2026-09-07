"""Unified report across all 3 layers of AI pattern detection.

Provides a single HumanizationReport that aggregates findings from:
  Layer 1: Stylistic patterns (vocabulary, phrases, punctuation, burstiness)
  Layer 2: Unicode watermarks (zero-width chars, exotic spaces, bidi, homoglyphs)
  Layer 3: Statistical signals (perplexity, Binoculars, green-list, repetition)
"""

from dataclasses import dataclass, field
from typing import Literal

from sai_humanizer.unicode_watermarks import UnicodeScanResult
from sai_humanizer.statistical import StatisticalScanResult


LayerName = Literal["stylistic", "unicode", "statistical"]


@dataclass
class LayerReport:
    """Per-layer results."""
    name: LayerName
    detected: int
    fixed: int
    details: list[str] = field(default_factory=list)
    is_clean: bool = True


@dataclass
class HumanizationReport:
    """Full report across all 3 layers.

    Attributes:
        original: Input text before processing
        humanized: Output text after all layers applied
        layers: Per-layer breakdown
        total_detected: Total AI patterns/signals detected across all layers
        total_fixed: Total patterns fixed/removed across all layers
        ai_probability: Overall AI probability from statistical layer (0.0-1.0)
        burstiness_before: Burstiness score before processing
        burstiness_after: Burstiness score after processing
        unicode_findings: Raw Unicode scan findings (Layer 2)
        statistical_findings: Raw statistical scan findings (Layer 3)
        stylistic_changes: List of stylistic changes made (Layer 1)
    """
    original: str
    humanized: str = ""
    layers: dict[str, LayerReport] = field(default_factory=dict)
    total_detected: int = 0
    total_fixed: int = 0
    ai_probability: float = 0.0
    burstiness_before: float = 0.0
    burstiness_after: float = 0.0
    unicode_findings: list = field(default_factory=list)
    statistical_findings: list = field(default_factory=list)
    stylistic_changes: list[str] = field(default_factory=list)
    mode: str = "technical"

    @property
    def is_clean(self) -> bool:
        """True if no AI patterns detected across any layer."""
        return self.total_detected == 0

    @property
    def overall_verdict(self) -> str:
        """Human-readable overall verdict."""
        if self.total_detected == 0:
            return "CLEAN: No AI patterns detected across any layer."
        parts = [f"Detected {self.total_detected} AI signal(s) across {sum(1 for l in self.layers.values() if not l.is_clean)} layer(s):"]
        for name, layer in self.layers.items():
            if not layer.is_clean:
                parts.append(f"  {name}: {layer.detected} detected, {layer.fixed} fixed")
        if self.ai_probability > 0.5:
            parts.append(f"  Statistical AI probability: {self.ai_probability:.1%}")
        return "\n".join(parts)

    def summary(self) -> str:
        """One-line summary suitable for CLI output."""
        if self.is_clean:
            return "OK: 0 AI patterns detected across all 3 layers."
        return (
            f"FOUND: {self.total_detected} AI patterns across "
            f"{sum(1 for l in self.layers.values() if not l.is_clean)} layer(s), "
            f"{self.total_fixed} fixed. "
            f"AI probability: {self.ai_probability:.0%}"
        )

    def detailed_report(self) -> str:
        """Full multi-line report for --report mode."""
        lines = [
            "=" * 70,
            "sai-humanizer: Full 3-Layer Report",
            "=" * 70,
            f"Mode: {self.mode}",
            f"Input length: {len(self.original)} chars",
            f"Output length: {len(self.humanized)} chars",
            f"Total detected: {self.total_detected}",
            f"Total fixed: {self.total_fixed}",
            f"AI probability: {self.ai_probability:.1%}",
            f"Burstiness: {self.burstiness_before:.2f} -> {self.burstiness_after:.2f}",
            "=" * 70,
        ]

        for layer_name in ["stylistic", "unicode", "statistical"]:
            layer = self.layers.get(layer_name)
            if not layer:
                continue
            status = "CLEAN" if layer.is_clean else f"{layer.detected} detected, {layer.fixed} fixed"
            lines.append(f"\n[{layer_name.upper()}] {status}")
            for detail in layer.details:
                lines.append(f"  - {detail}")

        lines.append("\n" + "=" * 70)
        lines.append(self.overall_verdict)
        lines.append("=" * 70)
        return "\n".join(lines)
