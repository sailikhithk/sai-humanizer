"""sai-humanizer: Three-layer de-AI text humanizer.

Layer 1 - Stylistic: 57+ AI-tell patterns (vocabulary, phrases, punctuation)
Layer 2 - Unicode: invisible watermark detection and removal
Layer 3 - Statistical: perplexity, Binoculars, burstiness, green-list scoring

Modes: technical, marketing, resume
"""

from sai_humanizer.pipeline import Humanizer
from sai_humanizer.patterns import AI_TELL_TAXONOMY, BANNED_WORDS, BANNED_PHRASES
from sai_humanizer.stats import BurstinessScorer
from sai_humanizer.unicode_watermarks import UnicodeWatermarkScanner
from sai_humanizer.statistical import StatisticalDetector
from sai_humanizer.report import HumanizationReport, LayerReport

__version__ = "0.2.0"
__all__ = [
    "Humanizer",
    "HumanizationReport",
    "LayerReport",
    "AI_TELL_TAXONOMY",
    "BANNED_WORDS",
    "BANNED_PHRASES",
    "BurstinessScorer",
    "UnicodeWatermarkScanner",
    "StatisticalDetector",
]
