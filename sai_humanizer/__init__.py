"""sai-humanizer: De-AI text humanizer with 55+ pattern taxonomy.

Two-pass pipeline:
  1. Deterministic: regex-based pattern removal (em-dashes, AI vocabulary, filler)
  2. Statistical: burstiness enforcement, sentence length variance, triadic breaking

Modes: technical, marketing, resume
"""

from sai_humanizer.pipeline import Humanizer
from sai_humanizer.patterns import AI_TELL_TAXONOMY, BANNED_WORDS, BANNED_PHRASES
from sai_humanizer.stats import BurstinessScorer

__version__ = "0.1.0"
__all__ = ["Humanizer", "AI_TELL_TAXONOMY", "BANNED_WORDS", "BANNED_PHRASES", "BurstinessScorer"]
