"""Tests for sai-humanizer."""

import pytest
from sai_humanizer.pipeline import Humanizer
from sai_humanizer.patterns import BANNED_WORDS, BANNED_PHRASES, AI_TELL_TAXONOMY
from sai_humanizer.stats import BurstinessScorer


class TestPass1Deterministic:
    """Pass 1: deterministic pattern removal."""

    def test_em_dash_removal(self):
        h = Humanizer()
        text = "The tool is fast\u2014reliable\u2014and cheap."
        result = h.humanize(text)
        assert "\u2014" not in result.humanized
        assert "\u2013" not in result.humanized
        assert any("dash" in c.lower() for c in result.pass1_changes)

    def test_banned_word_removal(self):
        h = Humanizer()
        text = "Let's delve into the tapestry of data."
        result = h.humanize(text)
        assert "delve" not in result.humanized.lower()
        assert "tapestry" not in result.humanized.lower()
        assert result.total_changes >= 2

    def test_seamless_removal(self):
        h = Humanizer()
        text = "The integration is seamless and robust."
        result = h.humanize(text)
        assert "seamless" not in result.humanized.lower()

    def test_sycophantic_opener_removal(self):
        h = Humanizer()
        text = "Great question! The answer is simple."
        result = h.humanize(text)
        assert not result.humanized.lower().startswith("great question")
        assert any("sycophantic" in c.lower() for c in result.pass1_changes)

    def test_filler_phrase_removal(self):
        h = Humanizer()
        text = "In order to run the test, you need pytest."
        result = h.humanize(text)
        assert "in order to" not in result.humanized.lower()

    def test_copula_avoidance_fix(self):
        h = Humanizer()
        text = "The tool serves as a proxy for the API."
        result = h.humanize(text)
        assert "serves as" not in result.humanized.lower()
        assert "is" in result.humanized.lower()

    def test_curly_quotes_replaced(self):
        h = Humanizer()
        text = "He said \u201chello\u201d and left."
        result = h.humanize(text)
        assert "\u201c" not in result.humanized
        assert "\u201d" not in result.humanized
        assert '"' in result.humanized

    def test_banned_phrase_removal(self):
        h = Humanizer()
        text = "It is important to note that the system works."
        result = h.humanize(text)
        assert "it is important to note" not in result.humanized.lower()

    def test_clean_text_no_changes(self):
        h = Humanizer()
        text = "The function returns a list of tuples. It filters by name and sorts by date. Fast. We benchmarked it at 50ms per call."
        result = h.humanize(text)
        assert result.total_changes == 0

    def test_multiple_patterns_at_once(self):
        h = Humanizer()
        text = (
            "Great question! Let's delve into the seamless integration. "
            "It is important to note that the tool serves as a testament "
            "to the power of modern engineering. In order to use it, "
            "you need Python 3.10+."
        )
        result = h.humanize(text)
        assert "delve" not in result.humanized.lower()
        assert "seamless" not in result.humanized.lower()
        assert "it is important to note" not in result.humanized.lower()
        assert "serves as" not in result.humanized.lower()
        assert "testament" not in result.humanized.lower()
        assert "in order to" not in result.humanized.lower()
        assert not result.humanized.lower().startswith("great question")
        assert result.total_changes >= 7


class TestPass2Statistical:
    """Pass 2: statistical rhythm and cadence."""

    def test_burstiness_scoring_low(self):
        scorer = BurstinessScorer()
        # Uniform sentence lengths = AI-like
        text = "The system works well. The code runs fast. The tests pass now. The build is clean."
        result = scorer.score(text)
        assert result.burstiness < 0.5

    def test_burstiness_scoring_high(self):
        scorer = BurstinessScorer()
        # Varied sentence lengths = human-like
        text = (
            "It works. "
            "The system processes 4M requests per minute across 12 Kafka partitions with exactly-once semantics. "
            "Fast. "
            "We measured this over a 30-day window in production."
        )
        result = scorer.score(text)
        assert result.burstiness > 0.3

    def test_triadic_detection(self):
        h = Humanizer()
        text = "The system is scalable, resilient, and robust."
        result = h.humanize(text)
        assert any("triadic" in c.lower() for c in result.pass2_changes)

    def test_report_contains_burstiness(self):
        h = Humanizer()
        text = "The system works. The code runs. The tests pass. The build is clean."
        result = h.humanize(text)
        assert result.burstiness_before > 0
        assert result.burstiness_after >= 0


class TestModes:
    """Test different humanization modes."""

    def test_technical_mode_preserves_terms(self):
        h = Humanizer(mode="technical")
        text = "The API returns JSON. Latency is 50ms. Throughput is 10K req/s."
        result = h.humanize(text)
        assert "JSON" in result.humanized
        assert "50ms" in result.humanized
        assert "10K req/s" in result.humanized

    def test_marketing_mode(self):
        h = Humanizer(mode="marketing")
        text = "Our groundbreaking product revolutionizes the industry."
        result = h.humanize(text)
        assert "groundbreaking" not in result.humanized.lower()
        assert "revolutionize" not in result.humanized.lower()

    def test_resume_mode(self):
        h = Humanizer(mode="resume")
        text = "Built a scalable, reliable, and robust data pipeline."
        result = h.humanize(text)
        # Resume mode should still strip AI words
        assert "scalable" in result.humanized or "reliable" in result.humanized

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            Humanizer(mode="invalid")


class TestTaxonomy:
    """Test the AI-tell taxonomy."""

    def test_taxonomy_has_55_plus_entries(self):
        assert len(AI_TELL_TAXONOMY) >= 55

    def test_taxonomy_has_categories(self):
        categories = {e["category"] for e in AI_TELL_TAXONOMY}
        assert "Punctuation" in categories
        assert "AI Vocabulary" in categories
        assert "AI Phrase" in categories
        assert "Sycophantic" in categories
        assert "Structure" in categories

    def test_banned_words_not_empty(self):
        assert len(BANNED_WORDS) >= 40

    def test_banned_phrases_not_empty(self):
        assert len(BANNED_PHRASES) >= 20

    def test_every_taxonomy_entry_has_required_fields(self):
        for entry in AI_TELL_TAXONOMY:
            assert "id" in entry
            assert "category" in entry
            assert "name" in entry
            assert "pattern" in entry
