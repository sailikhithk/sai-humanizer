"""Tests for sai-humanizer three-layer pipeline."""

import pytest
from sai_humanizer.pipeline import Humanizer
from sai_humanizer.patterns import BANNED_WORDS, BANNED_PHRASES, AI_TELL_TAXONOMY
from sai_humanizer.stats import BurstinessScorer
from sai_humanizer.unicode_watermarks import UnicodeWatermarkScanner
from sai_humanizer.statistical import StatisticalDetector
from sai_humanizer.report import HumanizationReport


class TestLayer1Stylistic:
    """Layer 1: deterministic pattern removal."""

    def test_em_dash_removal(self):
        h = Humanizer(layers=["stylistic"])
        text = "The tool is fast\u2014reliable\u2014and cheap."
        result = h.humanize(text)
        assert "\u2014" not in result.humanized
        assert "\u2013" not in result.humanized
        assert result.layers["stylistic"].detected > 0

    def test_banned_word_removal(self):
        h = Humanizer(layers=["stylistic"])
        text = "Let's delve into the tapestry of data."
        result = h.humanize(text)
        assert "delve" not in result.humanized.lower()
        assert "tapestry" not in result.humanized.lower()
        assert result.layers["stylistic"].detected >= 2

    def test_seamless_removal(self):
        h = Humanizer(layers=["stylistic"])
        text = "The integration is seamless and robust."
        result = h.humanize(text)
        assert "seamless" not in result.humanized.lower()

    def test_sycophantic_opener_removal(self):
        h = Humanizer(layers=["stylistic"])
        text = "Great question! The answer is simple."
        result = h.humanize(text)
        assert not result.humanized.lower().startswith("great question")

    def test_filler_phrase_removal(self):
        h = Humanizer(layers=["stylistic"])
        text = "In order to run the test, you need pytest."
        result = h.humanize(text)
        assert "in order to" not in result.humanized.lower()

    def test_copula_avoidance_fix(self):
        h = Humanizer(layers=["stylistic"])
        text = "The tool serves as a proxy for the API."
        result = h.humanize(text)
        assert "serves as" not in result.humanized.lower()

    def test_curly_quotes_replaced(self):
        h = Humanizer(layers=["stylistic"])
        text = "He said \u201chello\u201d and left."
        result = h.humanize(text)
        assert "\u201c" not in result.humanized
        assert '"' in result.humanized

    def test_banned_phrase_removal(self):
        h = Humanizer(layers=["stylistic"])
        text = "It is important to note that the system works."
        result = h.humanize(text)
        assert "it is important to note" not in result.humanized.lower()

    def test_clean_text_no_changes(self):
        h = Humanizer(layers=["stylistic"])
        text = "The function returns a list of tuples. It filters by name and sorts by date. Fast. We benchmarked it at 50ms per call."
        result = h.humanize(text)
        assert result.layers["stylistic"].is_clean

    def test_multiple_patterns_at_once(self):
        h = Humanizer(layers=["stylistic"])
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
        assert result.layers["stylistic"].detected >= 7

    def test_detect_only_does_not_modify(self):
        h = Humanizer(layers=["stylistic"])
        text = "Let's delve into the seamless integration."
        result = h.detect(text)
        assert result.humanized == text
        assert result.layers["stylistic"].detected >= 2
        assert result.layers["stylistic"].fixed == 0


class TestLayer2Unicode:
    """Layer 2: Unicode watermark detection and removal."""

    def test_zero_width_space_detected(self):
        scanner = UnicodeWatermarkScanner()
        text = "Hello\u200bWorld"
        result = scanner.scan(text)
        assert result.total_found == 1
        assert "zero-width" in result.categories_found

    def test_zero_width_space_removed(self):
        scanner = UnicodeWatermarkScanner()
        text = "Hello\u200bWorld"
        result = scanner.clean(text)
        assert result.cleaned == "HelloWorld"
        assert result.total_found == 1

    def test_multiple_zero_width_chars(self):
        scanner = UnicodeWatermarkScanner()
        text = "A\u200bB\u200cC\u200dD"
        result = scanner.clean(text)
        assert result.cleaned == "ABCD"
        assert result.total_found == 3

    def test_exotic_space_detected(self):
        scanner = UnicodeWatermarkScanner()
        text = "Hello\u2009World"  # thin space
        result = scanner.scan(text)
        assert result.total_found == 1
        assert "exotic-space" in result.categories_found

    def test_bidi_control_detected(self):
        scanner = UnicodeWatermarkScanner()
        text = "Hello\u202eWorld"  # RIGHT-TO-LEFT OVERRIDE
        result = scanner.scan(text)
        assert result.total_found == 1
        assert "bidi" in result.categories_found

    def test_homoglyph_detected(self):
        scanner = UnicodeWatermarkScanner()
        text = "Cyrillic: \u0430nd"  # Cyrillic 'a'
        result = scanner.scan(text)
        assert any(f.category == "homoglyph" for f in result.findings)

    def test_homoglyph_replaced(self):
        scanner = UnicodeWatermarkScanner()
        text = "\u0430pple"  # Cyrillic a + pple
        result = scanner.clean(text, remove_homoglyphs=True)
        assert result.cleaned == "apple"

    def test_clean_text_no_findings(self):
        scanner = UnicodeWatermarkScanner()
        text = "Hello World, this is clean text."
        result = scanner.scan(text)
        assert result.is_clean
        assert result.total_found == 0

    def test_variation_selector_detected(self):
        scanner = UnicodeWatermarkScanner()
        text = "A\ufe0fB"  # VS16
        result = scanner.scan(text)
        assert result.total_found == 1
        assert "variation-selector" in result.categories_found

    def test_word_joiner_detected(self):
        scanner = UnicodeWatermarkScanner()
        text = "Hello\u2060World"  # WORD JOINER (Claude)
        result = scanner.scan(text)
        assert result.total_found == 1
        assert any("Claude" in f.vendor_hint for f in result.findings)

    def test_inspect_returns_readable_report(self):
        scanner = UnicodeWatermarkScanner()
        text = "A\u200bB\u202cC"
        report = scanner.inspect(text)
        assert "2 artifact(s)" in report
        assert "zero-width" in report
        assert "bidi" in report

    def test_pipeline_layer2_integration(self):
        h = Humanizer(layers=["unicode"])
        text = "Hello\u200bWorld\u202eTest"
        result = h.humanize(text)
        assert "\u200b" not in result.humanized
        assert "\u202e" not in result.humanized
        assert result.layers["unicode"].detected == 2
        assert result.layers["unicode"].fixed == 2


class TestLayer3Statistical:
    """Layer 3: statistical AI text detection."""

    def test_heuristic_mode_no_torch(self):
        detector = StatisticalDetector(use_model=False)
        result = detector.scan("The system works. The code runs. The tests pass.")
        assert result.method == "heuristic"
        assert len(result.findings) > 0

    def test_burstiness_signal_in_heuristic(self):
        detector = StatisticalDetector(use_model=False)
        text = "The system works. The code runs. The tests pass. The build is clean."
        result = detector.scan(text)
        burst_finding = [f for f in result.findings if f.signal == "burstiness"]
        assert len(burst_finding) == 1
        assert burst_finding[0].is_ai_like

    def test_repetition_signal(self):
        detector = StatisticalDetector(use_model=False)
        text = "the the the the the the the the the the test"
        result = detector.scan(text)
        rep_finding = [f for f in result.findings if f.signal == "repetition"]
        assert len(rep_finding) == 1

    def test_green_list_signal(self):
        detector = StatisticalDetector(use_model=False)
        result = detector.scan("This is a test of the system.")
        green_finding = [f for f in result.findings if f.signal == "green_list"]
        assert len(green_finding) == 1

    def test_ai_probability_range(self):
        detector = StatisticalDetector(use_model=False)
        result = detector.scan("The system works. The code runs. The tests pass. The build is clean.")
        assert 0.0 <= result.ai_probability <= 1.0

    def test_inspect_returns_readable_report(self):
        detector = StatisticalDetector(use_model=False)
        text = "The system works. The code runs. The tests pass. The build is clean."
        report = detector.inspect(text)
        assert "heuristic" in report
        assert "burstiness" in report

    def test_pipeline_layer3_integration(self):
        h = Humanizer(layers=["statistical"], use_model=False)
        text = "The system works. The code runs. The tests pass. The build is clean."
        result = h.humanize(text)
        assert "statistical" in result.layers
        assert result.ai_probability > 0.0


class TestFullPipeline:
    """Test all 3 layers running together."""

    def test_all_layers_run(self):
        h = Humanizer(use_model=False)
        text = "Great question! Let's delve into the seamless integration.\u200bIt is important to note that the tool works."
        result = h.humanize(text)
        assert "stylistic" in result.layers
        assert "unicode" in result.layers
        assert "statistical" in result.layers
        assert result.total_detected > 0

    def test_report_has_all_fields(self):
        h = Humanizer(use_model=False)
        text = "Let's delve into the seamless integration. It is important to note that the tool works. Fast. We measured it."
        result = h.humanize(text)
        assert result.original == text
        assert len(result.humanized) > 0
        assert result.total_detected > 0

    def test_detect_mode_all_layers(self):
        h = Humanizer(use_model=False)
        text = "Let's delve into the seamless integration.\u200b"
        result = h.detect(text)
        assert result.humanized == text  # unchanged
        assert result.total_detected > 0
        assert result.total_fixed == 0

    def test_summary_string(self):
        h = Humanizer(use_model=False)
        text = "Let's delve into the seamless integration."
        result = h.humanize(text)
        s = result.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_detailed_report_string(self):
        h = Humanizer(use_model=False)
        text = "Let's delve into the seamless integration."
        result = h.humanize(text)
        r = result.detailed_report()
        assert "STYLISTIC" in r
        assert "UNICODE" in r
        assert "STATISTICAL" in r

    def test_clean_text_all_layers(self):
        h = Humanizer(use_model=False)
        text = "The function returns a list of tuples. It filters by name and sorts by date. Fast. We benchmarked it at 50ms per call."
        result = h.humanize(text)
        assert result.is_clean or result.total_detected <= 1


class TestModes:
    """Test different humanization modes."""

    def test_technical_mode_preserves_terms(self):
        h = Humanizer(mode="technical", layers=["stylistic"])
        text = "The API returns JSON. Latency is 50ms. Throughput is 10K req/s."
        result = h.humanize(text)
        assert "JSON" in result.humanized
        assert "50ms" in result.humanized

    def test_marketing_mode(self):
        h = Humanizer(mode="marketing", layers=["stylistic"])
        text = "Our groundbreaking product revolutionizes the industry."
        result = h.humanize(text)
        assert "groundbreaking" not in result.humanized.lower()
        assert "revolutionize" not in result.humanized.lower()

    def test_resume_mode(self):
        h = Humanizer(mode="resume", layers=["stylistic"])
        text = "Built a scalable, reliable, and robust data pipeline."
        result = h.humanize(text)
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
