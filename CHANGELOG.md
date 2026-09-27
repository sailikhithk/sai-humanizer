# Changelog

All notable changes to sai-humanizer are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-09-27

### Added

- **Layer 2: Unicode watermark detection and removal** (`unicode_watermarks.py`)
  - Zero-width characters: ZWSP, ZWNJ, ZWJ, word joiner, BOM, invisible operators
  - Exotic spaces: thin, hair, narrow no-break, en/em quad, punctuation, figure
  - Bidi controls: LRE, RLE, PDF, LRO, RLO, LRI, RLI, FSI, PDI
  - Tag characters: U+E0000-E007F range (AI provenance systems)
  - Homoglyphs: Cyrillic (a, e, o, p, c, x, y) and Greek (o, p, a, e) lookalikes, replaced with ASCII
  - Variation selectors: VS1-VS16
  - Interlinear annotation characters
  - Vendor hints per finding (which provenance system likely embedded it)
  - `UnicodeWatermarkScanner.scan()` (detect only) and `.clean()` (detect + remove)
  - `inspect()` produces a human-readable per-category report

- **Layer 3: Statistical AI detection** (`statistical.py`)
  - Perplexity scoring via GPT-2 (optional, requires `torch` + `transformers`)
  - Binoculars cross-perplexity ratio (Hans et al., ICML 2024)
  - Burstiness coefficient of variation on sentence lengths
  - Green-list ratio heuristic (Kirchenbauer watermark signal)
  - Repetition ratio, type-token ratio, word-length CV
  - Ensemble `ai_probability` (0.0-1.0) with weighted signal fusion
  - Two modes: `heuristic` (default, stdlib only) and `model` (`pip install sai-humanizer[model]`)

- **Unified report** (`report.py`)
  - `HumanizationReport` aggregates all 3 layers
  - Per-layer `LayerReport` (detected, fixed, details, is_clean)
  - `summary()` one-liner, `detailed_report()` for `--report` output
  - `ai_probability`, `burstiness_before`, `burstiness_after` fields

- **CLI flags**
  - `-L/--layer` - run a specific layer (`stylistic`, `unicode`, `statistical`), repeatable
  - `--detect` - detect only, no rewrite
  - `--inspect-unicode` - Unicode watermark scan only
  - `--inspect-stat` - statistical analysis only
  - `--use-model` - enable GPT-2 perplexity/Binoculars

- **Tests**: 45 tests total (was 23)
  - `TestLayer2Unicode`: 12 tests covering zero-width, exotic spaces, bidi, homoglyphs, variation selectors, word joiner, inspect, pipeline integration
  - `TestLayer3Statistical`: 7 tests covering heuristic mode, burstiness, repetition, green-list, probability range, inspect, pipeline integration
  - `TestFullPipeline`: 6 tests for all-layers-together runs, detect-only, summary, detailed report

### Changed

- `Humanizer.__init__` accepts `layers` (list of layer names) and `use_model` (bool)
- `Humanizer.humanize()` now returns `HumanizationReport` (was `HumanizerReport`)
- `Humanizer.detect()` added for detection-only runs
- pyproject `license` moved to SPDX string format
- Removed invalid `Intended Audience :: Content Creators` trove classifier

## [0.1.0] - 2026-09-07

### Added

- **Layer 1: Stylistic pattern removal** (initial release)
  - Em-dash and en-dash replacement with hyphens
  - Curly quote straightening
  - Sycophantic opener removal ("Great question!", "Spot on!", etc.)
  - 40+ banned AI vocabulary replacements (delve, tapestry, seamless, foster, leverage, holistic, etc.)
  - 40+ banned AI phrase patterns ("ever-evolving landscape", "at its core", "in conclusion", etc.)
  - Copula avoidance fixes ("serves as" -> "is", "stands as" -> "is")
  - Filler phrase removal ("in order to" -> "to", "due to the fact that" -> "because")
  - Triadic parallelism detection (rule-of-three flagging)
  - Burstiness scoring and sentence-length variance enforcement
  - Bold-lead bullet detection, title-case heading detection

- **Modes**: `technical`, `marketing`, `resume`
- **CLI**: file/stdin/inline input, `--check` (CI gate), `--report`, `--taxonomy`, `-o` output
- **Package**: `pyproject.toml`, MIT license, zero runtime dependencies
- **Tests**: 23 tests across `TestPass1Deterministic`, `TestPass2Statistical`, `TestModes`, `TestTaxonomy`
- **Taxonomy**: 57 AI-tell patterns in `AI_TELL_TAXONOMY` data structure
