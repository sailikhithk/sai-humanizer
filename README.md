# sai-humanizer

Three-layer de-AI text humanizer. Detects and removes AI writing patterns, invisible Unicode watermarks, and statistical AI signals.

## Why

LLMs leave three kinds of traces in text:

1. **Stylistic patterns** - AI vocabulary (delve, seamless, tapestry), em-dashes, sycophantic openers, filler phrases, low burstiness
2. **Invisible Unicode watermarks** - Zero-width characters, exotic spaces, bidi controls, tag characters, homoglyphs embedded by Claude, Gemini/SynthID, OpenAI
3. **Statistical signals** - Low perplexity, low burstiness, high green-list token ratio, uniform word lengths

sai-humanizer detects all three and fixes layers 1 and 2 deterministically. Layer 3 provides detection and guidance.

Built from production use at Airbnb and synthesized from:
- `blader/humanizer` (Wikipedia "Signs of AI writing", MIT)
- `guillaumemeyer/watermarks-remover` (940 stars, multi-vendor Unicode stripping)
- `cyzanfar/text-watermark-remover` (dewatermark, detector-scoped mitigation)
- `PyModel/watermark-remover` (4-channel provenance removal)
- `virajshoor/ADAFAI` (multi-signal detection: stylometry, Binoculars, green-list)
- `umairinayat/AI-Detection` (GPTZero-style perplexity + burstiness)
- `HUMANIZED_CONTENT_GUARDRAILS.md` (57+ AI-tell taxonomy)

## What makes this different

| Existing tool | Layer 1 | Layer 2 | Layer 3 | CLI | Library |
|---------------|:-------:|:-------:|:-------:|:---:|:-------:|
| `blader/humanizer` | 29 patterns | - | - | - | - |
| `guillaumemeyer/watermarks-remover` | - | Unicode + files | - | agent | - |
| `cyzanfar/dewatermark` | - | Unicode | detector-guided | yes | yes |
| `PyModel/watermark-remover` | - | Unicode + images | - | yes | yes |
| `virajshoor/ADAFAI` | stylometry | Unicode | Binoculars | - | - |
| **sai-humanizer** | **57 patterns** | **Unicode** | **perplexity + Binoculars + burstiness** | **yes** | **yes** |

sai-humanizer is the only tool that combines all three layers in a single pip-installable package with zero required dependencies.

## Install

```bash
pip install sai-humanizer
```

For model-based statistical detection (optional):

```bash
pip install sai-humanizer[model]
# or manually: pip install torch transformers
```

From source:

```bash
git clone https://github.com/sailikhithk/sai-humanizer.git
cd sai-humanizer
pip install -e .
```

## Usage

### CLI

```bash
# Humanize a file (all 3 layers)
sai-humanizer draft.md

# Humanize inline text
sai-humanizer -t "Let's delve into the seamless integration."

# Pipe from stdin
cat draft.md | sai-humanizer

# Show detailed 3-layer report
sai-humanizer -r draft.md

# Check only (exit 1 if any AI patterns found)
sai-humanizer --check draft.md

# Detect only (print findings, no rewrite)
sai-humanizer --detect draft.md

# Run specific layer(s) only
sai-humanizer -L stylistic -t "Let's delve into the tapestry."
sai-humanizer -L unicode -L statistical draft.md

# Inspect Unicode watermarks only
sai-humanizer --inspect-unicode draft.md

# Inspect statistical signals only
sai-humanizer --inspect-stat draft.md

# Use GPT-2 model for perplexity/Binoculars (requires torch)
sai-humanizer --use-model --inspect-stat draft.md

# Marketing mode (aggressive)
sai-humanizer -m marketing -t "Our product revolutionizes the industry."

# Resume mode (preserve action verbs, strip filler)
sai-humanizer -m resume resume.tex

# Print the full 57+ AI-tell taxonomy
sai-humanizer --taxonomy
```

### Python API

```python
from sai_humanizer import Humanizer

# Full 3-layer pipeline
h = Humanizer(mode="technical")
report = h.humanize("Let's delve into the seamless integration.")

print(report.humanized)
print(report.total_detected)   # e.g. 5
print(report.ai_probability)   # e.g. 0.72
print(report.summary())        # one-line summary
print(report.detailed_report())  # full 3-layer breakdown

# Detection only (no modification)
report = h.detect("Some text to check.")
print(report.is_clean)         # True/False

# Run specific layers only
h = Humanizer(layers=["unicode"])
report = h.humanize("Hello\u200bWorld")
# Unicode watermarks stripped, stylistic patterns untouched
```

### Layer 2: Unicode watermark scanning

```python
from sai_humanizer import UnicodeWatermarkScanner

scanner = UnicodeWatermarkScanner()
result = scanner.scan("Hello\u200bWorld\u202eTest")
print(result.total_found)       # 2
print(result.categories_found)  # {"zero-width", "bidi"}

cleaned = scanner.clean("Hello\u200bWorld")
print(cleaned.cleaned)          # "HelloWorld"
```

### Layer 3: Statistical detection

```python
from sai_humanizer import StatisticalDetector

# Heuristic mode (no torch needed)
detector = StatisticalDetector(use_model=False)
result = detector.scan("The system works. The code runs. The tests pass.")
print(result.ai_probability)    # e.g. 0.65
print(result.method)            # "heuristic"

# Model mode (requires torch + transformers)
detector = StatisticalDetector(use_model=True)
result = detector.scan("Some text to analyze.")
print(result.perplexity)        # e.g. 15.2
print(result.binoculars)        # e.g. 0.82
```

## The three-layer pipeline

```
[Input text]
     |
     v
+-------------------------------------------+
| Layer 1: Stylistic (deterministic regex)  |
| - Replace em-dashes and en-dashes         |
| - Replace curly quotes                    |
| - Remove sycophantic openers              |
| - Replace 57+ banned AI vocabulary words  |
| - Replace 40+ banned AI phrases           |
| - Fix copula avoidance (serves as -> is)  |
| - Remove filler phrases                   |
| - Detect triadic parallelism              |
| - Score and enforce burstiness            |
| - Detect bold-lead monotony               |
| - Detect title case headings              |
+-------------------------------------------+
     |
     v
+-------------------------------------------+
| Layer 2: Unicode Watermarks (deterministic)|
| - Zero-width chars (ZWSP, ZWNJ, ZWJ)      |
| - Exotic spaces (thin, hair, narrow NBSP) |
| - Bidi controls (LRE, RLE, PDF, LRO, RLO) |
| - Tag characters (U+E0000-E007F)          |
| - Homoglyphs (Cyrillic, Greek lookalikes) |
| - Variation selectors (VS1-VS16)          |
| - Interlinear annotation chars            |
| - Vendor hints: Claude, Gemini, OpenAI    |
+-------------------------------------------+
     |
     v
+-------------------------------------------+
| Layer 3: Statistical (heuristic or model) |
| - Perplexity (GPT-2, requires torch)      |
| - Binoculars score (cross-perplexity)     |
| - Burstiness (sentence length CV)         |
| - Green-list ratio (Kirchenbauer)         |
| - Repetition ratio                        |
| - Type-token ratio (vocabulary diversity) |
| - Word length coefficient of variation    |
| - Ensemble AI probability (0.0-1.0)       |
+-------------------------------------------+
     |
     v
[Humanized output + 3-layer report]
```

## Modes

| Mode | Behavior | Use case |
|------|----------|----------|
| `technical` | Conservative. Preserves technical terms and metrics. | Docs, README, blog posts |
| `marketing` | Aggressive. Strips all AI patterns, punchier output. | Landing pages, ads, social |
| `resume` | Preserves action verbs and metrics. Strips filler. | Resumes, cover letters |

## The 57+ AI-tell taxonomy (Layer 1)

Run `sai-humanizer --taxonomy` to see all patterns. Categories:

| Category | Count | Examples |
|----------|------:|---------|
| Punctuation | 6 | Em-dashes, en-dashes, curly quotes, emoji, bold-lead |
| AI Vocabulary | 22 | delve, tapestry, testament, seamless, foster, leverage, holistic |
| AI Phrase | 15 | "ever-evolving landscape", "at its core", "in conclusion" |
| Sycophantic | 5 | "Great question!", "You're absolutely right", "I hope this helps" |
| Structure | 6 | Triadic parallelism, copula avoidance, false ranges, title case |
| Filler | 3 | "in order to", "at this point in time", "has the ability to" |

## Unicode watermark coverage (Layer 2)

| Category | Characters | Vendors |
|----------|-----------|---------|
| Zero-width | ZWSP, ZWNJ, ZWJ, WJ, invisible operators, BOM | Claude, various |
| Exotic spaces | NBSP, thin, hair, narrow NBSP, en/em quad | Gemini, various |
| Bidi controls | LRE, RLE, PDF, LRO, RLO, LRI, RLI, FSI, PDI | Various |
| Tag chars | U+E0000-E007F | AI provenance systems |
| Homoglyphs | Cyrillic a/e/o/p/c/x/y, Greek o/p/a/e | Spoofing |
| Variation selectors | VS1-VS16 | Various |
| Interlinear | Anchor, separator, terminator | Rare |

## Statistical signals (Layer 3)

| Signal | What it measures | Method | Needs torch |
|--------|-----------------|--------|:-----------:|
| Perplexity | Text predictability (low = AI-like) | GPT-2 | yes |
| Binoculars | Cross-perplexity ratio | Observer/performer models | yes |
| Burstiness | Sentence length variance (low = AI-like) | Coefficient of variation | no |
| Green-list | Kirchenbauer watermark token ratio | Heuristic token analysis | no |
| Repetition | Vocabulary repetition ratio | Unique/total word ratio | no |
| TTR | Type-token ratio (low = AI-like) | Unique words / total words | no |
| Word length CV | Word length uniformity (low = AI-like) | Coefficient of variation | no |

## Check mode for CI

```bash
# Block AI-generated content from being merged
sai-humanizer --check docs/**/*.md
# Exit 0: clean
# Exit 1: AI patterns found (prints all findings to stderr)
```

## Testing

```bash
pip install pytest
pytest tests/ -v
```

## License

MIT

## Author

Sai Likhith Kanuparthi ([github.com/sailikhithk](https://github.com/sailikhithk))

Built for the reliability layer of production AI in regulated industries.
