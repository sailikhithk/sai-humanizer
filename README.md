# sai-humanizer

De-AI text humanizer with 55+ pattern taxonomy. Two-pass pipeline that strips AI writing patterns and enforces human-like rhythm.

## Why

LLMs produce statistically likely text. That sameness is detectable. This tool removes the patterns that make text sound AI-generated, then checks the rhythm to make sure it reads like a human wrote it.

Built from production use at Airbnb (humanizing AI-generated content for LinkedIn, Substack, resumes) and synthesized from three open-source sources:

- `blader/humanizer` (Wikipedia "Signs of AI writing", MIT)
- `HUMANIZED_CONTENT_GUARDRAILS.md` (55+ AI-tell taxonomy, two-pass pipeline design)
- `jpeggdev/humanize-writing` (statistical burstiness approach)

## What makes this different

| Existing tool | Gap | sai-humanizer |
|---------------|-----|---------------|
| `blader/humanizer` | Markdown skill only, no CLI, no API | Python CLI + library |
| `Matt-Payne/content-humanizer` | Skill-only, no programmatic use | Importable, scriptable |
| `ChrisThoma/de-ai-text` | Claude Code only, no Python | Python 3.10+, pip installable |
| `lynote-ai/humanize-text` | Translation chain, heavy infra | Regex + statistics, zero deps |

**Your unfair advantage:** The most comprehensive pattern list (55+ entries) combined with a statistical burstiness scorer that measures whether your text reads like AI (uniform sentence lengths) or human (varied rhythm).

## Install

```bash
pip install sai-humanizer
```

Or from source:

```bash
git clone https://github.com/sailikhithk/sai-humanizer.git
cd sai-humanizer
pip install -e .
```

## Usage

### CLI

```bash
# Humanize a file
sai-humanizer draft.md

# Humanize inline text
sai-humanizer -t "Let's delve into the seamless integration of our groundbreaking platform."

# Pipe from stdin
cat draft.md | sai-humanizer

# Show detailed change report
sai-humanizer -r draft.md

# Check only (exit 1 if AI patterns found, no rewrite)
sai-humanizer --check draft.md

# Use marketing mode (more aggressive)
sai-humanizer -m marketing -t "Our product revolutionizes the industry."

# Use resume mode (preserve action verbs, strip filler)
sai-humanizer -m resume resume.tex

# Print the full 55+ AI-tell taxonomy
sai-humanizer --taxonomy

# Write to file
sai-humanizer draft.md -o draft-humanized.md
```

### Python API

```python
from sai_humanizer import Humanizer

h = Humanizer(mode="technical")
report = h.humanize("Let's delve into the seamless integration.")

print(report.humanized)
# "Let's explore, examine, inspect, look at the clean, direct, smooth, integrated integration."

print(report.total_changes)
# 2

print(report.pass1_changes)
# ['Replaced 1x AI word 'delve' -> 'explore, examine, inspect, look at'',
#  'Replaced 1x AI word 'seamless' -> 'clean, direct, smooth, integrated'']

print(f"Burstiness: {report.burstiness_before} -> {report.burstiness_after}")
```

### Burstiness scoring only

```python
from sai_humanizer import BurstinessScorer

scorer = BurstinessScorer()
result = scorer.score("The system works. The code runs. The tests pass. The build is clean.")

print(result.burstiness)     # 0.12 (AI-like, below 0.35)
print(result.is_ai_like)     # True
print(result.suggestion)     # "Burstiness 0.12 is below 0.35..."
```

## The two-pass pipeline

```
[Input text]
     |
     v
+-------------------------------------------+
| Pass 1: Deterministic (regex-based)       |
| - Replace em-dashes and en-dashes         |
| - Replace curly quotes with straight      |
| - Remove sycophantic openers              |
| - Replace 55+ banned AI vocabulary words  |
| - Replace 40+ banned AI phrases           |
| - Fix copula avoidance (serves as -> is)  |
| - Remove filler phrases (in order to)     |
+-------------------------------------------+
     |
     v
+-------------------------------------------+
| Pass 2: Statistical (structural)          |
| - Score burstiness (sentence length CV)   |
| - Flag triadic parallelism (rule of 3)    |
| - Detect bold-lead monotony in lists      |
| - Detect title case in headings           |
| - Suggest sentence splits for variance    |
+-------------------------------------------+
     |
     v
[Humanized output + change report]
```

## Modes

| Mode | Behavior | Use case |
|------|----------|----------|
| `technical` | Conservative. Preserves technical terms and metrics. | Docs, README, blog posts |
| `marketing` | Aggressive. Strips all AI patterns, punchier output. | Landing pages, ads, social |
| `resume` | Preserves action verbs and metrics. Strips filler. | Resumes, cover letters |

## The 55+ AI-tell taxonomy

Run `sai-humanizer --taxonomy` to see all patterns. Categories:

| Category | Count | Examples |
|----------|------:|---------|
| Punctuation | 6 | Em-dashes, en-dashes, curly quotes, emoji, bold-lead |
| AI Vocabulary | 22 | delve, tapestry, testament, seamless, foster, leverage, holistic |
| AI Phrase | 15 | "ever-evolving landscape", "at its core", "in conclusion" |
| Sycophantic | 5 | "Great question!", "You're absolutely right", "I hope this helps" |
| Structure | 6 | Triadic parallelism, copula avoidance, false ranges, title case |
| Filler | 3 | "in order to", "at this point in time", "has the ability to" |

## Check mode for CI

Use `--check` in CI pipelines to block AI-generated content from being merged:

```bash
# In a pre-commit hook or CI step
sai-humanizer --check docs/**/*.md
# Exit 0: clean
# Exit 1: AI patterns found (prints violations to stderr)
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
