#!/usr/bin/env python3
"""sai-humanizer CLI: Three-layer de-AI text humanizer.

Usage:
  sai-humanizer [options] [file]
  sai-humanizer -t "text to humanize"
  cat file.md | sai-humanizer

Options:
  -t, --text       Text to humanize (inline)
  -m, --mode       Mode: technical, marketing, resume (default: technical)
  -L, --layer      Run specific layer(s): stylistic, unicode, statistical
                   (default: all three. Can be specified multiple times)
  -r, --report     Show detailed change report
  -o, --output     Output file path (default: stdout)
  --check          Check only (exit 1 if AI patterns found, no rewrite)
  --detect         Detect only, print findings, no rewrite
  --taxonomy       Print the full 57+ AI-tell taxonomy
  --inspect-unicode  Inspect text for Unicode watermarks only
  --inspect-stat   Inspect text for statistical AI signals only
  --use-model      Use GPT-2 model for perplexity/Binoculars (requires torch)
"""

import argparse
import sys

from sai_humanizer.pipeline import Humanizer
from sai_humanizer.patterns import AI_TELL_TAXONOMY
from sai_humanizer.unicode_watermarks import UnicodeWatermarkScanner
from sai_humanizer.statistical import StatisticalDetector


def main():
    parser = argparse.ArgumentParser(
        prog="sai-humanizer",
        description="Three-layer de-AI text humanizer. Detects and removes stylistic patterns, Unicode watermarks, and statistical AI signals.",
    )
    parser.add_argument("file", nargs="?", help="Input file (reads from stdin if omitted)")
    parser.add_argument("-t", "--text", help="Text to humanize (inline)")
    parser.add_argument(
        "-m", "--mode",
        choices=["technical", "marketing", "resume"],
        default="technical",
        help="Humanization mode (default: technical)",
    )
    parser.add_argument(
        "-L", "--layer",
        choices=["stylistic", "unicode", "statistical"],
        action="append",
        help="Run specific layer(s). Can be specified multiple times. Default: all three.",
    )
    parser.add_argument("-r", "--report", action="store_true", help="Show detailed change report")
    parser.add_argument("-o", "--output", help="Output file path (default: stdout)")
    parser.add_argument("--check", action="store_true", help="Check only, no rewrite. Exit 1 if AI patterns found.")
    parser.add_argument("--detect", action="store_true", help="Detect only, print findings, no rewrite.")
    parser.add_argument("--taxonomy", action="store_true", help="Print the full 57+ AI-tell taxonomy")
    parser.add_argument("--inspect-unicode", action="store_true", help="Inspect text for Unicode watermarks only")
    parser.add_argument("--inspect-stat", action="store_true", help="Inspect text for statistical AI signals only")
    parser.add_argument("--use-model", action="store_true", help="Use GPT-2 model for perplexity/Binoculars (requires torch)")

    args = parser.parse_args()

    if args.taxonomy:
        print_taxonomy()
        return

    # Get input text
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    if not text.strip():
        print("No input text provided.", file=sys.stderr)
        sys.exit(1)

    # Specialized inspection modes
    if args.inspect_unicode:
        scanner = UnicodeWatermarkScanner()
        print(scanner.inspect(text))
        sys.exit(0 if scanner.scan(text).is_clean else 1)

    if args.inspect_stat:
        detector = StatisticalDetector(use_model=args.use_model)
        print(detector.inspect(text))
        sys.exit(0 if not detector.scan(text).is_ai_like else 1)

    # Determine layers
    layers = args.layer if args.layer else ["stylistic", "unicode", "statistical"]

    humanizer = Humanizer(mode=args.mode, layers=layers, use_model=args.use_model)

    if args.check or args.detect:
        report = humanizer.detect(text)
        if report.total_detected > 0:
            print(f"FAIL: {report.total_detected} AI patterns detected", file=sys.stderr)
            for layer_name, layer in report.layers.items():
                if not layer.is_clean:
                    print(f"\n  [{layer_name.upper()}] {layer.detected} detected:", file=sys.stderr)
                    for detail in layer.details:
                        print(f"    - {detail}", file=sys.stderr)
            if args.check:
                sys.exit(1)
        else:
            print("OK: No AI patterns detected across all layers.")
            sys.exit(0)

    report = humanizer.humanize(text)

    if args.report:
        print(report.detailed_report(), file=sys.stderr)

    output = report.humanized
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(output)


def print_taxonomy():
    """Print the full AI-tell taxonomy."""
    print(f"sai-humanizer AI-Tell Taxonomy ({len(AI_TELL_TAXONOMY)} patterns)\n")
    print("Layer 1: Stylistic Patterns\n")
    categories = {}
    for entry in AI_TELL_TAXONOMY:
        cat = entry["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(entry)

    for cat, entries in categories.items():
        print(f"\n  {cat} ({len(entries)} patterns)")
        for e in entries:
            print(f"    {e['id']:12s} {e['name']:40s} pattern: {e['pattern']}")

    print(f"\n\nLayer 2: Unicode Watermarks")
    print(f"  Detects: zero-width chars, exotic spaces, bidi controls,")
    print(f"  tag characters, homoglyphs, variation selectors")
    print(f"  Vendors: Claude, Gemini/SynthID, OpenAI, Kirchenbauer")

    print(f"\n\nLayer 3: Statistical Detection")
    print(f"  Signals: perplexity (GPT-2), Binoculars score, burstiness,")
    print(f"  green-list ratio, repetition, type-token ratio, word length CV")
    print(f"  Methods: heuristic (default) or model-based (with --use-model)")


if __name__ == "__main__":
    main()
