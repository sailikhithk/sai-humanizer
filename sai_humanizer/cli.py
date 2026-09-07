#!/usr/bin/env python3
"""sai-humanizer CLI: De-AI text humanizer.

Usage:
  sai-humanizer [options] [file]
  sai-humanizer -t "text to humanize"
  cat file.md | sai-humanizer

Options:
  -t, --text       Text to humanize (inline)
  -m, --mode       Mode: technical, marketing, resume (default: technical)
  -r, --report     Show detailed change report
  -o, --output     Output file path (default: stdout)
  --check          Check only (exit 1 if AI patterns found, no rewrite)
  --taxonomy       Print the full 55+ AI-tell taxonomy
"""

import argparse
import sys

from sai_humanizer.pipeline import Humanizer
from sai_humanizer.patterns import AI_TELL_TAXONOMY
from sai_humanizer.stats import BurstinessScorer


def main():
    parser = argparse.ArgumentParser(
        prog="sai-humanizer",
        description="De-AI text humanizer with 55+ pattern taxonomy. Two-pass pipeline.",
    )
    parser.add_argument("file", nargs="?", help="Input file (reads from stdin if omitted)")
    parser.add_argument("-t", "--text", help="Text to humanize (inline)")
    parser.add_argument(
        "-m", "--mode",
        choices=["technical", "marketing", "resume"],
        default="technical",
        help="Humanization mode (default: technical)",
    )
    parser.add_argument("-r", "--report", action="store_true", help="Show detailed change report")
    parser.add_argument("-o", "--output", help="Output file path (default: stdout)")
    parser.add_argument("--check", action="store_true", help="Check only, no rewrite. Exit 1 if AI patterns found.")
    parser.add_argument("--taxonomy", action="store_true", help="Print the full 55+ AI-tell taxonomy")

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

    humanizer = Humanizer(mode=args.mode)

    if args.check:
        report = humanizer.humanize(text)
        if report.total_changes > 0:
            print(f"FAIL: {report.total_changes} AI patterns detected", file=sys.stderr)
            for change in report.pass1_changes:
                print(f"  [pass1] {change}", file=sys.stderr)
            for change in report.pass2_changes:
                print(f"  [pass2] {change}", file=sys.stderr)
            sys.exit(1)
        else:
            print("OK: No AI patterns detected.")
            sys.exit(0)

    report = humanizer.humanize(text)

    if args.report:
        print("=" * 60, file=sys.stderr)
        print(f"MODE: {args.mode}", file=sys.stderr)
        print(f"CHANGES: {report.total_changes}", file=sys.stderr)
        print(f"BURSTINESS: {report.burstiness_before} -> {report.burstiness_after}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        for change in report.pass1_changes:
            print(f"  [pass1] {change}", file=sys.stderr)
        for change in report.pass2_changes:
            print(f"  [pass2] {change}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(file=sys.stderr)

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
    categories = {}
    for entry in AI_TELL_TAXONOMY:
        cat = entry["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(entry)

    for cat, entries in categories.items():
        print(f"\n## {cat} ({len(entries)} patterns)")
        for e in entries:
            print(f"  {e['id']:12s} {e['name']:40s} pattern: {e['pattern']}")


if __name__ == "__main__":
    main()
