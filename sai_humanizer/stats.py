"""Statistical pass: burstiness scoring and sentence length variance.

AI text tends toward uniform sentence lengths (low burstiness).
Human text varies: short punches mixed with longer flowing sentences.

Burstiness = coefficient of variation of sentence lengths.
Human writing typically has burstiness > 0.5.
AI writing typically has burstiness < 0.3.
"""

import re
import statistics
from dataclasses import dataclass


@dataclass
class BurstinessResult:
    burstiness: float
    mean_length: float
    stdev_length: float
    sentence_lengths: list[int]
    is_ai_like: bool
    suggestion: str | None


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, handling common abbreviations."""
    abbreviations = {
        "Mr.": "Mr<DOT>",
        "Mrs.": "Mrs<DOT>",
        "Dr.": "Dr<DOT>",
        "Ph.D.": "Ph<DOT>D<DOT>",
        "vs.": "vs<DOT>",
        "e.g.": "e<DOT>g<DOT>",
        "i.e.": "i<DOT>e<DOT>",
        "Inc.": "Inc<DOT>",
        "Ltd.": "Ltd<DOT>",
        "Corp.": "Corp<DOT>",
        "U.S.": "U<DOT>S<DOT>",
        "U.K.": "U<DOT>K<DOT>",
    }
    protected = text
    for orig, replacement in abbreviations.items():
        protected = protected.replace(orig, replacement)

    sentences = re.split(r'(?<=[.!?])\s+', protected)

    result = []
    for s in sentences:
        for orig, replacement in abbreviations.items():
            s = s.replace(replacement, orig)
        s = s.strip()
        if s:
            result.append(s)
    return result


class BurstinessScorer:
    """Score text burstiness and detect AI-like uniformity."""

    def score(self, text: str) -> BurstinessResult:
        sentences = split_sentences(text)
        if len(sentences) < 3:
            return BurstinessResult(
                burstiness=0.0,
                mean_length=0.0,
                stdev_length=0.0,
                sentence_lengths=[len(s) for s in sentences],
                is_ai_like=False,
                suggestion="Not enough sentences to score burstiness.",
            )

        lengths = [len(s) for s in sentences]
        mean_len = statistics.mean(lengths)
        stdev_len = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
        burstiness = stdev_len / mean_len if mean_len > 0 else 0.0

        is_ai = burstiness < 0.35
        suggestion = None
        if is_ai:
            suggestion = (
                f"Burstiness {burstiness:.2f} is below 0.35 (AI-like uniformity). "
                f"Vary sentence lengths: mix short punches ({int(mean_len * 0.4)} chars) "
                f"with longer flowing sentences ({int(mean_len * 1.6)} chars)."
            )

        return BurstinessResult(
            burstiness=round(burstiness, 3),
            mean_length=round(mean_len, 1),
            stdev_length=round(stdev_len, 1),
            sentence_lengths=lengths,
            is_ai_like=is_ai,
            suggestion=suggestion,
        )

    def enforce_variance(self, text: str, target_burstiness: float = 0.5) -> str:
        """Suggest sentence splits to increase burstiness.

        Lightweight heuristic: identifies the longest sentences
        and suggests splitting them at natural break points.
        """
        sentences = split_sentences(text)
        result = self.score(text)
        if result.burstiness >= target_burstiness:
            return text

        output = []
        for s in sentences:
            if len(s) > result.mean_length * 1.5:
                split_point = s.find(", ")
                if split_point == -1:
                    split_point = s.find("; ")
                if split_point > 0 and split_point < len(s) - 20:
                    output.append(s[:split_point + 1])
                    output.append(s[split_point + 2:])
                else:
                    output.append(s)
            else:
                output.append(s)

        return " ".join(output)
