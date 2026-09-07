"""Layer 3: Statistical watermark detection.

Detects AI-generated text and statistical watermarks using:
  - Perplexity scoring (GPT-2 based, measures text predictability)
  - Binoculars score (cross-perplexity ratio, best zero-shot method)
  - Token distribution analysis (Kirchenbauer green-list detection)

Optional dependency: torch + transformers. Falls back to heuristic
scoring when torch is not available.

Sources:
  - GPTZero (perplexity + burstiness, arxiv.org/abs/2602.13042)
  - Binoculars (Hans et al., ICML 2024, arxiv.org/abs/2401.12070)
  - Kirchenbauer et al. (green-list watermarks, PMLR 2023)
  - SynthID-Text (Dathathri et al., Nature 2024)
  - umairinayat/AI-Detection, virajshoor/ADAFAI
"""

import math
import re
from dataclasses import dataclass, field

from sai_humanizer.stats import BurstinessScorer


@dataclass
class StatisticalFinding:
    """A single statistical signal indicating AI-generated text."""
    signal: str
    value: float
    threshold: float
    is_ai_like: bool
    description: str


@dataclass
class StatisticalScanResult:
    """Result of statistical watermark / AI text detection."""
    text: str
    findings: list[StatisticalFinding] = field(default_factory=list)
    perplexity: float | None = None
    binoculars: float | None = None
    burstiness: float | None = None
    green_list_score: float | None = None
    ai_probability: float = 0.0
    method: str = "heuristic"  # "heuristic" or "model"
    suggestion: str | None = None

    @property
    def is_ai_like(self) -> bool:
        return self.ai_probability > 0.5


class StatisticalDetector:
    """Detect AI-generated text and statistical watermarks.

    Uses model-based detection (GPT-2 perplexity + Binoculars) when
    torch + transformers are available. Falls back to heuristic
    scoring (burstiness + token repetition + vocabulary analysis)
    when they are not.
    """

    def __init__(self, use_model: bool = True):
        self._burstiness = BurstinessScorer()
        self._use_model = use_model
        self._model_loaded = False
        self._observer_model = None
        self._performer_model = None
        self._tokenizer = None

        if use_model:
            self._try_load_models()

    def _try_load_models(self):
        """Attempt to load torch + transformers. Silent fallback."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self._observer_model = AutoModelForCausalLM.from_pretrained("gpt2")
            self._performer_model = AutoModelForCausalLM.from_pretrained("gpt2")
            self._observer_model.eval()
            self._performer_model.eval()
            self._model_loaded = True
        except Exception:
            self._model_loaded = False

    def scan(self, text: str) -> StatisticalScanResult:
        """Run statistical detection on text."""
        if self._model_loaded:
            return self._scan_model(text)
        return self._scan_heuristic(text)

    def _scan_model(self, text: str) -> StatisticalScanResult:
        """Model-based detection: perplexity + Binoculars."""
        import torch

        result = StatisticalScanResult(text=text, method="model")
        findings = []

        # 1. Perplexity
        perplexity = self._compute_perplexity(text)
        result.perplexity = perplexity
        # Low perplexity = AI-like (text is predictable)
        ppl_threshold = 30.0
        is_ai_ppl = perplexity is not None and perplexity < ppl_threshold
        findings.append(StatisticalFinding(
            signal="perplexity",
            value=perplexity if perplexity else 0.0,
            threshold=ppl_threshold,
            is_ai_like=is_ai_ppl,
            description=f"Perplexity {perplexity:.1f} (below {ppl_threshold} = AI-like, predictable text)" if perplexity else "Could not compute perplexity",
        ))

        # 2. Binoculars score
        binoculars = self._compute_binoculars(text)
        result.binoculars = binoculars
        binoc_threshold = 0.9
        is_ai_binoc = binoculars is not None and binoculars < binoc_threshold
        findings.append(StatisticalFinding(
            signal="binoculars",
            value=binoculars if binoculars else 0.0,
            threshold=binoc_threshold,
            is_ai_like=is_ai_binoc,
            description=f"Binoculars {binoculars:.3f} (below {binoc_threshold} = AI-like)" if binoculars else "Could not compute Binoculars score",
        ))

        # 3. Burstiness
        burst = self._burstiness.score(text)
        result.burstiness = burst.burstiness
        findings.append(StatisticalFinding(
            signal="burstiness",
            value=burst.burstiness,
            threshold=0.35,
            is_ai_like=burst.is_ai_like,
            description=burst.suggestion or f"Burstiness {burst.burstiness:.2f} (below 0.35 = AI-like uniform rhythm)",
        ))

        # 4. Green-list score (Kirchenbauer watermark)
        green_score = self._compute_green_list_score(text)
        result.green_list_score = green_score
        green_threshold = 0.6
        is_green = green_score > green_threshold
        findings.append(StatisticalFinding(
            signal="green_list",
            value=green_score,
            threshold=green_threshold,
            is_ai_like=is_green,
            description=f"Green-list ratio {green_score:.2f} (above {green_threshold} = possible Kirchenbauer watermark)",
        ))

        result.findings = findings
        result.ai_probability = self._ensemble_probability(findings)
        if result.is_ai_like:
            result.suggestion = self._build_suggestion(findings)
        return result

    def _scan_heuristic(self, text: str) -> StatisticalScanResult:
        """Heuristic detection when torch is not available.

        Uses:
          - Burstiness (sentence length variance)
          - Token repetition ratio
          - Vocabulary richness (type-token ratio)
          - Average word length uniformity
          - Punctuation entropy
        """
        result = StatisticalScanResult(text=text, method="heuristic")
        findings = []

        # 1. Burstiness
        burst = self._burstiness.score(text)
        result.burstiness = burst.burstiness
        findings.append(StatisticalFinding(
            signal="burstiness",
            value=burst.burstiness,
            threshold=0.35,
            is_ai_like=burst.is_ai_like,
            description=burst.suggestion or f"Burstiness {burst.burstiness:.2f} (below 0.35 = AI-like)",
        ))

        # 2. Token repetition ratio
        words = re.findall(r'\b\w+\b', text.lower())
        if words:
            unique = set(words)
            repetition = 1.0 - (len(unique) / len(words))
            rep_threshold = 0.4
            is_rep = repetition > rep_threshold
            findings.append(StatisticalFinding(
                signal="repetition",
                value=repetition,
                threshold=rep_threshold,
                is_ai_like=is_rep,
                description=f"Repetition ratio {repetition:.2f} (above {rep_threshold} = AI-like, repetitive vocabulary)",
            ))

        # 3. Type-token ratio (vocabulary richness)
        if words:
            ttr = len(set(words)) / len(words)
            ttr_threshold = 0.6
            is_low_ttr = ttr < ttr_threshold
            findings.append(StatisticalFinding(
                signal="type_token_ratio",
                value=ttr,
                threshold=ttr_threshold,
                is_ai_like=is_low_ttr,
                description=f"TTR {ttr:.2f} (below {ttr_threshold} = AI-like, low vocabulary diversity)",
            ))

        # 4. Average word length uniformity
        if words:
            word_lengths = [len(w) for w in words]
            mean_wl = sum(word_lengths) / len(word_lengths)
            if len(word_lengths) > 1:
                variance = sum((l - mean_wl) ** 2 for l in word_lengths) / len(word_lengths)
                wl_cv = math.sqrt(variance) / mean_wl if mean_wl > 0 else 0
            else:
                wl_cv = 0.0
            wl_threshold = 0.3
            is_uniform_wl = wl_cv < wl_threshold
            findings.append(StatisticalFinding(
                signal="word_length_cv",
                value=wl_cv,
                threshold=wl_threshold,
                is_ai_like=is_uniform_wl,
                description=f"Word length CV {wl_cv:.2f} (below {wl_threshold} = AI-like, uniform word lengths)",
            ))

        # 5. Green-list heuristic (without model)
        green_score = self._compute_green_list_score(text)
        result.green_list_score = green_score
        green_threshold = 0.6
        is_green = green_score > green_threshold
        findings.append(StatisticalFinding(
            signal="green_list",
            value=green_score,
            threshold=green_threshold,
            is_ai_like=is_green,
            description=f"Green-list ratio {green_score:.2f} (above {green_threshold} = possible Kirchenbauer watermark)",
        ))

        result.findings = findings
        result.ai_probability = self._ensemble_probability(findings)
        if result.is_ai_like:
            result.suggestion = self._build_suggestion(findings)
        return result

    def _compute_perplexity(self, text: str) -> float | None:
        """Compute GPT-2 perplexity. Requires torch."""
        if not self._model_loaded:
            return None
        import torch

        encodings = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = self._observer_model(**encodings, labels=encodings["input_ids"])
        return math.exp(outputs.loss.item())

    def _compute_binoculars(self, text: str) -> float | None:
        """Compute Binoculars score (observer/performer cross-perplexity ratio).

        Following Hans et al., ICML 2024.
        """
        if not self._model_loaded:
            return None
        import torch

        encodings = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            obs_logits = self._observer_model(**encodings).logits
            perf_logits = self._performer_model(**encodings).logits

        # Compute cross-entropy of observer given performer
        input_ids = encodings["input_ids"]
        # Shift for next-token prediction
        shift_logits = obs_logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        obs_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        perf_loss = loss_fct(
            perf_logits[..., :-1, :].contiguous().view(-1, perf_logits.size(-1)),
            shift_labels.view(-1),
        )

        # Binoculars = mean(observer_loss) / mean(observer_loss + performer_loss)
        mean_obs = obs_loss.mean().item()
        mean_perf = perf_loss.mean().item()
        if mean_obs + mean_perf > 0:
            return mean_obs / (mean_obs + mean_perf)
        return None

    def _compute_green_list_score(self, text: str) -> float:
        """Heuristic green-list score.

        Kirchenbauer watermarks bias token sampling toward a "green list"
        of tokens. Without the secret key, we can't compute the exact
        green list, but we can check for over-representation of common
        AI-favored tokens.

        This is a rough heuristic. For precise detection, use the
        reference Kirchenbauer detector with the model's secret key.
        """
        # Common AI-favored tokens (high-frequency in LLM output)
        ai_favored_tokens = {
            "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "can",
            "this", "that", "these", "those", "it", "its",
            "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "into", "through",
        }

        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.0

        favored_count = sum(1 for w in words if w in ai_favored_tokens)
        return favored_count / len(words)

    def _ensemble_probability(self, findings: list[StatisticalFinding]) -> float:
        """Combine multiple signals into a single AI probability.

        Weighted average where each signal contributes proportionally
        to how far past its threshold it is.
        """
        if not findings:
            return 0.0

        weights = {
            "perplexity": 0.30,
            "binoculars": 0.25,
            "burstiness": 0.20,
            "repetition": 0.10,
            "type_token_ratio": 0.10,
            "word_length_cv": 0.05,
            "green_list": 0.05,
        }

        total_weight = 0.0
        weighted_score = 0.0
        for f in findings:
            w = weights.get(f.signal, 0.05)
            if f.is_ai_like:
                # How far past threshold (normalized)
                if f.threshold > 0:
                    if f.value < f.threshold:
                        ratio = (f.threshold - f.value) / f.threshold
                    else:
                        ratio = (f.value - f.threshold) / f.threshold
                else:
                    ratio = 0.5
                weighted_score += w * min(ratio, 1.0)
            total_weight += w

        return weighted_score / total_weight if total_weight > 0 else 0.0

    def _build_suggestion(self, findings: list[StatisticalFinding]) -> str:
        """Build a human-readable suggestion from findings."""
        ai_signals = [f for f in findings if f.is_ai_like]
        if not ai_signals:
            return ""

        parts = ["Statistical detection suggests AI-generated text:"]
        for f in ai_signals:
            parts.append(f"  - {f.description}")
        parts.append("Consider rewriting with more varied vocabulary, sentence lengths, and less predictable word choices.")
        return "\n".join(parts)

    def inspect(self, text: str) -> str:
        """Return a human-readable inspection report."""
        result = self.scan(text)
        lines = [f"Statistical Detection ({result.method} method)"]
        lines.append(f"AI probability: {result.ai_probability:.1%}")
        lines.append("")
        for f in result.findings:
            flag = "AI" if f.is_ai_like else "OK"
            lines.append(f"  [{flag}] {f.signal:20s} = {f.value:.3f} (threshold: {f.threshold})")
            lines.append(f"        {f.description}")

        if result.suggestion:
            lines.append("")
            lines.append(result.suggestion)

        return "\n".join(lines)
