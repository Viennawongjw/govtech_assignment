"""Evaluation against a hand-labelled gold set.

Two design choices worth knowing:
- Ground truth is itself uncertain (trained markers disagree on AOs), so this
  measures AGREEMENT with a human reference, not right/wrong against an oracle.
- AOs are multi-label, so we report micro precision/recall/F1 and per-item
  Jaccard rather than plain accuracy. Content unit is closer to single-label,
  reported as set-overlap accuracy (any intersection counts).
- Confidence calibration: bucket predictions by confidence and check whether
  higher-confidence items actually agree with humans more often. Well-calibrated
  confidence is what would let a deployment auto-route only low-confidence
  items to human review.

gold_set.json schema: list of {question_id, content_units[], assessment_objectives[]}.
IDs must match those produced by the syllabus parser for THIS syllabus."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from exam_analyzer.mapper import Mapping


@dataclass
class EvalReport:
    n: int
    unit_top1_accuracy: float
    ao_micro_precision: float
    ao_micro_recall: float
    ao_micro_f1: float
    ao_mean_jaccard: float
    calibration: list[dict[str, Any]] = field(default_factory=list)
    per_item: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Evaluated items: {self.n}",
            f"Content-unit overlap accuracy: {self.unit_top1_accuracy:.2f}",
            f"AO micro-precision: {self.ao_micro_precision:.2f}",
            f"AO micro-recall:    {self.ao_micro_recall:.2f}",
            f"AO micro-F1:        {self.ao_micro_f1:.2f}",
            f"AO mean Jaccard:    {self.ao_mean_jaccard:.2f}",
            "Confidence calibration (bucket -> agreement):",
        ]
        for b in self.calibration:
            lines.append(
                f"  conf {b['lo']:.1f}-{b['hi']:.1f}: "
                f"n={b['n']}, mean_jaccard={b['mean_jaccard']:.2f}"
            )
        return "\n".join(lines)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_gold(path: str) -> dict[str, dict[str, set[str]]]:
    with open(path) as f:
        raw = json.load(f)
    gold: dict[str, dict[str, set[str]]] = {}
    for r in raw:
        if "question_id" not in r:
            continue  # skip metadata entries like {_note: ...}
        gold[r["question_id"]] = {
            "content_units": set(r.get("content_units", [])),
            "assessment_objectives": set(r.get("assessment_objectives", [])),
        }
    return gold


def evaluate(mappings: list[Mapping], gold_path: str) -> EvalReport:
    gold = load_gold(gold_path)
    evaluated = [m for m in mappings if m.question_id in gold]

    unit_hits = 0
    jaccards: list[float] = []
    tp = fp = fn = 0
    per_item: list[dict[str, Any]] = []
    conf_records: list[tuple[float, float]] = []  # (confidence, jaccard)

    for m in evaluated:
        g = gold[m.question_id]

        pred_u = set(m.content_units)
        gold_u = g["content_units"]
        # top-1 unit accuracy: does the predicted set intersect gold at all?
        unit_correct = bool(pred_u & gold_u)
        unit_hits += int(unit_correct)

        pred_a = set(m.assessment_objectives)
        gold_a = g["assessment_objectives"]
        j = _jaccard(pred_a, gold_a)
        jaccards.append(j)
        conf_records.append((m.confidence, j))

        tp += len(pred_a & gold_a)
        fp += len(pred_a - gold_a)
        fn += len(gold_a - pred_a)

        per_item.append({
            "question_id": m.question_id,
            "pred_units": sorted(pred_u),
            "gold_units": sorted(gold_u),
            "unit_correct": unit_correct,
            "pred_aos": sorted(pred_a),
            "gold_aos": sorted(gold_a),
            "ao_jaccard": round(j, 2),
            "confidence": m.confidence,
        })

    n = len(evaluated)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    # calibration: bucket by confidence, report mean jaccard per bucket
    buckets = [(0.0, 0.5), (0.5, 0.8), (0.8, 1.01)]
    calibration = []
    for lo, hi in buckets:
        items = [j for c, j in conf_records if lo <= c < hi]
        calibration.append({
            "lo": lo, "hi": min(hi, 1.0), "n": len(items),
            "mean_jaccard": round(mean(items), 2) if items else 0.0,
        })

    return EvalReport(
        n=n,
        unit_top1_accuracy=round(unit_hits / n, 2) if n else 0.0,
        ao_micro_precision=round(prec, 2),
        ao_micro_recall=round(rec, 2),
        ao_micro_f1=round(f1, 2),
        ao_mean_jaccard=round(mean(jaccards), 2) if jaccards else 0.0,
        calibration=calibration,
        per_item=per_item,
    )
