"""Aggregate per-question mappings into weightage and coverage.

Pure Python — no LLM. Judgement happens in the mapping step; this step is
arithmetic and should be exact. When a question maps to multiple AOs, its
marks are split evenly across them (avoids double-counting; can be replaced
with a weighted split later). Coverage gaps (units/AOs with zero marks) are
reported alongside the weightage."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .exam_parser import Question
from .mapper import Mapping
from .syllabus_parser import Syllabus


@dataclass
class WeightageRow:
    id: str
    name: str
    marks: float
    pct: float


@dataclass
class AnalysisResult:
    total_marks: int
    unit_weightage: list[WeightageRow] = field(default_factory=list)
    ao_weightage: list[WeightageRow] = field(default_factory=list)
    uncovered_units: list[str] = field(default_factory=list)
    uncovered_aos: list[str] = field(default_factory=list)
    unmapped_marks: float = 0.0


def _name_lookup(syllabus: Syllabus) -> dict[str, str]:
    d = {u.id: u.name for u in syllabus.content_units}
    d.update({a.id: a.name for a in syllabus.assessment_objectives})
    d["unmapped"] = "Unmapped / outside syllabus"
    return d


def aggregate(
    questions: list[Question],
    mappings: list[Mapping],
    syllabus: Syllabus,
    split_marks_across_aos: bool = True,
) -> AnalysisResult:
    qById = {q.id: q for q in questions}
    names = _name_lookup(syllabus)

    unit_marks: dict[str, float] = defaultdict(float)
    ao_marks: dict[str, float] = defaultdict(float)
    total = 0
    unmapped = 0.0

    for m in mappings:
        q = qById.get(m.question_id)
        if q is None:
            continue
        total += q.marks

        # content units: split marks evenly across mapped units
        units = m.content_units or ["unmapped"]
        share_u = q.marks / len(units)
        for u in units:
            unit_marks[u] += share_u
            if u == "unmapped":
                unmapped += share_u

        # assessment objectives: split (or duplicate) across mapped AOs
        aos = m.assessment_objectives or ["unmapped"]
        share_a = q.marks / len(aos) if split_marks_across_aos else q.marks
        for a in aos:
            ao_marks[a] += share_a

    def rows(marks: dict[str, float], denom: int) -> list[WeightageRow]:
        out = [
            WeightageRow(k, names.get(k, k), round(v, 1), round(100 * v / denom, 1) if denom else 0.0)
            for k, v in marks.items()
        ]
        return sorted(out, key=lambda r: r.marks, reverse=True)

    unit_rows = rows(unit_marks, total)
    ao_rows = rows(ao_marks, total)

    uncovered_units = [u.name for u in syllabus.content_units if unit_marks.get(u.id, 0) == 0]
    uncovered_aos = [a.name for a in syllabus.assessment_objectives if ao_marks.get(a.id, 0) == 0]

    return AnalysisResult(
        total_marks=total,
        unit_weightage=unit_rows,
        ao_weightage=ao_rows,
        uncovered_units=uncovered_units,
        uncovered_aos=uncovered_aos,
        unmapped_marks=round(unmapped, 1),
    )
