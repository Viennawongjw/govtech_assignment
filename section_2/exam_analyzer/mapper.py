"""Map each question to the syllabus rubric. This is the only LLM judgement
step in the pipeline, so each mapping carries a confidence score and a short
justification: low-confidence items can be flagged for human review, and
confidence calibration is then measurable in the evaluation harness.
Outputs are constrained to known IDs (or 'unmapped') to prevent the model
inventing categories."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .exam_parser import Question
from .llm import LLMClient
from .syllabus_parser import Syllabus

_SYS = """You are an assessment analyst mapping each examination question to a \
fixed syllabus rubric. Return ONLY valid JSON, no prose.

You are given the rubric (allowed content unit ids and assessment objective \
ids) and one question. Choose ONLY from the provided ids. If a question maps to \
nothing in the rubric, use "unmapped".

For the question return:
- "content_units": list of applicable content unit ids (usually 1)
- "assessment_objectives": list of applicable AO ids (1 or more)
- "confidence": a number in [0,1] for how certain the mapping is
- "justification": one sentence (<=25 words) explaining the mapping

Schema: {"content_units":[...],"assessment_objectives":[...],"confidence":...,"justification":...}"""


@dataclass
class Mapping:
    question_id: str
    content_units: list[str] = field(default_factory=list)
    assessment_objectives: list[str] = field(default_factory=list)
    confidence: float = 0.0
    justification: str = ""

    @classmethod
    def from_dict(cls, qid: str, d: dict[str, Any]) -> "Mapping":
        try:
            conf = float(d.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        return cls(
            question_id=qid,
            content_units=[str(x).strip() for x in d.get("content_units", [])],
            assessment_objectives=[str(x).strip() for x in d.get("assessment_objectives", [])],
            confidence=max(0.0, min(1.0, conf)),
            justification=str(d.get("justification", "")).strip(),
        )


def map_question(q: Question, syllabus: Syllabus, llm: LLMClient) -> Mapping:
    user = (
        f"RUBRIC:\n{syllabus.rubric_for_prompt()}\n\n"
        f'ALLOWED content unit ids: {syllabus.unit_ids() + ["unmapped"]}\n'
        f'ALLOWED assessment objective ids: {syllabus.ao_ids() + ["unmapped"]}\n\n'
        f"QUESTION {q.number} (command: {q.command}, marks: {q.marks}, "
        f"source_based: {q.source_based}):\n{q.text}"
    )
    data = llm.complete_json(_SYS, user, max_tokens=1024)
    m = Mapping.from_dict(q.id, data)
    # defensive: drop any ids the model hallucinated despite instructions
    valid_units = set(syllabus.unit_ids()) | {"unmapped"}
    valid_aos = set(syllabus.ao_ids()) | {"unmapped"}
    m.content_units = [u for u in m.content_units if u in valid_units] or ["unmapped"]
    m.assessment_objectives = [a for a in m.assessment_objectives if a in valid_aos] or ["unmapped"]
    return m


def map_all(questions: list[Question], syllabus: Syllabus, llm: LLMClient) -> list[Mapping]:
    return [map_question(q, syllabus, llm) for q in questions]
