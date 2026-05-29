"""Parse the exam paper into atomic, mark-bearing parts.
The unit of analysis is the sub-question (e.g. 1a, 1b), not the whole question:
sub-parts carry their own marks and test different skills, and aggregating at
'Question 1' would hide the signal we want."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .llm import LLMClient

_SYS = """You are an assessment analyst decomposing an examination paper into \
its atomic, mark-bearing questions. Return ONLY valid JSON, no prose.

Rules:
- The unit is the lowest-level part that carries marks. If Question 1 has \
parts (a),(b),(c), emit one record per part, not one for Question 1.
- Capture the marks shown in square brackets [n] as an integer.
- Capture the command word / question stem verb (e.g. "Describe", "Explain", \
"How far do you agree", "Study Source A").
- Include the full question text so it can be mapped to the syllabus later.
- If a part references a source (source-based case study), set "source_based" \
true.

For each question part provide:
- "id": stable label, e.g. "Q1a", "Q2", "Q5b"
- "number": the human-facing number/letter, e.g. "1(a)"
- "marks": integer marks (0 if genuinely none shown)
- "command": the command word/stem
- "source_based": boolean
- "text": the full question text

Schema: {"questions": [{"id","number","marks","command","source_based","text"}]}"""


@dataclass
class Question:
    id: str
    number: str
    marks: int
    command: str
    source_based: bool
    text: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Question":
        try:
            marks = int(d.get("marks", 0) or 0)
        except (TypeError, ValueError):
            marks = 0
        return cls(
            id=str(d.get("id", "")).strip(),
            number=str(d.get("number", "")).strip(),
            marks=marks,
            command=str(d.get("command", "")).strip(),
            source_based=bool(d.get("source_based", False)),
            text=str(d.get("text", "")).strip(),
        )


def parse_exam(exam_text: str, llm: LLMClient) -> list[Question]:
    data = llm.complete_json(_SYS, f"EXAMINATION PAPER TEXT:\n\n{exam_text}", max_tokens=8192)
    return [Question.from_dict(d) for d in data.get("questions", [])]
