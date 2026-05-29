"""Parse the syllabus into two lists: content units (topics) and assessment
objectives (skills). Together they form the rubric questions are mapped against.
Each item carries a verbatim evidence snippet for human verification."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .llm import LLMClient

_SYS = """You are a curriculum analyst extracting structure from a national \
examination syllabus. Return ONLY valid JSON, no prose, no markdown fences.

Extract two lists:
1. "content_units": the substantive topics/themes students are examined on.
2. "assessment_objectives": the skills/competencies the syllabus says are \
assessed (e.g. knowledge and understanding, source/evidence analysis, \
constructing and communicating an argument).

For each item provide:
- "id": a short stable slug (e.g. "unit_european_control", "ao_source_analysis")
- "name": a concise human-readable name
- "description": one sentence describing it, in your own words
- "evidence": a short verbatim snippet (<=15 words) from the syllabus that \
supports this extraction, for human verification

Schema:
{
  "content_units": [{"id","name","description","evidence"}],
  "assessment_objectives": [{"id","name","description","evidence"}]
}"""


@dataclass
class SyllabusItem:
    id: str
    name: str
    description: str
    evidence: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SyllabusItem":
        return cls(
            id=str(d.get("id", "")).strip(),
            name=str(d.get("name", "")).strip(),
            description=str(d.get("description", "")).strip(),
            evidence=str(d.get("evidence", "")).strip(),
        )


@dataclass
class Syllabus:
    content_units: list[SyllabusItem] = field(default_factory=list)
    assessment_objectives: list[SyllabusItem] = field(default_factory=list)

    def unit_ids(self) -> list[str]:
        return [u.id for u in self.content_units]

    def ao_ids(self) -> list[str]:
        return [a.id for a in self.assessment_objectives]

    def rubric_for_prompt(self) -> str:
        """Compact text rendering of the rubric for the mapping prompt."""
        lines = ["CONTENT UNITS:"]
        for u in self.content_units:
            lines.append(f"  - {u.id}: {u.name} — {u.description}")
        lines.append("ASSESSMENT OBJECTIVES:")
        for a in self.assessment_objectives:
            lines.append(f"  - {a.id}: {a.name} — {a.description}")
        return "\n".join(lines)


def parse_syllabus(syllabus_text: str, llm: LLMClient) -> Syllabus:
    data = llm.complete_json(_SYS, f"SYLLABUS TEXT:\n\n{syllabus_text}")
    return Syllabus(
        content_units=[SyllabusItem.from_dict(d) for d in data.get("content_units", [])],
        assessment_objectives=[
            SyllabusItem.from_dict(d) for d in data.get("assessment_objectives", [])
        ],
    )
