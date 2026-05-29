"""Offline smoke test using a mock LLM. No API key or network needed.
Exercises every pipeline stage and the evaluation harness end-to-end."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from exam_analyzer.syllabus_parser import parse_syllabus
from exam_analyzer.exam_parser import parse_exam
from exam_analyzer.mapper import map_all
from exam_analyzer.aggregate import aggregate
from evaluation.evaluate import evaluate


class MockLLM:
    """Returns canned JSON keyed by what kind of system prompt it sees."""
    def complete_json(self, system, user, max_tokens=4096):
        if "extracting structure" in system:  # syllabus
            return {
                "content_units": [
                    {"id": "unit_european_control", "name": "Extension of European control in SE Asia 1870s–1942",
                     "description": "European expansion and challenges to dominance in Southeast Asia.",
                     "evidence": "Extension of European control in Southeast Asia"},
                    {"id": "unit_cold_war", "name": "The Cold War and decolonisation 1940s–1991",
                     "description": "Post-war Cold War dynamics and decolonisation in SE Asia.",
                     "evidence": "the Cold War and decolonisation"},
                ],
                "assessment_objectives": [
                    {"id": "ao_knowledge", "name": "Knowledge & understanding",
                     "description": "Recall and understanding of historical knowledge.",
                     "evidence": "knowledge and understanding"},
                    {"id": "ao_source_analysis", "name": "Source/evidence analysis",
                     "description": "Critically examine sources to reach substantiated judgements.",
                     "evidence": "examine a range of sources critically"},
                    {"id": "ao_argument", "name": "Constructing historical arguments",
                     "description": "Organise and communicate a substantiated historical argument.",
                     "evidence": "organise and communicate their historical knowledge"},
                ],
            }
        if "decomposing an examination" in system:  # exam
            return {"questions": [
                {"id": "Q1a", "number": "1(a)", "marks": 4, "command": "Study Source A. What can you learn",
                 "source_based": True, "text": "Study Source A. What can you learn about European expansion?"},
                {"id": "Q1b", "number": "1(b)", "marks": 6, "command": "How reliable",
                 "source_based": True, "text": "How reliable is Source B for studying resistance? Explain."},
                {"id": "Q1c", "number": "1(c)", "marks": 12, "command": "How far do you agree",
                 "source_based": True, "text": "Using the sources, how far do you agree that economic motives drove expansion?"},
                {"id": "Q2", "number": "2", "marks": 13, "command": "Explain",
                 "source_based": False, "text": "Explain why European powers extended control over SE Asia."},
                {"id": "Q3", "number": "3", "marks": 13, "command": "How far",
                 "source_based": False, "text": "How far was the Cold War responsible for decolonisation in SE Asia?"},
            ]}
        # mapping: branch on question text in the user prompt
        if "QUESTION 1(a)" in user:
            return {"content_units": ["unit_european_control"], "assessment_objectives": ["ao_source_analysis"],
                    "confidence": 0.9, "justification": "Source comprehension of European expansion."}
        if "QUESTION 1(b)" in user:
            return {"content_units": ["unit_european_control"], "assessment_objectives": ["ao_source_analysis"],
                    "confidence": 0.85, "justification": "Evaluating source reliability."}
        if "QUESTION 1(c)" in user:
            return {"content_units": ["unit_european_control"],
                    "assessment_objectives": ["ao_source_analysis", "ao_argument"],
                    "confidence": 0.6, "justification": "Source-based judgement requiring an argument."}
        if "QUESTION 2" in user:
            return {"content_units": ["unit_european_control"],
                    "assessment_objectives": ["ao_knowledge", "ao_argument"],
                    "confidence": 0.8, "justification": "Causation essay on European control."}
        if "QUESTION 3" in user:
            return {"content_units": ["unit_cold_war"],
                    "assessment_objectives": ["ao_knowledge", "ao_argument"],
                    "confidence": 0.75, "justification": "Cold War causation essay."}
        return {"content_units": ["unmapped"], "assessment_objectives": ["unmapped"],
                "confidence": 0.2, "justification": "No clear match."}


def main():
    llm = MockLLM()
    syllabus = parse_syllabus("(mock syllabus text)", llm)
    questions = parse_exam("(mock paper text)", llm)
    mappings = map_all(questions, syllabus, llm)
    result = aggregate(questions, mappings, syllabus)

    print("=== PARSED ===")
    print(f"units={len(syllabus.content_units)} aos={len(syllabus.assessment_objectives)} "
          f"questions={len(questions)} total_marks={result.total_marks}")

    print("\n=== ALIGNMENT ===")
    for m in mappings:
        print(f"{m.question_id}: units={m.content_units} aos={m.assessment_objectives} "
              f"conf={m.confidence} :: {m.justification}")

    print("\n=== WEIGHTAGE (by unit) ===")
    for r in result.unit_weightage:
        print(f"  {r.name}: {r.marks} marks ({r.pct}%)")
    print("=== WEIGHTAGE (by AO) ===")
    for r in result.ao_weightage:
        print(f"  {r.name}: {r.marks} marks ({r.pct}%)")
    print(f"uncovered_units={result.uncovered_units} uncovered_aos={result.uncovered_aos} "
          f"unmapped_marks={result.unmapped_marks}")

    # build a small gold set (with a deliberate disagreement on Q2's AOs and Q1c)
    gold = [
        {"question_id": "Q1a", "content_units": ["unit_european_control"], "assessment_objectives": ["ao_source_analysis"]},
        {"question_id": "Q1b", "content_units": ["unit_european_control"], "assessment_objectives": ["ao_source_analysis"]},
        {"question_id": "Q1c", "content_units": ["unit_european_control"], "assessment_objectives": ["ao_source_analysis", "ao_argument"]},
        {"question_id": "Q2", "content_units": ["unit_european_control"], "assessment_objectives": ["ao_argument"]},  # human says argument only
        {"question_id": "Q3", "content_units": ["unit_cold_war"], "assessment_objectives": ["ao_knowledge", "ao_argument"]},
    ]
    with open("/tmp/gold_set.json", "w") as f:
        json.dump(gold, f)

    report = evaluate(mappings, "/tmp/gold_set.json")
    print("\n=== EVALUATION ===")
    print(report.summary())
    print("\nAll stages ran successfully.")


if __name__ == "__main__":
    main()
