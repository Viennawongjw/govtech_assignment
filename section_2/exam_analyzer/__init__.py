"""Exam-paper syllabus-alignment analyzer.

Upload an exam paper PDF and a syllabus PDF; analyse (1) how each question
aligns with syllabus assessment objectives and content units, and (2) the
balance of marks across those units and objectives."""
from .pdf_ingest import extract_text, has_text_layer
from .llm import LLMClient
from .syllabus_parser import parse_syllabus, Syllabus, SyllabusItem
from .exam_parser import parse_exam, Question
from .mapper import map_all, map_question, Mapping
from .aggregate import aggregate, AnalysisResult, WeightageRow

__all__ = [
    "extract_text", "has_text_layer", "LLMClient",
    "parse_syllabus", "Syllabus", "SyllabusItem",
    "parse_exam", "Question",
    "map_all", "map_question", "Mapping",
    "aggregate", "AnalysisResult", "WeightageRow",
    "run_pipeline",
]


def run_pipeline(exam_pdf: str, syllabus_pdf: str, llm: LLMClient | None = None):
    """Run all stages. Returns (questions, syllabus, mappings, result)."""
    llm = llm or LLMClient()
    exam_text = extract_text(exam_pdf)
    syllabus_text = extract_text(syllabus_pdf)
    syllabus = parse_syllabus(syllabus_text, llm)
    questions = parse_exam(exam_text, llm)
    mappings = map_all(questions, syllabus, llm)
    result = aggregate(questions, mappings, syllabus)
    return questions, syllabus, mappings, result
