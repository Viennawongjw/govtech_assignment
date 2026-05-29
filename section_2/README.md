# Exam–Syllabus Alignment Analyzer

A prototype for the MOE brief: upload an exam paper PDF and a syllabus PDF,
get back

1. **Alignment** — which syllabus content units and assessment objectives each
   question tests, and
2. **Topic weightage** — how marks are distributed across those units and
   objectives, including any coverage gaps.

Tested on the Singapore-Cambridge **O-Level History 2174** specimen Paper 1
and the **Upper Secondary History (2174)** syllabus.

> The brief says we are assessed on how we *think about evaluation*, not on
> raw model performance. The prototype uses a small, swappable model and a
> deliberately simple, inspectable pipeline.

## The idea

The brief asks for two analyses. They map cleanly onto the two things a
history syllabus actually contains:

| Brief asks for… | …which maps to | Driven by |
|---|---|---|
| Alignment with syllabus objectives | Assessment objectives (skills) + content units (topics) | LLM classification |
| Balance of topic weightage | Marks summed per unit / per objective | Plain Python |

So the tool does **structured extraction** on both PDFs, a **mapping** between
them, then **deterministic aggregation** — rather than asking the LLM one
fuzzy "does this match?" question.

## Architecture

```
PDF (paper) ─┐
             ├─► extract text (PyMuPDF, born-digital; OCR out of scope)
PDF (syllabus)┘
                    │
   ┌────────────────┴───────────────────┐
   ▼                                     ▼
parse_syllabus                       parse_exam
 → content units                      → atomic mark-bearing parts
 → assessment objectives                (1a, 1b…) with marks + command
 (+ verbatim evidence snippets)
   │                                     │
   └──────────────► map_question ◄───────┘
                     per question: units, AOs (multi-label),
                     confidence, one-sentence justification
                            │
                            ▼
                       aggregate  (pure Python, no LLM)
                       → marks & % per unit / per AO
                       → coverage gaps, unmapped marks
                            │
                            ▼
                    Streamlit UI / CLI / eval harness
```

## Key design choices

- **The unit of analysis is the sub-question, not the question.** Section A's
  Question 1 has parts (a), (b), (c) — each with its own marks and testing a
  different skill. Aggregating at "Question 1" would destroy the signal.
- **Judgement and arithmetic are separated.** The LLM does the hard, uncertain
  step (mapping). Python does the exact step (summing marks). Numbers are
  reproducible and cheap, and never hallucinated.
- **Every mapping carries a confidence and justification.** Turns an
  unverifiable label into an auditable one. The UI flags low-confidence items
  for human review, and the eval harness checks whether confidence is
  *calibrated* against agreement with humans.
- **Multi-label assessment objectives.** Real history questions test more than
  one skill. When a question maps to several AOs, marks are split evenly across
  them; this is documented and easy to swap for a weighted split later
  (e.g. from a mark scheme).
- **Constrained outputs.** The mapper may only emit known unit/AO IDs (or the
  explicit `unmapped` sentinel), so the model can't invent categories.
- **Model-agnostic.** Claude Sonnet 4.6 by default; swap via `ANALYZER_MODEL`.
  Temperature is pinned to 0 for run-to-run reproducibility.

## Evaluation method

The harness in `evaluation/` is built around one premise: **ground truth is
itself uncertain.** A history question can legitimately test multiple
objectives, and trained markers disagree. So we don't score against a single
oracle; we measure **agreement with a human reference**, with metrics
appropriate to multi-label answers:

- **Content-unit overlap accuracy** — unit is near single-label; any
  intersection between predicted and gold counts.
- **AO micro precision / recall / F1** over (question, AO) pairs, plus
  **mean Jaccard** per question.
- **Confidence calibration** — bucket predictions by confidence and check
  whether higher-confidence items actually agree with humans more often.

Workflow: do one real run, hand-label 10–15 question parts into
`evaluation/gold_set.json` (the IDs must match that run's syllabus parse),
then re-run with `--gold`.

## Running it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...

# UI
streamlit run app.py

# CLI
python run_cli.py --exam sample_data/paper.pdf --syllabus sample_data/syllabus.pdf \
    --gold evaluation/gold_set.json --out results.json
```

```bash
python test_pipeline.py
```

## Source PDFs

The two PDFs used here come from SEAB (seab.gov.sg):
- `sample_data/paper.pdf` — History 2174/01 specimen Paper 1
- `sample_data/syllabus.pdf` — O-Level History (2174) syllabus 2025

## Repository layout

```
exam_analyzer/        core library
  pdf_ingest.py        text extraction (+ scanned-PDF guard)
  llm.py               model-agnostic JSON client (Claude default)
  syllabus_parser.py   syllabus → rubric (units + AOs)
  exam_parser.py       paper → atomic mark-bearing parts
  mapper.py            question → rubric (confidence + justification)
  aggregate.py         deterministic weightage + coverage
evaluation/
  evaluate.py          multi-label metrics + confidence calibration
  gold_set.json        human reference labels
app.py                 Streamlit UI
run_cli.py             headless runner
test_pipeline.py       offline end-to-end test (mock LLM)
```

## Known limitations

- Born-digital PDFs only; scanned papers would need OCR (guarded, not
  implemented).
- Mark scheme / grade-descriptor ingestion is not yet included; adding it
  would sharpen AO classification.
- The gold set is small by design — the point is the evaluation *method*,
  which scales with more labels and an inter-annotator-agreement study.
