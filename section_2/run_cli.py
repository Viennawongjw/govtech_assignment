"""Command-line runner. Analyse a paper+syllabus, optionally evaluate against a gold set.

  export ANTHROPIC_API_KEY=...
  python run_cli.py --exam sample_data/paper.pdf --syllabus sample_data/syllabus.pdf
  python run_cli.py --exam sample_data/paper.pdf --syllabus sample_data/syllabus.pdf \\
      --gold evaluation/gold_set.json --out results.json
"""
from __future__ import annotations

import argparse
import json

from exam_analyzer import run_pipeline, LLMClient
from evaluation.evaluate import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exam", required=True)
    ap.add_argument("--syllabus", required=True)
    ap.add_argument("--gold", default=None, help="optional gold_set.json to evaluate against")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None, help="optional path to dump full results as JSON")
    args = ap.parse_args()

    llm = LLMClient(model=args.model) if args.model else LLMClient()
    questions, syllabus, mappings, result = run_pipeline(args.exam, args.syllabus, llm)

    print(f"Parsed {len(questions)} questions, total {result.total_marks} marks.\n")
    print("Weightage by content unit:")
    for r in result.unit_weightage:
        print(f"  {r.pct:5.1f}%  {r.name}")
    print("\nWeightage by assessment objective:")
    for r in result.ao_weightage:
        print(f"  {r.pct:5.1f}%  {r.name}")
    if result.uncovered_units:
        print("\nUncovered units:", ", ".join(result.uncovered_units))
    if result.uncovered_aos:
        print("Uncovered AOs:", ", ".join(result.uncovered_aos))

    if args.gold:
        report = evaluate(mappings, args.gold)
        print("\n" + report.summary())

    if args.out:
        payload = {
            "questions": [q.__dict__ for q in questions],
            "syllabus": {
                "content_units": [u.__dict__ for u in syllabus.content_units],
                "assessment_objectives": [a.__dict__ for a in syllabus.assessment_objectives],
            },
            "mappings": [m.__dict__ for m in mappings],
            "weightage_units": [r.__dict__ for r in result.unit_weightage],
            "weightage_aos": [r.__dict__ for r in result.ao_weightage],
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
