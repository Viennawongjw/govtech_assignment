"""Streamlit UI for the exam-syllabus alignment analyzer.

Run:  streamlit run app.py
Needs: ANTHROPIC_API_KEY in the environment.
Stages are shown progressively so the user sees the rubric, the parsed
questions, and each mapping's justification + confidence before the aggregate
charts. Showing the justification turns an unverifiable label into an
auditable one."""
from __future__ import annotations

import os
import tempfile

import pandas as pd
import streamlit as st

from exam_analyzer import (
    LLMClient, extract_text, has_text_layer,
    parse_syllabus, parse_exam, map_all, aggregate,
)

st.set_page_config(page_title="Exam–Syllabus Alignment Analyzer", layout="wide")

st.title("Exam–Syllabus Alignment Analyzer")
st.caption(
    "Upload an examination paper and its syllabus to analyse (1) alignment of "
    "each question with syllabus assessment objectives and content units, and "
    "(2) the balance of topic and skill weightage."
)

with st.sidebar:
    st.header("Setup")
    model = st.text_input("Model", value=os.environ.get("ANALYZER_MODEL", "claude-sonnet-4-6"))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("Set ANTHROPIC_API_KEY in your environment before running an analysis.")
    st.markdown("---")
    st.markdown(
        "**How it works**\n\n"
        "1. Extract text from both PDFs\n"
        "2. Parse syllabus → content units + assessment objectives (rubric)\n"
        "3. Parse paper → atomic mark-bearing questions\n"
        "4. Map each question → rubric (with confidence + justification)\n"
        "5. Aggregate marks → weightage & coverage (pure Python)"
    )


def _save_upload(uploaded) -> str:
    suffix = ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        return tmp.name


col1, col2 = st.columns(2)
with col1:
    exam_file = st.file_uploader("Examination paper (PDF)", type="pdf", key="exam")
with col2:
    syllabus_file = st.file_uploader("Syllabus (PDF)", type="pdf", key="syllabus")

run = st.button("Analyse", type="primary", disabled=not (exam_file and syllabus_file))

if run and exam_file and syllabus_file:
    llm = LLMClient(model=model)
    exam_path = _save_upload(exam_file)
    syl_path = _save_upload(syllabus_file)

    # Guard: warn on scanned PDFs (OCR is out of scope for this prototype)
    for label, path in [("paper", exam_path), ("syllabus", syl_path)]:
        if not has_text_layer(path):
            st.error(
                f"The {label} PDF has little or no extractable text — it may be "
                f"scanned. OCR is out of scope for this prototype."
            )
            st.stop()

    with st.status("Running analysis…", expanded=True) as status:
        st.write("Extracting text…")
        exam_text = extract_text(exam_path)
        syllabus_text = extract_text(syl_path)

        st.write("Parsing syllabus into rubric…")
        syllabus = parse_syllabus(syllabus_text, llm)

        st.write("Parsing examination paper into questions…")
        questions = parse_exam(exam_text, llm)

        st.write(f"Mapping {len(questions)} questions to the rubric…")
        mappings = map_all(questions, syllabus, llm)

        st.write("Aggregating weightage…")
        result = aggregate(questions, mappings, syllabus)
        status.update(label="Analysis complete", state="complete", expanded=False)

    # ---- Rubric ----
    st.subheader("1 · Extracted syllabus rubric")
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("**Content units**")
        st.dataframe(pd.DataFrame(
            [{"id": u.id, "name": u.name, "evidence": u.evidence} for u in syllabus.content_units]
        ), use_container_width=True, hide_index=True)
    with rc2:
        st.markdown("**Assessment objectives**")
        st.dataframe(pd.DataFrame(
            [{"id": a.id, "name": a.name, "evidence": a.evidence} for a in syllabus.assessment_objectives]
        ), use_container_width=True, hide_index=True)

    # ---- Alignment table ----
    st.subheader("2 · Question-by-question alignment")
    qById = {q.id: q for q in questions}
    align_rows = []
    for m in mappings:
        q = qById[m.question_id]
        align_rows.append({
            "Q": q.number,
            "Marks": q.marks,
            "Command": q.command,
            "Content unit(s)": ", ".join(m.content_units),
            "Assessment objective(s)": ", ".join(m.assessment_objectives),
            "Confidence": m.confidence,
            "Justification": m.justification,
        })
    align_df = pd.DataFrame(align_rows)
    st.dataframe(
        align_df,
        use_container_width=True, hide_index=True,
        column_config={"Confidence": st.column_config.ProgressColumn(
            "Confidence", min_value=0.0, max_value=1.0, format="%.2f")},
    )
    low = align_df[align_df["Confidence"] < 0.5]
    if len(low):
        st.info(f"{len(low)} mapping(s) below 0.5 confidence — these would be routed to human review in production.")

    # ---- Weightage ----
    st.subheader("3 · Weightage")
    wc1, wc2 = st.columns(2)
    with wc1:
        st.markdown(f"**By content unit** (total {result.total_marks} marks)")
        udf = pd.DataFrame([{"Unit": r.name, "Marks": r.marks, "%": r.pct} for r in result.unit_weightage])
        st.bar_chart(udf.set_index("Unit")["Marks"])
        st.dataframe(udf, use_container_width=True, hide_index=True)
    with wc2:
        st.markdown("**By assessment objective**")
        adf = pd.DataFrame([{"Objective": r.name, "Marks": r.marks, "%": r.pct} for r in result.ao_weightage])
        st.bar_chart(adf.set_index("Objective")["Marks"])
        st.dataframe(adf, use_container_width=True, hide_index=True)

    # ---- Coverage / gaps ----
    st.subheader("4 · Coverage & gaps")
    if result.uncovered_units:
        st.warning("Content units with **no marks** in this paper: " + ", ".join(result.uncovered_units))
    if result.uncovered_aos:
        st.warning("Assessment objectives with **no marks** in this paper: " + ", ".join(result.uncovered_aos))
    if result.unmapped_marks:
        st.warning(f"{result.unmapped_marks} marks could not be mapped to any syllabus item.")
    if not (result.uncovered_units or result.uncovered_aos or result.unmapped_marks):
        st.success("Every syllabus unit and objective is represented, and all marks mapped.")

    # ---- Export ----
    st.download_button(
        "Download alignment table (CSV)",
        align_df.to_csv(index=False).encode(),
        file_name="alignment.csv", mime="text/csv",
    )
