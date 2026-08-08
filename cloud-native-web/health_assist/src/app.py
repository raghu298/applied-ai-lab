"""Interactive application (requirement 6).

Streamlit front end for the Patient Voice Triage and Clinical Support Assistant.

Run from the project root:
    .venv/bin/streamlit run src/app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

import config
import pipeline
from llmops import metrics as llmops
from tasks import model_hub, subtask5_qa

st.set_page_config(
    page_title="Patient Voice Triage and Clinical Support Assistant",
    layout="wide",
)

URGENCY_STYLE = {
    "EMERGENCY": ("#b00020", "Emergency"),
    "HIGH": ("#b26a00", "High"),
    "ROUTINE": ("#1b5e20", "Routine"),
}

SAMPLE_CASES = {
    "Dengue-like fever": (
        "I have been having a really bad headache for the last four days and a high "
        "fever that will not come down. There is pain behind my eyes and my joints "
        "and muscles ache badly. Since yesterday a rash has appeared on my arms and "
        "I feel very weak and nauseous."
    ),
    "Cardiac emergency": (
        "I am having severe chest pain since this morning and it is spreading to my "
        "left arm and jaw. I am sweating a lot and I feel breathless when I try to "
        "walk even a few steps."
    ),
    "Gastro-intestinal": (
        "For the past three days I have had a lot of stomach pain and loose motions "
        "many times a day. I feel dizzy when I stand up and I have been vomiting "
        "after meals. I have not been able to keep any food down."
    ),
    "Chronic condition review": (
        "I have been feeling very tired for a few weeks now and I am passing urine "
        "much more often than before, especially at night. I feel thirsty all the "
        "time and I have lost some weight without trying."
    ),
}


def header():
    st.title("Patient Voice Triage and Clinical Support Assistant")
    st.caption(
        "Domain: Healthcare  |  Categories: Speech Recognition and Natural "
        "Language Processing  |  Six sub-tasks on one patient episode"
    )


def sidebar():
    with st.sidebar:
        st.header("Models in use")
        ready = config.FINETUNED_TRIAGE_DIR.exists()
        clf_label = (
            "bert-mini fine-tuned (cloud-lite)"
            if config.LITE_MODE
            else "ModernBERT fine-tuned (this project)"
        )
        rows = [
            ("1  Speech Recognition", config.ASR_MODEL),
            ("2  Text Classification", clf_label),
            ("3  Clinical NER", config.NER_MODEL),
            ("4  Summarisation", config.SUMMARISER_MODEL),
            ("5  Question Answering", config.QA_MODEL),
            ("6  Text Generation",
             config.GENERATOR_MODEL or "deterministic template (cloud-lite)"),
        ]
        for name, model in rows:
            st.markdown(f"**{name}**  \n`{model}`")
        if config.LITE_MODE:
            st.info(
                "Cloud demo: lightweight models sized for the free hosting "
                "tier. The full application runs locally with larger models "
                "(whisper-large-v3-turbo, ModernBERT, bart-large, Qwen2.5) - "
                "see the README."
            )
        st.divider()
        if ready:
            st.success("Fine-tuned triage model loaded")
        else:
            st.error(
                "Fine-tuned model missing. Run:\n\n"
                "`.venv/bin/python src/finetune/train_triage.py`"
            )
        st.divider()
        st.caption(
            "All models run locally through the Hugging Face transformers API. "
            "No patient text leaves this machine."
        )


def render_urgency(triage: dict):
    colour, label = URGENCY_STYLE.get(triage["urgency"], ("#333", triage["urgency"]))
    # An overridden emergency leads with the red flag that actually decided
    # the routing. Headlining the model's discredited label ("Emergency |
    # dengue") misread as the system diagnosing dengue; the prediction is
    # still shown below, explicitly marked as overridden, so the decision
    # stays auditable.
    if triage.get("rule_escalated"):
        headline = f"Red flag detected: {triage.get('red_flag') or 'emergency phrase'}"
    else:
        headline = (
            f"{triage['condition']} &nbsp;({triage['confidence']:.0%} confidence)"
        )
    st.markdown(
        f"<div style='background:{colour};color:#fff;padding:14px 18px;"
        f"border-radius:6px;font-size:19px;font-weight:600'>"
        f"{label} &nbsp;|&nbsp; {headline}</div>",
        unsafe_allow_html=True,
    )
    st.write(f"**Routing decision:** {triage['routing_action']}")
    if triage.get("rule_escalated"):
        # The overridden prediction is audit information, not a result, so it
        # sits behind a collapsed expander: invisible by default, available to
        # the nurse or an auditor who wants to see what the model suggested
        # before the rule layer discarded it.
        with st.expander("Audit trail: overridden model prediction"):
            st.write(
                f"The complaint contained the red flag "
                f"'{triage.get('red_flag') or 'emergency phrase'}', so the "
                f"rule layer decided the routing. The classifier's suggestion "
                f"of {triage['condition']} ({triage['confidence']:.0%}) was "
                f"overridden, was never shown to the patient, and is recorded "
                f"here only for clinical audit. Presentations such as cardiac "
                f"events are deliberately outside the classifier's 22 trained "
                f"conditions; emergencies are decided by rules, not by the "
                f"model."
            )
    if triage.get("review_required"):
        st.warning(
            f"Model confidence is below the "
            f"{config.TRIAGE_REVIEW_THRESHOLD:.0%} threshold. The case has "
            "been queued for a triage nurse to confirm before any action."
        )
    if triage.get("rule_escalated"):
        st.error(
            "The rule layer overrode the model. A red-flag phrase in the "
            "complaint forced an emergency route regardless of the prediction."
        )


def episode_tab():
    st.subheader("Patient intake")
    left, right = st.columns([1, 1])

    with left:
        mode = st.radio(
            "Input channel",
            ["Sample case", "Type the complaint", "Upload a voice note"],
            horizontal=False,
        )
        audio_path, audio_label, typed = None, None, None

        if mode == "Sample case":
            choice = st.selectbox("Select a case", list(SAMPLE_CASES))
            typed = st.text_area("Complaint", SAMPLE_CASES[choice], height=170)
        elif mode == "Type the complaint":
            typed = st.text_area(
                "Describe the symptoms in the patient's own words",
                height=170,
                placeholder="I have had a high fever for four days and a severe headache...",
            )
        else:
            uploaded = st.file_uploader(
                "Voice note", type=["wav", "mp3", "m4a", "flac", "ogg"]
            )
            if uploaded is not None:
                st.audio(uploaded)
                suffix = Path(uploaded.name).suffix
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(uploaded.getvalue())
                tmp.close()
                audio_path = tmp.name
                audio_label = uploaded.name

        # Identifies the inputs currently on screen. A cached result is only
        # re-shown while this matches the inputs that produced it, so changing
        # the case, editing the text or swapping the file clears the display.
        input_sig = (mode, (typed or "").strip(), audio_label)

        run = st.button("Run triage", type="primary", width='stretch')

    with right:
        st.markdown("**Pipeline stages**")
        st.markdown(
            "1. Speech Recognition converts the voice note to text\n"
            "2. Text Classification predicts the condition and urgency\n"
            "3. Clinical NER extracts symptoms, duration and severity\n"
            "4. Question Answering fetches the governing hospital policy\n"
            "5. Summarisation writes the physician handover note\n"
            "6. Text Generation drafts the reply sent to the patient"
        )

    if run:
        if not audio_path and not (typed or "").strip():
            st.warning("Provide a complaint or upload a voice note first.")
            return

        bar = st.progress(0, text="Starting")

        def progress(n, label):
            bar.progress(int(n / 6 * 100), text=f"Step {n} of 6: {label}")

        try:
            result = pipeline.run_episode(
                audio_path=audio_path, typed_text=typed, progress=progress
            )
        except Exception as exc:
            bar.empty()
            st.error(f"{type(exc).__name__}: {exc}")
            return

        bar.progress(100, text="Complete")
        st.session_state["last_result"] = result
        st.session_state["last_input_sig"] = input_sig
        show_result(result)
    elif (
        "last_result" in st.session_state
        and st.session_state.get("last_input_sig") == input_sig
    ):
        show_result(st.session_state["last_result"])


def show_result(result: dict):
    st.divider()
    render_urgency(result["triage"])

    st.markdown("### Transcript")
    st.info(result["transcript"])
    asr = result["asr"]
    if asr["audio_seconds"]:
        st.caption(
            f"{asr['audio_seconds']} s of audio, {asr['word_count']} words, "
            f"{asr['words_per_second']} words per second"
        )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Clinical entities")
        ents = result["ner"]["entities"]
        if ents:
            for label, items in ents.items():
                st.markdown(
                    f"**{label}:** " + ", ".join(f"{i['text']}" for i in items)
                )
            st.caption(
                f"{result['ner']['entity_count']} entities, mean confidence "
                f"{result['ner']['confidence']:.2f}"
            )
        else:
            st.write("No structured entities detected.")

        st.markdown("### Differential list")
        alts = result["triage"]["alternatives"]
        if alts:
            st.dataframe(
                pd.DataFrame(alts).rename(
                    columns={"condition": "Alternative condition", "score": "Score"}
                ),
                hide_index=True,
                width='stretch',
            )
        st.caption(f"Patient tone: {result['distress']['label']}")

    with c2:
        st.markdown("### Physician handover note")
        st.success(result["summary"]["summary"])
        s = result["summary"]
        st.caption(
            f"{s['source_words']} words condensed to {s['summary_words']} "
            f"(compression {s['compression_ratio']:.2f})"
        )

        st.markdown("### Applicable hospital policy")
        pol = result["policy"]
        st.write(pol["answer"])
        if pol["source_title"]:
            st.caption(
                f"Source: {pol['source_title']} ({pol['source_id']}), "
                f"retrieval score {pol['retrieval_score']:.2f}"
            )

    st.markdown("### Message sent to the patient")
    # Rendered as markdown rather than a text_area. A keyed widget holds its
    # value in session state and would keep showing the previous patient's
    # message after a new episode is run.
    # Both colours are fixed together: a hardcoded light background with an
    # inherited text colour turned white-on-white under the dark theme.
    st.markdown(
        "<div style='background:#f5f5f5;color:#1a1a1a;"
        "border-left:4px solid #1b5e20;"
        "padding:14px 18px;border-radius:4px;white-space:pre-wrap'>"
        f"{result['reply']['message']}</div>",
        unsafe_allow_html=True,
    )
    if result["reply"]["safety_banner_applied"]:
        st.warning("Emergency safety banner was prepended by the rule layer.")


def qa_tab():
    st.subheader("Patient question answering")
    st.caption(
        "Sub-task 5 on its own. Questions are answered only from the 15 approved "
        "hospital knowledge base passages, so the assistant cannot invent policy."
    )

    examples = [
        "What are the fasting requirements before a lipid profile?",
        "When is the emergency department open?",
        "What is the target HbA1c for an adult with diabetes?",
        "How long does it take to get a prescription refill?",
        "What should I avoid taking if I have dengue?",
    ]
    picked = st.selectbox("Example questions", [""] + examples)
    question = st.text_input("Question", value=picked)

    if st.button("Answer", type="primary") and question.strip():
        out = subtask5_qa.answer_question(question)
        if out["in_scope"]:
            st.success(out["answer"])
            st.caption(
                f"Source: {out['source_title']} ({out['source_id']})  |  "
                f"retrieval {out['retrieval_score']:.2f}  |  "
                f"span score {out.get('span_score', 0):.2f}"
            )
            with st.expander("Retrieved passages"):
                for p in out["retrieved"]:
                    st.markdown(f"**{p['title']}** (score {p['score']:.2f})")
                    st.write(p["text"])
        else:
            st.warning(out["answer"])


def llmops_tab():
    st.subheader("LLMOps metrics")
    st.caption(
        "Every sub-task invocation is instrumented. Metrics are written to "
        "artifacts/llmops_metrics.jsonl and mirrored into MLflow."
    )

    summary = llmops.summarise_metrics()
    if summary.empty:
        st.info("No invocations recorded yet. Run a patient episode first.")
        return

    raw = llmops.load_metrics()

    a, b, c, d = st.columns(4)
    a.metric("Total invocations", int(raw.shape[0]))
    b.metric("Mean latency", f"{raw['latency_ms'].mean():.0f} ms")
    c.metric("Success rate", f"{raw['success'].mean():.1%}")
    d.metric("Mean confidence", f"{raw[raw.confidence > 0]['confidence'].mean():.2f}")

    st.markdown("#### Metric 1 to 7, aggregated by sub-task")
    st.dataframe(
        summary.rename(
            columns={
                "task": "Sub-task",
                "invocations": "Calls",
                "avg_latency_ms": "Avg latency (ms)",
                "p95_latency_ms": "P95 latency (ms)",
                "avg_throughput_tps": "Throughput (tok/s)",
                "avg_confidence": "Avg confidence",
                "success_rate": "Success rate",
                "low_confidence_rate": "Low-confidence rate",
                "total_output_tokens": "Output tokens",
            }
        ),
        hide_index=True,
        width='stretch',
    )

    import plotly.express as px

    st.markdown("#### Latency by sub-task")
    st.plotly_chart(
        px.bar(
            summary.sort_values("avg_latency_ms"),
            x="avg_latency_ms",
            y="task",
            orientation="h",
            labels={"avg_latency_ms": "Average latency (ms)", "task": ""},
        ),
        width='stretch',
    )

    st.markdown("#### Latency trend over the session")
    raw = raw.reset_index().rename(columns={"index": "invocation"})
    st.plotly_chart(
        px.line(
            raw,
            x="invocation",
            y="latency_ms",
            color="task",
            labels={"latency_ms": "Latency (ms)", "invocation": "Invocation"},
        ),
        width='stretch',
    )

    calib = config.ARTIFACT_DIR / "threshold_calibration.csv"
    if calib.exists():
        st.markdown("#### Human-review threshold, chosen from the test split")
        st.caption(
            "The triage model does not need to be right on its own, it needs to "
            "know when it is unsure. At the chosen threshold of "
            f"{config.TRIAGE_REVIEW_THRESHOLD}, 88 percent of cases are routed "
            "automatically at 97.3 percent accuracy, and the remaining 12 "
            "percent go to a triage nurse."
        )
        st.dataframe(pd.read_csv(calib), hide_index=True, width='stretch')
        png = config.ARTIFACT_DIR / "threshold_calibration.png"
        if png.exists():
            st.image(str(png))

    with st.expander("Raw metric records"):
        st.dataframe(raw.tail(200), hide_index=True, width='stretch')


def finetune_tab():
    st.subheader("Fine-tuned model")
    summary_path = config.ARTIFACT_DIR / "finetune_summary.json"
    if not summary_path.exists():
        st.info("Run src/finetune/train_triage.py to produce the fine-tuning report.")
        return

    import json

    s = json.loads(summary_path.read_text())

    st.markdown(
        f"**Base model:** `{s['base_model']}`  \n"
        f"**Dataset:** `{s['dataset']}`  \n"
        f"**Records:** {s['train_records']} train, {s['test_records']} test, "
        f"{s['num_classes']} diagnosis classes  \n"
        f"**Training:** {s['epochs']} epochs, batch {s['batch_size']}, "
        f"learning rate {s['learning_rate']}, {s['train_seconds']} s"
    )

    base, tuned = s["baseline"], s["finetuned"]
    rows = []
    for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]:
        if key in tuned:
            rows.append(
                {
                    "Metric": key.replace("_", " "),
                    "Before fine-tuning": base.get(key, 0.0),
                    "After fine-tuning": tuned.get(key, 0.0),
                    "Gain": round(tuned.get(key, 0.0) - base.get(key, 0.0), 4),
                }
            )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

    cm = config.ARTIFACT_DIR / "finetune_confusion_matrix.png"
    if cm.exists():
        st.markdown("#### Confusion matrix on the held-out test split")
        st.image(str(cm))

    report = config.ARTIFACT_DIR / "finetune_classification_report.txt"
    if report.exists():
        with st.expander("Per-class classification report"):
            st.code(report.read_text())

    st.markdown("#### Try the fine-tuned classifier directly")
    text = st.text_area(
        "Symptom description",
        "I have a burning feeling when I pass urine and I need to go very often.",
        height=90,
    )
    if st.button("Classify"):
        clf = model_hub.triage_classifier()
        if clf is None:
            st.error("Fine-tuned model not found.")
        else:
            scores = sorted(
                clf(text, truncation=True, max_length=256)[0],
                key=lambda d: d["score"],
                reverse=True,
            )[:5]
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Condition": r["label"], "Probability": round(r["score"], 4)}
                        for r in scores
                    ]
                ),
                hide_index=True,
                width='stretch',
            )


def main():
    header()
    sidebar()
    t1, t2, t3, t4 = st.tabs(
        [
            "Patient episode",
            "Knowledge base QA",
            "LLMOps metrics",
            "Fine-tuned model",
        ]
    )
    with t1:
        episode_tab()
    with t2:
        qa_tab()
    with t3:
        llmops_tab()
    with t4:
        finetune_tab()


if __name__ == "__main__":
    main()
