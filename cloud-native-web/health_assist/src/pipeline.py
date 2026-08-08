"""Unified objective (requirement 5).

One patient episode flows through every sub-task in a single chain:

    voice note
      -> [1] Speech Recognition      transcript
      -> [2] Text Classification     provisional condition + urgency band
      -> [3] Named Entity Recognition structured clinical findings
      -> [5] Question Answering       the hospital policy that governs the case
      -> [4] Summarisation            handover note for the physician
      -> [6] Text Generation          reply sent back to the patient

Each stage consumes the output of the earlier stages, which is what makes the
six sub-tasks one application rather than six separate demonstrations.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from tasks import (
    subtask1_asr,
    subtask2_triage,
    subtask3_ner,
    subtask4_summary,
    subtask5_qa,
    subtask6_guidance,
)

# The policy question asked of the knowledge base depends on the urgency band,
# so the reply the patient receives is grounded in the right hospital document.
POLICY_QUESTION = {
    "EMERGENCY": "What should a patient do about emergency warning signs?",
    "HIGH": "How does a patient book a same-day specialist appointment?",
    "ROUTINE": "How does a patient book an out-patient appointment?",
}


def run_episode(
    audio_path: str | None = None,
    typed_text: str | None = None,
    progress=None,
) -> dict:
    """Run one complete patient episode and return every intermediate result."""

    # Per-stage wall time. The @tracked decorator already logs each model call
    # for the LLMOps tab, but that is aggregated across every episode ever run.
    # These are the timings for this one episode, which is what tells you which
    # stage made this particular patient wait.
    timings: dict[str, float] = {}
    open_stage: list = [None, time.perf_counter()]

    def step(n: int, label: str):
        now = time.perf_counter()
        if open_stage[0] is not None:
            timings[open_stage[0]] = round((now - open_stage[1]) * 1000, 1)
        open_stage[0] = f"{n}. {label}"
        open_stage[1] = now
        if progress:
            progress(n, label)

    def close_last():
        if open_stage[0] is not None:
            timings[open_stage[0]] = round(
                (time.perf_counter() - open_stage[1]) * 1000, 1
            )
            open_stage[0] = None

    # Sub-task 1 - Speech Recognition
    step(1, "Transcribing the voice note")
    if audio_path:
        asr = subtask1_asr.transcribe(audio_path)
    else:
        asr = subtask1_asr.accept_typed_text(typed_text or "")
    transcript = asr["transcript"]
    if not transcript:
        raise ValueError("No speech was recognised in the recording.")

    # Sub-task 2 - Text Classification (fine-tuned model) and distress check
    step(2, "Classifying the condition and urgency")
    triage = subtask2_triage.classify(transcript)
    distress = subtask2_triage.distress(transcript)

    # A machine-translated complaint is evidence about the patient, not the
    # patient's own words. Whisper identifies the spoken language reliably even
    # at tiny size, but a small checkpoint's translation is not clinically
    # trustworthy: on a Spanish test clip "me siento muy debil" (I feel very
    # weak) came back as "I feel very happy". Everything downstream reads
    # English, including the red-flag phrase list, so one inverted word can
    # flip the routing without lowering the model's confidence. Language
    # detection is therefore used as a gate rather than a green light, and a
    # translated episode always reaches a human whatever the model scored.
    if asr.get("translated"):
        triage["review_required"] = True
        triage["review_reason"] = (
            f"the complaint was spoken in "
            f"{(asr.get('language') or 'another language').title()} and machine-"
            f"translated before triage, so the wording triage acted on is not "
            f"the patient's own"
        )

    # Sub-task 3 - Named Entity Recognition
    step(3, "Extracting clinical entities")
    ner = subtask3_ner.extract_entities(transcript)
    clinical_note = subtask3_ner.as_clinical_note(ner)

    # Sub-task 5 - Question Answering, used here to fetch the governing policy
    step(4, "Retrieving the applicable hospital policy")
    policy = subtask5_qa.answer_question(
        POLICY_QUESTION.get(triage["urgency"], POLICY_QUESTION["ROUTINE"])
    )

    # Sub-task 4 - Summarisation. When the rule layer has escalated, the
    # classifier's condition is known to be unreliable (the emergency band is
    # only ever reached through a red-flag phrase), so the handover note
    # carries the override instead of presenting the prediction as the
    # assessment.
    step(5, "Writing the clinician handover note")
    handover_condition = triage["condition"]
    if triage.get("rule_escalated"):
        # The overridden label itself is deliberately NOT named here: it adds
        # no clinical value to the handover and could bias the physician. It
        # remains available in the audit panel and the metrics log.
        handover_condition = (
            f"a red-flag emergency ({triage.get('red_flag') or 'red-flag phrase'}); "
            "the model's provisional label was overridden by the rule layer "
            "and is retained in the audit trail"
        )
    elif triage.get("review_required"):
        # The physician should see at a glance that the label is provisional.
        handover_condition = (
            f"{triage['condition']} (low model confidence, "
            "queued for nurse review)"
        )
    summary = subtask4_summary.summarise(
        transcript, clinical_note=clinical_note, condition=handover_condition
    )

    # Sub-task 6 - Text Generation
    step(6, "Drafting the patient reply")
    reply = subtask6_guidance.draft_reply(
        transcript=transcript,
        condition=triage["condition"],
        urgency=triage["urgency"],
        routing=triage["routing_action"],
        policy=policy["answer"],
    )

    close_last()

    return {
        "transcript": transcript,
        "asr": asr,
        "triage": triage,
        "distress": distress,
        "ner": ner,
        "clinical_note": clinical_note,
        "policy": policy,
        "summary": summary,
        "reply": reply,
        "timings_ms": timings,
        "total_ms": round(sum(timings.values()), 1),
    }


def format_console(result: dict) -> str:
    """Plain-text rendering, used by the command line demonstration."""
    t, tri, s, r = (
        result["transcript"],
        result["triage"],
        result["summary"],
        result["reply"],
    )
    asr = result.get("asr") or {}
    # The translation warning leads the record. A clinician reading the
    # exported note has no other way to know the transcript is not what the
    # patient actually said.
    banner = ""
    if asr.get("translated"):
        banner = (
            f"\n*** MACHINE TRANSLATION FROM "
            f"{(asr.get('language') or 'UNKNOWN').upper()} ***\n"
            "The transcript below is not the patient's own words. Triage was "
            "computed from this translation.\nConfirm with the patient or an "
            "interpreter before acting.\n"
        )
    lines = [
        "=" * 72,
        "PATIENT EPISODE",
        "=" * 72,
        banner,
        f"\n[1] TRANSCRIPT\n{t}",
        f"\n[2] TRIAGE\n    Condition : {tri['condition']} ({tri['confidence']:.1%})"
        + (f"\n    Red flag  : {tri['red_flag']} (rule layer overrode the model)"
           if tri.get("red_flag") else "")
        + f"\n    Urgency   : {tri['urgency']}"
        f"\n    Action    : {tri['routing_action']}"
        f"\n    Patient   : {result['distress']['label']}"
        + ("\n    Review    : low confidence, queued for a triage nurse"
           if tri.get("review_required") else ""),
        f"\n[3] CLINICAL ENTITIES\n{result['clinical_note']}",
        f"\n[5] HOSPITAL POLICY ({result['policy']['source_title']})\n{result['policy']['answer']}",
        f"\n[4] HANDOVER NOTE\n{s['summary']}",
        f"\n[6] PATIENT REPLY\n{r['message']}",
        "=" * 72,
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    demo = (
        "I have been having a really bad headache for the last four days and a "
        "high fever that will not come down. There is pain behind my eyes and my "
        "joints and muscles ache badly. Since yesterday a rash has appeared on my "
        "arms and I feel very weak and nauseous."
    )
    text = " ".join(sys.argv[1:]) or demo
    print(format_console(run_episode(typed_text=text)))
