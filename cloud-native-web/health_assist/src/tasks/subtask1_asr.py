"""Sub-task 1 - Speech Recognition: transcribe a patient voice note.

Category: Speech Recognition
Model:    openai/whisper-large-v3-turbo
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

import config
from llmops.metrics import tracked
from tasks import model_hub


def _load_audio(path: str, target_sr: int = 16000):
    import librosa

    waveform, _ = librosa.load(path, sr=target_sr, mono=True)
    # Peak-normalise. Phone voice notes often arrive quiet, and Whisper's
    # accuracy drops on low-level audio; scaling to full range costs nothing
    # and never hurts a well-recorded clip.
    peak = float(np.max(np.abs(waveform))) if len(waveform) else 0.0
    if peak > 0:
        waveform = waveform / peak * 0.95
    return waveform.astype(np.float32)


@tracked("1_speech_recognition", config.ASR_MODEL)
def transcribe(audio_path: str) -> dict:
    """Convert a recorded patient complaint into text.

    Whisper does not expose a class probability, so the mean of the exponentiated
    average log-probability is unavailable in the simple pipeline call. We report
    a length-based reliability proxy instead: very short transcripts of long audio
    usually indicate a failed recording.
    """
    waveform = _load_audio(audio_path)
    duration_s = len(waveform) / 16000.0
    result = model_hub.asr()(waveform)
    text = (result.get("text") or "").strip()

    words = len(text.split())
    words_per_second = words / duration_s if duration_s > 0 else 0.0
    # Normal conversational speech is roughly 2 to 3 words per second.
    reliability = min(1.0, words_per_second / 2.0) if duration_s > 1 else 1.0

    return {
        "transcript": text,
        "audio_seconds": round(duration_s, 2),
        "word_count": words,
        "words_per_second": round(words_per_second, 2),
        "confidence": round(reliability, 4),
    }


@tracked("1_speech_recognition_text", "manual-entry")
def accept_typed_text(text: str) -> dict:
    """Fallback path when the patient types instead of speaking."""
    text = (text or "").strip()
    return {
        "transcript": text,
        "audio_seconds": 0.0,
        "word_count": len(text.split()),
        "words_per_second": 0.0,
        "confidence": 1.0,
    }
