"""Lazy, process-wide model loader.

Each model is instantiated on first use and cached, so the Streamlit app pays
the load cost once rather than on every interaction.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)

import config

_CACHE: dict[str, object] = {}

DEVICE = -1  # CPU keeps the demo reproducible on any machine


def _get(key: str, builder):
    if key not in _CACHE:
        _CACHE[key] = builder()
    return _CACHE[key]


class _Seq2Seq:
    """Minimal generation wrapper.

    transformers 5.x removed the ``summarization`` and ``text2text-generation``
    pipelines, so the encoder-decoder models are driven directly here. The call
    signature matches what the sub-task modules expect from the old pipelines.
    """

    def __init__(self, model_name: str, output_key: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
        self.output_key = output_key

    def __call__(self, text: str, **kwargs):
        kwargs.pop("truncation", None)
        max_new = kwargs.pop("max_new_tokens", None)
        min_new = kwargs.pop("min_new_tokens", None)
        max_len = kwargs.pop("max_length", None)
        min_len = kwargs.pop("min_length", None)

        gen_kwargs = dict(kwargs)
        if max_new is not None:
            gen_kwargs["max_new_tokens"] = max_new
        elif max_len is not None:
            gen_kwargs["max_length"] = max_len
        if min_new is not None:
            gen_kwargs["min_new_tokens"] = min_new
        elif min_len is not None:
            gen_kwargs["min_length"] = min_len

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.model.config.max_position_embeddings
            if hasattr(self.model.config, "max_position_embeddings")
            else 1024,
        )
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)
        text_out = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return [{self.output_key: text_out}]


class _ChatSLM:
    """Instruction-tuned causal model driven through its chat template."""

    def __init__(self, model_name: str):
        from transformers import AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32
        )
        self.model.eval()

    def __call__(self, system: str, user: str, max_new_tokens: int = 200) -> str:
        text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=1.05,
            )
        generated = out[0][enc["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


class _ExtractiveQA:
    """Span extraction over a context, replacing the removed QA pipeline."""

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.eval()

    def __call__(self, question: str, context: str, max_answer_tokens: int = 40):
        enc = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation="only_second",
            max_length=512,
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")[0]
        seq_ids = enc.sequence_ids(0)

        with torch.no_grad():
            out = self.model(**enc)

        start_logits = out.start_logits[0]
        end_logits = out.end_logits[0]

        # Restrict the span to tokens that belong to the context.
        mask = torch.tensor(
            [sid != 1 for sid in seq_ids], dtype=torch.bool
        )
        start_logits = start_logits.masked_fill(mask, -1e9)
        end_logits = end_logits.masked_fill(mask, -1e9)

        start_probs = torch.softmax(start_logits, dim=-1)
        end_probs = torch.softmax(end_logits, dim=-1)

        best_score, best_span = 0.0, (0, 0)
        top_starts = torch.topk(start_probs, k=20).indices.tolist()
        top_ends = torch.topk(end_probs, k=20).indices.tolist()
        for s in top_starts:
            for e in top_ends:
                if e < s or e - s + 1 > max_answer_tokens:
                    continue
                score = float(start_probs[s] * end_probs[e])
                if score > best_score:
                    best_score, best_span = score, (s, e)

        s, e = best_span
        char_start = int(offsets[s][0])
        char_end = int(offsets[e][1])
        answer = context[char_start:char_end].strip() if char_end > char_start else ""
        return {"answer": answer, "score": best_score}


def asr():
    return _get(
        "asr",
        lambda: pipeline(
            "automatic-speech-recognition",
            model=config.ASR_MODEL,
            device=DEVICE,
            chunk_length_s=30,
        ),
    )


def triage_classifier():
    """Return the fine-tuned triage classifier, or None if not yet trained."""
    if not config.FINETUNED_TRIAGE_DIR.exists():
        return None
    return _get(
        "triage",
        lambda: pipeline(
            "text-classification",
            model=str(config.FINETUNED_TRIAGE_DIR),
            tokenizer=str(config.FINETUNED_TRIAGE_DIR),
            device=DEVICE,
            top_k=None,
        ),
    )


def base_triage_classifier():
    """Untrained baseline, used to show the value added by fine-tuning."""
    return _get(
        "triage_base",
        lambda: pipeline(
            "text-classification",
            model=config.BASE_CLASSIFIER,
            device=DEVICE,
            top_k=None,
        ),
    )


def ner():
    return _get(
        "ner",
        lambda: pipeline(
            "token-classification",
            model=config.NER_MODEL,
            aggregation_strategy="simple",
            device=DEVICE,
        ),
    )


def summariser():
    return _get("sum", lambda: _Seq2Seq(config.SUMMARISER_MODEL, "summary_text"))


def qa():
    return _get("qa", lambda: _ExtractiveQA(config.QA_MODEL))


def generator():
    """The chat SLM, or None in lite mode where no generator model is set."""
    if not config.GENERATOR_MODEL:
        return None
    return _get("gen", lambda: _ChatSLM(config.GENERATOR_MODEL))


def sentiment():
    return _get(
        "sent",
        lambda: pipeline(
            "text-classification", model=config.SENTIMENT_MODEL, device=DEVICE
        ),
    )


def embedder():
    from sentence_transformers import SentenceTransformer

    return _get("emb", lambda: SentenceTransformer(config.EMBEDDING_MODEL))


def warm_up(include_triage: bool = True) -> list[str]:
    """Load every model up front and report which ones are ready."""
    loaded = []
    for name, fn in [
        ("ASR (Whisper)", asr),
        ("Biomedical NER", ner),
        ("Summariser", summariser),
        ("Question Answering", qa),
        ("Guidance Generator", generator),
        ("Distress Sentiment", sentiment),
        ("Retriever", embedder),
    ]:
        try:
            fn()
            loaded.append(name)
        except Exception as exc:
            loaded.append(f"{name} FAILED: {type(exc).__name__}")
    if include_triage and triage_classifier() is not None:
        loaded.append("Fine-tuned Triage Classifier")
    return loaded
