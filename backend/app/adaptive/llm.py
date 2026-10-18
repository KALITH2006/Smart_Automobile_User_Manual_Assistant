"""LLM-backed pieces: router classifier, query rewriter, and grounded structured generation."""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .types import Hit, Mode

# ---------------------------------------------------------------- router classifier / rewriter

_CLASSIFY_PROMPT = """You route vehicle-manual search queries to a retrieval mode.
Modes: M1 = exact part/term lookup, M2 = general question, M3 = vague or multi-symptom problem.
Reply with JSON only: {{"mode": "M1|M2|M3", "confidence": <0..1>}}

Query: {query}"""

_REWRITE_PROMPT = """Rewrite this vehicle-manual search query to retrieve better passages.
The previous search returned weak results. Use synonyms and the terms an owner's manual would use
(parts, warning lights, procedures). Keep it under 25 words. Reply with the query only.

Original complaint: {original}
Previous query: {previous}
Weak passages (page: start of text):
{passages}"""


def _content(msg: Any) -> str:
    return getattr(msg, "content", msg) if not isinstance(msg, str) else msg


def _json_object(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


def make_classifier(llm):
    """Returns classifier(text) -> (Mode, confidence). Falls back to low-confidence M3 on bad output."""

    def classify(text: str) -> tuple[Mode, float]:
        try:
            data = _json_object(_content(llm.invoke(_CLASSIFY_PROMPT.format(query=text))))
            return Mode(data["mode"]), float(data["confidence"])
        except (KeyError, ValueError, json.JSONDecodeError):
            return Mode.M3, 0.0

    return classify


def make_rewriter(llm):
    def rewrite(previous: str, evidence: list[Hit], original: str) -> str:
        passages = "\n".join(f"{h.page}: {h.text[:120]}" for h in evidence[:3]) or "(none)"
        out = _content(llm.invoke(_REWRITE_PROMPT.format(original=original, previous=previous, passages=passages)))
        out = out.strip().strip('"').splitlines()[0] if out.strip() else ""
        return out or previous

    return rewrite


# ---------------------------------------------------------------- grounded generation

class Cause(BaseModel):
    cause: str
    verification_steps: list[str] = Field(default_factory=list)
    citations: list[int] = Field(default_factory=list, description="page numbers of supporting sections")


class Diagnosis(BaseModel):
    causes: list[Cause]
    unsupported: list[str] = Field(default_factory=list, description="claims the evidence does not support")


_GEN_PROMPT = """You are a vehicle diagnostic assistant. Use ONLY the manual sections below.
Rank the probable causes, give verification steps to perform BEFORE replacing parts, and cite the
page number of the section supporting each cause. List anything the sections do not support under
"unsupported". Do not invent pages, parts or specifications.

Reply with JSON only, matching:
{{"causes": [{{"cause": str, "verification_steps": [str], "citations": [int]}}], "unsupported": [str]}}

Vehicle/context: {context}
Question: {question}

Manual sections:
{sections}"""

_SAFETY_TRIGGERS = {
    "brake": "braking system", "airbag": "airbag / restraint system", "fuel leak": "fuel system",
    "smell of fuel": "fuel system", "fuel smell": "fuel system", "smell fuel": "fuel system",
    "high-voltage": "high-voltage system", "high voltage": "high-voltage system",
    "overheat": "cooling system / overheating", "steering": "steering system",
}


def safety_notice(question: str, evidence: list[Hit] | None = None) -> str | None:
    """Deterministic safety notice for safety-critical systems (not left to the model).

    Keyed on the user's question only: matching against retrieved text produced false alarms,
    e.g. a tire-rotation answer whose page also mentioned steering.
    """
    blob = question.lower()
    systems = sorted({label for key, label in _SAFETY_TRIGGERS.items() if key in blob})
    if not systems:
        return None
    return (
        f"Safety notice: this involves the {', '.join(systems)}. "
        "Have a qualified technician inspect the vehicle before relying on this guidance."
    )


def citation_precision(diagnosis: Diagnosis, evidence: list[Hit]) -> float:
    """Fraction of cited pages that actually appear in the retrieved evidence."""
    pages = {h.page for h in evidence}
    cited = [p for c in diagnosis.causes for p in c.citations]
    return 1.0 if not cited else sum(p in pages for p in cited) / len(cited)


def generate_diagnosis(llm, question: str, evidence: list[Hit], context: str = "n/a") -> Diagnosis | None:
    sections = "\n\n".join(f"[Page {h.page}]\n{h.text}" for h in evidence)
    raw = _content(llm.invoke(_GEN_PROMPT.format(context=context, question=question, sections=sections)))
    try:
        return Diagnosis.model_validate(_json_object(raw))
    except (ValidationError, json.JSONDecodeError):
        return None


def render_markdown(diagnosis: Diagnosis, notice: str | None) -> str:
    lines: list[str] = []
    for i, c in enumerate(diagnosis.causes, start=1):
        pages = ", ".join(f"p.{p}" for p in sorted(set(c.citations)))
        lines.append(f"**{i}. {c.cause}**" + (f" ({pages})" if pages else ""))
        lines += [f"   - {s}" for s in c.verification_steps]
    if diagnosis.unsupported:
        lines.append("\n_Not supported by the manual:_ " + "; ".join(diagnosis.unsupported))
    if notice:
        lines.append(f"\n> ⚠️ {notice}")
    return "\n".join(lines)
