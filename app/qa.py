"""Employee-facing Q&A over plan documents (retrieval-augmented, with citations + human handoff).

Retrieval uses TF-IDF so it runs anywhere with zero setup. Swapping in embeddings
(Chroma / pgvector) only means replacing `_retrieve`; the rest of the flow stays the same.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import llm

CONFIDENCE_THRESHOLD = 0.12
# Words that appear in nearly every benefits question/document carry no signal and cause
# false-confident matches ("does the plan cover my dog?" matching "Plan Overview").
STOP_WORDS = ENGLISH_STOP_WORDS | {"plan", "plans", "cover", "covers", "covered", "coverage",
                                   "benefit", "benefits", "summary", "employee", "employees"}
HANDOFF_MSG = ("I'm not confident I can answer that from your plan documents, so I've routed it to a "
               "benefits specialist, who will follow up within one business day.")

# Employees don't talk like plan documents. Cheap query expansion bridges the vocabulary gap.
EXPANSIONS = {
    r"\b(kid|kids|son|daughter|children)\b": "child dependent",
    r"\b(\d+\s*(year|yr)s?[- ]old|age|aged)\b": "age dependent children",
    r"\b(stay on|keep|remain)\b": "remain covered",
    r"\b(wife|husband|partner)\b": "spouse dependent",
    r"\b(baby|newborn|pregnan\w*)\b": "newborn birth child qualifying life event",
    r"\b(married|marriage|divorce\w*)\b": "qualifying life event special enrollment",
    r"\b(start|begin|kick in|when.*covered)\b": "begins waiting period",
    r"\b(quit|fired|laid off|leave|leaving)\b": "termination employment COBRA",
    r"\b(meds|medication|pills|prescriptions?|pharmacy)\b": "prescription drugs generic brand",
    r"\b(doctor|physician|checkup)\b": "primary care office visit copay",
    r"\b(er|emergency)\b": "emergency room copay",
    r"\b(max|maximum|most i)\b": "out-of-pocket maximum",
    r"\b(part[- ]?time|hours)\b": "hours per week eligible",
}


@dataclass
class Chunk:
    page: int
    section: str
    text: str


def _is_heading(line: str) -> bool:
    line = line.strip()
    return 2 < len(line) < 60 and not line.endswith((".", ",", ":")) and line[0].isupper() and not re.search(r"\$\d", line)


def chunk_pages(pages: list[tuple[int, str]], max_words: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_no, text in pages:
        section, buf = "General", []

        def flush():
            body = " ".join(buf).strip()
            if body:
                words = body.split()
                for i in range(0, len(words), max_words):
                    chunks.append(Chunk(page_no, section, " ".join(words[i:i + max_words])))

        for line in text.splitlines():
            if _is_heading(line) and buf:
                flush(); buf = []; section = line.strip()
            elif _is_heading(line) and not buf:
                section = line.strip()
            else:
                buf.append(line.strip())
        flush()
    return chunks


def expand_query(q: str) -> str:
    extra = [v for pat, v in EXPANSIONS.items() if re.search(pat, q, re.I)]
    return q + " " + " ".join(extra)


class PlanQA:
    def __init__(self, pages: list[tuple[int, str]], plan_name: str = "your plan"):
        self.plan_name = plan_name
        self.chunks = chunk_pages(pages)
        docs = [f"{c.section} {c.section} {c.text}" for c in self.chunks]  # weight headings
        self.vec = TfidfVectorizer(stop_words=list(STOP_WORDS), ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(docs)

    def _retrieve(self, question: str, k: int = 3) -> list[tuple[Chunk, float]]:
        sims = cosine_similarity(self.vec.transform([expand_query(question)]), self.matrix)[0]
        top = sims.argsort()[::-1][:k]
        return [(self.chunks[i], float(sims[i])) for i in top]

    @staticmethod
    def _best_sentences(text: str, question: str, vec: TfidfVectorizer, n: int = 2) -> str:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if len(sents) <= n:
            return text
        sims = cosine_similarity(vec.transform([expand_query(question)]), vec.transform(sents))[0]
        keep = sorted(sims.argsort()[::-1][:n])
        return " ".join(sents[i] for i in keep)

    def ask(self, question: str) -> dict:
        hits = self._retrieve(question)
        best_score = hits[0][1] if hits else 0.0
        citations = [{"page": c.page, "section": c.section, "score": round(s, 3)} for c, s in hits if s > 0]

        if best_score < CONFIDENCE_THRESHOLD:
            return {"answer": HANDOFF_MSG, "citations": [], "confidence": round(best_score, 3),
                    "handoff": True, "method": "handoff"}

        if llm.available():
            context = "\n\n".join(f"[{c.section}, page {c.page}]\n{c.text}" for c, _ in hits)
            text = llm.complete(
                "You answer employee benefits questions using ONLY the provided plan excerpts. "
                "Be brief and plain-spoken. Cite sections like (Section, p.N). If the excerpts don't "
                "answer the question, reply exactly: HANDOFF",
                f"Plan: {self.plan_name}\n\nExcerpts:\n{context}\n\nQuestion: {question}",
                max_tokens=500,
            )
            if text:
                if text.strip() == "HANDOFF":
                    return {"answer": HANDOFF_MSG, "citations": [], "confidence": round(best_score, 3),
                            "handoff": True, "method": "llm"}
                return {"answer": text.strip(), "citations": citations, "confidence": round(best_score, 3),
                        "handoff": False, "method": "llm"}

        top = hits[0][0]
        excerpt = self._best_sentences(top.text, question, self.vec)
        return {"answer": f"{excerpt} ({top.section}, p.{top.page})", "citations": citations,
                "confidence": round(best_score, 3), "handoff": False, "method": "extractive"}
