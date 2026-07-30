"""Retrieval layer for the grounded Q&A feature.

Loads the markdown files in ``knowledge/``, splits them into one chunk per
``##`` section, and serves the top-scoring chunks for a question.

Why lexical retrieval rather than a dense vector store
------------------------------------------------------
The corpus is a few dozen short, hand-curated sections whose distinguishing
terms are rare and exact — ISUOG, Hadlock, Dice, Ramanujan, reliability,
calvarium. TF-IDF over word n-grams plus a prefix-stem branch matches those
directly and needs nothing beyond scikit-learn, already a dependency: no extra
install weight, no embedding model fetched over the network while the Space
container is starting, and a deterministic index that tests can assert on.

Dense embeddings would earn their keep on a large corpus with paraphrased
queries. If that changes, implement :class:`Retriever` with an embedding
backend and swap it in ``_build_index`` — nothing else in the module depends
on how similarity is computed.

Nothing here raises: a missing or empty ``knowledge/`` directory yields an
empty index and ``retrieve`` returns no chunks, which the endpoint reports as
"no supporting reference found" rather than letting the model answer unbacked.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# knowledge/ sits at the repo root, next to app/ and demo_subjects/.
KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge"

# README.md documents the corpus for maintainers; it is not reference material
# and must never be cited in an answer.
_EXCLUDED_FILES = {"readme.md"}

# A chunk still carrying this marker is a paraphrase, not sourced guideline
# text. Answers resting on one are flagged so the UI can say so.
_PROVISIONAL_MARKER = "TODO(verbatim)"

DEFAULT_TOP_K = 4
# Below this cosine score a chunk is noise rather than a weak match. Tuned so
# an off-topic question ("what is the capital of France") retrieves nothing.
MIN_SCORE = 0.05


@dataclass(frozen=True)
class Chunk:
    """One retrievable section of a reference file."""

    chunk_id: str  # "isuog_hc_measurement.md#correct-plane-for-..."
    source_file: str  # "isuog_hc_measurement.md"
    heading: str  # "Correct plane for head circumference measurement"
    text: str  # body, excluding the heading and the > Source: line
    source_note: str  # the "> Source: ..." line, if present
    provisional: bool  # body still contains TODO(verbatim)

    @property
    def citation(self) -> str:
        """Human-readable citation shown next to an answer."""
        return f"{self.source_file} § {self.heading}"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def parse_markdown(path: Path) -> list[Chunk]:
    """Split one markdown file into chunks, one per ``##`` heading.

    Content before the first ``##`` (the ``#`` title and any preamble) is
    dropped: it is framing, not a retrievable fact.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001 — an unreadable file must not break startup
        log.warning("knowledge: could not read %s", path, exc_info=True)
        return []

    chunks: list[Chunk] = []
    # Split on level-2 headings only; ### stays inside its parent chunk.
    sections = re.split(r"^##\s+(.+)$", raw, flags=re.MULTILINE)
    # sections = [preamble, heading1, body1, heading2, body2, ...]
    for i in range(1, len(sections) - 1, 2):
        heading = sections[i].strip()
        body = sections[i + 1].strip()
        if not body:
            continue

        source_lines = [ln for ln in body.splitlines() if ln.strip().startswith("> Source:")]
        source_note = source_lines[0].strip().lstrip("> ").strip() if source_lines else ""

        # Strip blockquote lines from the indexed text — they are provenance
        # metadata, not content to answer from.
        text = "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith(">")).strip()
        if not text:
            continue

        chunks.append(
            Chunk(
                chunk_id=f"{path.name}#{_slugify(heading)}",
                source_file=path.name,
                heading=heading,
                text=text,
                source_note=source_note,
                provisional=_PROVISIONAL_MARKER in body,
            )
        )
    return chunks


def load_chunks(directory: Path | None = None) -> list[Chunk]:
    """Load every chunk from the knowledge directory, sorted for determinism."""
    directory = directory or KNOWLEDGE_DIR
    if not directory.is_dir():
        log.warning("knowledge: directory %s does not exist — Q&A will be unavailable", directory)
        return []

    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        if path.name.lower() in _EXCLUDED_FILES:
            continue
        chunks.extend(parse_markdown(path))
    return chunks


def _stem_analyzer(doc: str) -> list[str]:
    """Tokenise to lowercase words and truncate each to a 6-character prefix.

    A deliberately crude stemmer: it collapses reliable/reliability,
    measure/measurement/measured and gestation/gestational onto a shared key
    without pulling in a stemming dependency. Truncation is safe on this
    corpus because the domain vocabulary does not collide in its first six
    characters; revisit if the knowledge base grows much larger.
    """
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    tokens = re.findall(r"[a-z]{3,}", doc.lower())
    return [t[:6] for t in tokens if t not in ENGLISH_STOP_WORDS]


class Retriever:
    """TF-IDF + cosine similarity over the chunk corpus.

    Swap this class for an embedding-backed implementation to change the
    retrieval strategy; the public surface is ``search(query, top_k)``.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._matrix = None
        self._vectorizer = None
        if not chunks:
            return
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.pipeline import FeatureUnion

            # The heading is repeated into the indexed text so a question
            # echoing a section title scores that section highly.
            corpus = [f"{c.heading}. {c.heading}. {c.text}" for c in chunks]

            # Two branches. Word features carry the topical signal; the stem
            # branch supplies the morphology matching that plain TF-IDF lacks.
            #
            # Without it, "how RELIABLE is this measurement" scores the
            # RELIABILITY sections at zero — "reliable" and "reliability" are
            # unrelated tokens — and the query lands on ISUOG text that merely
            # shares the word "measurement". Verified before and after.
            #
            # Character n-grams were tried first and rejected: they bridge the
            # stem but smear similarity everywhere, so "what is the capital of
            # France" scored 0.097 and cleared MIN_SCORE. Prefix stemming is
            # strictly better here — it fixes the ranking AND drops off-topic
            # queries to exactly 0.000, because their stems appear nowhere in
            # the corpus.
            self._vectorizer = FeatureUnion(
                [
                    (
                        "word",
                        TfidfVectorizer(
                            lowercase=True,
                            stop_words="english",
                            ngram_range=(1, 2),
                            sublinear_tf=True,
                        ),
                    ),
                    ("stem", TfidfVectorizer(analyzer=_stem_analyzer, sublinear_tf=True)),
                ]
            )
            self._matrix = self._vectorizer.fit_transform(corpus)
        except Exception:  # noqa: BLE001 — degrade to no-retrieval, never crash boot
            log.warning("knowledge: failed to build the TF-IDF index", exc_info=True)
            self._matrix = None
            self._vectorizer = None

    @property
    def ready(self) -> bool:
        return self._matrix is not None and bool(self.chunks)

    def search(
        self, query: str, top_k: int = DEFAULT_TOP_K, min_score: float = MIN_SCORE
    ) -> list[RetrievedChunk]:
        if not self.ready or not query.strip():
            return []
        try:
            from sklearn.metrics.pairwise import cosine_similarity

            q = self._vectorizer.transform([query])
            scores = cosine_similarity(q, self._matrix)[0]
        except Exception:  # noqa: BLE001
            log.warning("knowledge: retrieval failed", exc_info=True)
            return []

        ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        out: list[RetrievedChunk] = []
        for i in ranked[:top_k]:
            score = float(scores[i])
            if score < min_score:
                break
            out.append(RetrievedChunk(chunk=self.chunks[i], score=round(score, 4)))
        return out


# ── module-level singleton ────────────────────────────────────────────────────

_retriever: Retriever | None = None
_lock = threading.Lock()


def _build_index() -> Retriever:
    chunks = load_chunks()
    r = Retriever(chunks)
    log.info(
        "knowledge: indexed %d chunks from %d files (%d provisional)",
        len(chunks),
        len({c.source_file for c in chunks}),
        sum(1 for c in chunks if c.provisional),
    )
    return r


def get_retriever() -> Retriever:
    """Process-wide retriever, built once on first use."""
    global _retriever
    if _retriever is None:
        with _lock:
            if _retriever is None:
                _retriever = _build_index()
    return _retriever


def reset_retriever() -> None:
    """Drop the cached index. For tests, and for a future reload endpoint."""
    global _retriever
    with _lock:
        _retriever = None


def retrieve(
    query: str, top_k: int = DEFAULT_TOP_K, min_score: float = MIN_SCORE
) -> list[RetrievedChunk]:
    return get_retriever().search(query, top_k=top_k, min_score=min_score)
