# Knowledge base — retrieval corpus for the RAG Q&A layer

These markdown files are the **only** material the "Ask about this result"
feature is allowed to ground its answers in, alongside the numbers from the
measurement being asked about. If a fact is not in here, the assistant is
instructed to say it does not have a reference for it rather than answer from
memory. That is the whole point: an ungrounded LLM paragraph is not auditable.

## How chunking works

`app/api/knowledge_base.py` splits each file on `##` headings. **Each `##`
section becomes one retrievable chunk**, so:

- Keep every section short and self-contained — one fact or one definition.
- Do not rely on context from a neighbouring section; a chunk is retrieved
  alone and the model may see it without its siblings.
- The heading text is indexed too, so make headings descriptive.

The citation shown to the user is `file.md § Heading`, so headings are
user-facing. Write them as something a clinician would recognise.

## ⚠️ Placeholder policy — read before trusting these files

Sections marked **`TODO(verbatim)`** contain a *paraphrase or a structural
placeholder*, not sourced guideline text. They are deliberately conservative
and avoid stating specific numeric thresholds that I could not verify against
the primary document.

**Do not present TODO-marked content as an authoritative quotation.** Replace
it with the exact text from the source you have licensed access to, then delete
the TODO marker. Anything describing this project's *own* implementation
(`project_metrics.md`, and the Hadlock coefficients as coded) is derived
directly from the source in this repository and is accurate as written.

The retrieval layer surfaces a `provisional: true` flag on any chunk still
carrying a TODO marker, and the API echoes it in the response so the UI can
badge an answer as resting on unverified reference text.

## Adding a new reference file

1. Drop a `.md` file in this folder.
2. Give every fact its own `##` section.
3. Add a `> Source:` line under the heading — it is shown with the citation.
4. Restart the API (the index builds once at startup).
