# GraphRAG / Knowledge-Graph in DeepTutor — research notes

> WIP scratch doc. Question: is adding GraphRAG / graph entity extraction / KG
> information worth pursuing, given the project's open nature makes building an
> ontology hard?

## TL;DR recommendation

**Don't bolt on full GraphRAG as a retrieval backend right now.** The higher-leverage,
lower-risk move is to grow the *existing* book-layer `ConceptGraph` into a real,
lightweight, per-KB knowledge layer used for navigation + prerequisite sequencing
+ retrieval boosting — not a wholesale RAG replacement.

Two reasons this beats "add GraphRAG":
1. The team **already shipped graph RAG (LightRAG + RAG-Anything) and deliberately removed it.**
2. The "ontology is hard" worry is **mostly a non-issue** with modern schema-free GraphRAG,
   so it shouldn't be the deciding factor — the real blockers are cost/complexity and
   the recent consolidation direction.

## The decisive context: they tried this and pulled it out

- `f377913 fix: remove lightrag and unify llm & search config` (2026-03-13)
- `ver-1-0-0-beta1.md`: *"the RAG pipeline has been **simplified to LlamaIndex only**.
  LightRAG and RAG-Anything pipelines … have been **temporarily removed** to focus on
  stability. They will be re-introduced in upcoming releases."*
- `ver1-1-2.md`: removed **~2,600 lines of unused RAG scaffolding** (chunkers, embedders,
  indexers, parsers, retrievers, orchestrator) — *"placeholders for never-shipped backends.
  The RAG service is now a thin wrapper over the single LlamaIndex pipeline."* Legacy
  `lightrag` provider values are silently coerced to `llamaindex` + KB flagged for re-index.

**Read:** LightRAG *is* graph-RAG (LLM entity/relation extraction → KG → graph+vector
retrieval). So the project's trajectory has been *away* from graph RAG toward aggressive
simplification. Re-adding it cuts against that, and "we'll re-introduce later" has not
materialized. Before building anything, the first question for the maintainers is *why*
LightRAG was dropped beyond the stated "stability" (cost? quality across weak providers?
maintenance? KB re-index churn?).

## On the "ontology is hard" point

Largely solved by current tools — **modern GraphRAG is schema-free / emergent**:
- MS GraphRAG and LightRAG do open LLM-based entity+relation extraction with *no
  predefined ontology*, then cluster (Leiden community detection) + hierarchically
  summarize. You don't hand-author a schema.
- The cost of schema-free is the flip side: **entity-resolution / dedup noise,
  inconsistent labels, and high indexing cost** — not the absence of a schema.

So the open domain is *not* the real obstacle. The real obstacles below are.

## Real trade-offs for THIS project

**Against (why it was likely dropped):**
- **Indexing cost & latency**: GraphRAG = many LLM calls per chunk (extraction) +
  community summarization. DeepTutor KBs are *user-uploaded and mutate*, and run across
  **40+ providers incl. local/weak models** (Ollama, LM Studio, llama.cpp). Extraction
  quality and cost vary wildly; weak models produce garbage graphs.
- **Complexity/maintenance**: adds a graph store, entity resolution, community detection,
  incremental re-index — right after the team spent two releases deleting that surface area.
- **Re-index churn**: every KB edit potentially re-extracts/re-clusters.

**For (where graph genuinely wins):**
- **Global / sensemaking queries** ("main themes across this corpus", "how do X and Y
  relate") that flat vector + BM25 handles poorly.
- **Multi-hop relational** questions.
- **Pedagogical fit**: for a *tutor*, the high-value artifact is a **prerequisite /
  concept-dependency graph** ("learn A before B"). That maps directly onto structure
  the project already models.

## Existing assets to build on (don't start from scratch)

- `deeptutor/book/blocks/concept_graph.py` + models in `deeptutor/book/models.py`:
  `ConceptNode`, `ConceptEdge`, `ConceptGraph` with relations **`depends_on` / `extends`
  / `related`**, rendered as Mermaid. Built *deterministically from book-spine synthesis*
  (no NER) and used only for "living book" visualization today.
- `deeptutor/services/rag/smart_retriever.py`: multi-query expansion — natural seam to add
  entity-anchored expansion.
- `deeptutor/services/rag/factory.py`: still tolerates legacy `lightrag` provider strings;
  a second pipeline *could* slot back in here, but see "against".
- Three-layer memory + a memory graph already exist (`api/routers/memory.py`) — proves the
  team is comfortable with graph-shaped UX.

## Suggested path (if pursuing)

1. **Phase 0 — ask maintainers** why LightRAG was removed; that answer dominates everything.
2. **Phase 1 (recommended)** — promote `ConceptGraph` from book-only viz to a per-KB
   *concept layer*: LLM-extract emergent concept nodes + `depends_on` edges scoped to one
   KB (schema-free, so the open domain is fine). Use it for navigation, prerequisite
   sequencing, and as a *booster* in `smart_retriever` (entity-anchored expansion), **not**
   a replacement backend. Reuses existing models; low blast radius; clear tutor payoff.
3. **Phase 2 (only if global-query demand is real)** — reintroduce **LightRAG** (lighter,
   incremental) over MS GraphRAG (batch, expensive), gated **opt-in per-KB** because of
   cost. This is essentially the dual-pipeline they just removed — proceed deliberately.

## Architecture quick-map (for laptop continuation)

- RAG service facade: `deeptutor/services/rag/service.py`
- LlamaIndex pipeline (BM25/vector hybrid): `deeptutor/services/rag/pipelines/llamaindex/pipeline.py`
- Doc ingestion (PDF/Office/text/image/code): `…/llamaindex/document_loader.py`
- Embeddings: `deeptutor/services/embedding/client.py`; LLM factory: `deeptutor/services/llm/provider_factory.py`
- Provider catalog (40+): `deeptutor/services/provider_registry.py`
- Concept graph: `deeptutor/book/blocks/concept_graph.py`, `deeptutor/book/models.py`
