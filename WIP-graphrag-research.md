# GraphRAG / Knowledge-Graph in DeepTutor — a decision case study

> **Question:** Is adding GraphRAG / graph entity-extraction / knowledge-graph
> information to DeepTutor worth pursuing — especially given that the project's
> open, any-domain nature makes building an ontology hard?
>
> **Short answer:** The ontology worry is a *red herring* — modern graph RAG is
> schema-free. The real reasons to be cautious are different and stronger, and
> the team has already lived them once. **Don't reinstate a graph-RAG retrieval
> backend. Do build a per-KB concept/prerequisite graph** on top of the existing
> `ConceptGraph`, used for navigation + sequencing + retrieval boosting. Full
> reasoning below.

---

## 0. How to read this document

This is a weighing exercise across three evidence bases:
1. **What DeepTutor already tried** (git archaeology — the most decisive evidence).
2. **What it would actually cost to build today** (codebase integration map).
3. **What the field knows** (2026 SOTA, with measured numbers vs vendor claims).

Numbers are tagged `[measured]` (traceable to a paper/benchmark) vs `[blog/vendor]`
(directional only). Code references are `file:line` at current HEAD unless noted.

---

## 1. The decisive fact: DeepTutor already shipped graph RAG and tore it out

This is the single most important input to the decision, and it is not hypothetical.

- **LightRAG was present from the very first commit** (`fbb0b15`, 2025-12-28) and
  was removed on **2026-03-13** (`f377913 fix: remove lightrag and unify llm & search config`).
  Lifespan ≈ **2.5 months**, during which **~83 commits** touched LightRAG/RAG-Anything code.
- At its high-water mark (v0.5.0, `ver0-5-0.md`) DeepTutor advertised **three
  selectable RAG pipelines per KB**:
  - **LlamaIndex** — vector/direct, "Fastest"
  - **LightRAG** — knowledge graph, "Fast"
  - **RAG-Anything** — multimodal graph (MinerU + Docling), "Thorough"
- The removal commit was **239 files, +4,869 / −9,453 lines**. A follow-up cleanup
  (`ver1-1-2`, `100576d`) deleted ~2,600 *more* lines of orphaned RAG scaffolding.

### What was deleted (graph-specific)
`deeptutor/services/rag/pipelines/{lightrag.py, raganything.py, raganything_docling.py}`,
`components/indexers/{graph.py, lightrag.py}`, `components/retrievers/{lightrag.py, hybrid.py}`,
`parsers/mineru_api.py`, `utils/image_migration.py`, `logging/adapters/lightrag.py`
(a 190-line forwarder that existed *solely* to bridge LightRAG's logging into DeepTutor's
logger), and `tools/multi_kb_rag_tool.py`.

### The reusable abstraction they built (and would need again)
The plugin system from `bf21580` ("flexible RAG plugin system with UI") was an
**auto-discovery loader** scanning `src/rag/plugins/*.py`; each backend exposed three
functions — `initialize_rag()`, `search_rag()`, `delete_rag()` — plus a `CONFIG` dict
(`supported_modes`, `requires`, …). Drop-in modules, no inheritance. **If multiple RAG
backends ever return, this contract is what you rebuild.** Note it survives in spirit:
`RAGService` still calls the pipeline duck-typed (`hasattr(pipeline, "add_documents")`,
`service.py:57`; `delete`, `:185`), but `factory.get_pipeline` is welded to LlamaIndex
(`factory.py:24-37` — ignores the `name` arg entirely).

### Stated vs inferred reasons for removal

**Stated** (release notes): maintenance dead-weight. `ver1-1-2.md` verbatim:
*"Removed ~2,600 lines of unused RAG scaffolding … that existed as placeholders for
never-shipped backends. The RAG service is now a thin wrapper over the single LlamaIndex
pipeline."* `ver-1-0-0-beta1.md` frames it as architecture unification/stability.
> ⚠️ Correction to my first take: the "temporarily removed… will be re-introduced in
> upcoming releases" line I quoted earlier is a **paraphrase, not verbatim** in the
> committed notes. The committed framing is consolidation, not a promise to bring it back.

**Inferred, but corroborated by commit-level evidence — these are the real lessons:**

1. **Provider lock-in (the big one).** LightRAG is **OpenAI-hardcoded**. Commit `f6e6b9f`
   ("a wrapper on top of lightrag for providing new llm providers") added a **300-line
   `llm_factory.py`** whose entire job was to route around it — a hand-rolled
   `anthropic_complete()` over raw `aiohttp` because LightRAG had no Anthropic binding,
   plus brittle Ollama-detection heuristics (sniff for `:11434`, rewrite `/api`→`/v1`).
   A surviving inline comment is the smoking gun: *"Load LLM config early … because
   LightRAG reads `os.environ['OPENAI_API_KEY']` directly."* This is upstream-confirmed:
   LightRAG GitHub issues show `tiktoken_model_name` hardcoded to `gpt-4o-mini`, custom
   OpenAI URLs not detected, ingestion dying on transient OpenAI 400s (#2099, #2794).
2. **20× indexing latency.** Commit `1d985d2` records measured numbers in its own message:
   **"llamaindex 3s, lightrag 10-15s, raganything 60s+."**
3. **Monolith-vs-plugin friction.** `f771199` (a revert ~1 hour after the plugin
   integration landed): *"RAG-Anything already handles LightRAG internally; changed
   working_dir broke the normal document processing flow; plugin system is only needed
   for queries, not document upload."*
4. **Embedding-model coupling broke KBs.** The graph store bound to a fixed
   `EmbeddingFunc(embedding_dim, max_token_size)`; the team later had to add
   "embedding model mismatch" detection (`8255f8b`) because KBs broke on embedding swaps.
5. **General architectural thrash.** An entire from-scratch `src/lego_rag/` framework
   (~2,500 lines, `3836c5a`) was built and abandoned (not an ancestor of HEAD). The RAG
   architecture was rebuilt 3+ times in 2.5 months.

**One thing that was *never* built:** a knowledge-graph **visualization** for document KBs.
The only user-facing graph surface was a provider-dropdown + LightRAG's text query modes
(`naive`/`local`/`global`/`hybrid`). Graph-shaped UI in the repo today
(`web/components/memory/MemoryGraph.tsx`, the book `ConceptGraphBlock`) was built *fresh*
and post-removal for *other* subsystems. So "users loved the knowledge graph" is not a
sunk benefit you'd be restoring — it never reached them.

---

## 2. The open-domain / "ontology is hard" concern — largely a non-issue

This was the user's stated worry, and the research is clear: **it is not the blocker.**

- MS GraphRAG, LightRAG, and HippoRAG all do **open / emergent LLM extraction** then
  cluster — **no hand-authored schema** (GraphRAG paper arXiv 2404.16130; LightRAG 2410.05779).
- Staged pipelines (Extract-Define-Canonicalize, iText2KG) explicitly decouple raw
  extraction from any schema. An open-domain tutor is a *fine* fit for schema-free KG.

**But schema-free has its own tax** (this is the real cost of "no ontology"):
- **Entity resolution / dedup noise** — the same concept appears under variant surface
  forms; needs post-hoc vector clustering + LLM dedup, itself error-prone.
- **Label/type drift** across chunks; **hallucinated relations** + error propagation
  (the entire selling point of iText2KG is "near-zero hallucinations," i.e. the baseline
  has them).
- These costs **get worse on weak models** (see §4).

So: the open domain doesn't stop you, but don't expect a clean graph for free either.

---

## 3. What it would actually cost to build in DeepTutor today

DeepTutor's RAG is a thin, well-factored wrapper over LlamaIndex `VectorStoreIndex`
with an optional BM25 hybrid leg. Vector-first, file-backed (JSON docstores), versioned
by embedding signature, **zero graph dependencies** (no networkx/neo4j/kuzu/graphrag in
`pyproject.toml` — only `llama-index` + `llama-index-retrievers-bm25`).

### Integration cost map (seam → verdict → why)

| Seam | Where | Verdict | Why |
|---|---|---|---|
| **Entity-anchored query expansion** ⭐ | `smart_retriever.py:49-66` | **EASY** | `SmartRetriever` already owns its own LLM calls + aggregation; swap `_generate_queries` for "extract entities → graph-neighbour expansion." No pipeline surgery, no schema. Best low-risk proof-of-value. |
| Orchestration funnel | `add_documents.py:126-157` | **EASY** | Single incremental-ingest path; SHA256 content IDs already exist (`:74`) → free graph provenance. |
| Ingestion-time extraction | `ingestion.py:61-87`, `pipeline.py:87/238` | **EASY-MED** | One chunking funnel (512-tok `SentenceSplitter`); per-node text is right granularity. Gap: doc metadata is thin (filename only, no stable doc_id). |
| Reuse `ConceptGraph` model + Mermaid render | `book/models.py:207-245`, `book/blocks/concept_graph.py` | **MEDIUM** | Models (`depends_on`/`extends`/`related`), cycle-removal, topo-sort, Mermaid rendering are KB-agnostic and *free*. But today it's keyed by `book_id`, built from an `ExplorationReport` (not chunks), persisted in book storage — needs re-keying + a chunk-based extractor. |
| Retriever fusion (add graph leg) | `retrievers.py:98-130` | **MEDIUM** | Clean `QueryFusionRetriever` (RRF) composition; LlamaIndex `PropertyGraphIndex`+`SimpleGraphStore` fits the file-backed convention (no graph DB needed). |
| Per-KB graph persistence | `index_versioning.py:300-326`, `storage.py:69-74` | **MEDIUM** | `version-N/` dirs accept a `graph_store.json` sidecar cleanly. |
| Second pipeline (un-weld factory) | `factory.py:18-48` | **MEDIUM** | Duck-typed interface survives; must un-collapse `get_pipeline`/`normalize_provider_name`/`list_pipelines`. This is literally re-opening what they closed. |
| **Graph re-index signature** ⚠️ | `index_versioning.py:46-59` | **HARD** | The versioning key (`EmbeddingSignature` = SHA256 over binding/model/dim/base_url) is **embedding-only**. The extraction LLM isn't captured. Either extend the signature (forces wasteful *vector* re-index on extraction-model change) or add a separate `graph_signature` + sidecar versioning. **Decide this before building anything.** |
| Per-chunk extraction cost | `llm/client.py:63`, `embedding/client.py:66` | **MED cost / EASY wiring** | ~1 LLM `complete()` call per 512-tok chunk → hundreds of calls per medium PDF. `complete` facade is async-batchable across all providers. |

**Two non-obvious snags worth surfacing:**
- **No index caching** — the index is reloaded from disk on *every* query
  (`storage.py:176-177`). Any graph retriever inherits that latency, and traversing a
  JSON-deserialized graph store amplifies it.
- **The embedding-only signature (HARD row)** is the one true architectural decision; get
  it wrong and every embedding-model swap nukes the (expensive) graph too.

**Net:** A *graph-augmented retrieval booster* via `SmartRetriever` is genuinely easy and
schema-free. A *full second graph-RAG backend* re-opens the exact abstraction (`factory`
dispatch, per-provider plumbing, versioning) the team spent two releases welding shut.

---

## 4. The field in 2026 — with numbers

### The approaches, cheapest→heaviest

| Approach | Indexing cost | Query cost | Incremental update | Weak-model safe? | Where it wins |
|---|---|---|---|---|---|
| **Vector RAG** (today) | Lowest | Lowest | ✓ trivial | ✓ (no extraction) | Local factoid; *finding the source doc* |
| **RAPTOR** | Low-med (recursive *summarize*, **no NER**) | Low | ✗ tree rebuild | ✓ summarization | Cheap holistic context; **+20% QuALITY** `[measured]` |
| **LazyGraphRAG** | ≈ vector RAG (~**1000× cheaper** than GraphRAG) `[MS blog]` | Low-med (+2-8s) | ✓ light index | ✓ defers LLM | Best global cost/quality today; *matches* vector on local |
| **LightRAG** | Medium (per-chunk triples) | **<100 tok/query** `[measured]` | ✓ graph union, no rebuild | ✗ **fails on Ollama** | Multi-hop + diversity; cheap incremental |
| **HippoRAG / 2** | Medium (open KG triples) | Low (Personalized PageRank) | partial | ✗ extraction-bound | **Multi-hop** (+20% / +7% assoc.) `[measured]` |
| **MS GraphRAG** | **Highest** ($20-500/corpus `[blog]`; **281 min for 1M tokens** `[measured]`) | High (~**610k tok/global query** `[measured]`) | ✗ community restructuring (**16.6% time-sensitive accuracy drop** `[blog/study]`) | ✗ F1 −25pt on weak `[measured]` | Global sensemaking; high-entity aggregation (**3.4×** `[measured]`) |

### When graph actually beats vector
- **Multi-hop QA**: graph methods >50% better; HippoRAG up to **+20%** `[measured]`.
- **Global/sensemaking**: the whole premise of GraphRAG global search & LazyGraphRAG.
- **Schema-bound aggregation**: Diffbot benchmark — vector RAG scored **~0%** on
  aggregation queries and degraded to 0% past 5 entities; graph held at 10+ `[measured]`.

### When it does NOT pay
- Local factoid lookup (vector ties or wins, and is better at citing the *source doc*).
- Small/static corpora; cost/latency-sensitive deployments.
- **Frequently-mutating KBs** → re-index churn. **This is DeepTutor's exact profile**
  (user-uploaded, edited KBs).
- 2025-26 practitioner consensus: **layer graph on top of vector, don't replace it.**

### The killer constraint for DeepTutor: weak/local models
DeepTutor must run across **40+ providers incl. Ollama / llama.cpp / LM Studio**. Strict
triple extraction is exactly where these collapse:
- **LightRAG + Ollama extracts *zero* entities/relations** while appearing to run normally
  (HKUDS/LightRAG **issue #30**). Silent graph collapse on the local-model class.
- Extraction micro-F1 drops **up to 25 absolute points** low- vs high-resource `[measured]`;
  even GPT-4 zero-shot KGC hasn't beaten fully-supervised small models.
- JSON-schema brittleness fails extraction outright on weak models (LightRAG #287).

→ **Entity-graph backends (GraphRAG/LightRAG/HippoRAG) are fragile across DeepTutor's
fleet. Summarization/co-occurrence approaches (RAPTOR, LazyGraphRAG) are far safer.**
This, plus re-index churn, is the most likely *real* reason LightRAG was removed — beyond
the stated "stability."

---

## 5. The pedagogical angle — where graph info genuinely helps a *tutor*

For a learning app the valuable graph artifact is **not** an entity-retrieval backend —
it's a **concept-prerequisite graph** ("learn A before B"):
- Standard model: a **DAG of Knowledge Components with `depends_on` edges** driving
  adaptive curriculum sequencing.
- Outcome evidence: concept-map-guided learning **significantly improved post-test scores**
  vs control; prerequisite-tiered ITS guidance improves comprehension `[measured]`.
- **DeepTutor already models exactly this**: `ConceptNode`/`ConceptEdge`/`ConceptGraph`
  with `depends_on`/`extends`/`related` (`book/models.py:207-245`), built by an LLM
  Draft→Critique→Revise loop then deterministically de-cycled + topo-sorted
  (`spine_synthesizer.py`). It's currently book-scoped and viz-only.

This is the part worth investing in, and it sidesteps every §4 failure mode because it's
small, schema-free, and concept-level (not exhaustive entity extraction).

---

## 6. Recommendation

**Verdict: do not reinstate a graph-RAG retrieval backend. Build a per-KB concept layer.**

The open-domain concern is not the reason — schema-free extraction handles it (§2). The
reasons are: the team already paid this cost and retreated (§1); the dominant failure
modes (weak-model extraction collapse, re-index churn) are DeepTutor's *exact* operating
profile (§4); and the genuinely valuable graph for a tutor is a concept-prerequisite graph
it already half-owns (§5).

### Phased path (lowest risk → highest leverage)

- **Phase 0 — ask the maintainers** *why* LightRAG was really removed (cost? local-model
  quality? maintenance?). Their answer dominates everything. (Strongly suspect §1.1 + §4.)
- **Phase 1 — entity-anchored retrieval booster.** Modify `SmartRetriever._generate_queries`
  (`smart_retriever.py:49`) to extract concepts from the query/context and expand along
  graph neighbours. No schema, no new pipeline, no versioning change. Proves value cheaply
  and is fully reversible. **Start here.**
- **Phase 2 — per-KB concept-prerequisite graph.** Promote `ConceptGraph` from book-scoped
  to KB-scoped: extract emergent concepts + `depends_on` edges from KB chunks in
  `process_new_documents` (`add_documents.py:126`), persist a `graph_store.json` sidecar in
  each `version-N/` dir, reuse the existing model + Mermaid renderer. Use it for navigation,
  prerequisite sequencing, and as the Phase-1 booster's backing store. **Design the graph
  signature (§3 HARD row) before writing code.**
- **Phase 3 (only if real demand for global "themes across the whole corpus" queries
  emerges)** — add a **RAPTOR or LazyGraphRAG-style** layer (summarization/co-occurrence,
  **not** strict triple extraction), gated **opt-in per-KB** because of cost, via LlamaIndex's
  in-stack `PropertyGraphIndex` (no second framework). Explicitly **avoid** reinstating
  LightRAG/MS-GraphRAG given the provider-lock-in and weak-model evidence.

### What to avoid
- A full second graph-RAG backend (re-opens the welded-shut `factory` abstraction).
- Anything depending on reliable entity/relation **triple** extraction on local models.
- Coupling the graph to the embedding signature (forces wasteful full rebuilds).

---

## Appendix A — key file references
- RAG facade / factory: `deeptutor/services/rag/service.py`, `…/rag/factory.py`
- Pipeline / chunking / fusion: `…/pipelines/llamaindex/{pipeline,ingestion,retrievers}.py`
- Versioning (the HARD snag): `…/pipelines/llamaindex/index_versioning.py:46-59`
- Best seam: `deeptutor/services/rag/smart_retriever.py:49-66`
- Ingestion funnel: `deeptutor/knowledge/add_documents.py:126-157`
- Existing concept graph: `deeptutor/book/{models.py:207-245, blocks/concept_graph.py, spine_synthesizer.py}`
- LLM / embedding facades: `deeptutor/services/llm/client.py:63`, `deeptutor/services/embedding/client.py:66`
- Provider catalog (40+): `deeptutor/services/provider_registry.py`

## Appendix B — primary sources (selected)
- MS GraphRAG paper: https://arxiv.org/html/2404.16130v2
- LightRAG paper: https://arxiv.org/html/2410.05779v1 · repo: https://github.com/HKUDS/LightRAG
- LightRAG Ollama zero-extraction: https://github.com/HKUDS/LightRAG/issues/30
- LazyGraphRAG: https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/
- RAPTOR: https://arxiv.org/abs/2401.18059 · HippoRAG: https://arxiv.org/abs/2405.14831
- GraphRAG vs vector decision guide: https://tianpan.co/blog/2026-04-19-graphrag-vs-vector-rag-architecture-decision
- Weak-model extraction degradation: https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2025.1505877/full
- Schema-free KG construction survey: https://arxiv.org/html/2510.20345v1
- Concept-prerequisite graphs in ITS (outcomes): https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2026.1777749/full

## Appendix C — git evidence trail
- Removal: `f377913` (2026-03-13, 239 files, +4,869/−9,453)
- LightRAG provider wrapper (OpenAI lock-in): `f6e6b9f`
- Plugin system (reusable 3-fn contract): `bf21580`, `44acb1a`; revert `f771199`
- Latency datapoint (llamaindex 3s / lightrag 10-15s / raganything 60s+): `1d985d2`
- Per-KB provider selector UI: `9ef0a2a`; abandoned `lego_rag` framework: `3836c5a`
- Cleanup tail (~2,600 lines): `100576d`; rationale: `assets/releases/ver1-1-2.md`, `ver0-5-0.md`
