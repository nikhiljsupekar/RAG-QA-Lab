"""Streamlit UI: ask questions over uploaded PDF/TXT files, inspect evidence, and run the QA dashboard."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from llm.claude_provider import ClaudeAnswerer, ClaudeClient, LLMUnavailable, claude_available
from qa import deterministic as det
from qa.ragas_eval import (METRIC_LABELS, RagasUnavailable, exclusion_reason, ragas_available, run_ragas)
from qa.runner import DEFAULT_DATASET, load_dataset, run_eval
from qa.testgen import ROW_COLUMNS, case_to_row, cases_from_rows, draft_cases
from rag.chunking import validate_params
from rag.config import Settings
from rag.embeddings import get_embedder
from rag.loaders import Document, load_many, read_path
from rag.pipeline import RAGPipeline
from rag.textutils import md_escape

SAMPLE_DIR = Path(__file__).parent / "data" / "sample_docs"
MODE_RETRIEVAL = "Retrieval-only (no API key needed)"
MODE_CLAUDE = "Claude (optional, needs ANTHROPIC_API_KEY)"
QUESTIONS_BUNDLED = "Bundled sample questions"
QUESTIONS_CUSTOM = "My own questions (for my uploaded documents)"

st.set_page_config(page_title="RAG QA Lab", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model (first run downloads it)...")
def cached_embedder(backend: str, model: str):
    return get_embedder(backend, model)


def init_state() -> None:
    st.session_state.setdefault("docs", {})  # filename -> Document
    st.session_state.setdefault("pipeline", None)
    st.session_state.setdefault("built_with", None)
    st.session_state.setdefault("ingest_report", None)
    st.session_state.setdefault("last_response", None)
    st.session_state.setdefault("eval_report", None)
    st.session_state.setdefault("ragas_report", None)
    st.session_state.setdefault("custom_rows", [])  # the user's own test questions (table rows)
    st.session_state.setdefault("editor_version", 0)
    st.session_state.setdefault("draft_dropped", [])


def rebuild_index(settings: Settings) -> None:
    choice = cached_embedder(settings.embedding_backend, settings.embedding_model)
    pipe = RAGPipeline(settings, choice.embedder)
    pipe.fallback_reason = choice.fallback_reason
    bar = st.progress(0.0, text="Indexing...")
    report = pipe.ingest(list(st.session_state.docs.values()), progress=lambda f, m: bar.progress(f, text=m))
    bar.empty()
    st.session_state.pipeline = pipe
    st.session_state.ingest_report = report
    st.session_state.built_with = (settings.chunk_size, settings.chunk_overlap,
                                   settings.embedding_backend, settings.embedding_model, settings.min_score)
    st.session_state.last_response = None
    st.session_state.eval_report = None
    st.session_state.ragas_report = None


def add_documents(docs: list[Document]) -> None:
    for d in docs:
        st.session_state.docs[d.filename] = d  # same filename replaces the older upload


# ------------------------------------------------------------------------------------------ sidebar
def sidebar() -> tuple[Settings, str, str]:
    env = Settings.from_env()
    sb = st.sidebar
    sb.header("1 · Documents")
    uploads = sb.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    c1, c2 = sb.columns(2)
    add_clicked = c1.button("Index uploads", disabled=not uploads)
    sample_clicked = c2.button("Load samples")
    path_text, path_clicked = "", False
    if env.allow_local_paths:
        with sb.expander("File dialog not opening? Index by path"):
            st.caption("Type a file or a folder (reads its .pdf/.txt files) on the machine running this app. "
                       "Drag-and-drop onto the upload box also works without the dialog.")
            path_text = st.text_input("File or folder path", placeholder=r"C:\Users\you\Documents\policies",
                                      label_visibility="collapsed")
            path_clicked = st.button("Index path", disabled=not path_text.strip())
    if sb.button("Clear all documents"):
        st.session_state.docs.clear()
        st.session_state.pipeline = None
        st.session_state.ingest_report = None
        st.session_state.eval_report = None

    sb.header("2 · Retrieval settings")
    chunk_size = sb.number_input("Chunk size (characters)", 100, 5000, env.chunk_size, step=50)
    overlap = sb.number_input("Chunk overlap (characters)", 0, 2000, env.chunk_overlap, step=10)
    top_k = sb.slider("Top-K chunks", 1, 10, env.top_k)
    backend = sb.selectbox("Embedding backend", ["auto", "sentence-transformers", "hashing"],
                           index=["auto", "sentence-transformers", "hashing"].index(env.embedding_backend)
                           if env.embedding_backend in ("auto", "sentence-transformers", "hashing") else 0,
                           help="auto = Sentence Transformers if it loads, else the offline lexical fallback.")
    manual_thr = sb.checkbox("Set evidence threshold manually")
    min_score = sb.slider("Minimum similarity", 0.0, 1.0, 0.30, 0.01) if manual_thr else None
    budget = sb.number_input("Latency budget (ms) for QA", 50, 60000, int(env.latency_budget_ms), step=50)

    sb.header("3 · Answer mode")
    key_in_env = claude_available()
    mode = sb.radio("Generator", [MODE_RETRIEVAL, MODE_CLAUDE])
    api_key = ""
    if mode == MODE_CLAUDE:
        if not key_in_env:
            api_key = sb.text_input("Anthropic API key (kept in this session only)", type="password")
        else:
            sb.caption("Using the API key found in the environment.")
        sb.caption(f"Model: `{env.claude_model}`")

    settings = Settings(chunk_size=int(chunk_size), chunk_overlap=int(overlap), top_k=top_k,
                        min_score=min_score, embedding_backend=backend,
                        embedding_model=env.embedding_model, latency_budget_ms=float(budget),
                        claude_model=env.claude_model)

    try:
        validate_params(settings.chunk_size, settings.chunk_overlap)
        params_ok = True
    except ValueError as exc:
        sb.error(str(exc))
        params_ok = False

    if add_clicked and uploads and params_ok:
        docs, errors = load_many([(f.name, f.getvalue()) for f in uploads])
        add_documents(docs)
        for e in errors:
            sb.error(e)
        rebuild_index(settings)
    if path_clicked and params_ok:
        files, errors = read_path(path_text)
        docs, load_errors = load_many(files)
        for e in errors + load_errors:
            sb.error(e)
        if docs:
            add_documents(docs)
            rebuild_index(settings)
            sb.success(f"Indexed {len(docs)} file(s) from the path.")
    if sample_clicked and params_ok:
        docs, _ = load_many([(p.name, p.read_bytes()) for p in sorted(SAMPLE_DIR.iterdir())
                             if p.suffix.lower() in (".txt", ".pdf")])
        add_documents(docs)
        rebuild_index(settings)

    signature = (settings.chunk_size, settings.chunk_overlap, settings.embedding_backend,
                 settings.embedding_model, settings.min_score)
    if st.session_state.pipeline and st.session_state.built_with != signature:
        sb.warning("Settings changed since the index was built.")
        if sb.button("Rebuild index with new settings", disabled=not params_ok):
            rebuild_index(settings)
            st.rerun()
    return settings, mode, api_key


def make_generator(settings: Settings, mode: str, api_key: str):
    """Returns (generator_or_None, claude_client_or_None). Falls back to retrieval-only on any problem."""
    if mode != MODE_CLAUDE:
        return None, None
    try:
        client = ClaudeClient(api_key=api_key or None, model=settings.claude_model)
        return ClaudeAnswerer(client), client
    except LLMUnavailable as exc:
        st.warning(f"Claude unavailable ({exc}) - using retrieval-only mode.")
        return None, None


# ------------------------------------------------------------------------------------------ ask tab
def render_index_status() -> None:
    pipe: RAGPipeline | None = st.session_state.pipeline
    report = st.session_state.ingest_report
    if not pipe:
        st.info("Upload documents (or click **Load samples**) in the sidebar to begin.")
        return
    cols = st.columns(4)
    cols[0].metric("Documents indexed", len(report.documents_indexed))
    cols[1].metric("Chunks", report.chunk_count)
    cols[2].metric("Index build", f"{report.embed_seconds:.1f}s")
    cols[3].metric("Min similarity", f"{pipe.retriever.min_score:.2f}" if pipe.retriever else "n/a")
    st.caption(f"Embedder: **{pipe.embedder.name}**")
    if pipe.fallback_reason:
        st.warning("Sentence Transformers could not be loaded, so the offline **lexical** fallback is active. "
                   "It only matches shared words (no paraphrases). Reason: " + pipe.fallback_reason)
    elif not pipe.embedder.semantic:
        st.caption("Lexical embedder selected: matches shared words only, not meaning.")
    for skipped in report.documents_skipped:
        st.warning("Skipped " + skipped)


def render_response(resp, settings: Settings) -> None:
    if resp.answer.insufficient:
        st.warning(resp.answer.text)
        st.caption("Why: " + resp.retrieval.reason)
    else:
        st.success(md_escape(resp.answer.text))
    for w in resp.warnings:
        st.error(w)

    m = st.columns(5)
    m[0].metric("Retrieval", f"{resp.retrieval_ms:.1f} ms")
    m[1].metric("Generation", f"{resp.generation_ms:.1f} ms")
    m[2].metric("Total", f"{resp.total_ms:.1f} ms")
    m[3].metric("Top score", f"{resp.retrieval.top_score:.3f}")
    m[4].metric("Answer mode", resp.answer.mode)

    if not resp.answer.insufficient:
        checks = [det.check_source_attribution(resp), det.check_grounded(resp)]
        st.markdown("**Answer checks** · deterministic, no LLM")
        for c in checks:
            st.write(("✅ " if c.passed else "❌ ") + f"`{c.name}` - {c.detail}")

    st.subheader(f"Retrieved evidence (Top-{len(resp.retrieval.hits)})")
    if not resp.retrieval.hits:
        st.caption("Nothing retrieved.")
    for hit in resp.retrieval.hits:
        cited = hit.chunk.chunk_id in resp.answer.citations
        page = f" · page {hit.chunk.page}" if hit.chunk.page else ""
        label = f"#{hit.rank} · {hit.chunk.source} · {hit.chunk.chunk_id.split('::')[-1]}{page} · score {hit.score:.3f}"
        with st.expander(label + ("  ✔ cited" if cited else ""), expanded=cited):
            st.caption(f"chunk id: `{hit.chunk.chunk_id}` · characters {hit.chunk.start}-{hit.chunk.end}")
            st.text(hit.chunk.text)  # plain text: keeps line breaks, never interpreted as Markdown


def ask_tab(settings: Settings, mode: str, api_key: str) -> None:
    render_index_status()
    pipe: RAGPipeline | None = st.session_state.pipeline
    if not pipe:
        return
    with st.form("ask"):
        question = st.text_input("Ask a question about your documents",
                                 placeholder="e.g. How many days per week can employees work remotely?")
        submitted = st.form_submit_button("Ask")
    if submitted:
        generator, _ = make_generator(settings, mode, api_key)
        with st.spinner("Retrieving and answering..."):
            st.session_state.last_response = pipe.ask(question, generator=generator, top_k=settings.top_k)
    if st.session_state.last_response is not None:
        render_response(st.session_state.last_response, settings)


# ------------------------------------------------------------------------------------------ dashboard
def custom_cases_ui(pipe: RAGPipeline, settings: Settings, mode: str, api_key: str) -> list[dict]:
    """Editable table of the user's own test questions, with optional Claude-drafted suggestions."""
    st.caption("Write questions about the documents you indexed. **Reference answer** is the correct answer in "
               "your own words (RAGAS needs it for context precision/recall). **Evidence phrase** is optional: "
               "text copied exactly from the document that the retrieval should find. **Answerable** unticked = "
               "the documents do not contain the answer, so the app should refuse.")
    sources = sorted({c.source for c in pipe.chunks})
    version = st.session_state.editor_version
    df = pd.DataFrame(st.session_state.custom_rows, columns=ROW_COLUMNS).astype({"Answerable": bool})
    edited = st.data_editor(
        df, num_rows="dynamic", hide_index=True, key=f"custom_editor_{version}",
        column_config={
            "Question": st.column_config.TextColumn(required=True, width="large"),
            "Answerable": st.column_config.CheckboxColumn(default=True),
            "Reference answer": st.column_config.TextColumn(width="large"),
            "Evidence phrase": st.column_config.TextColumn(help="Exact text from the document (optional)"),
            "Required fact in answer": st.column_config.TextColumn(help="Short fact the answer must contain, "
                                                                   "e.g. '30 days' (optional)"),
            "Source file": st.column_config.SelectboxColumn(options=sources, help="Which document holds the "
                                                                                  "answer (optional)"),
        })
    rows = edited.to_dict("records")
    st.session_state.custom_rows = rows

    c1, c2 = st.columns([1, 3])
    n_draft = c1.number_input("Drafts to add", 1, 20, 5, help="One question per randomly chosen chunk.")
    drafted = c2.button("Draft questions with Claude", disabled=mode != MODE_CLAUDE,
                        help="Needs Claude mode. You must review the drafts: they are written by a model. "
                             "Drafts whose evidence quote is not word-for-word in the chunk are dropped.")
    if mode != MODE_CLAUDE:
        c2.caption("Drafting needs Answer mode = Claude (sidebar).")
    if drafted:
        _, client = make_generator(settings, mode, api_key)
        if client is not None:
            with st.spinner("Asking Claude to draft questions..."):
                result = draft_cases(client, pipe.chunks, n=int(n_draft), seed=len(rows))
            st.session_state.custom_rows = rows + [case_to_row(c) for c in result.cases]
            st.session_state.draft_dropped = result.dropped
            st.session_state.editor_version += 1  # new data -> fresh editor
            st.rerun()
    for reason in st.session_state.get("draft_dropped", []):
        st.caption(f"⚠️ Draft dropped: {reason}")

    cases, warnings = cases_from_rows(rows)
    for w in warnings:
        st.warning(w)
    if not cases:
        st.info("Add at least one question (use the empty row at the bottom of the table).")
    return cases


def dashboard_tab(settings: Settings, mode: str, api_key: str) -> None:
    st.markdown(
        "**How to read this dashboard.** Every pass/fail below comes from **deterministic checks** "
        "(string matching, set membership, thresholds): reproducible and free. "
        "An optional **LLM judge** column is separate, **non-deterministic**, and never changes pass/fail."
    )
    pipe: RAGPipeline | None = st.session_state.pipeline
    if not pipe:
        st.info("Index documents first (sidebar). To run the bundled dataset, click **Load samples**.")
        return

    choice = st.radio("Test questions", [QUESTIONS_BUNDLED, QUESTIONS_CUSTOM], horizontal=True,
                      help="The bundled 17 questions are about the sample policy documents only. To evaluate "
                           "your own uploaded documents, choose 'My own questions'.")
    generator_label = "Claude" if mode == MODE_CLAUDE else "extractive (retrieval-only)"
    if choice == QUESTIONS_BUNDLED:
        cases = load_dataset(DEFAULT_DATASET)
        indexed = {c.source for c in pipe.chunks}
        missing = sorted({c["expected_source"] for c in cases if c.get("expected_source")} - indexed)
        if missing:
            st.warning("The bundled questions are about the sample documents, which are not all indexed "
                       f"({', '.join(missing)}), so those cases will fail. Click **Load samples**, or choose "
                       "**My own questions** to evaluate the documents you uploaded.")
        st.caption(f"Dataset: `{DEFAULT_DATASET.name}` · {len(cases)} cases · generator: {generator_label}")
    else:
        cases = custom_cases_ui(pipe, settings, mode, api_key)
        st.caption(f"Your questions · {len(cases)} case(s) · generator: {generator_label}")

    use_judge = st.checkbox("Also run the LLM faithfulness judge (non-deterministic, uses API credits)",
                            disabled=mode != MODE_CLAUDE,
                            help="Requires Claude mode. Each answered case triggers one extra API call.")
    if st.button("Run evaluation", disabled=not cases):
        generator, client = make_generator(settings, mode, api_key)
        st.session_state.ragas_report = None  # scores belong to the previous run's answers
        with st.spinner("Running evaluation..."):
            st.session_state.eval_report = run_eval(
                pipe, cases, generator=generator, top_k=settings.top_k,
                latency_budget_ms=settings.latency_budget_ms, judge=client if use_judge else None)

    report = st.session_state.eval_report
    if report is None:
        st.info("Click **Run evaluation** to see the results. The optional **RAGAS metrics** section appears at "
                "the very bottom of this tab once the evaluation has run.")
        return
    s = report.summary
    st.caption(f"Embedder used: {report.embedder}")
    if s["cases"] == 0:
        st.error("No cases ran.")
        return
    cols = st.columns(5)
    cols[0].metric("Pass rate (deterministic)", f"{s['pass_rate']:.0%}", f"{s['passed']}/{s['cases']} cases")
    cov = s["mean_evidence_coverage"]
    cols[1].metric("Evidence coverage", f"{cov:.0%}" if cov is not None else "n/a",
                   help="Share of expected evidence phrases found in the retrieved chunks (answerable cases).")
    ab = s["abstention_accuracy"]
    cols[2].metric("Correct abstention", f"{ab:.0%}" if ab is not None else "n/a",
                   help="Unanswerable questions that got the explicit 'insufficient evidence' reply.")
    cols[3].metric("Mean latency", f"{s['mean_latency_ms']:.1f} ms")
    cols[4].metric("p95 latency", f"{s['p95_latency_ms']:.1f} ms")
    if s["skipped"]:
        st.info(f"{s['skipped']} case(s) skipped (not counted as passes): they need a semantic embedder.")
    if s["llm_judged"]:
        st.metric("LLM judge: faithful answers (non-deterministic)", f"{s['llm_faithful']}/{s['llm_judged']}")

    rows = []
    for r in report.results:
        rows.append({
            "Case": r.case_id, "Question": r.question,
            "Result": "SKIPPED" if r.skipped else ("PASS" if r.passed else "FAIL"),
            "Failed checks": ", ".join(c.name for c in r.checks if not c.passed),
            "Evidence coverage": None if r.skipped or not r.answerable else r.coverage,
            "Top score": r.top_score, "Latency (ms)": round(r.total_ms, 2),
            "Sources": ", ".join(r.sources),
            "LLM judge (non-deterministic)": ("" if not r.llm_judgement else
                                              {True: "faithful", False: "unfaithful", None: "n/a"}[r.llm_judgement.passed]),
        })
    df = pd.DataFrame(rows)
    if not use_judge and not any(r.llm_judgement for r in report.results):
        df = df.drop(columns=["LLM judge (non-deterministic)"])
    st.dataframe(df, hide_index=True)
    st.bar_chart(df[df["Result"] != "SKIPPED"].set_index("Case")["Latency (ms)"])

    st.subheader("Case details")
    for r in report.results:
        icon = "⏭️" if r.skipped else ("✅" if r.passed else "❌")
        with st.expander(f"{icon} {r.case_id} · {r.question}"):
            if r.skipped:
                st.write(r.skipped_reason)
                continue
            st.markdown("**Answer:**")
            st.text(r.answer)
            for c in r.checks:
                st.write(("✅ " if c.passed else "❌ ") + f"`{c.name}` ({c.kind}) - {c.detail}")
            if r.llm_judgement:
                j = r.llm_judgement
                st.write(f"🤖 `{j.name}` ({j.kind}, non-deterministic) - {j.passed}: {j.detail}")

    ragas_section(settings, mode, api_key, pipe, report)


def ragas_section(settings: Settings, mode: str, api_key: str, pipe: RAGPipeline, report) -> None:
    st.divider()
    st.subheader("RAGAS metrics · LLM-based · non-deterministic")
    st.caption("[RAGAS](https://docs.ragas.io) is an open-source RAG evaluation library. A judge model (Claude) "
               "reads each answer and its retrieved evidence and scores it. Scores can change between runs, "
               "depend on the judge, and **never affect the deterministic PASS/FAIL above**.")
    with st.expander("What the four scores mean"):
        st.markdown(
            "- **Faithfulness** - share of the answer's claims that are supported by the retrieved evidence "
            "(low = possible hallucination).\n"
            "- **Context precision** - are the useful chunks ranked near the top of the retrieved list? "
            "(needs a reference answer)\n"
            "- **Context recall** - does the retrieved evidence contain everything the reference answer says? "
            "(needs a reference answer)\n"
            "- **Answer relevancy** - does the answer address the question? Needs a semantic embedder.\n\n"
            "Only answerable cases that were answered are scored; refusals and unanswerable cases are excluded.")
    available, info = ragas_available()
    if not available:
        st.info("RAGAS is not available in this environment (`pip install -r requirements-ragas.txt`). "
                f"Reason: {info}")
        return
    st.caption(f"RAGAS {info} · evaluating the answers from the last run ({report.generator_mode}).")
    eligible = sum(exclusion_reason(r) is None for r in report.results)
    if mode != MODE_CLAUDE:
        st.info("Switch **Answer mode** to Claude (sidebar) to run RAGAS: it needs a judge model.")
    else:
        st.warning(f"Uses your Claude API credits: roughly 10 calls per case, about {eligible * 10} calls for "
                   f"{eligible} eligible cases.")
    if st.button("Run RAGAS metrics", disabled=mode != MODE_CLAUDE or eligible == 0):
        _, client = make_generator(settings, mode, api_key)
        if client is not None:
            with st.spinner("Running RAGAS (this can take a few minutes)..."):
                try:
                    st.session_state.ragas_report = run_ragas(report, client, pipe.embedder)
                except RagasUnavailable as exc:
                    st.error(f"RAGAS failed to start: {exc}")

    rr = st.session_state.ragas_report
    if rr is None:
        return
    st.markdown(f"**RAGAS results** · {rr.scored_cases} case(s) scored · `{rr.kind}`")
    means = rr.means
    cols = st.columns(max(len(means), 1))
    for col, (metric, value) in zip(cols, means.items()):
        col.metric(METRIC_LABELS.get(metric, metric), "n/a" if value is None else f"{value:.2f}")
    for note in rr.notes:
        st.warning(note)
    rows = [{"Case": c.case_id, "Question": c.question,
             **({METRIC_LABELS.get(m, m): v for m, v in c.scores.items()} if c.excluded_reason is None else {}),
             "Not scored because": c.excluded_reason or ""} for c in rr.cases]
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def main() -> None:
    init_state()
    st.title("🔎 RAG QA Lab")
    st.caption("Answers come only from your documents. If the evidence is insufficient, the app says so.")
    settings, mode, api_key = sidebar()
    tab_ask, tab_qa = st.tabs(["Ask", "QA dashboard"])
    with tab_ask:
        ask_tab(settings, mode, api_key)
    with tab_qa:
        dashboard_tab(settings, mode, api_key)


main()
