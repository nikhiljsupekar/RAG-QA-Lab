# RAG QA Lab

An interactive Retrieval-Augmented Generation app that answers questions **only from your PDF and TXT documents**, and a QA layer that measures how well it does that.

* Works **without any API key** (retrieval-only mode quotes sentences from your documents).
* Claude is an **optional** answer generator, kept in a separate package (`llm/`) that the retrieval code never imports.
* If the evidence is insufficient, the app says so instead of guessing.

## What is inside

```
rag-qa-lab/
├── app.py                  Streamlit UI (Ask tab + QA dashboard tab)
├── rag/                    Retrieval pipeline - no LLM imports
│   ├── loaders.py          PDF (PyMuPDF) / TXT extraction, text cleaning
│   ├── chunking.py         Configurable chunk size + overlap, stable chunk IDs, page numbers
│   ├── embeddings.py       Sentence Transformers, plus an offline lexical fallback
│   ├── vector_store.py     FAISS index (exact inner product on normalised vectors)
│   ├── retriever.py        Top-K retrieval + evidence-sufficiency decision
│   ├── answering.py        Extractive answerer (no LLM) + lexical grounding check
│   └── pipeline.py         Ingest + ask orchestration
├── llm/claude_provider.py  Optional Claude integration (reads ANTHROPIC_API_KEY from the environment)
├── qa/
│   ├── deterministic.py    Rule-based checks
│   ├── llm_judge.py        Optional LLM-based faithfulness judge (clearly labelled)
│   ├── ragas_eval.py       Optional RAGAS metrics (LLM-based, clearly labelled)
│   └── runner.py           Runs data/eval_dataset.json and computes metrics
├── data/sample_docs/       3 TXT policies + 1 PDF policy
├── data/eval_dataset.json  17 hand-written cases (13 answerable, 4 unanswerable)
├── scripts/make_sample_pdf.py
└── tests/                  pytest suite
```

## Setup (Windows PowerShell)

Requires Python 3.10+ (developed and tested on 3.14).

```powershell
cd "path\to\rag-qa-lab"

# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked:  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# 2. Install dependencies (PyTorch is pulled in by sentence-transformers: expect ~1-2 GB)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. (Optional) configure Claude. Skip this to stay in retrieval-only mode.
Copy-Item .env.example .env
notepad .env        # put your key after ANTHROPIC_API_KEY=   (.env is git-ignored)
```

You can also set the key for the current terminal only: `$env:ANTHROPIC_API_KEY = "your-key"`.
Never paste a key into source files. The app also offers a session-only password box in the sidebar.

## Run

```powershell
streamlit run app.py
```

Open the printed URL (normally http://localhost:8501), then:

1. Click **Load samples** in the sidebar (or upload your own PDF/TXT files and click **Index uploads**).
2. Ask a question in the **Ask** tab. Each answer shows source filename, chunk ID, page (PDFs), similarity score, the evidence text, and latency.
3. Open the **QA dashboard** tab and click **Run evaluation**.

The first run with Sentence Transformers downloads the `all-MiniLM-L6-v2` model (~90 MB) from huggingface.co. If that fails (offline, firewall), the app falls back to a lexical embedder and says so in the UI.

## Run the tests

```powershell
python -m pytest -q                      # everything (the 'semantic' test downloads the model once; it skips if it cannot)
python -m pytest -q -m "not semantic"    # fully offline
```

| Requirement | Test file |
|---|---|
| Chunking (size, overlap, offsets, IDs, invalid params) | `tests/test_chunking.py` |
| PDF/TXT loading, corrupt and empty files | `tests/test_loaders.py` |
| Retrieval (ranking, Top-K, determinism, threshold) | `tests/test_retrieval.py` |
| Source attribution | `tests/test_source_attribution.py` |
| Empty documents | `tests/test_empty_documents.py` |
| Irrelevant questions | `tests/test_irrelevant_questions.py` |
| Unsupported / hallucinated answers | `tests/test_unsupported_answers.py` |
| Eval runner, LLM judge (stubbed), Claude provider, no hardcoded keys | `tests/test_eval_and_safety.py` |
| RAGAS integration (stub judge; skipped if RAGAS is not installed) | `tests/test_ragas_eval.py` |
| Your-own-questions table and Claude-drafted questions (stub model) | `tests/test_testgen.py` |
| Markdown escaping for the UI | `tests/test_textutils.py` |

Offline tests use the deterministic lexical embedder so they never need network access. No test calls the real Claude API.

## Deterministic checks vs LLM-based evaluation

These are kept apart on purpose.

**Deterministic (`qa/deterministic.py`)** - same input, same result, free. These alone decide PASS/FAIL:

| Check | What it verifies |
|---|---|
| `correct_abstention` | Answerable questions are answered; unanswerable ones get the explicit "not enough evidence" reply |
| `source_retrieved` | The expected document is among the Top-K sources |
| `evidence_coverage` | Expected evidence phrases appear verbatim in the retrieved chunks |
| `answer_contains_facts` | Required facts (e.g. "14 characters") appear in the answer |
| `source_attribution` | Every cited chunk was actually retrieved; non-refusals must cite something |
| `grounded_in_evidence` | Each answer sentence's terms, and every number, appear in the evidence (lexical) |
| `latency_budget` | Measured wall-clock time is within the configured budget |

**LLM-based (`qa/llm_judge.py`)** - optional, needs Claude mode and a key, costs API credits, non-deterministic, and can be wrong. It is shown in its own column, labelled `llm`, and **never changes pass/fail**.

### Evaluating your own documents

The bundled 17 questions are about the sample policies only, so they say nothing about a PDF you uploaded (and the dashboard warns you when the sample documents are not indexed). To evaluate your own documents, open the **QA dashboard**, choose **My own questions**, and fill the table (one row per question):

| Column | Meaning |
|---|---|
| Question | What a user would ask. |
| Answerable | Untick if the documents do not contain the answer: the app should then refuse. |
| Reference answer | The correct answer in your own words. RAGAS needs it for context precision and recall. |
| Evidence phrase (optional) | Text copied exactly from the document; the check passes when retrieval returns it. |
| Required fact in answer (optional) | A short fact the answer must contain, e.g. `30 days`. |
| Source file (optional) | Which document holds the answer. |

In Claude mode, **Draft questions with Claude** adds suggested rows from random chunks of your indexed documents. They are model-written, so review them. One check is deterministic: a draft is dropped unless its evidence quote appears word for word in the chunk it came from. Then click **Run evaluation** (and **Run RAGAS metrics** if installed). The table lives in the browser session and is lost on refresh; copy it out if you want to keep it.

### Optional: RAGAS metrics (LLM-based)

[RAGAS](https://docs.ragas.io) is an open-source RAG evaluation library. When it is installed, the dashboard gets a **RAGAS metrics** section that scores the answerable, answered cases with Claude as the judge model:

| Metric | Question it answers | Needs |
|---|---|---|
| Faithfulness | Are the answer's claims supported by the retrieved evidence? | answer + contexts |
| Context precision | Are the useful chunks ranked near the top? | contexts + `reference` answer |
| Context recall | Does the retrieved evidence contain what the reference says? | contexts + `reference` answer |
| Answer relevancy | Does the answer address the question? | a semantic embedder |

* **Install** (optional, after the normal setup): `python -m pip install -r requirements-ragas.txt` then `python -m pip install --no-deps "ragas==0.4.3"`. The file explains why it is two commands and why LangChain is pinned below 0.4.
* **Run**: choose **Claude** in *Answer mode*, run the evaluation, then click **Run RAGAS metrics**. It costs API credits (about 10 calls per case; the UI shows an estimate first).
* **Labelled and separate**: these scores are LLM-based and non-deterministic. They are shown in their own section and never change the deterministic PASS/FAIL.
* Cases are excluded (with the reason shown) when unanswerable, when the system abstained, or when no `reference` answer exists in the dataset. Answer relevancy is skipped with the lexical fallback embedder.
* RAGAS is imported lazily: without it, everything else works and the section explains how to install it.
* **Verified vs not verified**: the integration is tested with a stub judge that answers RAGAS's own prompts (adapters, threading, error handling, mapping verdicts to scores). It has **not** been run against the live Claude API (no key was available), so real score values and Claude's compliance with RAGAS's JSON format are untested. RAGAS 0.4 marks the API used here as deprecated (it still works), so the version is pinned.
* Needs a machine that can load PyArrow's dataset DLL. Windows **Smart App Control** blocks it ("An Application Control policy has blocked this file").

All metrics on the dashboard (pass rate, coverage, abstention accuracy, latency mean/p95) are computed from real runs. Nothing is simulated or hardcoded. Cases that cannot run in the current mode are shown as **SKIPPED** and are not counted as passes.

## Known limitations (please read)

* **Lexical fallback is not semantic.** With it, "minimum password length" does not match "at least 14 characters". Case `SEC-01P` exists to show this: it is skipped in lexical mode and passes with Sentence Transformers.
* **Thresholds are calibrated on a tiny sample.** The semantic defaults (chunk similarity 0.30, sentence similarity 0.50) were chosen from the score gap on the 17 bundled cases with `all-MiniLM-L6-v2`. With 17 cases, 100% on this dataset does not mean 100% on your documents. Tune the sliders, and add your own cases to `data/eval_dataset.json`.
* **Retrieval-only mode is extractive.** It quotes the best-matching sentences; it does not synthesise or reason across sentences. Use Claude mode for that.
* **The grounding check is lexical.** A paraphrase that reuses few of the evidence's words can be flagged (false alarm), and an answer that reuses the evidence's words but distorts the meaning can pass. It is a safety net, not proof of truth.
* **No OCR.** Scanned PDFs with no text layer are skipped with a message.
* **The Claude path was not exercised against the live API** in development (no key was available); it is covered by stubbed unit tests only. It is a standard Messages API call. By default it also opts in to Anthropic's server-side refusal fallback beta and retries without it if the API rejects that; set `use_refusal_fallback=False` in `ClaudeClient` to disable.

## Configuration

Environment variables (all optional, see `.env.example`): `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` (default `claude-opus-5-5`), `EMBEDDING_BACKEND` (`auto` | `sentence-transformers` | `hashing`), `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`. Everything except the key can also be changed in the sidebar.

## Troubleshooting

* *`streamlit run app.py` asks for an email and hangs* - that is Streamlit's one-time onboarding prompt. Press Enter to skip it, or start with `streamlit run app.py --server.headless true` and open http://localhost:8501 yourself.

* *The Upload button does not open a file dialog* - some embedded browsers (for example an IDE's built-in preview panel) block native file dialogs. Open http://localhost:8501 in Chrome or Edge instead. Alternatives that need no dialog: drag-and-drop files onto the upload box, or open **File dialog not opening? Index by path** in the sidebar, type a file or folder path and click **Index path**. That option reads files from the machine running the app, so set `ALLOW_LOCAL_PATHS=0` for any shared or hosted deployment.
* *Code changes are not picked up* - the source file watcher is off (`.streamlit/config.toml`) to avoid noisy logs from transformers; restart `streamlit run app.py` after editing code.
* *First question after starting is slow* - the first index build in a fresh server imports PyTorch and loads the model (can take a minute); later builds reuse it.

* *"Settings changed since the index was built"* - click **Rebuild index with new settings**.
* *"An Application Control policy has blocked this file" when importing pyarrow/datasets/ragas* - Windows Smart App Control (or another code-integrity policy) is blocking PyArrow's `_dataset` DLL. It only matters for RAGAS; worse, once `datasets` is installed, Sentence Transformers also imports it and the app silently falls back to the lexical embedder. Either allow it in Windows Security, or do not install `requirements-ragas.txt` on that machine.
* *Hugging Face symlink warning on Windows* - harmless; silence it with `$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"`.
* *Embedding model will not download* - use `EMBEDDING_BACKEND=hashing` for the offline lexical mode.
* *Regenerate the sample PDF* - `python scripts/make_sample_pdf.py`.
