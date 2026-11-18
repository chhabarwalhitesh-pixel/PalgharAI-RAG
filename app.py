import html
import os
import platform
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import palghar_rag as rag  # noqa: E402

STORAGE_DIR = os.environ.get("PALGHAR_STORAGE", "/content/PalgharAI_storage")
EMBED_NAME = os.environ.get("PALGHAR_EMBED", rag.DEFAULT_EMBED_MODEL)
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", rag.DEFAULT_GEMINI_MODEL)

st.set_page_config(page_title="Palghar AI", page_icon="🏛️", layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------ design system
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Devanagari:wght@400;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', 'Noto Sans Devanagari', sans-serif; }
.stApp { background: radial-gradient(1200px 500px at 85% -10%, rgba(23,195,178,.12), transparent), #0B1220; }
[data-testid="stSidebar"] { background: #0E1A2E; border-right: 1px solid #1E3150; }
.block-container { padding-top: 1.6rem; max-width: 1180px; }
.hero { background: linear-gradient(120deg, #0F3A4A 0%, #14284A 60%, #1B2A55 100%); border: 1px solid #25507A;
        border-radius: 20px; padding: 28px 32px; margin-bottom: 18px; position: relative; overflow: hidden; }
.hero:after { content:""; position:absolute; right:-60px; top:-60px; width:260px; height:260px; border-radius:50%;
              background: radial-gradient(circle, rgba(242,163,58,.35), transparent 70%); }
.hero h1 { margin:0; font-size: 38px; font-weight: 700; color:#F6F4EE; letter-spacing:-.5px; }
.hero p { margin:6px 0 14px; color:#B9C9DC; font-size:16px; }
.chip { display:inline-block; padding:5px 12px; margin:0 8px 6px 0; border-radius:999px; font-size:12.5px; font-weight:600;
        background: rgba(23,195,178,.14); border:1px solid rgba(23,195,178,.45); color:#5EEAD4; }
.chip.warn { background: rgba(242,163,58,.14); border-color: rgba(242,163,58,.5); color:#F2C36B; }
.chip.off { background: rgba(148,163,184,.12); border-color: rgba(148,163,184,.35); color:#94A3B8; }
.card { background:#111C2E; border:1px solid #223656; border-radius:16px; padding:18px 20px; height:100%; }
.card h4 { margin:0 0 4px; color:#F6F4EE; font-size:16px; }
.card .sub { color:#8FA3BC; font-size:12.5px; margin-bottom:10px; }
.kv { display:flex; justify-content:space-between; gap:12px; padding:6px 0; border-bottom:1px dashed #223656; font-size:13.5px; }
.kv span:first-child { color:#8FA3BC; } .kv span:last-child { color:#E5E7EB; font-weight:600; text-align:right; word-break:break-all; }
.stat { background:#111C2E; border:1px solid #223656; border-radius:14px; padding:14px 16px; }
.stat b { display:block; font-size:26px; color:#5EEAD4; line-height:1.1; } .stat span { color:#8FA3BC; font-size:12.5px; }
.src { background:#0F1A2C; border:1px solid #223656; border-left:4px solid #F2A33A; border-radius:12px; padding:12px 14px; margin:8px 0; }
.src.low { border-left-color:#64748B; }
.src-top { display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
.src-n { background:#F2A33A; color:#0B1F3A; font-weight:700; border-radius:50%; width:22px; height:22px; display:inline-flex;
         align-items:center; justify-content:center; font-size:12px; }
.src-name { color:#F6F4EE; font-weight:600; font-size:13.5px; }
.tag { background:#1B2D4A; color:#9DB4D2; border-radius:6px; padding:2px 8px; font-size:11.5px; font-weight:600; }
.src-sim { margin-left:auto; color:#5EEAD4; font-weight:700; font-size:13px; }
.bar { height:5px; background:#1B2D4A; border-radius:4px; margin:8px 0; } .bar div { height:5px; border-radius:4px; background:linear-gradient(90deg,#17C3B2,#F2A33A); }
.src details summary { cursor:pointer; color:#B9C9DC; font-size:13.5px; line-height:1.5; }
.src details div { color:#D5DEE8; font-size:13.5px; line-height:1.6; margin-top:8px; padding-top:8px; border-top:1px solid #223656; }
.meta { color:#8FA3BC; font-size:12px; margin-top:6px; }
.step { background:#111C2E; border:1px solid #223656; border-radius:12px; padding:12px 8px; text-align:center; font-size:13px; color:#E5E7EB; }
.step b { display:block; color:#F2A33A; font-size:12px; margin-bottom:2px; }
.notice { background:rgba(242,163,58,.08); border:1px solid rgba(242,163,58,.4); border-radius:12px; padding:14px 16px; color:#F2D9A8; font-size:14px; }
div[data-testid="stChatMessage"] { background:#0F1A2C; border:1px solid #1E3150; border-radius:14px; padding:12px 14px; }
.stButton button, .stDownloadButton button { border-radius:10px; border:1px solid #2A4A73; font-weight:600; }
.stButton button[kind="primary"] { background:#0D9488; border-color:#0D9488; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner=False)
def get_storage(path):
    return rag.Storage(path)


@st.cache_resource(show_spinner=False)
def get_embedder(name):
    return rag.Embedder(name)


@st.cache_resource(show_spinner=False)
def get_store(path, embed_name):
    return rag.VectorStore(get_storage(path), embed_name)


@st.cache_resource(show_spinner=False)
def get_llm(api_key, model, temperature):
    return rag.GeminiLLM(api_key, model=model, temperature=temperature)


storage = get_storage(STORAGE_DIR)
store = get_store(STORAGE_DIR, EMBED_NAME)


def embedder_ready():
    try:
        return get_embedder(EMBED_NAME)
    except Exception as exc:
        st.error(f"Embedding model could not be loaded: {type(exc).__name__}: {exc}")
        st.stop()


# ------------------------------------------------------------------ sidebar
st.session_state.setdefault("page", "💬 Ask Palghar AI")
st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploader_key", 0)

with st.sidebar:
    st.markdown("## 🏛️ Palghar AI")
    st.caption("District Development Intelligence Platform")
    st.radio("Navigate", ["💬 Ask Palghar AI", "📚 Knowledge Base", "⚙️ Model & System"], key="page",
             label_visibility="collapsed")
    st.divider()

    st.markdown("**Gemini (answer generator)**")
    default_key = os.environ.get("GEMINI_API_KEY", "")
    typed_key = st.text_input("API key", type="password", value="", placeholder="set in notebook" if default_key else "paste key",
                              help="Optional here if the notebook already loaded GEMINI_API_KEY from Colab Secrets.")
    api_key = typed_key.strip() or default_key
    model_name = st.text_input("Model", value=DEFAULT_MODEL)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    llm = None
    if api_key:
        try:
            llm = get_llm(api_key, model_name, temperature)
        except Exception as exc:
            st.error(f"Gemini setup failed: {exc}")
    st.markdown(f'<span class="chip {"" if llm else "warn"}">{"Gemini connected" if llm else "Retrieval-only mode"}</span>',
                unsafe_allow_html=True)
    reg0 = storage.load_registry()
    st.markdown(f'<span class="chip {"" if store.count() else "off"}">{len(reg0)} docs · {store.count()} passages</span>',
                unsafe_allow_html=True)
    st.caption("Index is saved on disk. After a Colab reconnect nothing needs to be re-trained or re-indexed.")


def hero(title_sub):
    llm_chip = '<span class="chip">Gemini answers</span>' if llm else '<span class="chip warn">Retrieval-only (no API key)</span>'
    st.markdown(f"""
    <div class="hero"><h1>Palghar AI</h1><p>{html.escape(title_sub)}</p>
    <span class="chip">RAG</span><span class="chip">ChromaDB vector database</span>
    <span class="chip">मराठी · हिंदी · English</span><span class="chip">Sources on every answer</span>{llm_chip}</div>
    """, unsafe_allow_html=True)


def source_cards(passages, low=False):
    out = []
    for i, p in enumerate(passages, 1):
        sim = max(0.0, min(1.0, p["similarity"]))
        text = html.escape(p["text"]).replace("\n", "<br>")
        summary = html.escape(p["text"][:170].replace("\n", " ")) + ("…" if len(p["text"]) > 170 else "")
        ocr = '<span class="tag">OCR</span>' if p.get("ocr") else ""
        out.append(
            f'<div class="src {"low" if low else ""}"><div class="src-top"><span class="src-n">{i}</span>'
            f'<span class="src-name">{html.escape(str(p["source"]))}</span><span class="tag">Page {p["page"]}</span>{ocr}'
            f'<span class="src-sim">{sim:.0%} match</span></div><div class="bar"><div style="width:{sim * 100:.0f}%"></div></div>'
            f'<details><summary>{summary}</summary><div>{text}</div></details></div>')
    return "".join(out)


def render_sources(msg):
    srcs = msg.get("sources") or []
    if not srcs:
        st.markdown('<div class="meta">Sources: none retrieved (the knowledge base is empty).</div>', unsafe_allow_html=True)
        return
    label = "Closest passages (below the match threshold)" if msg.get("low") else f"Sources ({len(srcs)})"
    with st.expander(f"📎 {label}", expanded=not msg.get("low")):
        st.markdown(source_cards(srcs, low=msg.get("low", False)), unsafe_allow_html=True)
    m = msg.get("meta", {})
    if m:
        st.markdown(f'<div class="meta">Search {m.get("retrieval_ms", 0):.0f} ms · Answer {m.get("generation_ms", 0):.0f} ms · '
                    f'{len(srcs)} passages · mode: {m.get("mode", "")}</div>', unsafe_allow_html=True)


# ================================================================== PAGE 1: ASK
def page_ask():
    hero("Ask questions about Palghar district in Marathi, Hindi or English. Every answer shows where it came from.")
    reg = storage.load_registry()
    names = {h: e["name"] for h, e in reg.items() if EMBED_NAME in e.get("indexed", {})}

    if store.count() == 0:
        st.markdown('<div class="notice"><b>No documents indexed yet.</b> Open <b>Knowledge Base</b>, upload your PDFs '
                    '(Marathi or English) and press <b>Index</b>. This takes a few minutes once; the index is then saved.</div>',
                    unsafe_allow_html=True)
        st.button("Go to Knowledge Base →", on_click=lambda: st.session_state.update(page="📚 Knowledge Base"))
        return

    with st.expander("⚙️ Search settings"):
        c1, c2, c3 = st.columns(3)
        k = c1.slider("Passages to retrieve", 1, 10, 5)
        min_sim = c2.slider("Minimum match", 0.0, 1.0, 0.0, 0.05,
                            help="Passages below this similarity are not given to the model. e5 scores are compressed, typically 0.7–0.9.")
        rewrite = c3.checkbox("Understand follow-up questions", value=True,
                              help="Rewrites a follow-up into a standalone question first (Day 4 technique). Needs Gemini.")
        scope = st.multiselect("Search only in these documents (empty = all)", options=list(names.values()))
    scope_hashes = [h for h, n in names.items() if n in scope] or None

    question = None
    if not st.session_state.messages:
        st.markdown("##### Try a question")
        sugg = ["पालघर जिल्ह्यातील आरोग्य सुविधांबद्दल सांगा.", "शेती आणि सिंचनाबद्दल माहिती द्या.",
                "What are the major development challenges in Palghar?", "Summarise education in Palghar district."]
        cols = st.columns(2)
        for i, s in enumerate(sugg):
            if cols[i % 2].button(s, key=f"sg{i}"):
                question = s

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_sources(msg)

    typed = st.chat_input("प्रश्न विचारा / Ask a question about Palghar...")
    question = typed or question
    if not question:
        if st.session_state.messages and st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.rerun()
        return

    history = [m for m in st.session_state.messages if m["role"] in ("user", "assistant")]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        emb = embedder_ready()
        search_q = question
        if rewrite and history and llm:
            try:
                rq, _ = rag.timed(llm.generate, rag.build_rewrite_prompt(question, history))
                search_q = rq.strip().splitlines()[0] if rq.strip() else question
                st.caption(f"Searching for: {search_q}")
            except Exception:
                search_q = question
        (passages, closest), t_ret = rag.timed(rag.retrieve, store, emb, search_q, k, min_sim, scope_hashes)

        reply, t_gen, mode, low = "", 0.0, "retrieval-only", False
        if not passages:
            reply = ("I could not find this in the uploaded documents. Try rephrasing, lowering the minimum match, "
                     "or adding the relevant PDF in the Knowledge Base.")
            low, mode = True, "no match"
        elif llm is None:
            reply = ("**Retrieval-only mode** (no Gemini key). These are the most relevant passages from the documents. "
                     "Add a key in the sidebar to get a written answer.")
        else:
            prompt = rag.build_prompt(search_q, passages, history)
            t0 = time.perf_counter()
            try:
                reply = st.write_stream(llm.stream(prompt, rag.SYSTEM_PROMPT))
                mode = f"RAG · {llm.model}"
            except Exception as exc:
                reply = (f"Gemini could not answer ({type(exc).__name__}: {str(exc)[:200]}). "
                         "Check the API key and model name in the sidebar. Showing the retrieved passages instead.")
                st.markdown(reply)
                mode = "retrieval-only (LLM error)"
            t_gen = (time.perf_counter() - t0) * 1000
        if mode.startswith("retrieval") or mode == "no match":
            st.markdown(reply)
        msg = {"role": "assistant", "content": reply, "sources": passages if passages else closest, "low": low,
               "meta": {"retrieval_ms": t_ret, "generation_ms": t_gen, "mode": mode}}
        render_sources(msg)
        st.session_state.messages.append(msg)


# ================================================================== PAGE 2: KNOWLEDGE BASE
def page_kb():
    hero("Upload or choose the PDFs Palghar AI should learn from. Marathi and scanned documents are supported.")
    reg = storage.load_registry()
    docs = [e for e in reg.values() if embedder_name_ok(e)]
    c = st.columns(4)
    for col, (val, lab) in zip(c, [(len(docs), "Documents"), (store.count(), "Searchable passages"),
                                   (sum(e["pages"] for e in docs), "Pages read"), (sum(e.get("ocr_pages", 0) for e in docs), "Pages via OCR")]):
        col.markdown(f'<div class="stat"><b>{val}</b><span>{lab}</span></div>', unsafe_allow_html=True)
    st.write("")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### ⬆️ Upload PDFs")
        files = st.file_uploader("Drop PDF files here (Marathi / English)", type=["pdf"], accept_multiple_files=True,
                                 key=f"up{st.session_state.uploader_key}")
        force_ocr = st.checkbox("Force OCR (use when Marathi text looks broken or the PDF is scanned)")
        with st.expander("Advanced"):
            csize = st.slider("Passage size (characters)", 300, 1500, 700, 50)
            cover = st.slider("Overlap (characters)", 0, 300, 120, 10)
        if st.button("Index uploaded PDFs", type="primary", disabled=not files):
            emb = embedder_ready()
            with st.status("Indexing...", expanded=True) as status:
                bar = st.progress(0.0)
                for f in files:
                    st.write(f"**{f.name}**")
                    res = rag.ingest_pdf(storage, store, emb, f.getvalue(), f.name, force_ocr=force_ocr,
                                         chunk_size=csize, overlap=cover,
                                         progress=lambda m, fr=None: (st.write(m) if fr in (0.05, 0.25) else None,
                                                                      bar.progress(fr) if fr is not None else None))
                    if res["status"] == "indexed":
                        st.success(f"{res['pages']} pages → {res['chunks']} passages ({res.get('ocr_pages', 0)} via OCR)")
                    elif res["status"] == "already_indexed":
                        st.info(f"Already indexed ({res['chunks']} passages). Skipped.")
                    else:
                        st.warning("No readable text found.")
                    for w in res.get("warnings", []):
                        st.warning(w)
                status.update(label="Indexing finished", state="complete")
            st.session_state.uploader_key += 1
            st.rerun()
    with right:
        st.markdown("#### 🗂️ Choose from stored files")
        reg = storage.load_registry()
        if not reg:
            st.caption("No stored PDFs yet.")
        else:
            labels = {h: f"{e['name']}  ({e['pages']} pages)" for h, e in reg.items()}
            chosen = st.multiselect("Select files", options=list(labels), format_func=lambda h: labels[h])
            b1, b2 = st.columns(2)
            if b1.button("Re-index selected", disabled=not chosen):
                emb = embedder_ready()
                with st.status("Re-indexing...", expanded=True):
                    for h in chosen:
                        p = storage.pdf_path(h)
                        if p is None:
                            st.warning(f"{reg[h]['name']}: stored PDF missing, upload it again.")
                            continue
                        res = rag.ingest_pdf(storage, store, emb, p.read_bytes(), reg[h]["name"], force_ocr=force_ocr, reindex=True)
                        st.write(f"{reg[h]['name']}: {res['chunks']} passages")
                st.rerun()
            if b2.button("Delete selected", disabled=not chosen):
                for h in chosen:
                    rag.remove_document(storage, store, h)
                st.rerun()

    st.markdown("#### 📋 Indexed documents")
    if reg:
        rows = [{"Document": e["name"], "Pages": e["pages"], "OCR pages": e.get("ocr_pages", 0),
                 "Passages": e.get("indexed", {}).get(EMBED_NAME, {}).get("chunks", "not indexed"),
                 "Added": e.get("uploaded_at", "")} for e in reg.values()]
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.markdown("#### 💾 Backup and restore")
    st.caption("Everything (PDFs, vector index, registry) lives in one folder. Download a backup so you can restore it after a Colab reset.")
    k1, k2 = st.columns(2)
    with k1:
        if st.button("Prepare backup ZIP"):
            z = storage.make_backup(Path(STORAGE_DIR).parent / "PalgharAI_backup.zip")
            st.session_state.backup_bytes = z.read_bytes()
        if st.session_state.get("backup_bytes"):
            st.download_button("⬇️ Download backup", st.session_state.backup_bytes, "PalgharAI_backup.zip", "application/zip")
    with k2:
        up = st.file_uploader("Restore from backup ZIP", type=["zip"], key="restore")
        if up and st.button("Restore now"):
            try:
                n = storage.restore_backup(up.getvalue())
                st.cache_resource.clear()
                st.success(f"Restored {n} files. Reloading...")
                st.rerun()
            except Exception as exc:
                st.error(f"Restore failed: {exc}")


def embedder_name_ok(entry):
    return EMBED_NAME in entry.get("indexed", {})


# ================================================================== PAGE 3: MODEL & SYSTEM
def kv(rows):
    return "".join(f'<div class="kv"><span>{html.escape(str(a))}</span><span>{html.escape(str(b))}</span></div>' for a, b in rows)


def page_system():
    hero("Live specification of the models and database that power this page.")
    try:
        emb = get_embedder(EMBED_NAME)
        espec = emb.spec()
    except Exception as exc:
        espec = {"name": EMBED_NAME, "dimension": "not loaded", "max_tokens": "-", "device": f"error: {exc}"[:60], "query_prefix": "-"}
    reg = storage.load_registry()
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="card"><h4>🧠 Answer generator (LLM)</h4><div class="sub">Writes the answer from retrieved passages</div>'
                + kv([("Model", model_name), ("Provider", "Google Gemini API"), ("Temperature", temperature),
                      ("Status", "connected" if llm else "no API key"), ("Grounding", "context-only prompt"),
                      ("Citations", "[n] per fact")]) + "</div>", unsafe_allow_html=True)
    c2.markdown(f'<div class="card"><h4>🔎 Embedding model</h4><div class="sub">Turns Marathi / English text into vectors</div>'
                + kv([("Model", espec["name"]), ("Vector size", espec["dimension"]), ("Max tokens", espec["max_tokens"]),
                      ("Device", espec["device"]), ("Prefixes", espec["query_prefix"]), ("Languages", "100+ incl. Marathi, Hindi")])
                + "</div>", unsafe_allow_html=True)
    c3.markdown(f'<div class="card"><h4>🗄️ Vector database</h4><div class="sub">Stores passages and finds the closest ones</div>'
                + kv([("Engine", f"ChromaDB {store.chromadb_version}"), ("Index", "HNSW, cosine distance"),
                      ("Collection", store.name), ("Passages", store.count()), ("Documents", len(reg)),
                      ("Persistent", "yes (on disk)")]) + "</div>", unsafe_allow_html=True)

    st.markdown("#### How a question is answered")
    steps = [("1", "Question"), ("2", "Embed query"), ("3", "Search ChromaDB"), ("4", "Top passages"), ("5", "Gemini writes answer"), ("6", "Answer + sources")]
    cols = st.columns(6)
    for col, (n, t) in zip(cols, steps):
        col.markdown(f'<div class="step"><b>STEP {n}</b>{t}</div>', unsafe_allow_html=True)
    st.write("")

    last = next((m for m in reversed(st.session_state.messages) if m["role"] == "assistant" and m.get("meta")), None)
    a, b = st.columns(2)
    with a:
        st.markdown("#### Runtime")
        try:
            import torch
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            tv = torch.__version__
        except Exception:
            gpu, tv = "unknown", "-"
        rows = [("Python", platform.python_version()), ("PyTorch", tv), ("Compute", gpu), ("Streamlit", st.__version__),
                ("Storage folder", STORAGE_DIR)]
        ok, langs = rag.ocr_available()
        rows.append(("OCR (Tesseract)", ("ready: " + ", ".join(l for l in langs if l in ("mar", "eng", "hin"))) if ok else "not installed"))
        st.markdown('<div class="card">' + kv(rows) + "</div>", unsafe_allow_html=True)
    with b:
        st.markdown("#### Last answer")
        if last:
            m = last["meta"]
            st.markdown('<div class="card">' + kv([("Search time", f"{m['retrieval_ms']:.0f} ms"),
                                                    ("Generation time", f"{m['generation_ms']:.0f} ms"),
                                                    ("Passages used", len(last["sources"])), ("Mode", m["mode"])]) + "</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><div class="sub">Ask a question to see timings here.</div></div>', unsafe_allow_html=True)
    st.info("This system does not train a language model. Knowledge comes from your PDFs, which are indexed once and saved. "
            "Answers are written only from retrieved passages and always list their sources. Verify important figures in the original PDF.")


PAGES = {"💬 Ask Palghar AI": page_ask, "📚 Knowledge Base": page_kb, "⚙️ Model & System": page_system}
PAGES[st.session_state.page]()
st.divider()
st.caption("Palghar AI · RAG with ChromaDB · Academic project demonstration · AI can make mistakes, check the cited sources.")