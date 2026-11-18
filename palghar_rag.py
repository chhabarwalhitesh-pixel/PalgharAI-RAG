"""
palghar_rag.py - core of Palghar AI (RAG + vector database).

Pipeline (same building blocks the faculty taught, Day 1-4):
  PDF -> text (PyMuPDF, OCR for scanned Marathi pages) -> chunks -> multilingual embeddings
  -> ChromaDB (persistent) -> semantic search -> grounded prompt -> Gemini -> answer + sources
"""
import hashlib
import io
import json
import re
import shutil
import time
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ------------------------------------------------------------------ storage (everything that must survive a reconnect)
class Storage:
    def __init__(self, root):
        self.root = Path(root)
        self.pdfs = self.root / "pdfs"
        self.db = self.root / "chroma_db"
        self.registry_path = self.root / "registry.json"
        for p in (self.root, self.pdfs, self.db):
            p.mkdir(parents=True, exist_ok=True)

    def load_registry(self):
        if self.registry_path.exists():
            try:
                return json.loads(self.registry_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def save_registry(self, reg):
        self.registry_path.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")

    def save_pdf(self, file_hash, filename, data):
        safe = re.sub(r"[^\w.\- ()]+", "_", Path(filename).name, flags=re.UNICODE)
        path = self.pdfs / f"{file_hash[:8]}__{safe}"
        path.write_bytes(data)
        return path

    def pdf_path(self, file_hash):
        hits = list(self.pdfs.glob(f"{file_hash[:8]}__*"))
        return hits[0] if hits else None

    def make_backup(self, zip_path):
        zip_path = Path(zip_path)
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in self.root.rglob("*"):
                if f.is_file() and f.resolve() != zip_path.resolve():
                    z.write(f, arcname=str(f.relative_to(self.root)))
        return zip_path

    def restore_backup(self, zip_bytes_or_path):
        src = io.BytesIO(zip_bytes_or_path) if isinstance(zip_bytes_or_path, (bytes, bytearray)) else zip_bytes_or_path
        with zipfile.ZipFile(src) as z:
            names = z.namelist()
            if "registry.json" not in names and not any(n.startswith("chroma_db/") for n in names):
                raise ValueError("This ZIP does not look like a Palghar AI backup.")
            z.extractall(self.root)
        return len(names)


# ------------------------------------------------------------------ PDF text (with OCR fallback for scanned Marathi pages)
def normalize_text(text):
    text = (text or "").replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ocr_available():
    try:
        import pytesseract
        langs = pytesseract.get_languages(config="")
        return True, langs
    except Exception:
        return False, []


def ocr_page(page, lang="mar+eng", dpi=200):
    import pytesseract
    from PIL import Image
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img, lang=lang)


def extract_pages(pdf_path, force_ocr=False, ocr_lang="mar+eng", min_chars=40):
    """Returns (pages, warnings). Each page: {page, text, ocr}."""
    import fitz
    doc = fitz.open(str(pdf_path))
    pages, warnings, ocr_ok = [], [], None
    for i, page in enumerate(doc):
        text = "" if force_ocr else normalize_text(page.get_text("text", sort=True))
        used_ocr = False
        if force_ocr or len(text) < min_chars:
            if ocr_ok is None:
                ocr_ok, langs = ocr_available()
                if not ocr_ok:
                    warnings.append("OCR is not available (tesseract missing); scanned pages were skipped.")
                elif "mar" not in langs:
                    warnings.append("Marathi OCR data (tesseract-ocr-mar) is not installed; OCR will use English only.")
                    ocr_lang = "eng"
            if ocr_ok:
                try:
                    ocr_text = normalize_text(ocr_page(page, ocr_lang))
                    if len(ocr_text) > len(text):
                        text, used_ocr = ocr_text, True
                except Exception as exc:
                    warnings.append(f"OCR failed on page {i + 1}: {exc}")
        pages.append({"page": i + 1, "text": text, "ocr": used_ocr})
    doc.close()
    return pages, warnings


# ------------------------------------------------------------------ chunking (sentence-aware; Marathi danda aware)
def chunk_text(text, size=700, overlap=120):
    text = re.sub(r"[ \t]+", " ", text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"(?<=[।॥.?!])\s+|\n+", text) if p.strip()]
    def word_tail(s):                                      # last ~overlap chars, starting at a word boundary
        if not s or not overlap or len(s) <= overlap:
            return ""
        t = s[-overlap:]
        i = t.find(" ")
        return t[i + 1:] if i != -1 else ""

    def hard_cut(s):                                       # cut a very long unbroken text at a space
        cut = s.rfind(" ", 0, size)
        cut = cut if cut > size * 0.5 else size
        return s[:cut].strip(), s[max(1, cut - overlap):].lstrip() if overlap and cut > overlap else s[cut:].lstrip()

    chunks, cur = [], ""
    for sent in parts:
        if len(cur) + len(sent) + 1 <= size:
            cur = f"{cur} {sent}".strip()
            continue
        if cur:
            chunks.append(cur)
        tail = word_tail(cur)
        cur = f"{tail} {sent}".strip() if len(sent) < size else sent
        while len(cur) > int(size * 1.5):
            head, cur = hard_cut(cur)
            chunks.append(head)
    if cur:
        chunks.append(cur)
    return chunks


# ------------------------------------------------------------------ embeddings (multilingual: Marathi / Hindi / English)
class Embedder:

# ------------------------------------------------------------------
# Embeddings (direct Transformers loading - no SentenceTransformer
# AutoProcessor initialization)
    def __init__(self, name=DEFAULT_EMBED_MODEL, device=None, encode_fn=None):
        import os
        import torch
        from transformers import AutoTokenizer, AutoModel

        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._fn = encode_fn
        self.model = None
        self.tokenizer = None
        self.dim = None
        self.max_seq_length = 512
        self.e5 = "e5" in name.lower()

        if encode_fn is None:
            cache_dir = os.environ.get("HF_HOME")

            print(f"Loading tokenizer: {name}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                name,
                cache_dir=cache_dir,
                use_fast=True
            )

            print(f"Loading transformer model: {name}")

            self.model = AutoModel.from_pretrained(
                name,
                cache_dir=cache_dir,
                use_safetensors=True
            )

            self.model.to(self.device)
            self.model.eval()

            self.dim = self.model.config.hidden_size

            print("Embedding model loaded successfully!")
            print("Device:", self.device)
            print("Vector dimension:", self.dim)

    def _encode(self, texts, batch_size=16):
        if self._fn is not None:
            return self._fn(texts)

        import torch
        import torch.nn.functional as F

        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                outputs = self.model(**inputs)

                token_embeddings = outputs.last_hidden_state

                attention_mask = inputs["attention_mask"]
                mask = attention_mask.unsqueeze(-1).expand(
                    token_embeddings.size()
                ).to(token_embeddings.dtype)

                # Mean pooling
                summed = torch.sum(token_embeddings * mask, dim=1)
                counts = torch.sum(mask, dim=1).clamp(min=1e-9)

                embeddings = summed / counts

                # Normalize vectors
                embeddings = F.normalize(
                    embeddings,
                    p=2,
                    dim=1
                )

                all_embeddings.extend(
                    embeddings.cpu().tolist()
                )

        return all_embeddings

    def embed_passages(self, texts, batch_size=16):
        texts = [
            ("passage: " + t) if self.e5 else t
            for t in texts
        ]
        return self._encode(texts, batch_size)

    def embed_query(self, text):
        query = ("query: " + text) if self.e5 else text
        return self._encode([query])[0]

    def spec(self):
        return {
            "name": self.name,
            "dimension": self.dim,
            "max_tokens": self.max_seq_length,
            "device": self.device,
            "query_prefix": "query: / passage:" if self.e5 else "none"
        }



# ------------------------------------------------------------------ vector database (ChromaDB, persistent)
class VectorStore:
    def __init__(self, storage, embed_name):
        import chromadb
        self.chromadb_version = chromadb.__version__
        self.path = str(storage.db)
        self.client = chromadb.PersistentClient(path=self.path)
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", embed_name).strip("_")[-40:] or "model"
        self.name = f"palghar_{slug}"          # one collection per embedding model (vectors are not interchangeable)
        try:
            self.col = self.client.get_or_create_collection(name=self.name, metadata={"hnsw:space": "cosine"})
        except Exception:
            self.col = self.client.get_or_create_collection(name=self.name, configuration={"hnsw": {"space": "cosine"}})

    def count(self):
        return self.col.count()

    def add_chunks(self, file_hash, source, items, embeddings):
        ids = [f"{file_hash[:12]}_{it['page']}_{it['chunk']}" for it in items]
        self.col.upsert(
            ids=ids,
            documents=[it["text"] for it in items],
            embeddings=embeddings,
            metadatas=[{"file_hash": file_hash, "source": source, "page": it["page"],
                        "chunk": it["chunk"], "ocr": bool(it["ocr"])} for it in items],
        )

    def delete_document(self, file_hash):
        try:
            self.col.delete(where={"file_hash": file_hash})
        except Exception:
            pass

    def query(self, query_embedding, k=5, file_hashes=None):
        if self.count() == 0:
            return []
        kwargs = {"query_embeddings": [query_embedding], "n_results": min(k, self.count()),
                  "include": ["documents", "metadatas", "distances"]}
        if file_hashes:
            kwargs["where"] = {"file_hash": {"$in": list(file_hashes)}}
        res = self.col.query(**kwargs)
        out = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append({"text": doc, "source": meta.get("source", "?"), "page": meta.get("page", "?"),
                        "ocr": bool(meta.get("ocr")), "file_hash": meta.get("file_hash"),
                        "similarity": round(1.0 - float(dist), 4)})
        return out


# ------------------------------------------------------------------ ingestion (index once, reuse after every reconnect)
def ingest_pdf(storage, store, embedder, data, filename, force_ocr=False, reindex=False,
               chunk_size=700, overlap=120, progress=None):
    """Index one PDF. Skips work if this exact file is already indexed with this embedding model."""
    say = progress or (lambda msg, frac=None: None)
    file_hash = hashlib.sha256(data).hexdigest()
    reg = storage.load_registry()
    entry = reg.get(file_hash, {})
    if entry and embedder.name in entry.get("indexed", {}) and not reindex:
        return {"status": "already_indexed", "name": entry["name"], "hash": file_hash,
                "chunks": entry["indexed"][embedder.name]["chunks"], "warnings": []}

    path = storage.pdf_path(file_hash) or storage.save_pdf(file_hash, filename, data)
    say(f"Reading {filename} ...", 0.05)
    pages, warnings = extract_pages(path, force_ocr=force_ocr)
    items = []
    for p in pages:
        for j, ch in enumerate(chunk_text(p["text"], chunk_size, overlap)):
            items.append({"page": p["page"], "chunk": j, "text": ch, "ocr": p["ocr"]})
    chars = sum(len(p["text"]) for p in pages)
    if not items:
        return {"status": "no_text", "name": filename, "hash": file_hash, "chunks": 0,
                "warnings": warnings + ["No readable text found. If this is a scanned PDF, enable OCR."]}

    say(f"{len(pages)} pages, {len(items)} passages. Creating embeddings ...", 0.25)
    store.delete_document(file_hash)
    embeddings = []
    step = 32
    for i in range(0, len(items), step):
        embeddings += embedder.embed_passages([it["text"] for it in items[i:i + step]], step)
        say(f"Embedding passages {min(i + step, len(items))}/{len(items)}", 0.25 + 0.65 * min(1, (i + step) / len(items)))
    store.add_chunks(file_hash, Path(filename).name, items, embeddings)

    entry = reg.get(file_hash, {})
    entry.update({"name": Path(filename).name, "size": len(data), "pages": len(pages), "chars": chars,
                  "ocr_pages": sum(1 for p in pages if p["ocr"]), "uploaded_at": entry.get("uploaded_at", now_iso())})
    entry.setdefault("indexed", {})[embedder.name] = {"chunks": len(items), "at": now_iso()}
    reg[file_hash] = entry
    storage.save_registry(reg)
    say("Done", 1.0)
    return {"status": "indexed", "name": filename, "hash": file_hash, "chunks": len(items),
            "pages": len(pages), "ocr_pages": entry["ocr_pages"], "warnings": warnings}


def remove_document(storage, store, file_hash, delete_file=True):
    store.delete_document(file_hash)
    reg = storage.load_registry()
    reg.pop(file_hash, None)
    storage.save_registry(reg)
    if delete_file:
        p = storage.pdf_path(file_hash)
        if p:
            p.unlink(missing_ok=True)


# ------------------------------------------------------------------ retrieval + prompt (grounded, with numbered sources)
SYSTEM_PROMPT = (
    "You are Palghar AI, a district information assistant for government officials of Palghar, Maharashtra.\n"
    "Rules:\n"
    "1. Answer ONLY from the numbered context passages. Do not use outside knowledge and never guess.\n"
    "2. Cite the passage number after each fact, like [1] or [2][3].\n"
    "3. If the context does not contain the answer, say clearly that you could not find it in the uploaded "
    "documents. Do not invent figures.\n"
    "4. Keep numbers, units, years and names exactly as written in the passages. If passages disagree, say so.\n"
    "5. Reply in the same language as the question (Marathi, Hindi or English). Be concise and clear."
)


def format_context(passages):
    return "\n\n".join(f"[{i}] (Source: {p['source']}, page {p['page']})\n{p['text']}"
                       for i, p in enumerate(passages, 1))


def build_prompt(question, passages, history=None):
    hist = ""
    for m in (history or [])[-6:]:
        hist += f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:500]}\n"
    ctx = format_context(passages) if passages else "(no relevant passages found)"
    return (f"Previous conversation:\n{hist or '(none)'}\n\nContext passages:\n{ctx}\n\n"
            f"Question: {question}\n\nAnswer (with [n] citations):")


def build_rewrite_prompt(question, history):
    hist = "".join(f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:400]}\n" for m in history[-6:])
    return ("Rewrite the latest question as a complete standalone question, using the conversation for context. "
            "Keep the same language. Do not answer it. Return only the rewritten question.\n\n"
            f"Conversation:\n{hist}\nLatest question: {question}\n\nStandalone question:")


def retrieve(store, embedder, query, k=5, min_similarity=0.0, file_hashes=None):
    qv = embedder.embed_query(query)
    hits = store.query(qv, k=k, file_hashes=file_hashes)
    return [h for h in hits if h["similarity"] >= min_similarity], hits


# ------------------------------------------------------------------ Gemini (as taught in Day 3 / Day 4)
class GeminiLLM:
    def __init__(self, api_key, model=DEFAULT_GEMINI_MODEL, temperature=0.2, max_output_tokens=1500):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def _config(self, system):
        cfg = {"temperature": self.temperature, "max_output_tokens": self.max_output_tokens}
        if system:
            cfg["system_instruction"] = system
        return cfg

    def generate(self, prompt, system=None):
        r = self.client.models.generate_content(model=self.model, contents=prompt, config=self._config(system))
        return (r.text or "").strip()

    def stream(self, prompt, system=None):
        for chunk in self.client.models.generate_content_stream(model=self.model, contents=prompt,
                                                                config=self._config(system)):
            if getattr(chunk, "text", None):
                yield chunk.text

    def spec(self):
        return {"model": self.model, "temperature": self.temperature, "max_output_tokens": self.max_output_tokens}

    def list_models(self):
        names = []
        for m in self.client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if "generateContent" in actions or not actions:
                names.append(m.name.replace("models/", ""))
        return sorted(n for n in names if n.startswith("gemini"))


def timed(fn, *a, **kw):
    t = time.perf_counter()
    out = fn(*a, **kw)
    return out, (time.perf_counter() - t) * 1000.0