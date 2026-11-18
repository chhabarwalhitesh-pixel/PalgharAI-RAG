# PalgharAI-RAG
### Multilingual, source-grounded document question answering for Palghar district

PalgharAI-RAG is an experimental Retrieval-Augmented Generation (RAG) application built around district documents. It lets a user index PDF documents and ask questions in **Marathi, Hindi, or English** through a Streamlit interface.

The system extracts PDF text, applies OCR to scanned pages, creates sentence-aware chunks, embeds the chunks with a multilingual Sentence-Transformers model, stores vectors in ChromaDB, retrieves relevant passages, and (when configured) uses Google Gemini to draft an answer grounded in the retrieved context.

> **Important:** This project is a document assistant, not an official government service. OCR and generated answers can be wrong. Verify important figures, dates, and decisions against the cited original PDF pages.

## Features

- **Multilingual questions and documents:** designed for Marathi, Hindi, and English.
- **PDF text extraction:** PyMuPDF for digitally searchable PDFs.
- **OCR for scanned pages:** Tesseract OCR with Marathi (`mar`), Hindi (`hin`), and English (`eng`) language data.
- **Sentence-aware chunking:** handles the Marathi danda punctuation mark (`।`) as well as common sentence boundaries.
- **Multilingual embeddings:** `intfloat/multilingual-e5-base` by default; `intfloat/multilingual-e5-small` is an available lighter option.
- **Persistent vector search:** ChromaDB `PersistentClient`, cosine-distance retrieval, and separate collections for embedding models.
- **Grounded generation:** optional Google Gemini integration with an answer-only-from-context instruction.
- **Source transparency:** answer source cards show document name, page, match information, and retrieved passage.
- **Knowledge Base page:** upload, select, index, and delete documents; create or restore a knowledge-base backup.
- **Model & System page:** view the running model and vector database specifications.
- **Conversation support:** chat history and follow-up question rewriting.
- **Persistence options:** Google Drive storage or a downloadable ZIP backup for Colab sessions.

## Architecture

```text
PDF documents
    |
    +--> PyMuPDF text extraction
    |
    +--> Tesseract OCR (for scanned/image-only pages)
    |
    v
Page text + document/page metadata
    |
    v
Sentence-aware text chunking
    |
    v
Multilingual Sentence-Transformers embeddings
    |
    v
ChromaDB persistent vector store
    |
User question --> question rewriting (for follow-ups)
    |
    v
Semantic retrieval + minimum-match threshold
    |
    +--> Retrieved passages and source details
    |
    +--> Optional Gemini grounded answer
    |
    v
Streamlit UI: answer + cited source passages
```

## Technology stack

| Layer | Technology |
|---|---|
| User interface | Streamlit |
| PDF extraction | PyMuPDF |
| OCR | Tesseract OCR via `pytesseract` |
| Image handling | Pillow |
| Text chunking | Python sentence-aware chunking |
| Embeddings | Sentence-Transformers |
| Default embedding model | `intfloat/multilingual-e5-base` |
| Vector database | ChromaDB |
| Answer generation | Google Gemini via `google-genai` |
| Data handling | Python, pandas |
| Runtime shown in project notebook | Google Colab |

## Repository contents

```text
PalgharAI-RAG/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── CONTRIBUTING.md
├── app.py
├── palghar_rag.py
├── .streamlit/
│   └── config.toml
└── PalgharAI_RAG_Streamlit.ipynb
```

`app.py` contains the Streamlit interface, while `palghar_rag.py` contains the reusable document-ingestion, OCR, embedding, ChromaDB retrieval, and Gemini integration logic. The notebook is retained as the guided Google Colab workflow; it also writes these modules into the Colab runtime when run.

## Run the project

### Recommended: Google Colab

The supplied notebook is designed to be run in Google Colab.

1. Open `PalgharAI_RAG_Streamlit.ipynb` in Google Colab.
2. Choose **Runtime → Change runtime type**. A T4 GPU is recommended for faster embedding generation; CPU may also work, but indexing will be slower.
3. If you want generated answers, add a Google Gemini API key in Colab **Secrets** using the exact name `GEMINI_API_KEY`, then allow notebook access to it.
4. Run the notebook from top to bottom.
5. The notebook installs Tesseract and its Marathi, Hindi, and English language packs, installs the Python packages, prepares the RAG engine and Streamlit app, configures storage, and launches the app.
6. Open the temporary URL printed by the launch cell. Keep the notebook session running while using the app.

The notebook uses Google Drive by default (`USE_GOOGLE_DRIVE = True`) and stores the knowledge-base data under `MyDrive/Palghar_AI`. If Google Drive persistence is disabled, use the app's **Knowledge Base → Backup and restore** feature to download a backup ZIP before the Colab session ends.

### Gemini API key and retrieval-only mode

- The key is read from Colab Secrets as `GEMINI_API_KEY`; the notebook does not intentionally print or save the key.
- If no key is provided, the application can still run in **retrieval-only mode** and display matching passages with source information. It will not produce a Gemini-written answer.
- Never commit a real API key, `.env` file, backup archive, or private document to GitHub.

## Configuration

The notebook's Step 2 settings include:

| Setting | Default | Purpose |
|---|---|---|
| `USE_GOOGLE_DRIVE` | `True` | Store the index and related files in Google Drive |
| `RESTORE_BACKUP` | `False` | Restore a previously downloaded knowledge-base backup |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | Multilingual embedding model |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Gemini model identifier used by the notebook |
| `ASK_FOR_KEY` | `True` | Prompt for a key if no Colab Secret is available |
| `PORT` | `8504` | Local Streamlit port |

Model identifiers and API availability can change. If the configured Gemini model is unavailable to your API key, use a model identifier currently supported by your Google AI account and update the notebook setting.

## Local environment notes

The notebook is Colab-oriented: it uses Colab upload/Secrets/Drive features, notebook shell commands, and a Cloudflare quick tunnel for temporary sharing. The notebook is **not a one-command local deployment package**.

For a local run, install the system-level Tesseract OCR application and its Marathi, Hindi, and English language data for your operating system. Create and activate a Python virtual environment, install the packages in `requirements.txt`, and install a PyTorch build suitable for your hardware/runtime if it is not already available. Set the environment variables described in `.env.example` (or export them in your shell), then run `streamlit run app.py`. The repository deliberately does not install or replace PyTorch because the correct build depends on the runtime and hardware.

## Privacy and security

- Treat uploaded PDFs, extracted text, ChromaDB files, and backups as potentially sensitive.
- Do not commit source documents, indexed data, model caches, backups, logs, or API keys unless you have permission and have reviewed their contents.
- The notebook launches a temporary Cloudflare quick tunnel. Anyone who obtains the generated public URL may be able to reach the app while the tunnel is active. Do not expose confidential documents or use this setup for sensitive production workloads.
- The notebook disables some Streamlit server protections for its temporary tunnel setup. Review and harden deployment settings before using any persistent public deployment.
- Use an authenticated, access-controlled deployment for real organizational use.

## Limitations

- OCR quality depends on scan resolution, contrast, page rotation, and font quality. Poor OCR can misread names, dates, and numeric values.
- Legacy Marathi fonts or unusual encodings may require forcing OCR from the Knowledge Base page.
- Embedding similarity and match percentages are retrieval signals, not proof that a passage answers the question.
- Gemini can still misunderstand or omit details even when supplied with retrieved context.
- The source collection determines coverage; information absent from indexed PDFs cannot be reliably answered.
- The default embedding model is relatively large; the `multilingual-e5-small` option can reduce resource requirements at a possible quality trade-off.
- Colab storage is temporary unless Google Drive is enabled or a backup is downloaded.

## Troubleshooting

| Issue | Things to check |
|---|---|
| Marathi text appears corrupted | Enable **Force OCR** when indexing the affected PDF; check that Tesseract's `mar` language data is installed. |
| Gemini cannot answer | Check the API key and model identifier; try retrieval-only mode to verify that passages are being found. |
| Relevant text is not retrieved | Lower the minimum-match threshold, increase the number of passages retrieved, or rephrase using terms present in the document. |
| Indexing runs out of GPU memory | Change the embedding model to `intfloat/multilingual-e5-small` in Step 2 or use CPU. |
| Index disappears after reconnect | Enable Google Drive persistence or restore the downloaded knowledge-base ZIP. |
| Streamlit page does not open | Check the launch-cell output and Streamlit log; keep the Colab cell/session running while accessing the temporary URL. |

## Responsible use

PalgharAI-RAG is an experimental academic/research project. It is not affiliated with or endorsed by the Government of Maharashtra or any Palghar district authority. Do not rely on generated output as the sole basis for public policy, legal, financial, health, or other high-impact decisions. Check the original cited document and its reference period.

## Source documents and licensing

The notebook accepts PDFs supplied by the user through the application's Knowledge Base workflow. This repository does not bundle those PDFs or a prebuilt index. Before publishing source documents, extracted datasets, or backups, confirm that their copyright and terms permit redistribution.

No repository license is included by default. Add a license only after confirming that you have the right to license all code and other materials in the repository. Third-party models, libraries, and APIs remain subject to their own terms.

## Acknowledgements

- PyMuPDF
- Tesseract OCR
- Sentence-Transformers and the `intfloat/multilingual-e5-base` model
- ChromaDB
- Google Gemini API and `google-genai`
- Streamlit
- Google Colab

---

**Project:** PalgharAI-RAG  
**Domain:** Palghar district, Maharashtra, India  
**Approach:** Retrieval-Augmented Generation (RAG)  
**Languages:** Marathi, Hindi, English  
**Interface:** Streamlit