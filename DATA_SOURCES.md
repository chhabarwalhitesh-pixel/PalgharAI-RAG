# Data sources

The following public government portals were supplied as the source locations used to collect reference documents for the PalgharAI-RAG knowledge base.

> These links point to official portal pages and document repositories, not to a frozen copy of the dataset. The documents available on those pages may change over time. The repository does not redistribute the PDFs or claim ownership of them.

## Source portals

1. **Palghar District — Planning**  
   https://palghar.gov.in/en/planning/

2. **Zilla Parishad Palghar — Documents**  
   https://www.zppalghar.gov.in/en/documents/

3. **Palghar District — Documents**  
   https://palghar.gov.in/en/documents/

4. **Zilla Parishad Palghar — Rural Water Supply Department**  
   https://www.zppalghar.gov.in/en/rural-water-supply-department/

5. **Zilla Parishad Palghar — District Rural Development Agency (DRDA)**  
   https://www.zppalghar.gov.in/en/district-rural-development-agency-drda/

6. **Zilla Parishad Palghar — Budget Information**  
   https://www.zppalghar.gov.in/en/document-category/budget-information/

7. **Palghar District — Disaster Management**  
   https://palghar.gov.in/en/disaster-management/

## Dataset and attribution notes

- The project uses documents obtained from the above portals as inputs to its document-ingestion and retrieval workflow.
- No source PDFs, extracted text dumps, embeddings, or prebuilt vector database are included in this GitHub repository.
- For reproducibility, record the exact document title, issuing department, download URL, access date, language, and page count for each PDF that is actually indexed.
- Check the relevant portal's terms, document-specific copyright notices, and applicable government reuse policy before redistributing any downloaded file or extracted content.
- A portal being publicly accessible does not automatically mean every file on it can be republished or relicensed.
- If a source document is moved or removed, use the issuing department's portal to locate the current version and verify its publication date.

## Relationship to the project

The links above are source repositories/pages. PalgharAI-RAG processes selected PDFs from those sources using PDF text extraction and OCR, multilingual embeddings, ChromaDB retrieval, and optional Gemini answer generation. See [README.md](README.md) for setup and architecture.
