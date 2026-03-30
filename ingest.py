"""
ingest.py — Run this ONCE locally after adding/changing documents.

Usage:
    pip install -r requirements.txt
    python ingest.py

This script:
  1. Reads every PDF in the `documents/` folder
  2. Splits them into overlapping chunks
  3. Embeds them with a free local model (all-MiniLM-L6-v2)
  4. Saves the FAISS index to `vectorstore/`

Commit the `vectorstore/` folder to GitHub — the Streamlit app loads it at startup.
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


DOCS_DIR = Path("documents")
VECTORSTORE_DIR = Path("vectorstore")
CHUNK_SIZE = 1000       # characters per chunk
CHUNK_OVERLAP = 150     # overlap between consecutive chunks


def ingest():
    pdf_files = list(DOCS_DIR.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in `{DOCS_DIR}/`. Add your course files there and re-run.")
        return

    print(f"Found {len(pdf_files)} PDF(s). Loading…")
    all_docs = []

    for pdf_path in sorted(pdf_files):
        # Use relative path (e.g. "slides/lecture_03.pdf") so citations are informative
        relative_path = pdf_path.relative_to(DOCS_DIR)
        print(f"  -> {relative_path}")
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = str(relative_path)
        all_docs.extend(pages)

    print(f"\nLoaded {len(all_docs)} pages total. Splitting into chunks…")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} chunks.")

    print(f"\nEmbedding chunks with `all-MiniLM-L6-v2` (this may take a few minutes)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    VECTORSTORE_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    print(f"\nDone! Vector store saved to `{VECTORSTORE_DIR}/`.")
    print("Commit the `vectorstore/` folder to GitHub, then deploy on Streamlit Community Cloud.")



if __name__ == "__main__":
    ingest()
