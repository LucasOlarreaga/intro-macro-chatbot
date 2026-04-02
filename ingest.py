"""
ingest.py — Run this ONCE locally after adding/changing documents.

Usage:
    python ingest.py --lang en     # English course  → vectorstore_en/
    python ingest.py --lang fr     # French course   → vectorstore_fr/

Place documents in:
    documents/english/   (slides, textbooks, problem sets, exams, syllabus)
    documents/french/    (same structure, French versions)

Commit both vectorstore_en/ and vectorstore_fr/ to GitHub.
"""

import argparse
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150


def ingest(lang: str):
    folder_name    = "english" if lang == "en" else "french"
    docs_dir       = Path(f"documents/{folder_name}")
    vectorstore_dir = Path(f"vectorstore_{lang}")

    if not docs_dir.exists():
        print(f"Folder `{docs_dir}` not found. Create it and add your PDFs.")
        return

    pdf_files = list(docs_dir.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in `{docs_dir}/`.")
        return

    print(f"[{lang.upper()}] Found {len(pdf_files)} PDF(s). Loading...")
    all_docs = []

    for pdf_path in sorted(pdf_files):
        relative_path = pdf_path.relative_to(docs_dir)
        print(f"  -> {relative_path}")
        loader = PyPDFLoader(str(pdf_path))
        pages  = loader.load()
        for page in pages:
            page.metadata["source"] = str(relative_path)
        all_docs.extend(pages)

    print(f"\nLoaded {len(all_docs)} pages. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} chunks.")

    print(f"\nEmbedding with `all-MiniLM-L6-v2` (this may take a few minutes)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore_dir.mkdir(exist_ok=True)
    vectorstore.save_local(str(vectorstore_dir))
    print(f"\nDone! Saved to `{vectorstore_dir}/`.")
    print("Commit this folder to GitHub, then redeploy on Streamlit Community Cloud.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang",
        choices=["en", "fr"],
        required=True,
        help="Language of the course documents to ingest (en or fr)",
    )
    args = parser.parse_args()
    ingest(args.lang)
