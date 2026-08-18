"""
embeddings.py
-------------
STEP 6 from the doc: PDFs -> Chunking -> Embeddings -> Vector Store.

WHAT IS AN EMBEDDING?
An embedding turns a chunk of text into a list of numbers (a vector) —
e.g. [0.021, -0.114, 0.550, ...], usually 768+ numbers long. The key
property: chunks with SIMILAR MEANING end up as vectors that are close
together in that number-space, even if they don't share any exact
words. "Pooling reduces spatial dimensions" and "downsampling shrinks
the feature map" would land near each other, because they mean similar
things.

WHAT IS A VECTOR STORE?
A vector store is a database specialized for one operation: given a
query vector, quickly find the N stored vectors closest to it
("similarity search"). We use FAISS here — a library from Meta that
does this entirely on your local disk/memory, no external server
needed, which makes it perfect for a student project.

THE FULL IDEA:
  1. Embed every chunk of every lecture ONCE, store the vectors.
  2. When the student asks a question, embed the QUESTION with the
     same embedding model.
  3. Find the stored chunks whose vectors are closest to the question's
     vector -> these are (probably) the chunks that actually answer it.
  4. Only send THOSE chunks to the LLM, not the entire lecture library.

This is why RAG scales to many lectures where the "stuff everything in
the prompt" approach from v1 does not.
"""

import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

VECTORSTORE_DIR = "vectorstore/faiss_index"


def get_embeddings():
    """
    The embedding model is SEPARATE from the chat model (llm). Gemini's
    embedding model turns text into vectors; it doesn't generate text
    itself. Same API key, different model name.
    """
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


def build_vectorstore(chunks: list[Document]) -> FAISS:
    """
    Embeds every chunk and builds a brand-new FAISS index from scratch.
    Use this the very first time, or if you want to rebuild completely.
    """
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)


def add_to_vectorstore(vectorstore: FAISS, chunks: list[Document]) -> FAISS:
    """
    Adds MORE chunks (e.g. a newly uploaded lecture) to an EXISTING
    vector store, without re-embedding everything that's already in it.
    This is what makes "multi-lecture" practical — each new upload is
    a cheap incremental add, not a full rebuild.
    """
    vectorstore.add_documents(chunks)
    return vectorstore


def save_vectorstore(vectorstore: FAISS, path: str = VECTORSTORE_DIR) -> None:
    """Persist the index to disk so lectures survive an app restart."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vectorstore.save_local(path)


def load_vectorstore(path: str = VECTORSTORE_DIR) -> FAISS | None:
    """Load a previously saved index, or None if nothing's been saved yet."""
    if not os.path.exists(path):
        return None
    embeddings = get_embeddings()
    # allow_dangerous_deserialization=True is required by FAISS's loader
    # because unpickling can run arbitrary code in general — it's safe
    # here because WE are the only ones who ever write this file.
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
