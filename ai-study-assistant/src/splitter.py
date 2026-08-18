"""
splitter.py
-----------
STEP 2 of the pipeline: break a big lecture into smaller chunks.

Why do we need this at all?
Every LLM has a context window limit, and even within that limit,
stuffing an entire 40-page lecture into one prompt makes the model's
attention "spread thin" — quality drops. Splitting into chunks means:
  1. We can process long lectures without hitting token limits.
  2. Later, when you add RAG (Step 6-7 in the project doc), you NEED
     chunks anyway, because you embed and search over chunks, not
     whole documents.

For this beginner (non-RAG) version, we mostly use the splitter to keep
each LLM call's input a reasonable size.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_documents(
    documents: list[Document],
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    RecursiveCharacterTextSplitter is LangChain's "smart" default splitter.
    It tries to split on paragraph breaks first, then sentences, then
    words — only falling back to a hard character cut if it must. This
    keeps chunks semantically coherent instead of cutting mid-sentence.

    chunk_size: max characters per chunk (not tokens — a rough proxy)
    chunk_overlap: characters repeated between consecutive chunks, so a
                   concept split across a chunk boundary isn't lost
                   entirely from either chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)
