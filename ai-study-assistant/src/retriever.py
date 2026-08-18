"""
retriever.py
------------
STEP 7 from the doc: Question -> Retriever -> Relevant chunks.

A "retriever" in LangChain is a thin, standardized wrapper around a
vector store's similarity search. Wrapping it matters because:
  1. It gives every vector store the SAME interface (`.invoke(query)`),
     so if you later swap FAISS for Chroma or Pinecone, nothing else
     in your app has to change.
  2. It plugs directly into LCEL chains with `|`, same as an LLM does.

We also add LECTURE FILTERING here: since each chunk's metadata
records which lecture it came from (set in app.py when we upload),
a student can restrict search to just one lecture — e.g. "only search
Lecture 5" — instead of always searching everything.
"""

from langchain_community.vectorstores import FAISS


def get_retriever(vectorstore: FAISS, k: int = 4, lecture_filter: str | None = None):
    """
    k: how many chunks to retrieve per question. Too few and the answer
       might miss context; too many and irrelevant chunks dilute the
       prompt. 4 is a reasonable starting point for short factual
       questions.

    lecture_filter: if given, only search chunks whose metadata
       "lecture" field equals this value (see app.py, where we tag
       each chunk on upload).
    """
    search_kwargs = {"k": k}
    if lecture_filter:
        search_kwargs["filter"] = {"lecture": lecture_filter}

    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def list_available_lectures(vectorstore: FAISS) -> list[str]:
    """
    Scans the vector store's stored chunk metadata to find every unique
    lecture name that's been indexed so far. Used to populate the
    "filter by lecture" dropdown in the UI.
    """
    lectures = set()
    for doc in vectorstore.docstore._dict.values():
        lecture = doc.metadata.get("lecture")
        if lecture:
            lectures.add(lecture)
    return sorted(lectures)
