"""
loaders.py
----------
STEP 1 of the pipeline: get text OUT of the uploaded PDF and into
LangChain's "Document" format.

Why a separate file for this?
Because in real projects you keep each pipeline stage in its own module.
It makes the code easier to test, read, and later swap out (e.g. if you
later want to load .pptx or .docx lectures too, you just add a function
here — nothing else in the app has to change).
"""

import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(uploaded_file) -> list[Document]:
    """
    Takes a file uploaded through Streamlit (an in-memory file object)
    and returns a list of LangChain Document objects — one per page.

    A Document is just:
        Document(page_content="...text of the page...", metadata={"page": 0, ...})

    Why do we save it to a temp file first?
    PyPDFLoader (and most LangChain loaders) expect a FILE PATH on disk,
    not raw bytes. Streamlit gives us an in-memory file, so we write it
    to a temporary file just so the loader can read it.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()  # -> list[Document], one Document per PDF page

    os.remove(tmp_path)  # clean up, we don't need the temp file anymore
    return documents


def combine_documents_text(documents: list[Document]) -> str:
    """
    Many downstream steps (summarizing, generating MCQs) just want one
    big string of lecture text, not a list of per-page Documents.
    This helper flattens the list into a single string.
    """
    return "\n\n".join(doc.page_content for doc in documents)
