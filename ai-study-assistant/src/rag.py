"""
rag.py
------
STEP 8 from the doc, tied together: this is what lets a student ask

    "What did the teacher say about overfitting in Lecture 5?"

and get an answer pulled from just the relevant part of just that
lecture, instead of you manually re-reading everything.

THE FULL RAG FLOW (Retrieval-Augmented Generation):
  question
     -> retriever finds the most similar chunks (retriever.py)
     -> those chunks get formatted into the prompt as "context"
     -> LLM answers using ONLY that context (RAG_PROMPT in prompts.py)

This is different from summarizer.py's chains in one important way:
the "context" isn't fixed lecture text passed in by us — it's produced
by the retriever, freshly, for every single question.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from src.prompts import RAG_PROMPT


def format_docs(docs: list[Document]) -> str:
    """
    Turns the retrieved chunks into a labeled string the LLM can cite
    from, e.g.:

        [Lecture: cnn_lecture.pdf, page 3]
        Convolution applies a filter across the image...

        [Lecture: overfitting_lecture.pdf, page 7]
        Overfitting occurs when a model...

    Labeling with source metadata is what lets the LLM (and the UI,
    separately) tell the student WHERE an answer came from.
    """
    formatted = []
    for doc in docs:
        lecture = doc.metadata.get("lecture", "Unknown lecture")
        page = doc.metadata.get("page", "?")
        formatted.append(f"[Lecture: {lecture}, page {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def answer_question(llm, retriever, question: str) -> dict:
    """
    Runs the full retrieve -> generate flow for one question.

    Returns:
        {
            "answer": str,              # the LLM's answer
            "sources": list[Document],  # the chunks actually used,
                                         # so the UI can show "Sources"
        }

    Note: we do retrieval and generation as two explicit steps (rather
    than one long LCEL pipe) specifically so we can hand `sources` back
    to the UI. If we only needed the answer text, this could collapse
    into:
        chain = {"context": retriever | format_docs,
                 "question": RunnablePassthrough()} | RAG_PROMPT | llm | StrOutputParser()
        chain.invoke(question)
    but then we'd lose easy access to which chunks were used.
    """
    docs = retriever.invoke(question)
    context = format_docs(docs)

    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return {"answer": answer, "sources": docs}
