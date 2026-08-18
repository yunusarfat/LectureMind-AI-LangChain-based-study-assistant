"""
summarizer.py
-------------
STEP: "Chains" from the project doc.

A "chain" in modern LangChain is usually just: PROMPT -> LLM -> OUTPUT PARSER,
connected with the `|` (pipe) operator. This is called "LCEL"
(LangChain Expression Language). Reading it left to right:

    prompt | llm | parser

means: "fill the prompt template, send it to the llm, then parse the
llm's raw output." Each `|` passes its left side's output as input to
its right side.

We build two tiny chains here: one for summaries, one for "important
topics". Both reuse the same underlying LLM object (passed in), so
app.py only creates the LLM connection once.
"""

from langchain_core.output_parsers import StrOutputParser
from src.prompts import SUMMARY_PROMPT, TOPICS_PROMPT


def get_summary_chain(llm):
    """
    StrOutputParser() just takes the LLM's response object and pulls out
    the plain string content — we don't need structured data for a
    summary, plain text is fine.
    """
    return SUMMARY_PROMPT | llm | StrOutputParser()


def get_topics_chain(llm):
    return TOPICS_PROMPT | llm | StrOutputParser()


def summarize(llm, lecture_text: str) -> str:
    chain = get_summary_chain(llm)
    # .invoke() runs the chain. The dict keys must match the
    # PromptTemplate's input_variables (see prompts.py).
    return chain.invoke({"context": lecture_text})


def extract_topics(llm, lecture_text: str) -> str:
    chain = get_topics_chain(llm)
    return chain.invoke({"context": lecture_text})
