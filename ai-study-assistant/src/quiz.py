"""
quiz.py
-------
This file covers THREE LangChain/AI-engineering concepts from the doc:

1. Structured Output  -> MCQGenerator (Pydantic models + JSON parsing)
2. Chains              -> get_mcq_chain()
3. Tools / tool calling -> calculate_topic_performance()

------------------------------------------------------------------
1. WHY STRUCTURED OUTPUT?
------------------------------------------------------------------
If we just ask the LLM "generate an MCQ" and get back free text, our
Streamlit app has no reliable way to render "Option A / B / C / D" as
separate buttons, or to check the student's answer programmatically.

So instead we define the EXACT shape we want using Pydantic models,
ask the LLM (in prompts.py) to return matching JSON, and then parse
that JSON into real Python objects. If the LLM's output doesn't match
the shape, Pydantic raises a clear validation error instead of your
app crashing mysteriously later.

------------------------------------------------------------------
2. WHY A "TOOL" FOR SCORING?
------------------------------------------------------------------
Tool calling exists so the LLM doesn't have to do things it's BAD at
(like arithmetic) or things that must be 100% deterministic (like
"is this student's answer exactly equal to the correct answer?").

Here, `calculate_topic_performance` is a plain Python function — no
LLM involved at all. We compute the score and per-topic accuracy
ourselves (fast, free, always correct), and only call the LLM
afterwards to turn those numbers into encouraging, human-readable
feedback (see FEEDBACK_PROMPT in prompts.py). This is the same
pattern real tool-calling agents use: LLM decides *when* to use a
tool, tool does the deterministic work, LLM explains the result.
"""

import json
from typing import List
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from src.prompts import MCQ_PROMPT, FEEDBACK_PROMPT


# ---------------------------------------------------------------------
# 1. STRUCTURED OUTPUT SCHEMA
# ---------------------------------------------------------------------
class MCQItem(BaseModel):
    topic: str = Field(description="Short topic name this question tests")
    question: str
    options: List[str] = Field(description="Exactly 4 answer options")
    answer: str = Field(description="Exact text of the correct option")
    explanation: str


class MCQSet(BaseModel):
    questions: List[MCQItem]


# ---------------------------------------------------------------------
# 2. MCQ GENERATION CHAIN
# ---------------------------------------------------------------------
def get_mcq_chain(llm):
    # We parse to a plain string first, THEN manually json.loads it in
    # generate_mcqs() below. (Some LLM providers' native structured-output
    # modes differ slightly, so manual parsing here is the most portable
    # approach for a beginner project — you're seeing exactly what
    # happens instead of it being hidden inside a library helper.)
    return MCQ_PROMPT | llm | StrOutputParser()


def generate_mcqs(llm, lecture_text: str, num_questions: int = 5) -> MCQSet:
    chain = get_mcq_chain(llm)
    raw_output = chain.invoke(
        {"context": lecture_text, "num_questions": num_questions}
    )

    # LLMs sometimes wrap JSON in ```json ... ``` fences even when told
    # not to. Strip those defensively before parsing.
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1)

    data = json.loads(cleaned)
    return MCQSet(**data)  # Pydantic validates the shape here


# ---------------------------------------------------------------------
# 3. THE "TOOL": deterministic scoring, no LLM involved
# ---------------------------------------------------------------------
def calculate_topic_performance(
    questions: List[MCQItem], student_answers: List[str]
) -> dict:
    """
    Plain Python function = our "tool". Compares each student answer to
    the correct answer and buckets topics into strong/weak.

    Returns:
        {
            "score": int,
            "total": int,
            "strong_topics": [str, ...],
            "weak_topics": [str, ...],
        }
    """
    score = 0
    strong_topics, weak_topics = [], []

    for question, student_answer in zip(questions, student_answers):
        is_correct = student_answer.strip() == question.answer.strip()
        if is_correct:
            score += 1
            if question.topic not in strong_topics:
                strong_topics.append(question.topic)
        else:
            if question.topic not in weak_topics:
                weak_topics.append(question.topic)

    return {
        "score": score,
        "total": len(questions),
        "strong_topics": strong_topics,
        "weak_topics": weak_topics,
    }


def get_feedback_chain(llm):
    return FEEDBACK_PROMPT | llm | StrOutputParser()


def generate_feedback(llm, performance: dict) -> str:
    """
    This is the "tool result -> LLM" half of the pattern: we hand the
    LLM the deterministic numbers the tool computed, and ask it only to
    phrase them nicely. The LLM never touches the actual scoring logic.
    """
    chain = get_feedback_chain(llm)
    return chain.invoke(
        {
            "score": performance["score"],
            "total": performance["total"],
            "strong_topics": ", ".join(performance["strong_topics"]) or "None yet",
            "weak_topics": ", ".join(performance["weak_topics"]) or "None",
        }
    )
