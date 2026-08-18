"""
prompts.py
----------
STEP 3: Prompt Templates.

Why not just f-string the prompt directly wherever we need it?
A PromptTemplate:
  1. Validates that you passed in the variables it expects
     (catches bugs early — e.g. forgetting to pass `context`).
  2. Keeps prompt text separate from application logic, so you (or a
     teammate) can tune the wording here without touching app.py.
  3. Plugs directly into LangChain's chain syntax (the `|` operator
     you'll see in summarizer.py and quiz.py).

Every template here has `{placeholders}` that get filled in at runtime.
"""

from langchain_core.prompts import PromptTemplate

# ---------------------------------------------------------------------
# 1. SUMMARY PROMPT
# ---------------------------------------------------------------------
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="""You are a university tutor helping a student study.

Using the following lecture material, write a clear, well-structured
summary. Include:
- A one-paragraph overview of what the lecture is about
- A bulleted list of the main topics/components covered
- A short explanation of each main topic

Lecture material:
{context}

Summary:""",
)

# ---------------------------------------------------------------------
# 2. IMPORTANT TOPICS PROMPT
# ---------------------------------------------------------------------
TOPICS_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="""You are a university tutor. Read the lecture material below
and extract the most important topics a student should know for an exam.

Return ONLY a bullet list of topic names (no explanations), ordered from
most fundamental to most advanced.

Lecture material:
{context}

Important topics:""",
)

# ---------------------------------------------------------------------
# 3. MCQ GENERATION PROMPT
# ---------------------------------------------------------------------
# Note: this asks for STRICT JSON. See quiz.py for how we parse it into
# a Pydantic model — that's the "Structured Output" concept from the doc.
MCQ_PROMPT = PromptTemplate(
    input_variables=["context", "num_questions"],
    template="""You are a university tutor creating exam-style multiple
choice questions (MCQs) for a Computer Science student, based ONLY on the
lecture material below.

Generate exactly {num_questions} MCQs.

Return ONLY valid JSON (no markdown fences, no extra commentary) matching
this exact structure:

{{
  "questions": [
    {{
      "topic": "short topic name this question tests, e.g. 'Pooling'",
      "question": "the question text",
      "options": ["option A", "option B", "option C", "option D"],
      "answer": "the exact text of the correct option",
      "explanation": "1-2 sentence explanation of why this is correct"
    }}
  ]
}}

Lecture material:
{context}

JSON:""",
)

# ---------------------------------------------------------------------
# 4. VIVA / ORAL QUESTIONS PROMPT
# ---------------------------------------------------------------------
VIVA_PROMPT = PromptTemplate(
    input_variables=["context", "num_questions"],
    template="""You are an examiner preparing oral ("viva") exam questions
based on the lecture material below. Generate {num_questions} short-answer
conceptual questions that test real understanding, not memorization.

Return them as a numbered list, one question per line.

Lecture material:
{context}

Viva questions:""",
)

# ---------------------------------------------------------------------
# 5. FEEDBACK PROMPT (used after the quiz is scored — see quiz.py)
# ---------------------------------------------------------------------
FEEDBACK_PROMPT = PromptTemplate(
    input_variables=["weak_topics", "strong_topics", "score", "total"],
    template="""You are a supportive tutor. A student just finished a quiz.

Score: {score}/{total}
Topics they answered correctly: {strong_topics}
Topics they answered incorrectly: {weak_topics}

Write a short (3-4 sentence), encouraging study report. Mention what
they're strong in, what to review, and one concrete next step (e.g. which
topic to re-read).""",
)

# ---------------------------------------------------------------------
# 6. RAG (multi-lecture question answering) PROMPT
# ---------------------------------------------------------------------
# Unlike SUMMARY_PROMPT/TOPICS_PROMPT, `context` here is NOT the whole
# lecture — it's just the handful of chunks the retriever picked as
# most relevant to the student's question (see rag.py).
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a tutor answering a student's question using ONLY
the lecture excerpts provided below. Each excerpt is labeled with which
lecture and page it came from.

If the excerpts don't contain enough information to answer confidently,
say so honestly instead of guessing.

Lecture excerpts:
{context}

Student's question: {question}

Answer (mention which lecture(s) you drew from):""",
)
