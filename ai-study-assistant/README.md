# AI Study Assistant

A LangChain + Streamlit app with two modes:

1. **Single Lecture Study** (Steps 1–4): upload one PDF, get a summary,
   MCQs, viva questions, or take a scored interactive quiz.
2. **Search Across Lectures — RAG** (Steps 6–8): upload any number of
   lecture PDFs over time; each is embedded into a shared vector
   store, and you can ask natural-language questions answered from
   just the relevant chunks — optionally filtered to one lecture —
   with source citations (lecture name + page number).

## Setup

```bash
cd ai-study-assistant
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a free Gemini API key from https://aistudio.google.com/app/apikey,
then create a `.env` file (copy `.env.example`) and paste your key in:

```
GOOGLE_API_KEY=your_key_here
```

## Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## How the code maps to the project doc

| Doc concept          | File                          |
|-----------------------|-------------------------------|
| Document Loader       | `src/loaders.py`               |
| Text Splitter         | `src/splitter.py`               |
| Prompt Templates       | `src/prompts.py`                |
| Chains                | `src/summarizer.py`, `src/quiz.py`, `src/rag.py` |
| Structured Output      | `src/quiz.py` (`MCQItem`, `MCQSet` Pydantic models) |
| Tool (Topic Performance) | `src/quiz.py` (`calculate_topic_performance`) |
| Embeddings + Vector Store | `src/embeddings.py` (FAISS, saved to `vectorstore/`) |
| Retriever              | `src/retriever.py`              |
| RAG chain              | `src/rag.py`                    |
| Streamlit UI           | `app.py`                        |

## How RAG mode works

1. Every uploaded PDF is loaded, chunked, and each chunk is tagged
   with `metadata["lecture"] = filename` (in `app.py`).
2. Chunks are embedded (`src/embeddings.py`) and added to a single
   shared FAISS index, persisted to `vectorstore/faiss_index/` so it
   survives app restarts — upload a lecture once, it stays searchable
   forever after.
3. When you ask a question, the retriever (`src/retriever.py`) embeds
   the question and finds the 4 most similar chunks — across all
   lectures, or just one if you pick a filter.
4. Those chunks (labeled with lecture + page) are handed to the LLM
   (`src/rag.py`), which is instructed to answer using ONLY that
   context and to say so honestly if the excerpts aren't enough.
5. The UI shows the answer plus an expandable "Sources" section so you
   can verify exactly which lecture/page it came from.

## What's next

1. **Topic filtering + difficulty levels** for MCQ generation.
2. **Delete/re-index a lecture** — currently the vector store only
   grows; removing a specific lecture's chunks would need FAISS's
   `delete()` by id, tracked per-lecture.
3. **Combine modes** — e.g. generate a quiz using RAG-retrieved
   chunks about one specific topic across all lectures, instead of
   one lecture's full text.
