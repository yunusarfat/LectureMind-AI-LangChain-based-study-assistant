# 🧠 LectureMind AI

An AI-powered university lecture assistant built with **LangChain, Gemini, FAISS, and Streamlit**.

LectureMind AI allows students to upload lecture PDFs, automatically process the content, generate summaries and exam questions, and ask questions using Retrieval-Augmented Generation (RAG).

---
🔗 Live Demo: [https://lecturemind-ai.streamlit.app/]

## ✨ Features

### 📄 PDF Lecture Processing
- Upload university lecture PDFs
- Extract text from PDFs
- Preserve page-level information
- Automatically split large lectures into smaller chunks

### 📝 AI Summarization
Generate:
- Lecture overview
- Important topics
- Explanations of key concepts

### ❓ MCQ Generator
Automatically generate exam-style multiple-choice questions from lecture material.

Each MCQ includes:
- Topic
- Question
- 4 answer options
- Correct answer
- Explanation

### 🎤 Viva Question Generator
Generate conceptual viva/oral examination questions based on the uploaded lecture.

### 📊 Quiz Evaluation
After completing a quiz, the system:
- Calculates the score
- Identifies strong topics
- Identifies weak topics
- Generates personalized AI feedback

### 🔎 RAG-based Question Answering

Ask questions about your lectures and retrieve the most relevant sections before generating an answer.

Example:

> "What is overfitting?"

The system:

```text
Question
   ↓
Query Embedding
   ↓
FAISS Similarity Search
   ↓
Top-K Relevant Chunks
   ↓
Gemini
   ↓
Answer + Sources
