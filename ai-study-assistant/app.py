# """
# app.py
# ------
# This is the entry point that ties every module together into the UI
# described in the doc. There are now TWO modes:

#   1. "Single Lecture Study" (Steps 1-4): upload one PDF, get a summary,
#      MCQs, viva questions, or take a scored quiz — all working directly
#      off that one lecture's full text.

#   2. "Search Across Lectures" (Steps 6-8, RAG): upload as many PDFs as
#      you want over time, each gets embedded into a shared vector store,
#      and you can ask natural-language questions that get answered using
#      only the most relevant chunks, from the right lecture(s).

# Run this with:  streamlit run app.py
# """

# import os
# import streamlit as st
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI

# from src.loaders import load_pdf, combine_documents_text
# from src.splitter import split_documents
# from src.summarizer import summarize, extract_topics
# from src.prompts import VIVA_PROMPT
# from src.quiz import generate_mcqs, calculate_topic_performance, generate_feedback
# from src.embeddings import (
#     build_vectorstore,
#     add_to_vectorstore,
#     save_vectorstore,
#     load_vectorstore,
# )
# from src.retriever import get_retriever, list_available_lectures
# from src.rag import answer_question
# from langchain_core.output_parsers import StrOutputParser

# load_dotenv()  # reads GOOGLE_API_KEY from a local .env file

# st.set_page_config(page_title="AI Study Assistant", page_icon="📚")
# st.title("📚 AI Study Assistant")
# st.caption("Upload a lecture PDF and turn it into a summary, MCQs, or a live quiz.")


# # -----------------------------------------------------------------
# # LLM connection (created once, reused by every chain)
# # -----------------------------------------------------------------
# @st.cache_resource
# def get_llm():
#     api_key = os.getenv("GOOGLE_API_KEY")
#     if not api_key:
#         st.error(
#             "No GOOGLE_API_KEY found. Create a .env file with:\n"
#             "GOOGLE_API_KEY=your_key_here"
#         )
#         st.stop()
#     return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, api_key=api_key)


# llm = get_llm()

# # -----------------------------------------------------------------
# # Streamlit "session_state" is how we remember data BETWEEN button
# # clicks/reruns. Streamlit reruns the whole script top-to-bottom on
# # every interaction, so anything you want to persist (extracted text,
# # quiz progress, score) must live in st.session_state, not a normal
# # Python variable.
# # -----------------------------------------------------------------
# if "lecture_text" not in st.session_state:
#     st.session_state.lecture_text = None
# if "mcq_set" not in st.session_state:
#     st.session_state.mcq_set = None
# if "current_q" not in st.session_state:
#     st.session_state.current_q = 0
# if "student_answers" not in st.session_state:
#     st.session_state.student_answers = []
# if "quiz_finished" not in st.session_state:
#     st.session_state.quiz_finished = False

# # Vector store lives in session_state too, but we ALSO persist it to
# # disk (save_vectorstore/load_vectorstore) so uploaded lectures aren't
# # lost when you restart the app — only re-embedding is expensive, not
# # loading an already-built index.
# if "vectorstore" not in st.session_state:
#     st.session_state.vectorstore = load_vectorstore()
# if "indexed_files" not in st.session_state:
#     st.session_state.indexed_files = set()


# mode = st.sidebar.radio(
#     "Mode",
#     ["Single Lecture Study", "Search Across Lectures (RAG)"],
#     help="Single-lecture tools work on one PDF at a time. RAG search "
#     "lets you upload many lectures and ask questions across all of them.",
# )

# if mode == "Search Across Lectures (RAG)":
#     # =================================================================
#     # MODE 2: Multi-lecture RAG search (Steps 6-8)
#     # =================================================================
#     st.header("🔎 Search Across Lectures")
#     st.caption(
#         "Upload one or more lecture PDFs. Each is chunked and embedded "
#         "into a shared vector store, so you can ask questions across "
#         "all of them — or filter to just one lecture."
#     )

#     rag_files = st.file_uploader(
#         "Upload lecture PDF(s)", type=["pdf"], accept_multiple_files=True, key="rag_upload"
#     )

#     if rag_files:
#         new_files = [f for f in rag_files if f.name not in st.session_state.indexed_files]
#         if new_files:
#             with st.spinner(f"Embedding {len(new_files)} new lecture(s)..."):
#                 for f in new_files:
#                     documents = load_pdf(f)
#                     chunks = split_documents(documents)
#                     # Tag every chunk with WHICH lecture it came from.
#                     # This metadata is what powers source citations
#                     # (rag.py) and the "filter by lecture" dropdown
#                     # (retriever.py's list_available_lectures).
#                     for chunk in chunks:
#                         chunk.metadata["lecture"] = f.name

#                     if st.session_state.vectorstore is None:
#                         st.session_state.vectorstore = build_vectorstore(chunks)
#                     else:
#                         st.session_state.vectorstore = add_to_vectorstore(
#                             st.session_state.vectorstore, chunks
#                         )
#                     st.session_state.indexed_files.add(f.name)

#                 save_vectorstore(st.session_state.vectorstore)
#             st.success(f"Indexed: {', '.join(f.name for f in new_files)}")

#     if st.session_state.vectorstore is None:
#         st.info("Upload at least one lecture PDF to start searching.")
#     else:
#         lectures = list_available_lectures(st.session_state.vectorstore)
#         st.caption(f"Indexed lectures: {', '.join(lectures)}")

#         lecture_choice = st.selectbox(
#             "Search within:", ["All lectures"] + lectures
#         )
#         lecture_filter = None if lecture_choice == "All lectures" else lecture_choice

#         question = st.text_input(
#             "Ask a question", placeholder="What did the lecture say about overfitting?"
#         )

#         if question and st.button("Ask"):
#             retriever = get_retriever(
#                 st.session_state.vectorstore, k=4, lecture_filter=lecture_filter
#             )
#             with st.spinner("Searching lectures and writing an answer..."):
#                 result = answer_question(llm, retriever, question)

#             st.markdown(f"**Answer:** {result['answer']}")

#             with st.expander(f"Sources ({len(result['sources'])} chunk(s) used)"):
#                 for doc in result["sources"]:
#                     lecture = doc.metadata.get("lecture", "Unknown")
#                     page = doc.metadata.get("page", "?")
#                     st.markdown(f"**{lecture} — page {page}**")
#                     st.text(doc.page_content[:400] + "...")

#     st.stop()  # don't fall through to single-lecture mode below


# # =====================================================================
# # MODE 1: Single-lecture study tools (Steps 1-4)
# # =====================================================================
# # -----------------------------------------------------------------
# # STEP 1: Upload + extract
# # -----------------------------------------------------------------
# uploaded_file = st.file_uploader("Upload your lecture PDF", type=["pdf"])

# if uploaded_file and st.session_state.lecture_text is None:
#     with st.spinner("Reading and chunking your lecture..."):
#         documents = load_pdf(uploaded_file)
#         chunks = split_documents(documents)
#         st.session_state.lecture_text = combine_documents_text(chunks)
#     st.success(f"Loaded {len(documents)} page(s), split into {len(chunks)} chunk(s).")

# lecture_text = st.session_state.lecture_text

# if lecture_text:
#     with st.expander("Preview extracted text"):
#         st.text(lecture_text[:2000] + ("..." if len(lecture_text) > 2000 else ""))

#     st.divider()
#     action = st.radio(
#         "What do you want to do?",
#         ["Summarize", "Important Topics", "Generate MCQs", "Viva Questions", "Start Quiz"],
#         horizontal=True,
#     )

#     # -----------------------------------------------------------
#     # Summarize
#     # -----------------------------------------------------------
#     if action == "Summarize":
#         if st.button("Generate Summary"):
#             with st.spinner("Summarizing..."):
#                 result = summarize(llm, lecture_text)
#             st.markdown(result)

#     # -----------------------------------------------------------
#     # Important Topics
#     # -----------------------------------------------------------
#     elif action == "Important Topics":
#         if st.button("Extract Topics"):
#             with st.spinner("Finding key topics..."):
#                 result = extract_topics(llm, lecture_text)
#             st.markdown(result)

#     # -----------------------------------------------------------
#     # Generate MCQs (view only, no scoring)
#     # -----------------------------------------------------------
#     elif action == "Generate MCQs":
#         num_q = st.select_slider("How many questions?", options=[5, 10, 20], value=5)
#         if st.button("Generate"):
#             with st.spinner("Writing questions..."):
#                 mcq_set = generate_mcqs(llm, lecture_text, num_q)
#             for i, q in enumerate(mcq_set.questions, start=1):
#                 st.markdown(f"**Q{i}. {q.question}**")
#                 for opt in q.options:
#                     st.write(f"- {opt}")
#                 with st.expander("Show answer"):
#                     st.write(f"**Answer:** {q.answer}")
#                     st.write(q.explanation)

#     # -----------------------------------------------------------
#     # Viva Questions
#     # -----------------------------------------------------------
#     elif action == "Viva Questions":
#         num_q = st.select_slider("How many questions?", options=[5, 10, 20], value=5, key="viva_n")
#         if st.button("Generate Viva Questions"):
#             chain = VIVA_PROMPT | llm | StrOutputParser()
#             with st.spinner("Writing viva questions..."):
#                 result = chain.invoke({"context": lecture_text, "num_questions": num_q})
#             st.markdown(result)

#     # -----------------------------------------------------------
#     # Interactive Quiz  <-- the "interesting part" from the doc
#     # -----------------------------------------------------------
#     elif action == "Start Quiz":
#         if st.session_state.mcq_set is None:
#             num_q = st.select_slider("How many questions?", options=[5, 10, 20], value=5, key="quiz_n")
#             if st.button("Begin Quiz"):
#                 with st.spinner("Preparing your quiz..."):
#                     st.session_state.mcq_set = generate_mcqs(llm, lecture_text, num_q)
#                 st.session_state.current_q = 0
#                 st.session_state.student_answers = []
#                 st.session_state.quiz_finished = False
#                 st.rerun()

#         elif not st.session_state.quiz_finished:
#             questions = st.session_state.mcq_set.questions
#             i = st.session_state.current_q
#             q = questions[i]

#             st.progress((i) / len(questions))
#             st.subheader(f"Question {i + 1} of {len(questions)}")
#             st.write(q.question)

#             choice = st.radio("Choose one:", q.options, key=f"choice_{i}")

#             if st.button("Submit Answer"):
#                 st.session_state.student_answers.append(choice)

#                 is_correct = choice.strip() == q.answer.strip()
#                 if is_correct:
#                     st.success("Correct! ✅")
#                 else:
#                     st.error(f"Incorrect. Correct answer: {q.answer}")
#                 st.info(q.explanation)

#                 if i + 1 < len(questions):
#                     st.session_state.current_q += 1
#                     st.button("Next question ->", on_click=lambda: None)
#                     # A rerun happens automatically on the next interaction;
#                     # we just nudge the user to click "Next" above.
#                 else:
#                     st.session_state.quiz_finished = True
#                 st.rerun()

#         else:
#             # -----------------------------------------------------
#             # Quiz finished: use the TOOL (deterministic scoring),
#             # then ask the LLM to phrase the feedback.
#             # -----------------------------------------------------
#             questions = st.session_state.mcq_set.questions
#             performance = calculate_topic_performance(
#                 questions, st.session_state.student_answers
#             )

#             st.header("──── Study Report ────")
#             st.metric("Score", f"{performance['score']}/{performance['total']}")

#             col1, col2 = st.columns(2)
#             with col1:
#                 st.markdown("**Strong topics**")
#                 for t in performance["strong_topics"]:
#                     st.write(f"✓ {t}")
#             with col2:
#                 st.markdown("**Needs improvement**")
#                 for t in performance["weak_topics"]:
#                     st.write(f"✗ {t}")

#             with st.spinner("Writing your personalized feedback..."):
#                 feedback = generate_feedback(llm, performance)
#             st.markdown(f"**Recommendation:** {feedback}")

#             if st.button("Restart Quiz"):
#                 st.session_state.mcq_set = None
#                 st.session_state.current_q = 0
#                 st.session_state.student_answers = []
#                 st.session_state.quiz_finished = False
#                 st.rerun()
# else:
#     st.info("Upload a lecture PDF above to get started.")




# """
# app.py
# ------
# This is the entry point that ties every module together into the UI
# described in the doc. There are now TWO modes:

#   1. "Single Lecture Study" (Steps 1-4): upload one PDF, get a summary,
#      MCQs, viva questions, or take a scored quiz — all working directly
#      off that one lecture's full text.

#   2. "Search Across Lectures" (Steps 6-8, RAG): upload as many PDFs as
#      you want over time, each gets embedded into a shared vector store,
#      and you can ask natural-language questions that get answered using
#      only the most relevant chunks, from the right lecture(s).

# Run this with:  streamlit run app.py
# """

# import os
# import streamlit as st
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI

# from src.loaders import load_pdf, combine_documents_text
# from src.splitter import split_documents
# from src.summarizer import summarize, extract_topics
# from src.prompts import VIVA_PROMPT
# from src.quiz import generate_mcqs, calculate_topic_performance, generate_feedback
# from src.embeddings import (
#     build_vectorstore,
#     add_to_vectorstore,
#     save_vectorstore,
#     load_vectorstore,
# )
# from src.retriever import get_retriever, list_available_lectures
# from src.rag import answer_question
# from langchain_core.output_parsers import StrOutputParser

# load_dotenv()  # reads GOOGLE_API_KEY from a local .env file

# # ============================================================
# # CUSTOM CSS FOR BOLD, MODERN, ATTRACTIVE DESIGN
# # ============================================================
# st.set_page_config(
#     page_title="AI Study Assistant", 
#     page_icon="📚",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# st.markdown("""
# <style>
#     /* Import Google Fonts */
#     @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
    
#     /* Global Styles */
#     .stApp {
#         background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
#         font-family: 'Inter', sans-serif;
#     }
    
#     /* Main Title */
#     .main-title {
#         font-size: 3.5rem;
#         font-weight: 900;
#         background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #4facfe 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         background-clip: text;
#         text-shadow: 0 0 40px rgba(245, 87, 108, 0.3);
#         padding: 1rem 0;
#         letter-spacing: -1px;
#         animation: glowPulse 3s ease-in-out infinite;
#     }
    
#     @keyframes glowPulse {
#         0%, 100% { text-shadow: 0 0 40px rgba(245, 87, 108, 0.3); }
#         50% { text-shadow: 0 0 60px rgba(79, 172, 254, 0.5); }
#     }
    
#     .sub-title {
#         font-size: 1.2rem;
#         font-weight: 400;
#         color: rgba(255,255,255,0.7);
#         padding-bottom: 1.5rem;
#         border-bottom: 2px solid rgba(255,255,255,0.1);
#         margin-bottom: 2rem;
#         letter-spacing: 0.5px;
#     }
    
#     /* SIDEBAR - FIXED FOR DARK THEME MATCHING */
#     [data-testid="stSidebar"] {
#         background: linear-gradient(180deg, #0a1628, #1a1a3e, #0f0c29) !important;
#         border-right: 2px solid rgba(79, 172, 254, 0.2) !important;
#         box-shadow: inset -5px 0 30px rgba(79, 172, 254, 0.05) !important;
#     }
    
#     [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
#         background: transparent !important;
#     }
    
#     /* Sidebar radio buttons */
#     [data-testid="stSidebar"] .stRadio > div {
#         background: rgba(255, 255, 255, 0.05) !important;
#         border-radius: 15px !important;
#         padding: 0.5rem !important;
#         border: 1px solid rgba(255, 255, 255, 0.05) !important;
#     }
    
#     [data-testid="stSidebar"] .stRadio label {
#         color: rgba(255, 255, 255, 0.85) !important;
#         font-weight: 600 !important;
#         padding: 0.6rem 1.2rem !important;
#         border-radius: 10px !important;
#         transition: all 0.3s ease;
#         font-size: 0.95rem !important;
#     }
    
#     [data-testid="stSidebar"] .stRadio label:hover {
#         background: rgba(255, 255, 255, 0.08) !important;
#     }
    
#     [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] {
#         accent-color: #f5576c !important;
#     }
    
#     /* Sidebar text */
#     [data-testid="stSidebar"] p, 
#     [data-testid="stSidebar"] div,
#     [data-testid="stSidebar"] span {
#         color: rgba(255, 255, 255, 0.8) !important;
#     }
    
#     /* Sidebar captions */
#     [data-testid="stSidebar"] .stCaption {
#         color: rgba(255, 255, 255, 0.4) !important;
#     }
    
#     /* Sidebar divider */
#     [data-testid="stSidebar"] hr {
#         border: none !important;
#         height: 1px !important;
#         background: linear-gradient(90deg, transparent, rgba(79, 172, 254, 0.2), transparent) !important;
#         margin: 1.5rem 0 !important;
#     }
    
#     /* Sidebar title */
#     .sidebar-title {
#         color: rgba(255, 255, 255, 0.6) !important;
#         font-weight: 700 !important;
#         font-size: 1.1rem !important;
#         text-transform: uppercase;
#         letter-spacing: 2px;
#         padding: 1rem 0 0.5rem 0;
#     }
    
#     /* Cards / Containers */
#     .fancy-card {
#         background: rgba(255,255,255,0.06);
#         backdrop-filter: blur(10px);
#         border: 1px solid rgba(255,255,255,0.1);
#         border-radius: 20px;
#         padding: 1.5rem;
#         margin: 1rem 0;
#         transition: all 0.3s ease;
#     }
    
#     .fancy-card:hover {
#         background: rgba(255,255,255,0.1);
#         border-color: rgba(255,255,255,0.2);
#         transform: translateY(-2px);
#         box-shadow: 0 10px 40px rgba(0,0,0,0.3);
#     }
    
#     /* Buttons */
#     .stButton > button {
#         background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
#         color: white !important;
#         font-weight: 700 !important;
#         font-size: 1rem !important;
#         padding: 0.6rem 2rem !important;
#         border: none !important;
#         border-radius: 50px !important;
#         transition: all 0.3s ease !important;
#         letter-spacing: 0.5px;
#         box-shadow: 0 4px 20px rgba(245, 87, 108, 0.4);
#         width: 100%;
#     }
    
#     .stButton > button:hover {
#         transform: translateY(-2px) scale(1.02);
#         box-shadow: 0 8px 30px rgba(245, 87, 108, 0.6) !important;
#     }
    
#     .stButton > button:active {
#         transform: scale(0.98);
#     }
    
#     /* Headers */
#     h1, h2, h3, h4, h5 {
#         color: #fff !important;
#         font-weight: 700 !important;
#     }
    
#     h1 {
#         font-size: 2.5rem !important;
#         font-weight: 900 !important;
#     }
    
#     h2 {
#         font-size: 2rem !important;
#         font-weight: 800 !important;
#         background: linear-gradient(135deg, #4facfe, #00f2fe);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         background-clip: text;
#     }
    
#     h3 {
#         font-size: 1.5rem !important;
#         font-weight: 700 !important;
#         color: #4facfe !important;
#     }
    
#     /* Text elements */
#     p, li, div, span {
#         color: rgba(255,255,255,0.85) !important;
#     }
    
#     .stCaption {
#         color: rgba(255,255,255,0.5) !important;
#         font-size: 0.9rem !important;
#     }
    
#     /* File Uploader */
#     /* Fix for Drag and Drop text visibility */
# .stFileUploader .st-emotion-cache-1y4p8pa {
#     color: rgba(255, 255, 255, 0.9) !important;
#     font-size: 1.2rem !important;
#     font-weight: 600 !important;
#     margin-bottom: 0.5rem !important;
#     text-shadow: 0 2px 10px rgba(0,0,0,0.3) !important;
# }

# /* The actual drag and drop text */
# .stFileUploader .st-emotion-cache-1y4p8pa p {
#     color: rgba(255, 255, 255, 0.9) !important;
# }

# /* File limit text */
# .stFileUploader .st-emotion-cache-1j7hq4o {
#     color: rgba(255, 255, 255, 0.5) !important;
#     font-size: 0.9rem !important;
# }

# /* Any text inside the uploader */
# .stFileUploader div[data-testid="stFileUploaderDropzone"] p {
#     color: rgba(255, 255, 255, 0.9) !important;
#     font-weight: 500 !important;
# }

# /* Upload icon */
# .stFileUploader div[data-testid="stFileUploaderDropzone"] svg {
#     fill: rgba(79, 172, 254, 0.8) !important;
# }
#     .stFileUploader > div {
#         background: rgba(255,255,255,0.05) !important;
#         border: 2px dashed rgba(255,255,255,0.15) !important;
#         border-radius: 20px !important;
#         padding: 2rem !important;
#         transition: all 0.3s ease;
#     }
    
#     .stFileUploader > div:hover {
#         border-color: rgba(79, 172, 254, 0.5) !important;
#         background: rgba(255,255,255,0.08) !important;
#     }
    
#     /* Radio Buttons */
#     .stRadio > div {
#         background: rgba(255,255,255,0.05) !important;
#         border-radius: 15px !important;
#         padding: 0.5rem !important;
#     }
    
#     .stRadio label {
#         color: rgba(255,255,255,0.8) !important;
#         font-weight: 600 !important;
#         padding: 0.5rem 1rem !important;
#         border-radius: 10px !important;
#         transition: all 0.3s ease;
#     }
    
#     .stRadio label:hover {
#         background: rgba(255,255,255,0.08) !important;
#     }
    
#     .stRadio [data-baseweb="radio"] {
#         accent-color: #f5576c !important;
#     }
    
#     /* Select Box */
#     .stSelectbox > div {
#         background: rgba(255,255,255,0.05) !important;
#         border-radius: 15px !important;
#         border: 1px solid rgba(255,255,255,0.1) !important;
#     }
    
#     .stSelectbox select {
#         color: #fff !important;
#         font-weight: 600 !important;
#     }
    
#     /* Expanders */
#     .streamlit-expanderHeader {
#         background: rgba(255,255,255,0.05) !important;
#         border-radius: 15px !important;
#         font-weight: 600 !important;
#         color: #4facfe !important;
#         border: 1px solid rgba(255,255,255,0.08) !important;
#     }
    
#     .streamlit-expanderContent {
#         background: rgba(255,255,255,0.03) !important;
#         border-radius: 0 0 15px 15px !important;
#         border: 1px solid rgba(255,255,255,0.05) !important;
#         border-top: none !important;
#     }
    
#     /* Metrics */
#     .stMetric {
#         background: rgba(255,255,255,0.06) !important;
#         border-radius: 20px !important;
#         padding: 1rem !important;
#         border: 1px solid rgba(255,255,255,0.08) !important;
#     }
    
#     .stMetric label {
#         color: rgba(255,255,255,0.6) !important;
#         font-weight: 600 !important;
#         text-transform: uppercase;
#         letter-spacing: 1px;
#         font-size: 0.8rem !important;
#     }
    
#     .stMetric div[data-testid="stMetricValue"] {
#         font-size: 2.5rem !important;
#         font-weight: 900 !important;
#         background: linear-gradient(135deg, #4facfe, #00f2fe);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         background-clip: text;
#     }
    
#     /* Progress Bar */
#     .stProgress > div > div {
#         background: linear-gradient(90deg, #f093fb, #f5576c, #4facfe) !important;
#         border-radius: 50px !important;
#         height: 8px !important;
#     }
    
#     /* Success/Error/Info Messages */
#     .stAlert {
#         border-radius: 15px !important;
#         border: 1px solid rgba(255,255,255,0.1) !important;
#         backdrop-filter: blur(10px);
#         font-weight: 600 !important;
#     }
    
#     .stAlertSuccess {
#         background: rgba(0, 255, 100, 0.15) !important;
#         border-color: rgba(0, 255, 100, 0.3) !important;
#         color: #00ff64 !important;
#     }
    
#     .stAlertError {
#         background: rgba(255, 50, 50, 0.15) !important;
#         border-color: rgba(255, 50, 50, 0.3) !important;
#         color: #ff6b6b !important;
#     }
    
#     .stAlertInfo {
#         background: rgba(79, 172, 254, 0.15) !important;
#         border-color: rgba(79, 172, 254, 0.3) !important;
#         color: #4facfe !important;
#     }
    
#     /* Code blocks */
#     .stCodeBlock {
#         background: rgba(0,0,0,0.3) !important;
#         border-radius: 15px !important;
#         border: 1px solid rgba(255,255,255,0.05) !important;
#     }
    
#     /* Tabs */
#     .stTabs [data-baseweb="tab-list"] {
#         gap: 1rem !important;
#         background: rgba(255,255,255,0.05) !important;
#         border-radius: 15px !important;
#         padding: 0.5rem !important;
#     }
    
#     .stTabs [data-baseweb="tab"] {
#         border-radius: 10px !important;
#         padding: 0.5rem 1.5rem !important;
#         font-weight: 600 !important;
#         color: rgba(255,255,255,0.6) !important;
#     }
    
#     .stTabs [data-baseweb="tab"][aria-selected="true"] {
#         background: rgba(255,255,255,0.1) !important;
#         color: #fff !important;
#     }
    
#     /* Divider */
#     hr {
#         border: none !important;
#         height: 2px !important;
#         background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent) !important;
#         margin: 2rem 0 !important;
#     }
    
#     /* Custom scrollbar */
#     ::-webkit-scrollbar {
#         width: 8px;
#         height: 8px;
#     }
    
#     ::-webkit-scrollbar-track {
#         background: rgba(255,255,255,0.05);
#         border-radius: 50px;
#     }
    
#     ::-webkit-scrollbar-thumb {
#         background: linear-gradient(180deg, #f093fb, #f5576c);
#         border-radius: 50px;
#     }
    
#     ::-webkit-scrollbar-thumb:hover {
#         background: linear-gradient(180deg, #f5576c, #4facfe);
#     }
    
#     /* Glow effects for containers */
#     .glow-box {
#         background: rgba(255,255,255,0.03);
#         border-radius: 20px;
#         padding: 1.5rem;
#         border: 1px solid rgba(255,255,255,0.05);
#         position: relative;
#         overflow: hidden;
#     }
    
#     .glow-box::before {
#         content: '';
#         position: absolute;
#         top: -50%;
#         left: -50%;
#         width: 200%;
#         height: 200%;
#         background: radial-gradient(circle at center, rgba(79, 172, 254, 0.05), transparent 70%);
#         animation: rotateGlow 10s linear infinite;
#     }
    
#     @keyframes rotateGlow {
#         0% { transform: rotate(0deg); }
#         100% { transform: rotate(360deg); }
#     }
# </style>
# """, unsafe_allow_html=True)

# # ============================================================
# # MAIN APP
# # ============================================================

# # Custom Title
# st.markdown('<div class="main-title">📚 AI Study Assistant</div>', unsafe_allow_html=True)
# st.markdown('<div class="sub-title">🚀 Upload your lecture PDF and transform it into summaries, quizzes, and interactive study tools</div>', unsafe_allow_html=True)

# # -----------------------------------------------------------------
# # LLM connection (created once, reused by every chain)
# # -----------------------------------------------------------------
# @st.cache_resource
# def get_llm():
#     api_key = os.getenv("GOOGLE_API_KEY")
#     if not api_key:
#         st.error(
#             "❌ **No GOOGLE_API_KEY found.**\n\n"
#             "Create a `.env` file with:\n"
#             "```\nGOOGLE_API_KEY=your_key_here\n```"
#         )
#         st.stop()
#     return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, api_key=api_key)


# llm = get_llm()

# # -----------------------------------------------------------------
# # Streamlit "session_state" is how we remember data BETWEEN button
# # clicks/reruns. Streamlit reruns the whole script top-to-bottom on
# # every interaction, so anything you want to persist (extracted text,
# # quiz progress, score) must live in st.session_state, not a normal
# # Python variable.
# # -----------------------------------------------------------------
# if "lecture_text" not in st.session_state:
#     st.session_state.lecture_text = None
# if "mcq_set" not in st.session_state:
#     st.session_state.mcq_set = None
# if "current_q" not in st.session_state:
#     st.session_state.current_q = 0
# if "student_answers" not in st.session_state:
#     st.session_state.student_answers = []
# if "quiz_finished" not in st.session_state:
#     st.session_state.quiz_finished = False

# # Vector store lives in session_state too, but we ALSO persist it to
# # disk (save_vectorstore/load_vectorstore) so uploaded lectures aren't
# # lost when you restart the app — only re-embedding is expensive, not
# # loading an already-built index.
# if "vectorstore" not in st.session_state:
#     st.session_state.vectorstore = load_vectorstore()
# if "indexed_files" not in st.session_state:
#     st.session_state.indexed_files = set()

# # ============================================================
# # SIDEBAR - Dark Theme Matching
# # ============================================================
# st.sidebar.markdown('<div class="sidebar-title">⚡ Mode Selection</div>', unsafe_allow_html=True)

# mode = st.sidebar.radio(
#     "",
#     ["📖 Single Lecture Study", "🔍 Search Across Lectures (RAG)"],
#     help="Single-lecture tools work on one PDF at a time. RAG search "
#     "lets you upload many lectures and ask questions across all of them.",
#     index=0,
# )

# st.sidebar.markdown("---")
# st.sidebar.markdown("""
# <div style="padding: 1rem 0.5rem;">
#     <p style="font-size: 0.8rem; opacity: 0.4; text-align: center; color: rgba(255,255,255,0.4);">
#         ⚡ Powered by Gemini 2.5 Flash<br>
#         Built with ❤️ using Streamlit
#     </p>
# </div>
# """, unsafe_allow_html=True)

# if mode == "🔍 Search Across Lectures (RAG)":
#     # =================================================================
#     # MODE 2: Multi-lecture RAG search (Steps 6-8)
#     # =================================================================
#     st.markdown('<h2>🔎 Search Across Lectures</h2>', unsafe_allow_html=True)
#     st.markdown(
#         '<p style="color: rgba(255,255,255,0.6); margin-bottom: 2rem;">'
#         'Upload one or more lecture PDFs. Each is chunked and embedded '
#         'into a shared vector store, so you can ask questions across '
#         'all of them — or filter to just one lecture.</p>',
#         unsafe_allow_html=True
#     )

#     with st.container():
#         st.markdown('<div class="fancy-card">', unsafe_allow_html=True)
#         rag_files = st.file_uploader(
#             "📤 Upload lecture PDF(s)", type=["pdf"], accept_multiple_files=True, key="rag_upload"
#         )
#         st.markdown('</div>', unsafe_allow_html=True)

#     if rag_files:
#         new_files = [f for f in rag_files if f.name not in st.session_state.indexed_files]
#         if new_files:
#             with st.spinner(f"🧠 Embedding {len(new_files)} new lecture(s)..."):
#                 for f in new_files:
#                     documents = load_pdf(f)
#                     chunks = split_documents(documents)
#                     # Tag every chunk with WHICH lecture it came from.
#                     # This metadata is what powers source citations
#                     # (rag.py) and the "filter by lecture" dropdown
#                     # (retriever.py's list_available_lectures).
#                     for chunk in chunks:
#                         chunk.metadata["lecture"] = f.name

#                     if st.session_state.vectorstore is None:
#                         st.session_state.vectorstore = build_vectorstore(chunks)
#                     else:
#                         st.session_state.vectorstore = add_to_vectorstore(
#                             st.session_state.vectorstore, chunks
#                         )
#                     st.session_state.indexed_files.add(f.name)

#                 save_vectorstore(st.session_state.vectorstore)
#             st.success(f"✅ Indexed: {', '.join(f.name for f in new_files)}")

#     if st.session_state.vectorstore is None:
#         st.info("📌 Upload at least one lecture PDF to start searching.")
#     else:
#         lectures = list_available_lectures(st.session_state.vectorstore)
#         st.caption(f"📚 Indexed lectures: {', '.join(lectures)}")

#         with st.container():
#             st.markdown('<div class="fancy-card">', unsafe_allow_html=True)
#             lecture_choice = st.selectbox(
#                 "🔍 Search within:", ["📚 All lectures"] + lectures
#             )
#             lecture_filter = None if lecture_choice == "📚 All lectures" else lecture_choice

#             question = st.text_input(
#                 "💭 Ask a question", 
#                 placeholder="e.g., What did the lecture say about overfitting?"
#             )

#             if question and st.button("🔍 Ask", use_container_width=True):
#                 retriever = get_retriever(
#                     st.session_state.vectorstore, k=4, lecture_filter=lecture_filter
#                 )
#                 with st.spinner("🔎 Searching lectures and writing an answer..."):
#                     result = answer_question(llm, retriever, question)

#                 st.markdown("---")
#                 st.markdown(f"**💡 Answer:** {result['answer']}")

#                 with st.expander(f"📄 Sources ({len(result['sources'])} chunk(s) used)"):
#                     for doc in result["sources"]:
#                         lecture = doc.metadata.get("lecture", "Unknown")
#                         page = doc.metadata.get("page", "?")
#                         st.markdown(f"**📖 {lecture} — Page {page}**")
#                         st.text(doc.page_content[:400] + "...")
#             st.markdown('</div>', unsafe_allow_html=True)

#     st.stop()  # don't fall through to single-lecture mode below


# # =====================================================================
# # MODE 1: Single-lecture study tools (Steps 1-4)
# # =====================================================================
# # -----------------------------------------------------------------
# # STEP 1: Upload + extract
# # -----------------------------------------------------------------
# st.markdown('<h2>📖 Single Lecture Study</h2>', unsafe_allow_html=True)
# st.markdown(
#     '<p style="color: rgba(255,255,255,0.6); margin-bottom: 2rem;">'
#     'Upload a lecture PDF and turn it into a summary, MCQs, or a live quiz.'
#     '</p>',
#     unsafe_allow_html=True
# )

# with st.container():
#     st.markdown('<div class="fancy-card">', unsafe_allow_html=True)
#     uploaded_file = st.file_uploader("📤 Upload your lecture PDF", type=["pdf"])
#     st.markdown('</div>', unsafe_allow_html=True)

# if uploaded_file and st.session_state.lecture_text is None:
#     with st.spinner("📖 Reading and chunking your lecture..."):
#         documents = load_pdf(uploaded_file)
#         chunks = split_documents(documents)
#         st.session_state.lecture_text = combine_documents_text(chunks)
#     st.success(f"✅ Loaded {len(documents)} page(s), split into {len(chunks)} chunk(s).")

# lecture_text = st.session_state.lecture_text

# if lecture_text:
#     with st.expander("📝 Preview extracted text"):
#         st.text(lecture_text[:2000] + ("..." if len(lecture_text) > 2000 else ""))

#     st.divider()
#     action = st.radio(
#         "🎯 What do you want to do?",
#         ["📄 Summarize", "🏷️ Important Topics", "📝 Generate MCQs", "🎤 Viva Questions", "🎯 Start Quiz"],
#         horizontal=True,
#     )

#     # -----------------------------------------------------------
#     # Summarize
#     # -----------------------------------------------------------
#     if action == "📄 Summarize":
#         if st.button("✨ Generate Summary", use_container_width=True):
#             with st.spinner("📝 Summarizing..."):
#                 result = summarize(llm, lecture_text)
#             st.markdown("---")
#             st.markdown(f"**📄 Summary:**\n\n{result}")

#     # -----------------------------------------------------------
#     # Important Topics
#     # -----------------------------------------------------------
#     elif action == "🏷️ Important Topics":
#         if st.button("🔍 Extract Topics", use_container_width=True):
#             with st.spinner("🔎 Finding key topics..."):
#                 result = extract_topics(llm, lecture_text)
#             st.markdown("---")
#             st.markdown(f"**🏷️ Key Topics:**\n\n{result}")

#     # -----------------------------------------------------------
#     # Generate MCQs (view only, no scoring)
#     # -----------------------------------------------------------
#     elif action == "📝 Generate MCQs":
#         num_q = st.select_slider(
#             "📊 How many questions?", 
#             options=[5, 10, 20], 
#             value=5,
#             format_func=lambda x: f"{x} questions"
#         )
#         if st.button("📝 Generate MCQs", use_container_width=True):
#             with st.spinner("✍️ Writing questions..."):
#                 mcq_set = generate_mcqs(llm, lecture_text, num_q)
#             st.markdown("---")
#             for i, q in enumerate(mcq_set.questions, start=1):
#                 with st.container():
#                     st.markdown(f"**Q{i}. {q.question}**")
#                     for opt in q.options:
#                         st.write(f"  • {opt}")
#                     with st.expander("🔍 Show answer"):
#                         st.write(f"**✅ Answer:** {q.answer}")
#                         st.write(f"**💡 Explanation:** {q.explanation}")

#     # -----------------------------------------------------------
#     # Viva Questions
#     # -----------------------------------------------------------
#     elif action == "🎤 Viva Questions":
#         num_q = st.select_slider(
#             "📊 How many questions?", 
#             options=[5, 10, 20], 
#             value=5, 
#             key="viva_n",
#             format_func=lambda x: f"{x} questions"
#         )
#         if st.button("🎤 Generate Viva Questions", use_container_width=True):
#             chain = VIVA_PROMPT | llm | StrOutputParser()
#             with st.spinner("✍️ Writing viva questions..."):
#                 result = chain.invoke({"context": lecture_text, "num_questions": num_q})
#             st.markdown("---")
#             st.markdown(f"**🎤 Viva Questions:**\n\n{result}")

#     # -----------------------------------------------------------
#     # Interactive Quiz  <-- the "interesting part" from the doc
#     # -----------------------------------------------------------
#     elif action == "🎯 Start Quiz":
#         if st.session_state.mcq_set is None:
#             num_q = st.select_slider(
#                 "📊 How many questions?", 
#                 options=[5, 10, 20], 
#                 value=5, 
#                 key="quiz_n",
#                 format_func=lambda x: f"{x} questions"
#             )
#             if st.button("🎯 Begin Quiz", use_container_width=True):
#                 with st.spinner("📝 Preparing your quiz..."):
#                     st.session_state.mcq_set = generate_mcqs(llm, lecture_text, num_q)
#                 st.session_state.current_q = 0
#                 st.session_state.student_answers = []
#                 st.session_state.quiz_finished = False
#                 st.rerun()

#         elif not st.session_state.quiz_finished:
#             questions = st.session_state.mcq_set.questions
#             i = st.session_state.current_q
#             q = questions[i]

#             st.progress((i) / len(questions))
#             st.markdown(f"### Question {i + 1} of {len(questions)}")
#             st.markdown(f"**{q.question}**")

#             choice = st.radio("Choose one:", q.options, key=f"choice_{i}", index=None)

#             if st.button("✅ Submit Answer", use_container_width=True):
#                 if choice is None:
#                     st.warning("⚠️ Please select an answer before submitting.")
#                 else:
#                     st.session_state.student_answers.append(choice)

#                     is_correct = choice.strip() == q.answer.strip()
#                     if is_correct:
#                         st.success("✅ Correct! Well done! 🎉")
#                     else:
#                         st.error(f"❌ Incorrect. Correct answer: {q.answer}")
#                     st.info(f"💡 {q.explanation}")

#                     if i + 1 < len(questions):
#                         st.session_state.current_q += 1
#                         if st.button("➡️ Next Question", use_container_width=True):
#                             st.rerun()
#                     else:
#                         st.session_state.quiz_finished = True
#                         st.rerun()

#         else:
#             # -----------------------------------------------------
#             # Quiz finished: use the TOOL (deterministic scoring),
#             # then ask the LLM to phrase the feedback.
#             # -----------------------------------------------------
#             questions = st.session_state.mcq_set.questions
#             performance = calculate_topic_performance(
#                 questions, st.session_state.student_answers
#             )

#             st.markdown("---")
#             st.markdown("## 🏆 Study Report")

#             col1, col2, col3 = st.columns([1, 2, 1])
#             with col2:
#                 st.metric(
#                     "📊 Score",
#                     f"{performance['score']}/{performance['total']}",
#                     f"{(performance['score']/performance['total']*100):.0f}%"
#                 )

#             col1, col2 = st.columns(2)
#             with col1:
#                 st.markdown("### ✅ Strong topics")
#                 for t in performance["strong_topics"]:
#                     st.write(f"✓ {t}")
#             with col2:
#                 st.markdown("### 📈 Needs improvement")
#                 for t in performance["weak_topics"]:
#                     st.write(f"✗ {t}")

#             with st.spinner("✍️ Writing your personalized feedback..."):
#                 feedback = generate_feedback(llm, performance)
#             st.markdown("### 💡 Recommendation")
#             st.markdown(feedback)

#             if st.button("🔄 Restart Quiz", use_container_width=True):
#                 st.session_state.mcq_set = None
#                 st.session_state.current_q = 0
#                 st.session_state.student_answers = []
#                 st.session_state.quiz_finished = False
#                 st.rerun()
# else:
#     with st.container():
#         st.markdown('<div class="fancy-card" style="text-align: center; padding: 3rem;">', unsafe_allow_html=True)
#         st.markdown("""
#         <div style="font-size: 4rem; margin-bottom: 1rem;">📚</div>
#         <h3 style="color: rgba(255,255,255,0.6);">Ready to study?</h3>
#         <p style="color: rgba(255,255,255,0.4);">Upload a lecture PDF above to get started with summaries, quizzes, and more.</p>
#         </div>
#         """, unsafe_allow_html=True)
#         st.markdown('</div>', unsafe_allow_html=True)



# """
# app.py
# ------
# This is the entry point that ties every module together into the UI
# described in the doc. There are now TWO modes:

#   1. "Single Lecture Study" (Steps 1-4): upload one PDF, get a summary,
#      MCQs, viva questions, or take a scored quiz — all working directly
#      off that one lecture's full text.

#   2. "Search Across Lectures" (Steps 6-8, RAG): upload as many PDFs as
#      you want over time, each gets embedded into a shared vector store,
#      and you can ask natural-language questions that get answered using
#      only the most relevant chunks, from the right lecture(s).

# Run this with:  streamlit run app.py
# """

import os
import streamlit as st
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.loaders import load_pdf, combine_documents_text
from src.splitter import split_documents
from src.summarizer import summarize, extract_topics
from src.prompts import VIVA_PROMPT
from src.quiz import generate_mcqs, calculate_topic_performance, generate_feedback
from src.embeddings import (
    build_vectorstore,
    add_to_vectorstore,
    save_vectorstore,
    load_vectorstore,
)
from src.retriever import get_retriever, list_available_lectures
from src.rag import answer_question
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  # reads GOOGLE_API_KEY from a local .env file

# ============================================================
# CUSTOM CSS FOR BOLD, MODERN, ATTRACTIVE DESIGN - WHITE THEME
# ============================================================
st.set_page_config(
    page_title="LectureMind AI : A Study Assistant", 
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
    
    /* Global Styles - White Main Page */
    .stApp {
        background: #ffffff !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Title */
    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding: 1rem 0;
        letter-spacing: -1px;
        animation: glowPulse 3s ease-in-out infinite;
    }
    
    @keyframes glowPulse {
        0%, 100% { filter: drop-shadow(0 0 20px rgba(102, 126, 234, 0.2)); }
        50% { filter: drop-shadow(0 0 40px rgba(118, 75, 162, 0.3)); }
    }
    
    .sub-title {
        font-size: 1.2rem;
        font-weight: 500;
        color: #555555;
        padding-bottom: 1.5rem;
        border-bottom: 2px solid #e8ecf1;
        margin-bottom: 2rem;
        letter-spacing: 0.5px;
    }
    
    /* SIDEBAR - White Ash / Light Gray */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f2f5, #e8ecf1, #f5f6f8) !important;
        border-right: 2px solid rgba(102, 126, 234, 0.15) !important;
        box-shadow: inset -5px 0 30px rgba(0, 0, 0, 0.03) !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        background: transparent !important;
    }
    
    /* Sidebar radio buttons */
    [data-testid="stSidebar"] .stRadio > div {
        background: rgba(255, 255, 255, 0.7) !important;
        border-radius: 15px !important;
        padding: 0.5rem !important;
        border: 1px solid rgba(102, 126, 234, 0.1) !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03) !important;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        color: #333333 !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.2rem !important;
        border-radius: 10px !important;
        transition: all 0.3s ease;
        font-size: 0.95rem !important;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(102, 126, 234, 0.08) !important;
    }
    
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] {
        accent-color: #667eea !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] span {
        color: #444444 !important;
    }
    
    /* Sidebar captions */
    [data-testid="stSidebar"] .stCaption {
        color: #888888 !important;
    }
    
    /* Sidebar divider */
    [data-testid="stSidebar"] hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.2), transparent) !important;
        margin: 1.5rem 0 !important;
    }
    
    /* Sidebar title */
    .sidebar-title {
        color: #444444 !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        padding: 1rem 0 0.5rem 0;
    }
    
    /* Cards / Containers - Light Theme */
    .fancy-card {
        background: #f8f9fc;
        backdrop-filter: blur(10px);
        border: 1px solid #e8ecf1;
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
    }
    
    .fancy-card:hover {
        background: #f0f2f8;
        border-color: #d0d5e0;
        transform: translateY(-2px);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.06);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.6rem 2rem !important;
        border: none !important;
        border-radius: 50px !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:active {
        transform: scale(0.98);
    }
    
    /* Headers - Dark for white bg */
    h1, h2, h3, h4, h5 {
        color: #1a1a2e !important;
        font-weight: 700 !important;
    }
    
    h1 {
        font-size: 2.5rem !important;
        font-weight: 900 !important;
    }
    
    h2 {
        font-size: 2rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h3 {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #667eea !important;
    }
    
    /* Text elements - Dark for white bg */
    p, li, div, span {
        color: #333333 !important;
    }
    
    .stCaption {
        color: #888888 !important;
        font-size: 0.9rem !important;
    }
    
    /* File Uploader */
    .stFileUploader div[data-testid="stFileUploaderDropzone"] p {
        color: #444444 !important;
        font-weight: 600 !important;
    }
    
    .stFileUploader > div {
        background: #f8f9fc !important;
        border: 2px dashed #d0d5e0 !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #667eea !important;
        background: #f0f2f8 !important;
    }
    
    /* Radio Buttons - Light */
    .stRadio > div {
        background: #f8f9fc !important;
        border-radius: 15px !important;
        padding: 0.5rem !important;
        border: 1px solid #e8ecf1 !important;
    }
    
    .stRadio label {
        color: #333333 !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        border-radius: 10px !important;
        transition: all 0.3s ease;
    }
    
    .stRadio label:hover {
        background: rgba(102, 126, 234, 0.06) !important;
    }
    
    .stRadio [data-baseweb="radio"] {
        accent-color: #667eea !important;
    }
    
    /* Select Box - Light */
    .stSelectbox > div {
        background: #f8f9fc !important;
        border-radius: 15px !important;
        border: 1px solid #e8ecf1 !important;
    }
    
    .stSelectbox select {
        color: #1a1a2e !important;
        font-weight: 600 !important;
    }
    
    /* Expanders - Light */
    .streamlit-expanderHeader {
        background: #f8f9fc !important;
        border-radius: 15px !important;
        font-weight: 700 !important;
        color: #667eea !important;
        border: 1px solid #e8ecf1 !important;
    }
    
    .streamlit-expanderContent {
        background: #fafbfc !important;
        border-radius: 0 0 15px 15px !important;
        border: 1px solid #e8ecf1 !important;
        border-top: none !important;
    }
    
    /* Metrics - Light */
    .stMetric {
        background: #f8f9fc !important;
        border-radius: 20px !important;
        padding: 1rem !important;
        border: 1px solid #e8ecf1 !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.02);
    }
    
    .stMetric label {
        color: #888888 !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.8rem !important;
    }
    
    .stMetric div[data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb) !important;
        border-radius: 50px !important;
        height: 8px !important;
    }
    
    /* Success/Error/Info Messages - Light */
    .stAlert {
        border-radius: 15px !important;
        border: 1px solid #e8ecf1 !important;
        font-weight: 600 !important;
        backdrop-filter: blur(10px);
    }
    
    .stAlertSuccess {
        background: rgba(46, 213, 115, 0.08) !important;
        border-color: rgba(46, 213, 115, 0.2) !important;
        color: #2ed573 !important;
    }
    
    .stAlertError {
        background: rgba(255, 71, 87, 0.08) !important;
        border-color: rgba(255, 71, 87, 0.2) !important;
        color: #ff4757 !important;
    }
    
    .stAlertInfo {
        background: rgba(102, 126, 234, 0.08) !important;
        border-color: rgba(102, 126, 234, 0.2) !important;
        color: #667eea !important;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background: #f8f9fc !important;
        border-radius: 15px !important;
        border: 1px solid #e8ecf1 !important;
    }
    
    /* Tabs - Light */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem !important;
        background: #f8f9fc !important;
        border-radius: 15px !important;
        padding: 0.5rem !important;
        border: 1px solid #e8ecf1 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        color: #888888 !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #667eea !important;
        color: #ffffff !important;
    }
    
    /* Divider */
    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, #e8ecf1, transparent) !important;
        margin: 2rem 0 !important;
    }
    
    /* Custom scrollbar - Light */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f0f2f5;
        border-radius: 50px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea, #764ba2);
        border-radius: 50px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #764ba2, #f093fb);
    }
    
    /* Glow effects for containers - Light */
    .glow-box {
        background: #f8f9fc;
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid #e8ecf1;
        position: relative;
        overflow: hidden;
    }
    
    .glow-box::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle at center, rgba(102, 126, 234, 0.03), transparent 70%);
        animation: rotateGlow 10s linear infinite;
    }
    
    @keyframes rotateGlow {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# MAIN APP
# ============================================================

# Custom Title
st.markdown('<div class="main-title">LectureMind-AI: a Study Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">🚀 Upload your lecture PDF and transform it into summaries, quizzes, and interactive study tools</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------
# LLM connection (created once, reused by every chain)
# -----------------------------------------------------------------
@st.cache_resource
def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error(
            "❌ **No GOOGLE_API_KEY found.**\n\n"
            "Create a `.env` file with:\n"
            "```\nGOOGLE_API_KEY=your_key_here\n```"
        )
        st.stop()
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, api_key=api_key)


llm = get_llm()

# -----------------------------------------------------------------
# Streamlit "session_state" is how we remember data BETWEEN button
# clicks/reruns. Streamlit reruns the whole script top-to-bottom on
# every interaction, so anything you want to persist (extracted text,
# quiz progress, score) must live in st.session_state, not a normal
# Python variable.
# -----------------------------------------------------------------
if "lecture_text" not in st.session_state:
    st.session_state.lecture_text = None
if "mcq_set" not in st.session_state:
    st.session_state.mcq_set = None
if "current_q" not in st.session_state:
    st.session_state.current_q = 0
if "student_answers" not in st.session_state:
    st.session_state.student_answers = []
if "quiz_finished" not in st.session_state:
    st.session_state.quiz_finished = False

# Vector store lives in session_state too, but we ALSO persist it to
# disk (save_vectorstore/load_vectorstore) so uploaded lectures aren't
# lost when you restart the app — only re-embedding is expensive, not
# loading an already-built index.
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore()
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

# ============================================================
# SIDEBAR - White Ash Theme Matching
# ============================================================
st.sidebar.markdown('<div class="sidebar-title">⚡ Mode Selection</div>', unsafe_allow_html=True)

mode = st.sidebar.radio(
    "",
    ["Single Lecture Study", "🔍 Search Across Lectures (RAG)"],
    help="Single-lecture tools work on one PDF at a time. RAG search "
    "lets you upload many lectures and ask questions across all of them.",
    index=0,
)

st.sidebar.markdown("---")
# st.sidebar.markdown("""
# <div style="padding: 1rem 0.5rem;">
#     <p style="font-size: 0.8rem; opacity: 0.4; text-align: center; color: #888888;">
#         ⚡ Powered by Gemini 2.5 Flash<br>
#         Built with ❤️ using Streamlit
#     </p>
# </div>
# """, unsafe_allow_html=True)

if mode == "🔍 Search Across Lectures (RAG)":
    # =================================================================
    # MODE 2: Multi-lecture RAG search (Steps 6-8)
    # =================================================================
    st.markdown('<h2> Search Across Lectures</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color: #555555; margin-bottom: 1rem;">'
        'Upload one or more lecture PDFs.</p>',
        unsafe_allow_html=True
    )

    with st.container():
        st.markdown('<div class="fancy-card">', unsafe_allow_html=True)
        rag_files = st.file_uploader(
            "📤 Upload lecture PDF(s)", type=["pdf"], accept_multiple_files=True, key="rag_upload"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if rag_files:
        new_files = [f for f in rag_files if f.name not in st.session_state.indexed_files]
        if new_files:
            with st.spinner(f"🧠 Embedding {len(new_files)} new lecture(s)..."):
                for f in new_files:
                    documents = load_pdf(f)
                    chunks = split_documents(documents)
                    # Tag every chunk with WHICH lecture it came from.
                    # This metadata is what powers source citations
                    # (rag.py) and the "filter by lecture" dropdown
                    # (retriever.py's list_available_lectures).
                    for chunk in chunks:
                        chunk.metadata["lecture"] = f.name

                    if st.session_state.vectorstore is None:
                        st.session_state.vectorstore = build_vectorstore(chunks)
                    else:
                        st.session_state.vectorstore = add_to_vectorstore(
                            st.session_state.vectorstore, chunks
                        )
                    st.session_state.indexed_files.add(f.name)

                save_vectorstore(st.session_state.vectorstore)
            st.success(f"✅ Indexed: {', '.join(f.name for f in new_files)}")

    if st.session_state.vectorstore is None:
        st.info("📌 Upload at least one lecture PDF to start searching.")
    else:
        lectures = list_available_lectures(st.session_state.vectorstore)
        # st.caption(f"📚 Indexed lectures: {', '.join(lectures)}")

        with st.container():
            st.markdown('<div class="fancy-card">', unsafe_allow_html=True)
            lecture_choice = st.selectbox(
                "🔍 Search within:", ["📚 All lectures"] + lectures
            )
            lecture_filter = None if lecture_choice == "📚 All lectures" else lecture_choice

            question = st.text_input(
                "💭 Ask a question", 
                placeholder="e.g., What did the lecture say about overfitting?"
            )

            if question and st.button("🔍 Ask", use_container_width=True):
                retriever = get_retriever(
                    st.session_state.vectorstore, k=4, lecture_filter=lecture_filter
                )
                with st.spinner("🔎 Searching lectures and writing an answer..."):
                    result = answer_question(llm, retriever, question)

                st.markdown("---")
                st.markdown(f"**💡 Answer:** {result['answer']}")

                with st.expander(f"📄 Sources ({len(result['sources'])} chunk(s) used)"):
                    for doc in result["sources"]:
                        lecture = doc.metadata.get("lecture", "Unknown")
                        page = doc.metadata.get("page", "?")
                        st.markdown(f"**📖 {lecture} — Page {page}**")
                        st.text(doc.page_content[:400] + "...")
            st.markdown('</div>', unsafe_allow_html=True)

    st.stop()  # don't fall through to single-lecture mode below


# =====================================================================
# MODE 1: Single-lecture study tools (Steps 1-4)
# =====================================================================
# -----------------------------------------------------------------
# STEP 1: Upload + extract
# -----------------------------------------------------------------
st.markdown('<h2> Single Lecture Study</h2>', unsafe_allow_html=True)
st.markdown(
    '<p style="color: #555555; margin-bottom: 2rem;">'
    'Upload a lecture PDF and turn it into a summary, MCQs, or a live quiz.'
    '</p>',
    unsafe_allow_html=True
)

with st.container():
    st.markdown('<div class="fancy-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Upload your lecture PDF", type=["pdf"])
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file and st.session_state.lecture_text is None:
    with st.spinner("📖 Reading and chunking your lecture..."):
        documents = load_pdf(uploaded_file)
        chunks = split_documents(documents)
        st.session_state.lecture_text = combine_documents_text(chunks)
    st.success(f"✅ Loaded {len(documents)} page(s), split into {len(chunks)} chunk(s).")

lecture_text = st.session_state.lecture_text

if lecture_text:
    with st.expander("📝 Preview extracted text"):
        st.text(lecture_text[:2000] + ("..." if len(lecture_text) > 2000 else ""))

    st.divider()
    action = st.radio(
        "🎯 What do you want to do?",
        ["📄 Summarize", "🏷️ Important Topics", "📝 Generate MCQs", "🎤 Viva Questions", "🎯 Start Quiz"],
        horizontal=True,
    )

    # -----------------------------------------------------------
    # Summarize
    # -----------------------------------------------------------
    if action == "📄 Summarize":
        if st.button("✨ Generate Summary", use_container_width=True):
            with st.spinner("📝 Summarizing..."):
                result = summarize(llm, lecture_text)
            st.markdown("---")
            st.markdown(f"**📄 Summary:**\n\n{result}")

    # -----------------------------------------------------------
    # Important Topics
    # -----------------------------------------------------------
    elif action == "🏷️ Important Topics":
        if st.button("🔍 Extract Topics", use_container_width=True):
            with st.spinner("🔎 Finding key topics..."):
                result = extract_topics(llm, lecture_text)
            st.markdown("---")
            st.markdown(f"**🏷️ Key Topics:**\n\n{result}")

    # -----------------------------------------------------------
    # Generate MCQs (view only, no scoring)
    # -----------------------------------------------------------
    elif action == "📝 Generate MCQs":
        num_q = st.select_slider(
            "📊 How many questions?", 
            options=[5, 10, 20], 
            value=5,
            format_func=lambda x: f"{x} questions"
        )
        if st.button("📝 Generate MCQs", use_container_width=True):
            with st.spinner("✍️ Writing questions..."):
                mcq_set = generate_mcqs(llm, lecture_text, num_q)
            st.markdown("---")
            for i, q in enumerate(mcq_set.questions, start=1):
                with st.container():
                    st.markdown(f"**Q{i}. {q.question}**")
                    for opt in q.options:
                        st.write(f"  • {opt}")
                    with st.expander("🔍 Show answer"):
                        st.write(f"**✅ Answer:** {q.answer}")
                        st.write(f"**💡 Explanation:** {q.explanation}")

    # -----------------------------------------------------------
    # Viva Questions
    # -----------------------------------------------------------
    elif action == "🎤 Viva Questions":
        num_q = st.select_slider(
            "📊 How many questions?", 
            options=[5, 10, 20], 
            value=5, 
            key="viva_n",
            format_func=lambda x: f"{x} questions"
        )
        if st.button("🎤 Generate Viva Questions", use_container_width=True):
            chain = VIVA_PROMPT | llm | StrOutputParser()
            with st.spinner("✍️ Writing viva questions..."):
                result = chain.invoke({"context": lecture_text, "num_questions": num_q})
            st.markdown("---")
            st.markdown(f"**🎤 Viva Questions:**\n\n{result}")

    # -----------------------------------------------------------
    # Interactive Quiz  <-- the "interesting part" from the doc
    # -----------------------------------------------------------
    elif action == "🎯 Start Quiz":
        if st.session_state.mcq_set is None:
            num_q = st.select_slider(
                "📊 How many questions?", 
                options=[5, 10, 20], 
                value=5, 
                key="quiz_n",
                format_func=lambda x: f"{x} questions"
            )
            if st.button("🎯 Begin Quiz", use_container_width=True):
                with st.spinner("📝 Preparing your quiz..."):
                    st.session_state.mcq_set = generate_mcqs(llm, lecture_text, num_q)
                st.session_state.current_q = 0
                st.session_state.student_answers = []
                st.session_state.quiz_finished = False
                st.rerun()

        elif not st.session_state.quiz_finished:
            questions = st.session_state.mcq_set.questions
            i = st.session_state.current_q
            q = questions[i]

            st.progress((i) / len(questions))
            st.markdown(f"### Question {i + 1} of {len(questions)}")
            st.markdown(f"**{q.question}**")

            choice = st.radio("Choose one:", q.options, key=f"choice_{i}", index=None)

            if st.button("✅ Submit Answer", use_container_width=True):
                if choice is None:
                    st.warning("⚠️ Please select an answer before submitting.")
                else:
                    st.session_state.student_answers.append(choice)

                    is_correct = choice.strip() == q.answer.strip()
                    if is_correct:
                        st.success("✅ Correct! Well done! 🎉")
                    else:
                        st.error(f"❌ Incorrect. Correct answer: {q.answer}")
                    st.info(f"💡 {q.explanation}")

                    if i + 1 < len(questions):
                        st.session_state.current_q += 1
                        if st.button("➡️ Next Question", use_container_width=True):
                            st.rerun()
                    else:
                        st.session_state.quiz_finished = True
                        st.rerun()

        else:
            # -----------------------------------------------------
            # Quiz finished: use the TOOL (deterministic scoring),
            # then ask the LLM to phrase the feedback.
            # -----------------------------------------------------
            questions = st.session_state.mcq_set.questions
            performance = calculate_topic_performance(
                questions, st.session_state.student_answers
            )

            st.markdown("---")
            st.markdown("## 🏆 Study Report")

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.metric(
                    "📊 Score",
                    f"{performance['score']}/{performance['total']}",
                    f"{(performance['score']/performance['total']*100):.0f}%"
                )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### ✅ Strong topics")
                for t in performance["strong_topics"]:
                    st.write(f"✓ {t}")
            with col2:
                st.markdown("### 📈 Needs improvement")
                for t in performance["weak_topics"]:
                    st.write(f"✗ {t}")

            with st.spinner("✍️ Writing your personalized feedback..."):
                feedback = generate_feedback(llm, performance)
            st.markdown("### 💡 Recommendation")
            st.markdown(feedback)

            if st.button("🔄 Restart Quiz", use_container_width=True):
                st.session_state.mcq_set = None
                st.session_state.current_q = 0
                st.session_state.student_answers = []
                st.session_state.quiz_finished = False
                st.rerun()
else:
    with st.container():
        st.markdown('<div class="fancy-card" style="text-align: center; padding: 3rem;">', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size: 4rem; margin-bottom: 1rem;">📚</div>
        <h3 style="color: #888888;">Ready to study?</h3>
        <p style="color: #aaaaaa;">Upload a lecture PDF above to get started with summaries, quizzes, and more.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
