import streamlit as st
import os
import tempfile

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils_1.doc_loader import loadpdf, chunk_text
from utils_1.Summarizer import summarize



# --- Page Config ---
st.set_page_config(page_title="AI Research Assistant", layout="wide")

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stTextInput > label {
            font-weight: 500 !important;
        }
    </style>
""", unsafe_allow_html=True)   

cols = st.columns([4, 1])

with cols[0]:
    st.title("AI Research Assistant")
    st.caption("Leverage AI to explore, query, and test your understanding of any academic document.")

with cols[1]:
    st.markdown("###")
   
st.divider()

# --- Upload Section ---
with st.container():
    st.subheader("1. Upload Document")
    uploaded_file = st.file_uploader("Upload a .pdf or .txt file", type=["pdf", "txt"])

    if uploaded_file:
        file_name, file_ext = os.path.splitext(uploaded_file.name)
        st.markdown(f"**File:** `{file_name}{file_ext}`")

        if file_ext not in [".pdf", ".txt"]:
            st.error("Unsupported file type. Only .pdf and .txt are allowed.")
            st.stop()

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        raw_text = loadpdf(tmp_path)
        st.success("Document uploaded and parsed.")

        # --- Chunking adn Embedding ---
        chunks = chunk_text(raw_text)
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_texts(chunks, embedding=embedding)
        db.save_local("vector_index")
        st.success("Document indexed for semantic search.")

        st.divider()

        # --- You ask Questions ---
        with st.container():
            st.subheader("2. Ask Questions")
            st.caption("Ask specific or conceptual questions based on the uploaded document.")
            user_question = st.text_input("Your question")

            if user_question:
                from utils_1.qa_handler import ask_question
                with st.spinner("Retrieving answer..."):
                    answer = ask_question(db, user_question)
                st.markdown("**Answer:**")
                st.write(answer)

        st.divider()

        # --- AI will ask questions ---
        with st.container():
            st.subheader("3. Challenge Yourself")
            st.caption("AI generated reasoning based questions based on the uploaded document.")

            

            cols = st.columns([3, 1])
            with cols[0]:
                num_questions = st.number_input("Number of questions", min_value=1, max_value=10, value=3)

            with cols[1]:
                st.markdown("###")
                generate_clicked = st.button("Generate", type="primary")




            if generate_clicked:
                from utils_1.challenge_me import generate_questions
                with st.spinner("Generating questions..."):
                    logic_questions = generate_questions(raw_text[:10000], num_questions)
                st.session_state.logic_questions = logic_questions.split("\n")

            if "logic_questions" in st.session_state:
                st.markdown("#### Questions")
                user_answers = []

                for i, q in enumerate(st.session_state.logic_questions):
                    if q.strip():
                        st.markdown(f"**Q{i+1}:** {q.strip()}")
                        ans = st.text_input(f"Answer {i+1}", key=f"answer_{i}")
                        user_answers.append((q, ans))

                if st.button("Submit Answers"):
                    from utils_1.challenge_me import evaluate_answer
                    for i, (question, user_answer) in enumerate(user_answers):
                        with st.spinner(f"Evaluating answer {i+1}..."):
                            feedback = evaluate_answer(db, question, user_answer)
                        st.markdown(f"**Feedback for Q{i+1}:**")
                        st.write(feedback)

        st.divider()

        # --- Summary ---
        with st.container():
            st.subheader("4. Document Summary")
            st.caption("AI-generated summary of the uploaded document.")
            with st.spinner("Summarizing..."):
                summary = summarize(raw_text)
            st.success("Summary generated.")
            st.markdown(summary)
