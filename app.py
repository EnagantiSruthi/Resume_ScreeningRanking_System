import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    return cosine_similarities

# Streamlit app
st.title("AI Resume Screening & Ranking System")

# Job description input
st.header("Job Description")
job_description = st.text_area("Enter the job description:")

# Resume upload
st.header("Upload Resumes (PDF format)")
uploaded_files = st.file_uploader("Upload one or multiple resumes", type=["pdf"], accept_multiple_files=True)

if st.button("Rank Resumes"):
    if not job_description:
        st.warning("Please enter the job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        resumes_text = [extract_text_from_pdf(file) for file in uploaded_files]
        scores = rank_resumes(job_description, resumes_text)

        ranking_df = pd.DataFrame({
            "Resume Name": [file.name for file in uploaded_files],
            "Matching Score": scores
        }).sort_values(by="Matching Score", ascending=False)

        st.subheader("Resume Ranking Results")
        st.dataframe(ranking_df)

        # Visualization
        st.subheader("Resume Matching Scores")
        fig, ax = plt.subplots()
        ax.barh(ranking_df["Resume Name"], ranking_df["Matching Score"], color='skyblue')
        ax.set_xlabel("Matching Score")
        ax.set_ylabel("Resume Name")
        ax.set_title("Resume Screening Ranking")
        plt.gca().invert_yaxis()
        st.pyplot(fig)
