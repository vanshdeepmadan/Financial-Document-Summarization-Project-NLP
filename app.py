import streamlit as st
from transformers import pipeline
import torch
from PyPDF2 import PdfReader

# Set the device to MPS (Apple GPU) if available, otherwise fallback to CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Load the pre-trained summarization model with MPS support
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if device.type == "mps" else -1)

# Function to extract text from a PDF file
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Dynamic summarization with adjusted max_length
def abstractive_summary(text):
    input_length = len(text.split())  # Calculate input length in tokens (approximation)
    max_length = min(200, max(30, int(input_length * 0.5)))  # Set max_length to 50% of input_length, with bounds
    return summarizer(text, max_length=max_length, min_length=int(max_length * 0.5), do_sample=False)[0]['summary_text']

# Summarize a large PDF
def summarize_large_pdf(uploaded_file, chunk_size=1024):
    pdf_text = extract_text_from_pdf(uploaded_file)
    chunks = [pdf_text[i:i+chunk_size] for i in range(0, len(pdf_text), chunk_size)]
    summaries = [abstractive_summary(chunk) for chunk in chunks]
    return " ".join(summaries)

# Streamlit App
st.title("Financial Document Summarization")
st.write("Upload a financial report (PDF) to generate a concise summary.")

# File uploader
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("Extracting and summarizing the document..."):
        final_summary = summarize_large_pdf(uploaded_file)
    st.success("Summarization complete!")
    st.write("### Summary:")
    st.write(final_summary)