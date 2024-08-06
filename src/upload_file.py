from langchain_community.document_loaders import PyMuPDFLoader
import streamlit as st

st.title("Knowledge Bot")

#Uploads the file from the pdf
def get_doc():
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file is not None:

        # Save the uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load the document using PyMuPDFLoader
        loader = PyMuPDFLoader("temp.pdf")
        data = loader.load()
        st.write("File successfully loaded!")
        return data
    else:
        st.write("Please upload a PDF file.")