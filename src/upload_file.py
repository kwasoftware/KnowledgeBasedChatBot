from langchain_community.document_loaders import PyMuPDFLoader
import streamlit as st

st.title("Knowledge Bot")

#Uploads the file from the pdf
def get_doc():
    uploaded_file = st.file_uploader("Choose a file")
    while uploaded_file is not None:
        loader = PyMuPDFLoader(uploaded_file.name)
        data = loader.load()
        return data