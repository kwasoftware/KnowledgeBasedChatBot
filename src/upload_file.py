from langchain_community.document_loaders import PyMuPDFLoader
import streamlit as st

st.title("Knowledge Bot")

#Uploads the file from the pdf
def get_doc():
    uploaded_file = st.sidebar.file_uploader(label="Choose a file", 
                                     accept_multiple_files=False, 
                                     type=['pdf'])
    while uploaded_file is None:
        uploaded_file = st.sidebar.file_uploader(label="Choose a file", 
                                     accept_multiple_files=False, 
                                     type=['pdf'])
    loader = PyMuPDFLoader(uploaded_file.name)
    data = loader.load()
    return data