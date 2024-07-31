from langchain_community.document_loaders import PyMuPDFLoader
import streamlit as st

st.title("Knowledge Bot")
#Uploads the file from the pdf
def get_doc():
    loader = PyMuPDFLoader("./data/fy24_acquisition_guide_fy2024_v4.pdf")
    data = loader.load()
    return data