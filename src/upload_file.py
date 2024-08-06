from langchain_community.document_loaders import PyMuPDFLoader
import streamlit as st

st.title("Knowledge Bot")

#Uploads the file from the pdf
def get_doc(uploaded_file):
    loader = PyMuPDFLoader(uploaded_file)
    data = loader.load()
    return data

def send_data():
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file is not None:

        # Save the uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load the document using PyMuPDFLoader
        data = get_doc("temp.pdf")
        st.write("File successfully loaded!")
        return data
    else:
        st.write("Please upload a PDF file.")