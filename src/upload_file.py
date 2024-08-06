from langchain_community.document_loaders import PyMuPDFLoader
import streamlit as st

st.title("Knowledge Bot")

def get_doc():

    # Check if the file is already uploaded
    if "uploaded_file" not in st.session_state:

        # Function to handle file upload
        uploaded_file = st.file_uploader("Choose a file", key="file_uploader")
        if uploaded_file is not None:
            st.session_state["uploaded_file"] = uploaded_file
        else:
            st.stop() #Waits for a file to be uploaded
    else:

        uploaded_file = st.session_state["uploaded_file"]
        st.write("File uploaded successfully!")
        
        # Process the uploaded file
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load the document using PyMuPDFLoader
        loader = PyMuPDFLoader("temp.pdf")
        data = loader.load()

        if data is None:
            st.stop() #Waits for data to be processed
        return data
