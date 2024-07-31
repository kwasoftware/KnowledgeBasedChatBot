import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as st

#Loads environment variables such as the API key
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")

#Sets llm model
@st.cache_resource
def init_model():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return llm