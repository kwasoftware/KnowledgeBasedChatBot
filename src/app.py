from langchain.chains import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st
import get_chain_elements


history_aware_retriever = get_chain_elements.get_history_aware_retriever()
question_answer_chain = get_chain_elements.get_question_answer_chain()

### Statefully manage chat history ###
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

def display_messages():

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

session_id = "default_session"

# Accept user input
if user_input := st.chat_input("Say something"):
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    #Generates the response from the chatbot
    history = get_session_history(session_id)
    response  = conversational_rag_chain.invoke(
    {"input": user_input},
    config={
        "configurable": {"session_id": session_id}
    },
    )["answer"]

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

display_messages()