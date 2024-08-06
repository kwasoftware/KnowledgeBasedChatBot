from langchain.chains import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
import set_env, set_retriever, upload_file
import streamlit as st

#Grabs model and retriever from imported files  
model = set_env.init_model()
data = upload_file.get_doc()
retriever = set_retriever.init_retriever(data)

### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    model, retriever, contextualize_q_prompt
)

system_prompt = (
    "You are an assistant for question-answering tasks"
    "You are an expert on the Department of Energy and information given to you"
    "When answering questions, be specific and in depth"
    "State the source of your data at the end of each and every question unless it is a greeting"
    "If they ask you something that is not in your data source say you dont know"
    "Only and exclusively answer questions based on the documents provided."
    "When ask what you know don't state specifics, but be very general and broad"
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

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