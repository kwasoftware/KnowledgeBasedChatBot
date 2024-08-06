from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
import set_env, set_retriever, upload_file

#Grabs model and retriever from imported files  
model = set_env.init_model()
data = upload_file.send_data()
retriever = set_retriever.init_retriever(data)

#Creates a retriever that is aware of chat history and previous context
def get_history_aware_retriever():

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
    return history_aware_retriever


#Creates a simple question and answer chain based off of the prompt
def get_question_answer_chain():
    ### Answer question ###
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
    return question_answer_chain