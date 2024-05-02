import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores.faiss import FAISS
from PyPDF2 import PdfReader
import time

HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN


if "chat_history" not in st.session_state :
    st.session_state.chat_history = []

st.set_page_config(page_title="LangChain App 🦜", page_icon="🦜", layout="wide")

def get_vectorstore_from_url(url) :
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HUGGINGFACEHUB_API_TOKEN, model_name="sentence-transformers/all-MiniLM-l6-v2"
    )

    vector_store = FAISS.from_documents(document_chunks, embeddings)

    return vector_store

def get_vectorstore_from_pdfs(pdf_docs) :
    text = ""
    for pdf in pdf_docs :
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages :
            text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HUGGINGFACEHUB_API_TOKEN, model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)

    return vector_store

def get_context_retriever_chain(vector_store) :
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, max_length=128, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN
    ) 

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        #("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

def get_conversatinal_rag_chain(retriever_chain) :

    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, max_length=128, temperature=0.5, token=HUGGINGFACEHUB_API_TOKEN
    ) 

    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_query) :
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversatinal_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })

    #print(response)

    #final_response = response["answer"].index(':')

    #final_response = response["answer"][final_response+2:]

    return response

st.header("Chatbot with LangChain 🦜")
st.markdown("<hr/>", unsafe_allow_html=True)

with st.sidebar:
    st.sidebar.markdown('''
        🧑🏻‍💻 `LangChain App with FAISS, HugginFace and Stream 🦜`
        ''')

    st.markdown("---------")
    st.title("Chatbot with LangChain 🦜")
    st.subheader("Load the Chatbot with PDFs or URLs")
    st.markdown("---------")
    st.header("Settings ⚙️")
    option = st.sidebar.radio("Options:", ["URL", "PDFs"], horizontal=True)

for message in st.session_state.chat_history :
    if isinstance(message, HumanMessage) :
        with st.chat_message("user") :
            st.write(message.content)
    else :
        with st.chat_message("assistant") :
            st.write(message.content)

if option == "URL" :
    website_url = st.sidebar.text_input("Website URL: ")

    if website_url is None or website_url == "":
        st.info("Please Enter a Website URL")
    else :
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

        user_query = st.chat_input("Type your message here...")

        if user_query is not None and user_query != "":
            st.session_state.chat_history.append(HumanMessage(content=user_query))

            with st.chat_message("user") :
                st.write(user_query)

            with st.chat_message("assistant") :
                ai_response = get_response(user_query)
                ai_response = ai_response["answer"]
                message_placeholder = st.empty()
                full_response = ""
                # Simulate a streaming response with a slight delay
                for chunk in ai_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)

                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
                
                # Display the full response
                message_placeholder.info(full_response)

            st.session_state.chat_history.append(AIMessage(content=ai_response))

elif option == "PDFs" :
    pdf_docs = st.sidebar.file_uploader("Upload PDF Files:", accept_multiple_files=True)
    if len(pdf_docs) == 0 :
        st.info("Please Load a PDF File")
    else :
        st.session_state.vector_store = get_vectorstore_from_pdfs(pdf_docs)

        user_query = st.chat_input("Type your message here...")
        
        if user_query is not None and user_query != "":
            st.session_state.chat_history.append(HumanMessage(content=user_query))
        
            with st.chat_message("user") :
                st.write(user_query)

            with st.chat_message("assistant") :
                ai_response = get_response(user_query)
                ai_response = ai_response["answer"]
                message_placeholder = st.empty()
                full_response = ""
                # Simulate a streaming response with a slight delay
                for chunk in ai_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)

                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
                
                # Display the full response
                message_placeholder.info(full_response)
                #st.markdown(ai_response)

            st.session_state.chat_history.append(AIMessage(content=ai_response))
