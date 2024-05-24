import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_google_genai import GoogleGenerativeAI 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores.faiss import FAISS
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
import time
from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.tools import Tool
from langchain import hub
from langchain.agents import create_structured_chat_agent
from langchain.agents import AgentExecutor
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.chains import LLMMathChain
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import CSVLoader
import pandas as pd
import uuid

HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["GOOGLE_CSE_ID"] = st.secrets['GOOGLE_CSE_ID']
os.environ["GOOGLE_API_KEY"] = st.secrets['GOOGLE_SEARCH_API']


if "chat_history" not in st.session_state :
    st.session_state.chat_history = [AIMessage(content="Hello! I'm a Chatbot assistant. Ask me anything about your Web Page URL or PDF Files."),]

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

def get_vectorstore_from_csv(csv_file) :
    df = pd.read_csv(csv_file)
    csv_name = './csv_files/{}.csv'.format(uuid.uuid1())
    df.to_csv(csv_name)
    loader = CSVLoader(file_path=csv_name, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)

    
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HUGGINGFACEHUB_API_TOKEN, model_name="sentence-transformers/all-MiniLM-l6-v2"
    )

    vector_store = FAISS.from_documents(text_chunks, embeddings)

    return vector_store

def get_context_retriever_chain(vector_store) :
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

def get_conversatinal_rag_chain(retriever_chain) :
    llm = ChatGroq(model="llama3-8b-8192", temperature=0.3, api_key=GROQ_API_KEY)

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

    final_response = response["answer"]
    return final_response

def get_search_response(user_query) :
    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
    google_search = GoogleSearchAPIWrapper()
    google_tool = Tool(
        name="google-search",
        description="Search Google for recent results.",
        func=google_search.run
    )

    wikipedia = WikipediaAPIWrapper()
    wikipedia_tool = Tool(name="Wikipedia",
                        func=wikipedia.run,
                        description="A useful tool for searching the Internet to find information on world events, issues, dates, years, etc. Worth using for general topics. Use precise questions.")

    # Set up the tool for responding to mathematical questions
    problem_chain = LLMMathChain.from_llm(llm=llm)
    math_tool = Tool.from_function(name="Calculator",
                    func=problem_chain.run,
                    description="Useful for when you need to answer questions about math. This tool is only for math questions and nothing else. Only input math expressions.")
    
    # Set up the tool for performing internet searches
    search = DuckDuckGoSearchRun()
    search_tool = Tool(
        name="DuckDuckGo Search",
        func=search.run,
        description="Useful for when you need to do a search on the internet to find information that another tool can't find. Be specific with your input or ask about something that is new and latest.",
    )
    chat_model = ChatGroq(temperature=0, model_name="llama3-8b-8192", groq_api_key=GROQ_API_KEY)
    prompt = hub.pull("hwchase17/structured-chat-agent")
    agent=create_structured_chat_agent(chat_model, [google_tool, math_tool, search_tool, wikipedia_tool], prompt)
    agent_executor=AgentExecutor(agent=agent, tools=[google_tool, math_tool, search_tool, wikipedia_tool], verbose=True, handle_parsing_errors=True, max_iterations=10)
    st_callback = StreamlitCallbackHandler(st.container())
    result = agent_executor.invoke({"input": user_query}, {"callbacks": [st_callback]})
    result = result["output"]
    return result

def get_chatbot_simple_response(user_query, chat_history ) :
    template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation:

    Chat history: {chat_history}

    User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatGroq(model="llama3-8b-8192", temperature=0.3, api_key=GROQ_API_KEY)

    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({
        "chat_history": chat_history,
        "user_question": user_query,
    })

st.header("Agent Chatbot with LangChain 🦜")
st.write("Load the Chatbot with PDFs, CSV or URL and Asks Questions About it, or just Chat Normally with the A.I. and also you can Ask Questions using the Agent Tools like the Google Search Engine, Math Problems or Wikipedia Posts.")
st.info("All the Different Options are Available in the Sidebar 🛠️")
st.markdown("<hr/>", unsafe_allow_html=True)

with st.sidebar:
    st.sidebar.markdown('''
        🧑🏻‍💻 Created by [Luis Jose Mendez](https://github.com/mendez-luisjose)
        ''')

    st.markdown("---------")
    st.title("Agent with LangChain 🦜")
    st.subheader("Load the Chatbot with PDFs, CSV or URL 📚")
    st.markdown("---------")

    st.write(
        """
        Load the Chatbot with PDFs, CSV or URL and Asks Questions About it. \n
        """
    )

    st.info("It Can be Selected Only One Option at Time ❗")

    st.header("Settings ⚙️")
    option = st.multiselect("Options:", ["URL",  "PDFs", "CSV", "Tools", "Conversation"], default=["Conversation"], max_selections=1)
    if len(option) !=0 :
        option = option[0]

for message in st.session_state.chat_history :
    if isinstance(message, HumanMessage) :
        with st.chat_message("user") :
            st.markdown(message.content)
    else :
        with st.chat_message("assistant") :
            st.markdown(message.content)

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
                st.markdown(user_query)

            with st.chat_message("assistant") :
                ai_response = get_response(user_query)
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
                st.markdown(user_query)

            with st.chat_message("assistant") :
                ai_response = get_response(user_query)
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

elif option == "Tools" :
    user_query = st.chat_input("Type your message here...")
        
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))
    
        with st.chat_message("user") :
            st.markdown(user_query)

        with st.chat_message("assistant") :
            ai_response = get_search_response(user_query)
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

elif option == "Conversation" :
    user_query = st.chat_input("Type your message here...")
        
    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))
    
        with st.chat_message("user") :
            st.markdown(user_query)

        with st.chat_message("assistant") :
            ai_response = get_chatbot_simple_response(user_query, st.session_state.chat_history)
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
        
elif option == "CSV" :
    csv_file = st.sidebar.file_uploader("Upload a CSV File:", accept_multiple_files=False)
    if csv_file == None :
        st.info("Please Load a CSV File")
    else :
        st.session_state.vector_store = get_vectorstore_from_csv(csv_file)

        user_query = st.chat_input("Type your message here...")
        
        if user_query is not None and user_query != "":
            st.session_state.chat_history.append(HumanMessage(content=user_query))
        
            with st.chat_message("user") :
                st.markdown(user_query)

            with st.chat_message("assistant") :
                ai_response = get_response(user_query)
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
