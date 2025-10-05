import os
import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

# Load .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Force library to use your Gemini API key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Streamlit Page Config
st.set_page_config(page_title=" PDF Chatbot (Gemini 2.5)", layout="centered")

st.title("Chat with your PDF (Gemini-2.0-Flash)")

# Upload PDF
pdf_file = st.file_uploader("Upload your PDF file", type="pdf")

if pdf_file:
    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        pdf_path = tmp.name

    # Load and split PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    # Create Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    vectordb = Chroma.from_documents(chunks, embeddings)
    retriever = vectordb.as_retriever()

    # Gemini 2.0 Flash Model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY
    )

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # User Input
    st.subheader("Ask Questions about your PDF:")
    user_q = st.text_input(" Type your question here:")

    if user_q:
        with st.spinner("Thinking..."):
            response = qa.invoke({"query": user_q})
            st.success("Answer:")
            st.write(response["result"])
