from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st


def process_user_query(file_path, user_query):
  """Backend function to handle PDF processing, retrieval, and LLM response."""
  # Initialize retriever in session state if not already present
  if "retriever" not in st.session_state:
    # 1. Load the PDF file
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # 2. Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(docs)

    # 3. Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Store chunks in Chroma vector database
    vector_store = Chroma.from_documents(split_docs, embeddings)

    # 5. Create retriever using MMR (Maximal Marginal Relevance)
    st.session_state.retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 3}
    )

  # Retrieve relevant chunks using MMR retriever from session state
  retriever = st.session_state.retriever
  retrieved_docs = retriever.invoke(user_query)

  # Combine context from retrieved chunks
  context = "\n\n".join([doc.page_content for doc in retrieved_docs])

  # Setup Local Ollama LLM
  llm = ChatOllama(model="llama3.2:1b", temperature=0.1)

  # Formulate Prompt
  prompt = f"""
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, say that you don't know.

    Context:
    {context}

    Question:
    {user_query}

    Answer:
    """

  # Get response from Local LLM
  response = llm.invoke(prompt)

  return response.content, context