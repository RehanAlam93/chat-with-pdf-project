from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st


def process_pdf_file(file_path):
  """Loads, splits, and embeds the PDF immediately upon upload."""
  if "retriever" not in st.session_state:
    with st.spinner(
        "Processing PDF, creating embeddings, and setting up database..."
    ):
      # 1. Load the PDF file
      loader = PyPDFLoader(file_path)
      docs = loader.load()

      # 2. Split text into smaller chunks
      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=500, chunk_overlap=100
      )
      split_docs = text_splitter.split_documents(docs)

      # 3. Initialize HuggingFace embeddings
      embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

      # 4. Store chunks in Chroma vector database
      vector_store = Chroma.from_documents(split_docs, embeddings)

      # 5. Create retriever using MMR with k=5
      st.session_state.retriever = vector_store.as_retriever(
          search_type="mmr", search_kwargs={"k": 5}
      )


def process_user_query(user_query):
  """Generates an answer using the pre-built retriever and Ollama LLM immediately."""
  retriever = st.session_state.retriever
  retrieved_docs = retriever.invoke(user_query)

  # Combine context from retrieved chunks
  context = "\n\n".join([doc.page_content for doc in retrieved_docs])

  # Setup Local Ollama LLM
  llm = ChatOllama(model="llama3.2:1b", temperature=0.1)

  # Formulate Prompt
  prompt = f"""
    You are an intelligent assistant. Read the provided context carefully and answer the question accurately.
    If the question asks for specific details like email ID, name, phone number, or skills, extract and provide them clearly from the context.
    If the information is not present in the context, state: "I could not find this information in the uploaded PDF."
    Do not assume or use external knowledge.

    Context:
    {context}

    Question:
    {user_query}

    Answer:
    """

  # Get response from Local LLM
  response = llm.invoke(prompt)

  return response.content, context