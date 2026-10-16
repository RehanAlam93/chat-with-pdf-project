import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Page Configuration & Headline
st.set_page_config(page_title="Chat with PDF Dashboard", layout="centered")
st.title("Chat with PDF")
st.write("Upload your PDF document and chat with it easily.")

# Upload Section
st.subheader("Upload PDF Document")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Logic to save PDF in local folder
if uploaded_file is not None:
  # Local folder name "uploads"
  save_folder = "uploads"
  os.makedirs(save_folder, exist_ok=True)

  file_path = os.path.join(save_folder, uploaded_file.name)

  # Write/save file to local directory
  with open(file_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

  st.success(
      f"File successfully saved in local folder: `{save_folder}/"
      f"{uploaded_file.name}`"
  )

  # Backend Pipeline: Load, Split, Embed, Vector Store, and MMR Retriever
  if "retriever" not in st.session_state:
    with st.spinner("Processing PDF through pipeline..."):
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

    st.success("Vector database and MMR retriever are ready!")

  # Chat/Query section below
  st.markdown("---")
  st.subheader("Chat with your PDF")

  # User query input box
  user_query = st.text_input("Ask a question about your uploaded PDF:")

  if user_query:
    with st.spinner("Generating answer using local Ollama (llama3.2:1b)..."):
      # Retrieve relevant chunks using MMR retriever from session state
      retriever = st.session_state.retriever
      retrieved_docs = retriever.invoke(user_query)

      # Combine context from retrieved chunks
      context = "\n\n".join([doc.page_content for doc in retrieved_docs])

      # ---> DEBUG: Yeh check karne ke liye ki PDF se kya text nikal kar aaya hai <---
      with st.expander("🔍 View Retrieved PDF Context (Debug)"):
        st.write(context)

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

      # Display Answer
      st.markdown("### Answer:")
      st.write(response.content)

else:
  st.info("Please upload a PDF file first.")