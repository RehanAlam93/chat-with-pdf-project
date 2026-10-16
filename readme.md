# Chat with PDF (RAG Application)

A fully local **Retrieval-Augmented Generation (RAG)** application built with **Streamlit**, **LangChain**, **ChromaDB**, and **Ollama (llama3.2:1b)**. This application allows users to upload any PDF document, securely store its vector embeddings, and chat with it locally while strictly answering questions based only on the provided document context.

---

## Features

* **PDF Document Upload & Processing**: Seamlessly loads and parses PDF files using PyPDFLoader.
* **Smart Text Splitting**: Breaks down large documents into manageable chunks using `RecursiveCharacterTextSplitter`.
* **Local Embeddings**: Utilizes HuggingFace sentence-transformers (`all-MiniLM-L6-v2`) for generating vector representations.
* **Vector Database**: Efficiently stores and searches document embeddings using **ChromaDB**.
* **MMR Retriever**: Implements Maximal Marginal Relevance (MMR) retrieval to fetch the most relevant and diverse context chunks.
* **Local LLM Integration**: Powered by **Ollama (`llama3.2:1b`)** running completely offline for privacy and zero API costs.
* **Strict Context Prompting**: Engineered prompts ensure the model avoids hallucinations and explicitly states when an answer is missing from the PDF.
* **Interactive Dashboard**: Built with **Streamlit** featuring real-time status indicators and a Debug view to inspect retrieved PDF contexts.

---

## Tech Stack

* **Frontend/UI**: Streamlit
* **Orchestration**: LangChain
* **Embeddings**: HuggingFace (`sentence-transformers`)
* **Vector Store**: Chroma
* **LLM**: Ollama (`llama3.2:1b`)
* **Language**: Python 3.11+

---

## Project Directory Structure

```text
chat_with_pdf/
│
├── uploads/                # Stores uploaded PDF documents locally
├── __pycache__/            # Python compiled bytecode cache
├── app.py                  # Main Streamlit application script
├── dashboard.py            # Additional dashboard interface file
└── requirements.txt        # Project dependencies and libraries