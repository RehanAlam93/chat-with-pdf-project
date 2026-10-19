import os
import streamlit as st
from app import process_pdf_file, process_user_query

# Page Configuration & Headline
st.set_page_config(page_title="Chat with PDF Dashboard", layout="centered")
st.title("Chat with PDF")
st.write("Upload your PDF document and chat with it easily.")

# Upload Section
st.subheader("Upload PDF Document")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Logic to save PDF in local folder
if uploaded_file is not None:
  save_folder = "uploads"
  os.makedirs(save_folder, exist_ok=True)
  file_path = os.path.join(save_folder, uploaded_file.name)

  # Save file and process immediately if not already done
  if "file_path" not in st.session_state or st.session_state.file_path != file_path:
    with open(file_path, "wb") as f:
      f.write(uploaded_file.getbuffer())
    st.session_state.file_path = file_path

    # Call backend function to process PDF and build embeddings right away
    process_pdf_file(file_path)
    st.success(f"File successfully saved and processed in local folder: `{save_folder}/{uploaded_file.name}`")

  # Chat/Query section below
  st.markdown("---")
  st.subheader("Chat with your PDF")

  # User query input box
  user_query = st.text_input("Ask a question about your uploaded PDF:")

  if user_query:
    with st.spinner("Generating answer using local Ollama (llama3.2:1b)..."):
      # Call backend query function with only 1 argument (user_query)
      response_text, context = process_user_query(user_query)

      # Display Debug Context
      with st.expander("🔍 View Retrieved PDF Context (Debug)"):
        st.write(context)

      # Display Answer
      st.markdown("### Answer:")
      st.write(response_text)

else:
  if "retriever" in st.session_state:
    del st.session_state.retriever
  st.info("Please upload a PDF file first.")