import os
import streamlit as st

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

  with open(file_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

  st.success(
      f"File successfully saved in local folder: `{save_folder}/"
      f"{uploaded_file.name}`"
  )

  # Chat/Query section below
  st.markdown("---")
  st.subheader("Chat with your PDF")
  user_query = st.text_input("Ask a question about your uploaded PDF:")

  if user_query:
    # Yahan hum backend function ko call karenge jo app.py mein banaya hai
    from app import process_user_query

    with st.spinner("Generating answer using local Ollama (llama3.2:1b)..."):
      response_text, context = process_user_query(file_path, user_query)

      # Display Debug Context
      with st.expander("🔍 View Retrieved PDF Context (Debug)"):
        st.write(context)

      # Display Answer
      st.markdown("### Answer:")
      st.write(response_text)

else:
  st.info("Please upload a PDF file first.")