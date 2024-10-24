import streamlit as st
import model
import getpdf
import vectorstore
import chains
import os


from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]

# Title of the app
st.markdown("<h1 style='color: #3498db; text-align: center;'>SmallPDF: Chat with PDF using LLaMA</h1>", unsafe_allow_html=True)

# Input API Keys
st.subheader("API Keys")
st.write("Make sure the Pinecone API and the OpenAI are included in the keys section")



# Step 3: Input the Pinecone Index Name
st.session_state.index_name = st.text_input("Enter a name for the Pinecone Index:")

# Step 3: Input the Pinecone Index Name
st.session_state.MODEL = st.text_input("Please enter the model you want to use (gpt-4o-mini or mistral:7b)."
                                       "Bear in mind that if using a LLaMA model you can only **run this script locally** on your computer and you will need to **have it installed locally**")

# Step 1: Input YouTube Link
with st.sidebar:
        st.markdown("<h3 style='color: #2ecc71;'>Menu:</h3>", unsafe_allow_html=True)

        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)

# Ensure all necessary inputs are provided
if pdf_docs and st.button("Submit & Process"):
    with st.spinner("Processing..."):
        try:
            text = getpdf.getpdf(pdf_docs)
            text_chunks = getpdf.get_text_chunks(text)
            st.session_state.vector = vectorstore.vectorstore(index_name=st.session_state.index_name, all_chunks=text_chunks, MODEL=st.session_state.MODEL)
            st.success("Processing complete!")
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
                

# Step 6: Ask Questions
question = st.text_input("Ask a question about the PDF:")

if question:
    if st.button("Get Answer"):
        # Step 7: Retrieve the answer using the question
        chain = chains.chains(pinecone=st.session_state.vector, question=question, model=st.session_state.MODEL)
        st.write("Answer:", chain)