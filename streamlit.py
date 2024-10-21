import streamlit as st
import model
import getpdf
import vectorstore
import chains


from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(".env")

# Title of the app
st.markdown("<h1 style='color: #3498db; text-align: center;'>SmallPDF: Chat with PDF using LLaMA</h1>", unsafe_allow_html=True)

# Input API Keys
st.subheader("API Keys")
st.write("Make sure the Pinecone API and the OpenAI are included in the keys section")

# Step 1: Input YouTube Link
with st.sidebar:
        st.markdown("<h3 style='color: #2ecc71;'>Menu:</h3>", unsafe_allow_html=True)

        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)

# Step 3: Input the Pinecone Index Name
index_name = st.text_input("Enter a name for the Pinecone Index:")

# Step 3: Input the Pinecone Index Name
MODEL = st.text_input("Please enter the model you want to use in quotes (gpt-4o-mini or mistral:7b)")

model_output=model.model(MODEL=MODEL)

# Ensure all necessary inputs are provided
if st.button("Submit & Process", style="background-color: #9b59b6; color: white; padding: 10px 20px; border: none; border-radius: 5px;"):  # Styled button
            with st.spinner("Processing..."):
                MODEL=model.model(MODEL)
                text = getpdf.getpdf(pdf_docs)
                text_chunks = getpdf.get_text_chunks(text)
                vector = vectorstore.vectorstore(index_name=index_name, all_chunks=text_chunks, MODEL=MODEL)
                

# Step 6: Ask Questions
question = st.text_input("Ask a question about the PDF:")

if question:
    if st.button("Get Answer"):
        # Step 7: Retrieve the answer using the question
        chain= chains.chains(vector, question, MODEL)