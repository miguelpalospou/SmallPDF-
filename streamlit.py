def vectorstore(index_name, all_chunks, MODEL):
    import os
    from langchain_openai import OpenAIEmbeddings
    from langchain_openai.chat_models import ChatOpenAI  # Correct import for ChatOpenAI
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone
    from pinecone import ServerlessSpec
    import streamlit as st

    # Load API keys
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or st.secrets["PINECONE_API_KEY"]
    
    if MODEL.startswith("gpt"):
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
        print("Using OpenAI API Key:", OPENAI_API_KEY is not None)  # Debugging output
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    else:
        print("Using Ollama embeddings (if needed) for model:", MODEL)  # Debugging output
        from langchain_community.embeddings import OllamaEmbeddings
        embeddings = OllamaEmbeddings()

    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create index in Pinecone
    try:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Pinecone index created successfully.")
    except Exception as e:
        print("Error creating Pinecone index:", e)

    # Create Pinecone vector store
    try:
        pinecone = PineconeVectorStore.from_texts(
            texts=all_chunks, 
            embedding=embeddings, 
            index_name=index_name,
            pinecone_api_key=PINECONE_API_KEY
        )
        print("Pinecone vector store created successfully.")
    except Exception as e:
        print("Error creating Pinecone vector store:", e)

    return pinecone
