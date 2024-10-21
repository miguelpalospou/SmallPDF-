def vectorstore(index_name, all_chunks, MODEL):
    import os
    from langchain_openai import OpenAIEmbeddings
    from langchain_openai.chat_models import ChatOpenAI  # Correct import for ChatOpenAI
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone
    from pinecone import ServerlessSpec
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.llms import Ollama


    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    
    # Initialize embeddings with OpenAI API Key
    if MODEL.startswith("gpt"):
        embeddings= OpenAIEmbeddings()
    else:
        embeddings = OllamaEmbeddings()

    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Create index in Pinecone
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    # Create Pinecone vector store
    pinecone = PineconeVectorStore.from_texts(
        texts=all_chunks, 
        embedding=embeddings, 
        index_name=index_name,
        pinecone_api_key=PINECONE_API_KEY
    )

    return pinecone