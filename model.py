def model(MODEL):
    import os
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI
    from langchain_openai.chat_models import ChatOpenAI
    from langchain_openai.embeddings import OpenAIEmbeddings
    from langchain_community.llms import Ollama
    from langchain_community.embeddings import OllamaEmbeddings

    # Load from .env file if API keys are not provided (for standalone script)
    load_dotenv(".env")  # Load environment variables from .env file
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


    # Initialize the OpenAI model with the API key
    if MODEL.startswith("gpt"):
        model= ChatOpenAI(api_key=OPENAI_API_KEY, model=MODEL)
        embeddings= OpenAIEmbeddings()
    else:
        model= Ollama(model=MODEL)
        embeddings = OllamaEmbeddings()
    
    
    return model, embeddings, OPENAI_API_KEY, PINECONE_API_KEY