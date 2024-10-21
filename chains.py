def chains(pinecone, question, model):
    from langchain_core.output_parsers import StrOutputParser
    from langchain.prompts import ChatPromptTemplate
    from operator import itemgetter
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough
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
    
    template = """
    Answer the question based on the context below.
    If you can't answer the question, reply "I don't know".

    Context: {context}

    Question: {question}
    """
    prompt= ChatPromptTemplate.from_template(template)
    
    parser=StrOutputParser()

    if model.startswith("gpt"):
        model= ChatOpenAI(api_key=OPENAI_API_KEY, model=model)

    else:
        model= Ollama(model=model)

    retriever = pinecone.as_retriever()

    chain= (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question")
    }
    | prompt
    | model 
    | parser
)

    answer=chain.invoke({"question":question})
    return answer
