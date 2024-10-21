def chains(pinecone, question, model):
    from langchain_core.output_parsers import StrOutputParser
    from langchain.prompts import ChatPromptTemplate
    from operator import itemgetter
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough
    
    template = """
    Answer the question based on the context below.
    If you can't answer the question, reply "I don't know".

    Context: {context}

    Question: {question}
    """
    prompt= ChatPromptTemplate.from_template(template)
    
    parser=StrOutputParser()

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
