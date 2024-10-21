def getpdf(pdf_docs):
    from PyPDF2 import PdfReader
    
    text= ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size= 500, chunk_overlap = 30)
    chunks = text_splitter.split_text(text)
    return chunks


