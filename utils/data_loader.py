# utils/data_loader.py
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk_documents(urls, chunk_size, chunk_overlap):
    """
    Load documents from URLs and split them into chunks.
    
    Args:
        urls (list): List of URLs to load documents from.
        chunk_size (int): Size of each document chunk.
        chunk_overlap (int): Overlap between chunks.
    
    Returns:
        list: List of document chunks.
    """
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    doc_splits = text_splitter.split_documents(docs_list)
    print(f"Length of document chunks generated: {len(doc_splits)}")
    return doc_splits