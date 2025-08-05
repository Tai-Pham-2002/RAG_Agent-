# utils/embedding.py
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_ollama import OllamaEmbeddings

def initialize_vectorstore(documents, model_name):
    """
    Initialize the vectorstore with documents and embedding model.
    
    Args:
        documents (list): List of document chunks.
        model_name (str): Name of the embedding model.
        collection_name (str): Name of the vectorstore collection.
    
    Returns:
        FAISS: Initialized vectorstore.
        OllamaEmbeddings: Initialized embedding model.
    """
    embed_model = OllamaEmbeddings(model=model_name, num_gpu=100)
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embed_model,
    )
    return vectorstore, embed_model