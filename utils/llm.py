# utils/llm.py
from langchain_groq import ChatGroq

def initialize_llm(model_name, api_key, temperature=0):
    """
    Initialize the Groq LLM.
    
    Args:
        model_name (str): Name of the LLM model.
        api_key (str): Groq API key.
        temperature (float): Sampling temperature for the LLM.
    
    Returns:
        ChatGroq: Initialized LLM.
    """
    return ChatGroq(
        temperature=temperature,
        model_name=model_name,
        api_key=api_key
    )