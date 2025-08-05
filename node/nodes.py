# graph/nodes.py
from langchain.schema import Document
# from langchain_community.tools.tavily_search import TavilySearchResults

def retrieve(state, retriever):
    """
    Retrieve documents from vectorstore.
    
    Args:
        state (dict): The current graph state.
        retriever: The vectorstore retriever.
    
    Returns:
        dict: Updated state with retrieved documents.
    """
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state, rag_chain):
    """
    Generate answer using RAG on retrieved documents.
    
    Args:
        state (dict): The current graph state.
        rag_chain: The RAG chain for generation.
    
    Returns:
        dict: Updated state with LLM generation.
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state, retrieval_grader):
    """
    Determines whether retrieved documents are relevant to the question.
    
    Args:
        state (dict): The current graph state.
        retrieval_grader: The retrieval grader chain.
    
    Returns:
        dict: Updated state with filtered documents and web_search flag.
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score['score']
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def web_search(state, web_search_tool):
    """
    Perform web search based on the question.
    
    Args:
        state (dict): The current graph state.
        web_search_tool: The Tavily search tool.
    
    Returns:
        dict: Updated state with web search results.
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])
    # docs = web_search_tool.invoke({"query": question})
    docs = web_search_tool.invoke({"query": question})["results"]
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)
    return {"documents": documents, "question": question}