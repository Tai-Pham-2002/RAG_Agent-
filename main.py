# main.py
from langgraph.graph import StateGraph, END
from config.settings import DATA_URLS, EMBEDDING_MODEL, LLM_MODEL, GROQ_API_KEY, TAVILY_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_K
from utils.data_loader import load_and_chunk_documents
from utils.embedding import initialize_vectorstore
from utils.llm import initialize_llm
from utils.prompts import get_router_prompt, get_generate_prompt, get_retrieval_grader_prompt, get_hallucination_grader_prompt, get_answer_grader_prompt
from node.state import GraphState
from node.nodes import retrieve, generate, grade_documents, web_search
from node.edges import route_question, decide_to_generate, grade_generation_v_documents_and_question
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from pprint import pprint
from langchain_tavily import TavilySearch
from IPython.display import Image

def main():
    # Initialize components
    documents = load_and_chunk_documents(DATA_URLS, CHUNK_SIZE, CHUNK_OVERLAP)
    vectorstore, embed_model = initialize_vectorstore(documents, EMBEDDING_MODEL)
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    llm = initialize_llm(LLM_MODEL, GROQ_API_KEY)
    web_search_tool = TavilySearch(max_results=3, api_key=TAVILY_API_KEY)

    # Initialize chains
    question_router = get_router_prompt() | llm | JsonOutputParser()
    rag_chain = get_generate_prompt() | llm | StrOutputParser()
    retrieval_grader = get_retrieval_grader_prompt() | llm | JsonOutputParser()
    hallucination_grader = get_hallucination_grader_prompt() | llm | JsonOutputParser()
    answer_grader = get_answer_grader_prompt() | llm | JsonOutputParser()

    # Define workflow
    workflow = StateGraph(GraphState)
    workflow.add_node("websearch", lambda state: web_search(state, web_search_tool))
    workflow.add_node("retrieve", lambda state: retrieve(state, retriever))
    workflow.add_node("grade_documents", lambda state: grade_documents(state, retrieval_grader))
    workflow.add_node("generate", lambda state: generate(state, rag_chain))

    # Set entry and conditional edges
    workflow.set_conditional_entry_point(
        lambda state: route_question(state, question_router),
        {
            "websearch": "websearch",
            "vectorstore": "retrieve",
        },
    )
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_conditional_edges(
        "generate",
        lambda state: grade_generation_v_documents_and_question(state, hallucination_grader, answer_grader),
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
        },
    )

    # Compile and test the workflow
    app = workflow.compile()
    test_questions = [
        "What is prompt engineering?",
        "Who are the Bears expected to draft first in the NFL draft?",
        "What are the types of agent memory?"
    ]
    
    for question in test_questions:
        print(f"\nTesting question: {question}")
        inputs = {"question": question}
        for output in app.stream(inputs):
            for key, value in output.items():
                pprint(f"Finished running: {key}:")
        pprint(value["generation"])

if __name__ == "__main__":
    main()
