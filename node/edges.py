# graph/edges.py
def route_question(state, question_router):
    """
    Route question to web search or RAG.
    
    Args:
        state (dict): The current graph state.
        question_router: The question router chain.
    
    Returns:
        str: Next node to call.
    """
    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    source = question_router.invoke({"question": question})
    print(source)
    print(source['datasource'])
    if source['datasource'] == 'web_search':
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source['datasource'] == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def decide_to_generate(state):
    """
    Determines whether to generate an answer or perform web search.
    
    Args:
        state (dict): The current graph state.
    
    Returns:
        str: Next node to call.
    """
    print("---ASSESS GRADED DOCUMENTS---")
    web_search = state["web_search"]
    filtered_documents = state["documents"]
    if web_search == "Yes":
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state, hallucination_grader, answer_grader):
    """
    Determines whether the generation is grounded and answers the question.
    
    Args:
        state (dict): The current graph state.
        hallucination_grader: The hallucination grader chain.
        answer_grader: The answer grader chain.
    
    Returns:
        str: Decision for next node to call.
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score['score']
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score['score']
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"