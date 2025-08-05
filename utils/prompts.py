# utils/prompts.py
from langchain.prompts import PromptTemplate

def get_router_prompt():
    """Return the prompt template for routing questions."""
    return PromptTemplate(
        template="""You are an expert at routing a 
        user question to a vectorstore or web search. Use the vectorstore for questions on LLM agents, 
        prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords 
        in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
        or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
        no premable or explaination. Question to route: {question}./no_think""",
        input_variables=["question"],
    )

def get_generate_prompt():
    """Return the prompt template for generating answers."""
    return PromptTemplate(
        template="""You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context} 
        Answer:./no_think""",
        input_variables=["question", "context"],
    )

def get_retrieval_grader_prompt():
    """Return the prompt template for grading retrieved documents."""
    return PromptTemplate(
        template="""You are a grader assessing relevance 
        of a retrieved document to a user question. If the document contains keywords related to the user question, 
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explaination. 
        Here is the retrieved document: \n\n {document} \n\n 
        Here is the user question: {question} \n./no_think""",
        input_variables=["question", "document"],
    )

def get_hallucination_grader_prompt():
    """Return the prompt template for grading hallucinations."""
    return PromptTemplate(
        template="""You are a grader assessing whether 
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
        single key 'score' and no preamble or explanation.
        Here are the facts: 
        \n ------- \n 
        {documents} 
        \n ------- \n 
        Here is the answer: {generation}./no_think""",
        input_variables=["generation", "documents"],
    )

def get_answer_grader_prompt():
    """Return the prompt template for grading answer usefulness."""
    return PromptTemplate(
        template="""You are a grader assessing whether an 
        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. 
        Here is the answer: 
        \n ------- \n 
        {generation} 
        \n ------- \n 
        Here is the question: {question}./no_think""",
        input_variables=["generation", "question"],
    )