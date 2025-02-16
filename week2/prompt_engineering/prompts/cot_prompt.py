from langchain.prompts import PromptTemplate

COT_SUMMARIZATION_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="""
    Summarize the following paragraph in one sentence. 
    First, identify the key points, then combine them into a concise summary.
    
    {context}
    
    Step-by-step reasoning:
    1. Identify key points in the paragraph.
    2. Combine key points into a single, concise sentence.
    
    Summary:
    """
)
