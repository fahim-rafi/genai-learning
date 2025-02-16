from langchain.prompts import PromptTemplate

ZERO_SHOT_SUMMARIZATION_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="""
    Summarize the following paragraph in one sentence.
    
    {context}
    
    Summary:
    """
)
