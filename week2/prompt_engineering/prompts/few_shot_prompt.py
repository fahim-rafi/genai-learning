from langchain.prompts import PromptTemplate

FEW_SHOT_SUMMARIZATION_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="""
    Summarize the following paragraphs in one sentence.
    
    Example 1: 
    'The sun had just begun to set, casting an orange glow over the horizon. People were gathering by the beach, setting up picnic blankets and chairs. The sound of waves crashing against the shore was soothing, and a gentle breeze kept the air cool. Children were running around, laughing as they played tag and built sandcastles, while some adults were chatting in groups, enjoying the peaceful evening. A few boats could be seen in the distance, gently bobbing on the water as the sky changed colors.'
    → 'The sun set as people gathered by the beach, enjoying the waves, breeze, and peaceful evening.'
    
    'She woke up early, brewed herself a hot cup of coffee, and sat down at the kitchen table to start her work. The morning light streamed through the window, illuminating the pages of the notebook in front of her. As she sipped her coffee, she made a to-do list for the day, prioritizing the most important tasks first. Her phone buzzed with a few reminders and messages, but she silenced it, wanting to focus on her work. The quiet atmosphere helped her concentrate, and soon she was deep in thought, tackling one task after another.'
    → 'She woke up early, made coffee, and focused on her work, creating a to-do list for the day.'
    
    Now summarize this:
    {context}
    
    Summary:
    """
)
