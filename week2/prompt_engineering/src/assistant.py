from langchain_ollama.llms import OllamaLLM
from prompts.zero_shot_prompt import ZERO_SHOT_SUMMARIZATION_PROMPT
from prompts.few_shot_prompt import FEW_SHOT_SUMMARIZATION_PROMPT
from prompts.cot_prompt import COT_SUMMARIZATION_PROMPT

class ChatAssistant:
    def __init__(self, models=None):
        if models is None:
            models = ["deepseek-r1:32b"]  # Default model
        
        self.models = {}
        for model_name in models:
            self.models[model_name] = OllamaLLM(
                model=model_name,
                temperature=0.7,
                verbose=True
            )
        
    def get_response(self, context="", prompt_type="zero_shot", model_name=None):
        """
        Get response using different prompting strategies and models
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not initialized")
                
            prompt_map = {
                "zero_shot": ZERO_SHOT_SUMMARIZATION_PROMPT,
                "few_shot": FEW_SHOT_SUMMARIZATION_PROMPT,
                "cot": COT_SUMMARIZATION_PROMPT
            }
            
            prompt = prompt_map.get(prompt_type)
            if not prompt:
                raise ValueError(f"Unknown prompt type: {prompt_type}")
                
            chain = prompt | self.models[model_name]
            return chain.invoke({"context": context})
            
        except Exception as e:
            return f"Error: {str(e)}"

    def compare_prompts(self, context=""):
        """
        Compare responses between different prompting strategies and models
        """
        results = {}
        for model_name in self.models:
            results[model_name] = {
                "zero_shot_response": self.get_response(context, "zero_shot", model_name),
                "few_shot_response": self.get_response(context, "few_shot", model_name),
                "cot_response": self.get_response(context, "cot", model_name)
            }
        return results
