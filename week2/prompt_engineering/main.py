from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import sys
from src.assistant import ChatAssistant
from src.utils import ExperimentLogger

def run_experiment():
    models = ["deepseek-r1:32b", "llama3.1:8b"]  # Add models for comparison
    assistant = ChatAssistant(models=models)
    logger = ExperimentLogger()
    
    # Test context
    context = """Artificial intelligence (AI) is revolutionizing the way we interact with technology, enabling machines to perform tasks that typically require human intelligence, such as problem-solving, decision-making, and language understanding. From automating repetitive processes to analyzing vast amounts of data for actionable insights, AI is driving efficiency and innovation across industries like healthcare, finance, transportation, and entertainment. It powers personalized recommendations, virtual assistants, and predictive analytics, enhancing user experiences and business outcomes. However, as AI continues to advance, it also brings challenges related to data privacy, security, and ethical concerns around bias and accountability, necessitating thoughtful regulation and responsible development."""

    results = assistant.compare_prompts(
        context=context
    )
    
    for model_name, model_results in results.items():
        print(f"\n=== Results for {model_name} ===")
        print("Zero-shot Response:", model_results["zero_shot_response"])
        print("\nFew-shot Response:", model_results["few_shot_response"])
        print("\nChain of Thought Response:", model_results["cot_response"])
    
    json_path, csv_path = logger.save_results(
        results=results,
        context=context
    )
    
    print(f"\nResults appended to:")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")

if __name__ == "__main__":
    run_experiment()

