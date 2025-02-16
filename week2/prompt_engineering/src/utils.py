import json
import csv
import os
from datetime import datetime

class ExperimentLogger:
    def __init__(self, output_dir="outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.json_path = os.path.join(output_dir, "experiments.json")
        self.csv_path = os.path.join(output_dir, "experiments.csv")
        
        # Create CSV with headers if it doesn't exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'model', 'prompt_type', 'context', 'response'])
    
    def save_results(self, results, context=""):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_data = {
            "timestamp": timestamp,
            "context": context,
            "results": {}
        }
        
        # Prepare data for both JSON and CSV
        csv_rows = []
        
        for model_name, model_results in results.items():
            experiment_data["results"][model_name] = {
                "zero_shot_response": model_results["zero_shot_response"],
                "few_shot_response": model_results["few_shot_response"],
                "cot_response": model_results["cot_response"]
            }
            
            # Prepare CSV rows
            for prompt_type, response in model_results.items():
                csv_rows.append([
                    timestamp,
                    model_name,
                    prompt_type.replace("_response", ""),
                    context,
                    response
                ])
        
        # Append to JSON
        existing_data = []
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = [existing_data]
                except json.JSONDecodeError:
                    existing_data = []
                    
        existing_data.append(experiment_data)
        with open(self.json_path, 'w') as f:
            json.dump(existing_data, f, indent=2)
            
        # Append to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        
        return self.json_path, self.csv_path 