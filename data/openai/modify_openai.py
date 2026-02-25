import json
import pandas as pd
import os

def main():
    input_path = 'data/openai/tea_data_with_embeddings.json'
    output_path = 'data/openai/tea_data_final.json'
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run embed_documents_openai.py first.")
        return

    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # In a real scenario, this might involve more complex transformations.
    # For now, we'll ensure it's a clean JSON for the recommender.
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Final data prepared at {output_path}")

if __name__ == "__main__":
    main()
