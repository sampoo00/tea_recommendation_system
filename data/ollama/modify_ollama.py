import json
import os

def main():
    input_path = 'data/ollama/tea_data_with_embeddings.json'
    output_path = 'data/ollama/tea_data_final.json'
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run embed_documents_ollama.py first.")
        return

    with open(input_path, 'r') as f:
        data = json.load(f)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Final data prepared at {output_path}")

if __name__ == "__main__":
    main()
