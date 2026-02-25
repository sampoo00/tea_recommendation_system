import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

def get_embedding(text):
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": OLLAMA_EMBEDDING_MODEL, "prompt": text}
    )
    response.raise_for_status()
    return response.json()["embedding"]

def main():
    with open('data/mock_tea_data.json', 'r') as f:
        teas = json.load(f)
    
    data = []
    for tea in teas:
        content = f"Name: {tea['name']}. Type: {tea['type']}. Flavors: {', '.join(tea['flavors'])}. Description: {tea['description']}"
        try:
            embedding = get_embedding(content)
            tea['embedding'] = embedding
            data.append(tea)
        except Exception as e:
            print(f"Error embedding {tea['name']}: {e}")
    
    os.makedirs('data/ollama', exist_ok=True)
    with open('data/ollama/tea_data_with_embeddings.json', 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Embeddings generated using {OLLAMA_EMBEDDING_MODEL} and saved to data/ollama/tea_data_with_embeddings.json")

if __name__ == "__main__":
    main()
