import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class TeaEmbeddingSearcherOpenAI:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.retrieval_n = int(os.getenv("RETRIEVAL_N", "3"))
        self.system_context = self._load_system_context()
        self.data_path = 'data/openai/tea_data_with_embeddings.json'
        self.teas = self._load_data()

    def _load_system_context(self):
        context_path = os.path.join(os.path.dirname(__file__), 'system_context.txt')
        if os.path.exists(context_path):
            with open(context_path, 'r') as f:
                return f.read().strip()
        return "You are a helpful tea searcher."

    def _load_data(self):
        """Loads the tea data with pre-computed embeddings."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Missing data file: {self.data_path}. Please run 'data/openai/embed_documents_openai.py' first.")
        
        with open(self.data_path, 'r') as f:
            return json.load(f)

    def get_embedding(self, text, model="text-embedding-ada-002"):
        """Generates an embedding for the user's query using OpenAI."""
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=model).data[0].embedding

    def cosine_similarity(self, v1, v2):
        """Calculates cosine similarity between two vectors."""
        v1 = np.array(v1)
        v2 = np.array(v2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def search(self, query, top_k=3):
        """Finds the top K most similar teas to the user's query."""
        print(f"\nSearching for: '{query}'...")
        query_vec = self.get_embedding(query)
        
        results = []
        for tea in self.teas:
            similarity = self.cosine_similarity(query_vec, tea['embedding'])
            results.append({
                "name": tea['name'],
                "type": tea['type'],
                "flavors": tea['flavors'],
                "description": tea['description'],
                "score": similarity
            })
        
        # Sort by similarity score in descending order
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

def main():
    try:
        searcher = TeaEmbeddingSearcherOpenAI()
        print("--- Tea Recommendation Search (OpenAI Embeddings) ---")
        print("Tell me what kind of tea you are looking for (e.g., 'Something bold and citrusy')")
        
        while True:
            user_input = input("\nYour preference (or 'exit'): ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            recommendations = searcher.search(user_input, top_k=searcher.retrieval_n)
            
            print("\nTop Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec['name']} (Match Score: {rec['score']:.4f})")
                print(f"   Type: {rec['type']}")
                print(f"   Flavors: {', '.join(rec['flavors'])}")
                print(f"   Description: {rec['description']}")
                print("-" * 30)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
