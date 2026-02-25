import os
import json
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

class TeaRecommenderOllama:
    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
        self.embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        self.retrieval_n = int(os.getenv("RETRIEVAL_N", "3"))
        self.system_context = self._load_system_context()
        self.data_path = 'data/ollama/tea_data_final.json'
        self.load_data()
        
    def _load_system_context(self):
        context_path = os.path.join(os.path.dirname(__file__), 'system_context.txt')
        if os.path.exists(context_path):
            with open(context_path, 'r') as f:
                return f.read().strip()
        return "You are a helpful tea recommender."

    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}. Please run data preparation scripts.")
        with open(self.data_path, 'r') as f:
            self.teas = json.load(f)
            
    def get_embedding(self, text):
        response = requests.post(
            f"{self.ollama_url}/api/embeddings",
            json={"model": self.embedding_model, "prompt": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def retrieve(self, query, k=2):
        query_embedding = self.get_embedding(query)
        similarities = []
        for tea in self.teas:
            sim = self.cosine_similarity(query_embedding, tea['embedding'])
            similarities.append((tea, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in similarities[:k]]

    def chat(self, user_input):
        results = self.retrieve(user_input, k=self.retrieval_n)
        
        context = "Relevant Tea Blends:\n"
        for tea in results:
            context += f"- {tea['name']}: {tea['description']} (Flavors: {', '.join(tea['flavors'])})\n"
            
        prompt = f"""System: {self.system_context} Use the following context to recommend exactly {self.retrieval_n} teas to the user.

Context:
{context}

User: {user_input}
TeaBot:"""
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"]

def main():
    try:
        recommender = TeaRecommenderOllama()
        print(f"TeaBot (Ollama: {recommender.model}) is ready! Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            response = recommender.chat(user_input)
            print(f"TeaBot: {response}")
    except Exception as e:
        print(f"Error starting TeaBot: {e}")

if __name__ == "__main__":
    main()
