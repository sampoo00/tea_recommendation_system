import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class TeaRecommenderOpenAI:
    def __init__(self, retrieval_n=None):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.retrieval_n = int(retrieval_n) if retrieval_n is not None else int(os.getenv("RETRIEVAL_N", "3"))
        self.system_context = self._load_system_context()
        self.data_path = 'data/openai/tea_data_final.json'
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
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model="text-embedding-ada-002").data[0].embedding

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
        # 1. Retrieve relevant tea blends
        results = self.retrieve(user_input, k=self.retrieval_n)
        
        # 2. Construct context for LLM
        context = "Relevant Tea Blends:\n"
        for tea in results:
            context += f"- {tea['name']}: {tea['description']} (Flavors: {', '.join(tea['flavors'])})\n"
            
        system_message = {
            "role": "system",
            "content": f"""{self.system_context} Use the following context to recommend exactly {self.retrieval_n} teas to the user.

Context:
{context}"""
        }
        
        messages = [
            system_message,
            {"role": "user", "content": user_input}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0
        )
        
        return response.choices[0].message.content

def main():
    recommender = TeaRecommenderOpenAI()
    print("TeaBot (OpenAI) is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = recommender.chat(user_input)
        print(f"TeaBot: {response}")

if __name__ == "__main__":
    main()
