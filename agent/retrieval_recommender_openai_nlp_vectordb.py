import os
import json
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class TeaChromaOpenAIRecommender:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.retrieval_n = int(os.getenv("RETRIEVAL_N", "3"))
        self.system_context = self._load_system_context()
        
        # Initialize ChromaDB (In-memory)
        self.chroma_client = chromadb.Client()
        # Use cosine similarity for better text search performance
        self.collection = self.chroma_client.get_or_create_collection(
            name="tea_inventory_openai",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.data_path = 'data/mock_tea_data.json'
        self.teas = self._load_data()

    def _load_system_context(self):
        context_path = os.path.join(os.path.dirname(__file__), 'system_context.txt')
        if os.path.exists(context_path):
            with open(context_path, 'r') as f:
                return f.read().strip()
        return "You are a professional tea sommelier."

    def _load_data(self):
        """Loads the raw tea data from JSON."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        with open(self.data_path, 'r') as f:
            return json.load(f)

    def get_embedding(self, text, model="text-embedding-ada-002"):
        """Generates an embedding using OpenAI."""
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=model).data[0].embedding

    def build_vectordb(self):
        """Embeds all tea data and stores it in ChromaDB."""
        print("Building ChromaDB using OpenAI embeddings...")
        
        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for tea in self.teas:
            # Optimize data format into natural language sentences for the embedding model
            content = f"The {tea['name']} is a {tea['type']} variety. It has a flavor profile featuring {', '.join(tea['flavors'])}. {tea['description']} This tea has a {tea['caffeine']} caffeine level."
            embedding = self.get_embedding(content)
            
            ids.append(tea['id'])
            documents.append(content)
            embeddings.append(embedding)
            metadatas.append({
                "name": tea['name'],
                "type": tea['type'],
                "flavors": ", ".join(tea['flavors']),
                "description": tea['description']
            })

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Successfully added {self.collection.count()} teas to ChromaDB.")

    def recommend(self, user_query):
        """Retrieves relevant context from ChromaDB and generates a recommendation via OpenAI LLM."""
        # 1. Embed user query
        query_embedding = self.get_embedding(user_query)
        
        # 2. Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.retrieval_n
        )
        
        # 3. Construct context from results
        context = "Top relevant teas from our collection:\n"
        for i in range(len(results['metadatas'][0])):
            meta = results['metadatas'][0][i]
            context += f"- {meta['name']} ({meta['type']}): {meta['description']} (Flavors: {meta['flavors']})\n"

        # 4. Generate recommendation using GPT-3.5
        system_msg = self.system_context
        prompt = f"""Context:
{context}

User's Request: "{user_query}"

Task:
- Recommend the top {self.retrieval_n} best matching teas from the context.
- Explain why they fit the user's request.
- Keep the response friendly and professional.

Response:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error during recommendation generation: {e}"

def main():
    try:
        recommender = TeaChromaOpenAIRecommender()
        recommender.build_vectordb()
        
        print("\n--- TeaBot ChromaDB + OpenAI (Embeddings & LLM) ---")
        print("I'm ready to find your perfect tea blend.")
        
        while True:
            user_input = input("\nHow can I help you find a tea today? (or 'exit'): ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            print("\nSearching and analyzing...")
            recommendation = recommender.recommend(user_input)
            print(f"\n{recommendation}")
            print("-" * 50)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
