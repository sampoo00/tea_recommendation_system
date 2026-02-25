import os
import json
import requests
import chromadb
from dotenv import load_dotenv

load_dotenv()

class TeaChromaRecommender:
    def __init__(self):
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
        self.embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        self.retrieval_n = int(os.getenv("RETRIEVAL_N", "3"))
        self.system_context = self._load_system_context()
        
        # Initialize ChromaDB client (In-memory for this example)
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(
            name="tea_inventory",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.data_path = 'data/mock_tea_data.json'
        self.teas = self._load_data()

    def _load_system_context(self):
        context_path = os.path.join(os.path.dirname(__file__), 'system_context.txt')
        if os.path.exists(context_path):
            with open(context_path, 'r') as f:
                return f.read().strip()
        return "You are a helpful tea assistant."

    def _load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        with open(self.data_path, 'r') as f:
            return json.load(f)

    def get_ollama_embedding(self, text, is_query=False):
        """Generates embedding using Ollama's embedding API with instructional prefixes."""
        prefix = "search_query: " if is_query else "search_document: "
        response = requests.post(
            f"{self.ollama_url}/api/embeddings",
            json={"model": self.embedding_model, "prompt": prefix + text}
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def build_vectordb(self):
        """Builds ChromaDB by embedding all teas."""
        print(f"Building ChromaDB collection using Ollama model: {self.embedding_model}...")
        
        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for tea in self.teas:
            # Create a more descriptive natural language string for better embedding quality
            content = f"{tea['name']} is a {tea['type']} tea. It features flavors like {', '.join(tea['flavors'])}. {tea['description']}"
            embedding = self.get_ollama_embedding(content, is_query=False)
            
            ids.append(tea['id'])
            documents.append(content)
            embeddings.append(embedding)
            # Store full info in metadata for easy retrieval
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
        print(f"ChromaDB built with {self.collection.count()} entries.")

    def recommend(self, user_query):
        """RAG pipeline: ChromaDB Retrieval -> Ollama Generation."""
        # 1. Embed Query
        query_embedding = self.get_ollama_embedding(user_query, is_query=True)
        
        # 2. Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.retrieval_n
        )
        
        # 3. Construct Context
        context = "Top Matching Teas:\n"
        for i in range(len(results['metadatas'][0])):
            meta = results['metadatas'][0][i]
            context += f"- {meta['name']} ({meta['type']}): {meta['description']} Flavors: {meta['flavors']}\n"

        # 4. Generate via Ollama
        prompt = f"""System: {self.system_context} Use the following tea information to answer the user's request.

Context:
{context}

User's Request: "{user_query}"

Instructions:
- Recommend the top {self.retrieval_n} teas from the provided context.
- Keep the response concise and friendly.

Response:"""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0
                    }
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            return f"Error during generation: {e}"

def main():
    try:
        recommender = TeaChromaRecommender()
        recommender.build_vectordb()
        
        print("\n--- TeaBot ChromaDB + Ollama ---")
        while True:
            user_input = input("\nWhat kind of tea are you looking for? (or 'exit'): ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            print("Searching and thinking...")
            answer = recommender.recommend(user_input)
            print(f"\n{answer}")
            print("-" * 50)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
