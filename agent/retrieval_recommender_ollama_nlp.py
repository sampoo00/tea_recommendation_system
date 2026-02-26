import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

class TeaRecommenderOllamaNLP:
    def __init__(self, retrieval_n=None):
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model = os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
        self.retrieval_n = int(retrieval_n) if retrieval_n is not None else int(os.getenv("RETRIEVAL_N", "3"))
        self.system_context = self._load_system_context()
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

    def recommend(self, user_query):
        """Uses LLM reasoning to find the best tea match from the full list."""
        tea_list_str = json.dumps(self.teas, indent=2)
        
        # We provide the full context to the LLM to act as a 'searchable database'
        prompt = f"""System: {self.system_context} Below is our current inventory of tea blends:

{tea_list_str}

User Preference: "{user_query}"

Task:
1. Identify the top {self.retrieval_n} best matching teas from the list above.
2. Explain why each tea is a good fit for the user's specific request.

Response:"""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
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
            return f"Error communicating with Ollama: {e}"

def main():
    try:
        recommender = TeaRecommenderOllamaNLP()
        print("--- TeaBot NLP Recommendation (Ollama) ---")
        print("Describe the tea you're looking for, and I'll find the best match from our list.")
        
        while True:
            user_input = input("\nYour preference (or 'exit'): ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            print("\nAnalyzing our inventory...")
            recommendation = recommender.recommend(user_input)
            print(f"\n{recommendation}")
            print("-" * 40)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
