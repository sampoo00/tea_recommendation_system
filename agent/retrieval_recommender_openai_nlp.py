import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class TeaRecommenderOpenAINLP:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.retrieval_n = int(os.getenv("RETRIEVAL_N", "3"))
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
        """Uses OpenAI reasoning to find the best tea match from the inventory."""
        tea_list_str = json.dumps(self.teas, indent=2)
        
        system_msg = self.system_context
        
        prompt = f"""Inventory:
{tea_list_str}

User's Request: "{user_query}"

Task:
1. Identify the top {self.retrieval_n} best matches from the inventory.
2. Explain the flavor profiles and why they match the user's request.
3. Keep the tone friendly and elegant.

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
            return f"Error communicating with OpenAI: {e}"

def main():
    try:
        recommender = TeaRecommenderOpenAINLP()
        print("--- TeaBot NLP Recommendation (OpenAI) ---")
        print("Tell me about the kind of tea experience you are looking for.")
        
        while True:
            user_input = input("\nYour preference (or 'exit'): ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            print("\nAnalyzing inventory and preferences...")
            recommendation = recommender.recommend(user_input)
            print(f"\n{recommendation}")
            print("-" * 40)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
