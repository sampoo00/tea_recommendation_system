import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def main():
    with open('data/mock_tea_data.json', 'r') as f:
        teas = json.load(f)
    
    data = []
    for tea in teas:
        # Combine relevant fields for embedding
        content = f"Name: {tea['name']}. Type: {tea['type']}. Flavors: {', '.join(tea['flavors'])}. Description: {tea['description']}"
        embedding = get_embedding(content)
        tea['embedding'] = embedding
        data.append(tea)
    
    os.makedirs('data/openai', exist_ok=True)
    with open('data/openai/tea_data_with_embeddings.json', 'w') as f:
        json.dump(data, f, indent=2)
    print("Embeddings generated and saved to data/openai/tea_data_with_embeddings.json")

if __name__ == "__main__":
    main()
