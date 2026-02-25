import requests
import os
from dotenv import load_dotenv

load_dotenv()

def test_ollama_health():
    url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    try:
        response = requests.get(f"{url}/api/tags")
        if response.status_code == 200:
            print(f"Ollama is healthy at {url}")
            models = response.json().get("models", [])
            print("Available models:")
            for m in models:
                print(f" - {m['name']}")
        else:
            print(f"Ollama returned status code {response.status_code}")
    except Exception as e:
        print(f"Could not connect to Ollama at {url}: {e}")

if __name__ == "__main__":
    test_ollama_health()
