import json
import os
import sys

# Add project root to path to import agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.retrieval_recommender_ollama import TeaRecommenderOllama

def evaluate():
    try:
        recommender = TeaRecommenderOllama()
    except Exception as e:
        print(f"Failed to initialize recommender: {e}")
        return

    with open('evaluation/test_data.json', 'r') as f:
        test_data = json.load(f)
    
    total = len(test_data)
    correct = 0
    
    print(f"Evaluating Ollama Recommender on {total} samples...")
    
    for item in test_data:
        query = item['query']
        expected = item['expected_names']
        
        retrieved = recommender.retrieve(query, k=1)
        retrieved_names = [t['name'] for t in retrieved]
        
        is_correct = any(name in retrieved_names for name in expected)
        if is_correct:
            correct += 1
        
        print(f"Query: {query}")
        print(f"  Expected: {expected}")
        print(f"  Retrieved: {retrieved_names}")
        print(f"  Match: {'Yes' if is_correct else 'No'}")
        print("-" * 20)
    
    precision = (correct / total) * 100
    print(f"Evaluation Complete.")
    print(f"Precision@1: {precision:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    evaluate()
