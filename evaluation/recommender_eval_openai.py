import json
import os
import sys

# Add project root to path to import agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.retrieval_recommender_openai_nlp_vectordb import TeaChromaOpenAIRecommender

def load_eval_context():
    context_path = os.path.join(os.path.dirname(__file__), 'system_context_eval.txt')
    if os.path.exists(context_path):
        with open(context_path, 'r') as f:
            return f.read().strip()
    return "Output ONLY a JSON array of tea names."

def evaluate():
    try:
        # Initialize the OpenAI VectorDB based recommender
        recommender = TeaChromaOpenAIRecommender()
        recommender.build_vectordb()
        # Override system context for evaluation (JSON output)
        recommender.system_context = load_eval_context()
    except Exception as e:
        print(f"Failed to initialize recommender: {e}")
        return

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data.json')
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    total = len(test_data)
    correct = 0
    
    print(f"Evaluating OpenAI VectorDB Recommender (N={recommender.retrieval_n}) on {total} samples...")
    
    for item in test_data:
        query = item['query']
        expected = item['expected_names']
        
        # Get recommendation (should be JSON string due to eval context)
        response_str = recommender.recommend(query)
        
        try:
            # Parse the JSON response
            retrieved_names = json.loads(response_str)
            if not isinstance(retrieved_names, list):
                retrieved_names = [str(retrieved_names)]
        except:
            # Fallback if LLM fails to output valid JSON
            retrieved_names = [response_str.strip()]
        
        # Success if any expected name is in the retrieved results
        is_correct = any(exp in retrieved_names for exp in expected)
        if is_correct:
            correct += 1
        
        print(f"Query: {query}")
        print(f"  Expected: {expected}")
        print(f"  Retrieved: {retrieved_names}")
        print(f"  Match: {'Yes' if is_correct else 'No'}")
        print("-" * 20)
    
    precision = (correct / total) * 100
    print(f"Evaluation Complete.")
    print(f"Precision@{recommender.retrieval_n}: {precision:.2f}% ({correct}/{total})")

if __name__ == "__main__":
    evaluate()
