import json
import os
import sys
import numpy as np

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
        recommender = TeaChromaOpenAIRecommender()
        recommender.build_vectordb()
        recommender.system_context = load_eval_context()
    except Exception as e:
        print(f"Failed to initialize recommender: {e}")
        return

    test_data_path = os.path.join(os.path.dirname(__file__), 'test_data.json')
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    total = len(test_data)
    correct_count = 0
    all_match_similarities = []
    prediction_rates = []
    
    print(f"Evaluating OpenAI VectorDB Recommender (N={recommender.retrieval_n}) on {total} samples...")
    
    for item in test_data:
        query = item['query']
        expected = item['expected_names']
        
        # 1. Manually perform retrieval to get similarity scores
        query_embedding = recommender.get_embedding(query) # Note: method name is get_embedding in OpenAI class
        results = recommender.collection.query(
            query_embeddings=[query_embedding],
            n_results=recommender.retrieval_n
        )
        
        # ChromaDB 'cosine' distance is 1 - similarity
        retrieved_metas = results['metadatas'][0]
        distances = results['distances'][0]
        retrieved_info = {} # name -> similarity
        for meta, dist in zip(retrieved_metas, distances):
            retrieved_info[meta['name']] = 1.0 - dist

        # Top similarity for prediction rate calculation if no match
        top_similarity = max(retrieved_info.values()) if retrieved_info else 0.0

        # 2. Get LLM recommendation using evaluation context
        response_str = recommender.recommend(query)
        
        try:
            llm_output_names = json.loads(response_str)
            if not isinstance(llm_output_names, list):
                llm_output_names = [str(llm_output_names)]
        except:
            llm_output_names = [response_str.strip()]
        
        # 3. Calculate matches and collect similarities
        case_match_found = False
        
        for name in llm_output_names:
            if name in expected:
                case_match_found = True
                # Get the similarity score for this matched tea
                score = retrieved_info.get(name, 0.0)
                all_match_similarities.append(score)
        
        if case_match_found:
            correct_count += 1
            prediction_rates.append(1.0)
        else:
            prediction_rates.append(top_similarity)
        
        print(f"Query: {query}")
        print(f"  Expected: {expected}")
        print(f"  LLM Output: {llm_output_names}")
        print(f"  Match: {'Yes' if case_match_found else 'No'} (Rate: {prediction_rates[-1]:.4f})")
        print("-" * 20)
    
    precision = (correct_count / total) * 100
    avg_similarity = np.mean(all_match_similarities) if all_match_similarities else 0.0
    avg_prediction_rate = np.mean(prediction_rates) if prediction_rates else 0.0
    
    print(f"Evaluation Complete.")
    print(f"Precision@{recommender.retrieval_n}: {precision:.2f}% ({correct_count}/{total})")
    print(f"Average Similarity of Matched Items: {avg_similarity:.4f}")
    print(f"Overall Prediction Rate: {avg_prediction_rate:.4f}")

if __name__ == "__main__":
    evaluate()
