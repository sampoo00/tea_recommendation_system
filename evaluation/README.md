# TeaBot Evaluation Framework

This directory contains the tools and data used to quantitatively measure the performance of the TeaBot recommendation engines. The framework focuses on the **VectorDB-based** agents to assess retrieval accuracy and LLM reasoning.

## Metrics Defined

The evaluation scripts calculate three primary metrics:

1.  **Precision@N**:
    *   Measures the percentage of test cases where the agent successfully recommended at least one of the `expected_names`.
    *   `N` is determined by the `RETRIEVAL_N` value in your `.env` file.
2.  **Average Similarity of Matched Items**:
    *   Calculates the mean cosine similarity score (confidence) of the items that were correctly identified by the LLM.
    *   This provides insight into how "mathematically close" the correct matches were in the vector space.
3.  **Overall Prediction Rate**:
    *   A hybrid performance metric.
    *   If a match is found, the score for that case is **1.0**.
    *   If no match is found, the score is the **similarity score** of the top retrieved item.
    *   The overall rate is the average of these scores, rewarding exact matches while giving partial credit for "near misses" in retrieval.

---

## Key Files

- **`test_data.json`**: The ground truth dataset containing various user queries and the names of the teas that *should* be recommended for each.
- **`system_context_eval.txt`**: A specialized system prompt that forces the LLM to output results as a raw JSON array of strings. This is critical for automated parsing and comparison.
- **`recommender_eval_ollama.py`**: Runs the evaluation suite against the Ollama VectorDB engine (`TeaChromaRecommender`).
- **`recommender_eval_openai.py`**: Runs the evaluation suite against the OpenAI VectorDB engine (`TeaChromaOpenAIRecommender`).

---

## How to Run Evaluation

Ensure your environment is configured and the necessary embeddings have been generated (if using pre-embedded files, though these scripts build the DB in-memory from `mock_tea_data.json`).

Run from the project root:

```bash
# Evaluate the local Ollama engine
python3 evaluation/recommender_eval_ollama.py

# Evaluate the OpenAI engine
python3 evaluation/recommender_eval_openai.py
```

## Customizing Tests
To add more test scenarios, simply append new query objects to `evaluation/test_data.json`:
```json
{
  "query": "Your test query here",
  "expected_names": ["Name of Tea 1", "Name of Tea 2"]
}
```
