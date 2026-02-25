# Tea Recommendation System (TeaBot)

TeaBot is a Retrieval-Augmented Generation (RAG) based recommendation system designed to suggest the perfect tea blends through conversation. It supports both local LLMs via **Ollama** and cloud LLMs via **OpenAI**, with multiple retrieval strategies including **VectorDB (ChromaDB)**.

## Project Structure

This project is organized into the following specialized directories:

- [**`agent/`**](./agent/): Contains multiple recommendation engine implementations. Includes local (Ollama) and cloud (OpenAI) versions, using NLP reasoning, manual RAG, or ChromaDB-backed vector search.
- [**`data/`**](./data/): The knowledge base. Stores the raw tea data (`mock_tea_data.json`) and scripts to generate embeddings for both local and cloud environments.
- [**`evaluation/`**](./evaluation/): Quantitative benchmarking tools. Measures Precision@N, Average Similarity, and Prediction Rates to verify the system's accuracy.
- [**`test/`**](./test/): Health check utilities to verify service connectivity (e.g., Ollama server status).

---

## Getting Started

### 1. Prerequisites
- Python 3.8+
- Ollama (for local mode)
- OpenAI API Key (for cloud mode)

### 2. Setup
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure your environment:
   Create a `.env` file based on `.env.example`:
   ```env
   OPENAI_API_KEY=your_key_here
   OLLAMA_URL=http://localhost:11434
   OLLAMA_MODEL=gpt-oss:20b
   OLLAMA_EMBEDDING_MODEL=nomic-embed-text
   RETRIEVAL_N=3
   ```

### 3. Data Preparation
If you are using agents that rely on pre-computed embeddings, run the preparation scripts:
```bash
python3 data/ollama/embed_documents_ollama.py
python3 data/openai/embed_documents_openai.py
```

---

## Usage

Run any of the agents from the root directory. For example, to use the most advanced local agent:
```bash
python3 agent/retrieval_recommender_ollama_nlp_vectordb.py
```

For more details on the different agent types, see the [Agent README](./agent/README.md).

---

## Evaluation

To benchmark the system performance:
```bash
python3 evaluation/recommender_eval_ollama.py
python3 evaluation/recommender_eval_openai.py
```
For metric definitions, see the [Evaluation README](./evaluation/README.md).

---

## Configuration

- **Persona**: You can change TeaBot's personality by editing [`agent/system_context.txt`](./agent/system_context.txt).
- **Result Count**: Adjust the `RETRIEVAL_N` variable in your `.env` to change how many recommendations you receive.
