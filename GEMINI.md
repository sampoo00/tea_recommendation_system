# Tea Recommendation System (TeaBot) - Project Context

This project is a RAG (Retrieval-Augmented Generation) based recommendation system designed to suggest tea blends through casual conversation. It supports both cloud-based LLMs (OpenAI) and local LLMs (Ollama).

## 1. Project Overview
- **Purpose**: Implicitly understand user preferences for tea through dialogue and provide data-backed recommendations.
- **Main Technologies**: 
    - **Language**: Python 3
    - **LLMs**: OpenAI (GPT-3.5 Turbo), Ollama (gpt-oss:20b or custom)
    - **Embeddings**: OpenAI (text-embedding-ada-002), Ollama (nomic-embed-text)
    - **Data Handling**: Pandas, Numpy, JSON
    - **API interaction**: Requests, OpenAI Python SDK
- **Architecture**:
    - **Agent**: Core conversational logic and retrieval engine.
    - **Data**: Knowledge base management and embedding generation.
    - **Evaluation**: Quantitative performance measurement (Precision@K, Prediction Rate).
    - **Test**: Connectivity and health checks for local services.

## 2. Directory Structure
- `agent/`: Contains recommendation engines (`retrieval_recommender_*.py`) and system prompts.
- `data/`: Knowledge base (`mock_tea_data.json`) and scripts for vectorizing data.
    - `data/openai/`: Files processed for OpenAI environment.
    - `data/ollama/`: Files processed for Ollama environment.
- `evaluation/`: Scripts to evaluate system performance using `test_data.json`.
- `test/`: Smoke tests for API servers (Ollama).

## 3. Building and Running

### Prerequisites
- Python 3.x and virtual environment recommended.
- Local Ollama server (if using Ollama mode).

### Setup
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Environment Variables**: Create a `.env` file in the root directory:
    ```env
    OPENAI_API_KEY=your_key
    OLLAMA_URL=http://localhost:11434
    OLLAMA_MODEL=gpt-oss:20b
    OLLAMA_EMBEDDING_MODEL=nomic-embed-text
    ```

### Running the System
1.  **Prepare Data**:
    - For OpenAI: Run `python3 data/openai/embed_documents_openai.py` followed by `python3 data/openai/modify_openai.py`.
    - For Ollama: Run `python3 data/ollama/embed_documents_ollama.py` followed by `python3 data/ollama/modify_ollama.py`.
2.  **Start Recommender**:
    - OpenAI: `python3 agent/retrieval_recommender_openai.py`
    - Ollama: `python3 agent/retrieval_recommender_ollama.py`

### Evaluation
- Run evaluation scripts from the root:
    - `python3 evaluation/recommender_eval_openai.py`
    - `python3 evaluation/recommender_eval_ollama.py`

## 4. Development Conventions
- **Pathing**: Python scripts use paths relative to the project root directory.
- **LLM Communication**: All LLM interactions are guided by strict JSON schemas defined in `system_message_retrieval` and `system_message_eval`.
- **Modularity**: Implementations are split between `ollama` and `openai` versions to maintain clear dependency boundaries.
- **Type Hinting**: Use Python type hints where possible for better readability and maintenance.
- **Error Handling**: Graceful degradation when LLM responses fail to parse as JSON.
