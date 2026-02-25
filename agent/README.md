# TeaBot Agents

This directory contains various implementations of the Tea Recommendation Engine (TeaBot). The agents are categorized by their LLM provider (**Ollama** vs. **OpenAI**) and their retrieval strategy (**NLP**, **Basic RAG**, or **VectorDB**).

## Core Components

- **`system_context.txt`**: Defines the persona and behavioral rules for all agents. Every LLM-based agent in this directory loads this file to maintain a consistent "Tea Sommelier" identity.
- **`RETRIEVAL_N`**: All agents respect the `RETRIEVAL_N` environment variable (defined in `.env`), which controls how many tea blends are considered or recommended.

---

## Agent Implementations

### 1. Ollama-based Agents (Local LLM)
These agents use a locally running Ollama server for processing.

- **`retrieval_recommender_ollama.py`**: A standard RAG implementation. It performs a manual cosine similarity search using local embeddings and then generates a response.
- **`retrieval_recommender_ollama_nlp.py`**: An NLP-only approach. It loads the entire tea inventory into the LLM's context window and lets the model reason through the data directly.
- **`retrieval_recommender_ollama_nlp_vectordb.py`**: The most advanced local agent. It uses **ChromaDB** for efficient vector storage and search, utilizing `nomic-embed-text` with optimized search prefixes (`search_query:`, `search_document:`) for high precision.
- **`retrieval_recommender_ollama_embedding.py`**: A pure search tool that outputs the top N matching teas with their mathematical similarity scores, without generating a conversational response.

### 2. OpenAI-based Agents (Cloud LLM)
These agents use OpenAI's API for processing.

- **`retrieval_recommender_openai.py`**: A standard RAG implementation using `text-embedding-ada-002` for manual vector search and `gpt-3.5-turbo` for response generation.
- **`retrieval_recommender_openai_nlp.py`**: An NLP-only approach that passes the raw tea list to GPT-3.5, leveraging its large context window for direct reasoning.
- **`retrieval_recommender_openai_nlp_vectordb.py`**: A sophisticated RAG implementation using **ChromaDB** and OpenAI embeddings. It features optimized data formatting (natural language sentences) to improve retrieval accuracy.
- **`retrieval_recommender_openai_embedding.py`**: A pure search tool using OpenAI embeddings to find and score the best matches.

---

## How to Run

1. **Set up Environment**: Ensure your `.env` file is configured with the necessary API keys and URLs.
2. **Execute Agent**: Run the desired script directly from the project root:
   ```bash
   python3 agent/retrieval_recommender_ollama_nlp_vectordb.py
   ```

## Configuration
- To change the number of results: Modify `RETRIEVAL_N` in `.env`.
- To change the bot's personality: Edit `agent/system_context.txt`.
- To update the knowledge base: Modify `data/mock_tea_data.json`.
