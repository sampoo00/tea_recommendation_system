# TeaBot Knowledge Base

This directory contains the core tea data and the pipelines for preparing embeddings.

## Contents

- **`mock_tea_data.json`**: The master dataset containing IDs, names, types, flavors, descriptions, and caffeine levels for various tea blends.
- **`ollama/`**: Tools for the local LLM environment.
  - `embed_documents_ollama.py`: Script to generate embeddings using the local Ollama API.
  - `tea_data_with_embeddings.json`: Pre-computed embeddings using the `nomic-embed-text` model.
- **`openai/`**: Tools for the cloud LLM environment.
  - `embed_documents_openai.py`: Script to generate embeddings using the OpenAI API.
  - `tea_data_with_embeddings.json`: Pre-computed embeddings using the `text-embedding-ada-002` model.

---

## Data Preparation

To refresh the embeddings after modifying `mock_tea_data.json`, run the corresponding script from the project root:

```bash
# For Ollama
python3 data/ollama/embed_documents_ollama.py

# For OpenAI
python3 data/openai/embed_documents_openai.py
```
