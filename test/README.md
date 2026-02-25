# TeaBot Health Checks

This directory contains scripts for verifying the connectivity and operational health of the external services used by the Tea Recommendation System.

## Contents

- **`test_ollama_health.py`**: A smoke test for the local Ollama service. It verifies that the server is reachable and lists all models currently downloaded and available for use.

---

## How to Run

Before running the agents or evaluation scripts, it is recommended to verify your local service status.

Run from the project root:

```bash
python3 test/test_ollama_health.py
```

## Expected Output

If successful, you will see a confirmation message and a list of available models:
```text
Ollama is healthy at http://localhost:11434
Available models:
 - gpt-oss:20b
 - nomic-embed-text
 - ...
```

If it fails, check that your Ollama server is running and that the `OLLAMA_URL` in your `.env` file is correct.
