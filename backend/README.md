# TeaBot Backend APIs

This directory contains the FastAPI backend implementations for the Tea Recommendation System.

## Prerequisites

Ensure you have the following installed:
- `fastapi`
- `uvicorn`
- `pydantic`

You can install them via pip:
```bash
pip install fastapi uvicorn pydantic
```

## Running the APIs

### Ollama Version
To run the Ollama-based recommendation API:
```bash
python3 backend/main_ollama.py
```
Or using uvicorn directly:
```bash
uvicorn backend.main_ollama:app --reload --port 8000
```

### OpenAI Version
To run the OpenAI-based recommendation API:
```bash
python3 backend/main_openai.py
```
Or using uvicorn directly:
```bash
uvicorn backend.main_openai:app --reload --port 8001
```

## API Endpoints

### 1. Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Description**: Checks if the recommender and VectorDB are initialized.

### 2. Recommend Tea
- **URL**: `/recommend`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "query": "I want something citrusy and bold."
  }
  ```
- **Response**:
  ```json
  {
    "names": ["Earl Grey"]
  }
  ```

## Error Handling
The APIs include error handling for:
- Service initialization failures (503 Service Unavailable)
- Processing errors during embedding or generation (500 Internal Server Error)
- Invalid request formats (422 Unprocessable Entity - handled by FastAPI/Pydantic)
