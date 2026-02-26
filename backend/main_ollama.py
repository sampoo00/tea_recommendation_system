import os
import sys
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Add project root to path to import agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.retrieval_recommender_ollama_nlp_vectordb import TeaChromaRecommender

app = FastAPI(title="TeaBot Ollama API")

# Request and Response models
class QueryRequest(BaseModel):
    query: str

class RecommendResponse(BaseModel):
    names: List[str]

# Global recommender instance
recommender = None

def load_eval_context():
    # Look for evaluation context in the evaluation folder
    context_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'evaluation', 'system_context_eval.txt')
    if os.path.exists(context_path):
        with open(context_path, 'r') as f:
            return f.read().strip()
    return "Output ONLY a JSON array of tea names."

@app.on_event("startup")
async def startup_event():
    global recommender
    try:
        print("Initializing Ollama Recommender...")
        recommender = TeaChromaRecommender(retrieval_n=2)
        recommender.build_vectordb()
        recommender.system_context = load_eval_context()
        print("Ollama Recommender initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize recommender during startup: {e}")

@app.get("/health")
async def health_check():
    if recommender is None:
        return {"status": "error", "message": "Recommender not initialized"}
    return {"status": "ok"}

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: QueryRequest):
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender service is not ready")
    
    try:
        query = request.query
        print(f"Processing query: {query}")
        
        response_str = recommender.recommend(query)
        
        # Try to parse the response as JSON (list of names)
        try:
            llm_output_names = json.loads(response_str)
            if not isinstance(llm_output_names, list):
                llm_output_names = [str(llm_output_names)]
        except Exception:
            # Fallback if it's not JSON (e.g. error message or raw string)
            llm_output_names = [response_str.strip()]
            
        return RecommendResponse(names=llm_output_names)
        
    except Exception as e:
        print(f"Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8021)
