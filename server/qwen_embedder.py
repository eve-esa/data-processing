from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from vllm import LLM
from typing import List, Union
import uvicorn

app = FastAPI()

# Initialize vLLM model at startup
print("Loading vLLM model...")
llm = LLM(
    model="Qwen/Qwen3-Embedding-4B",
    task="embed",
    max_model_len=8192,
    gpu_memory_utilization=0.9,
    enforce_eager=False  # Set to True if you get CUDA graph errors
)
print("Model loaded successfully!")

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "Qwen/Qwen3-Embedding-4B"

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    try:
        # Convert single string to list
        texts = request.input if isinstance(request.input, list) else [request.input]

        # Generate embeddings with vLLM
        outputs = llm.embed(texts)

        # Extract embeddings
        embeddings = torch.tensor([output.outputs.embedding for output in outputs])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Convert to list for JSON response
        embeddings_list = embeddings.cpu().tolist()

        return {
            'object': 'list',
            'data': [
                {
                    'object': 'embedding',
                    'embedding': emb,
                    'index': i
                }
                for i, emb in enumerate(embeddings_list)
            ],
            'model': request.model,
            'usage': {
                'prompt_tokens': len(texts),
                'total_tokens': len(texts)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "Qwen/Qwen3-Embedding-4B"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)