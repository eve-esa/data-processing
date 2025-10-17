from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from vllm import LLM
from typing import List, Union
import uvicorn
import argparse
import multiprocessing
import time
import sys
import os

app = FastAPI()

# Global variable for the model (initialized per worker)
llm = None

def initialize_model(gpu_id: int = None, gpu_memory_utilization: float = 0.9):
    """Initialize vLLM model with optional GPU specification."""
    global llm

    if llm is not None:
        return

    print(f"[Worker] Loading vLLM model on GPU {gpu_id if gpu_id is not None else 'auto'}...")

    # Set CUDA device if specified
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    llm = LLM(
        model="Qwen/Qwen3-Embedding-4B",
        task="embed",
        max_model_len=8192,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=False  # Set to True if you get CUDA graph errors
    )
    print(f"[Worker] Model loaded successfully on GPU {gpu_id if gpu_id is not None else 'auto'}!")

@app.on_event("startup")
async def startup_event():
    """Initialize model when the server starts."""
    # GPU ID and memory utilization will be set via environment variables
    gpu_id = int(os.environ.get("GPU_ID", -1))
    gpu_memory = float(os.environ.get("GPU_MEMORY_UTIL", 0.9))

    if gpu_id >= 0:
        initialize_model(gpu_id=gpu_id, gpu_memory_utilization=gpu_memory)
    else:
        initialize_model(gpu_memory_utilization=gpu_memory)

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

def run_server(port: int, gpu_id: int = None, gpu_memory_util: float = 0.9):
    """Run a single embedding server instance."""
    # Set environment variables for this process
    env = os.environ.copy()
    if gpu_id is not None:
        env["GPU_ID"] = str(gpu_id)
    env["GPU_MEMORY_UTIL"] = str(gpu_memory_util)

    print(f"Starting server on port {port} (GPU: {gpu_id if gpu_id is not None else 'auto'})")

    # Update environment for this process
    os.environ.update(env)

    uvicorn.run(app, host="0.0.0.0", port=port, workers=1, log_level="info")

def start_multiple_servers(num_servers: int, start_port: int = 8000,
                          gpu_ids: List[int] = None, gpu_memory_util: float = 0.9):
    """Start multiple embedding servers on consecutive ports.

    Args:
        num_servers: Number of server instances to start
        start_port: Starting port number (default: 8000)
        gpu_ids: List of GPU IDs to use (if None, auto-assign or use all available)
        gpu_memory_util: GPU memory utilization per server (default: 0.9)
    """
    processes = []

    # Get available GPUs if not specified
    if gpu_ids is None:
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            # Distribute servers across available GPUs
            gpu_ids = [i % num_gpus for i in range(num_servers)]
            print(f"Detected {num_gpus} GPUs. Distributing {num_servers} servers across them.")
        else:
            gpu_ids = [None] * num_servers
            print("No GPUs detected. Running on CPU.")

    # Adjust GPU memory utilization based on servers per GPU
    if gpu_ids and None not in gpu_ids:
        servers_per_gpu = {}
        for gpu_id in gpu_ids:
            servers_per_gpu[gpu_id] = servers_per_gpu.get(gpu_id, 0) + 1

        # Adjust memory utilization for shared GPUs
        max_servers_per_gpu = max(servers_per_gpu.values())
        if max_servers_per_gpu > 1:
            gpu_memory_util = gpu_memory_util / max_servers_per_gpu
            print(f"Adjusted GPU memory utilization to {gpu_memory_util:.2f} per server")

    print(f"\n{'='*60}")
    print(f"Starting {num_servers} embedding servers:")
    print(f"{'='*60}")

    urls = []
    for i in range(num_servers):
        port = start_port + i
        gpu_id = gpu_ids[i] if i < len(gpu_ids) else None

        url = f"http://localhost:{port}"
        urls.append(url)

        # Create process for each server
        process = multiprocessing.Process(
            target=run_server,
            args=(port, gpu_id, gpu_memory_util),
            name=f"EmbeddingServer-{port}"
        )
        process.start()
        processes.append(process)

        print(f"  [{i+1}] Port {port} - GPU {gpu_id if gpu_id is not None else 'auto'} - {url}")

        # Small delay to stagger startup
        time.sleep(1)

    print(f"{'='*60}")
    print(f"\nAll {num_servers} servers started successfully!")
    print(f"\nConfiguration for your pipeline:")
    print(f"embedder:")
    print(f"  url:")
    for url in urls:
        print(f"    - '{url}/'")
    print(f"{'='*60}\n")

    print("Press Ctrl+C to stop all servers...")

    try:
        # Wait for all processes
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        print("\n\nShutting down all servers...")
        for process in processes:
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
        print("All servers stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qwen embedding server(s)")
    parser.add_argument("--num-servers", type=int, default=1,
                       help="Number of server instances to start (default: 1)")
    parser.add_argument("--start-port", type=int, default=8000,
                       help="Starting port number (default: 8000)")
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=None,
                       help="GPU IDs to use (e.g., --gpu-ids 0 1 2). If not specified, auto-distribute.")
    parser.add_argument("--gpu-memory", type=float, default=0.9,
                       help="GPU memory utilization per server (default: 0.9)")

    args = parser.parse_args()

    if args.num_servers > 1:
        start_multiple_servers(
            num_servers=args.num_servers,
            start_port=args.start_port,
            gpu_ids=args.gpu_ids,
            gpu_memory_util=args.gpu_memory
        )
    else:
        # Single server mode (backward compatible)
        port = args.start_port
        gpu_id = args.gpu_ids[0] if args.gpu_ids else None
        run_server(port, gpu_id, args.gpu_memory)