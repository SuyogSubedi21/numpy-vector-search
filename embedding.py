import os
from pathlib import Path
import json
import urllib.request   
from dotenv import load_dotenv
import numpy as np

load_dotenv(Path(__file__).parent / ".env")
API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/embeddings"
MODEL = "openai/text-embedding-3-large"


def build_headers():
    if not API_KEY:
        raise ValueError("API key is not set. Please set the OPENROUTER_API_KEY environment variable.")
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
def embed(texts, model=MODEL):
    """Turn a list of texts into a numpy array of vectors, one row per text."""
    data = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request(API_URL, data=data, headers=build_headers())
    with urllib.request.urlopen(req) as res:
        body = json.load(res)
    return np.array([item["embedding"] for item in body["data"]], dtype=np.float32)
 