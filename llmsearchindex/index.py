import os
import torch
import faiss
import json
import pickle
import numpy as np
import httpx
import requests
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import IncrementalPCA

class LLMIndex:
    """
    A high-performance static index for LLM Retrieval-Augmented Generation (RAG).
    Optimized for 200M+ documents using Memory-Mapped I/O to minimize RAM usage.
    """
    REPO_ID = "zakerytclarke/llmindex"
    
    def __init__(self, device: Optional[str] = None):
        """Initializes the index by downloading and loading assets from Hugging Face."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models_from_hf()
        self._read_loaded_models()
        self.async_client = httpx.AsyncClient(timeout=10.0)

    def _load_models_from_hf(self):
        """Downloads required static index files and model weights."""
        self.pca_path = hf_hub_download(repo_id=self.REPO_ID, filename="pca_model_64.pkl")
        self.index_path = hf_hub_download(repo_id=self.REPO_ID, filename="fineweb_full_final.index")
        self.mapping_path = hf_hub_download(repo_id=self.REPO_ID, filename="fineweb_mapping_full.npy")

    def _read_loaded_models(self):
        """Loads models using memory-mapping to handle 200M rows on consumer hardware."""
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        # IncrementalPCA unpickling requires sklearn to be imported in the namespace
        with open(self.pca_path, "rb") as f:
            self.pca = pickle.load(f)
        
        # FAISS MMAP: Keeps the binary index footprint minimal
        self.index = faiss.read_index_binary(
            self.index_path, 
            faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
        )
        
        # NUMPY MMAP: mmap_mode='r' keeps the 200M row mapping on disk.
        self.mapping = np.load(self.mapping_path, mmap_mode='r')

    def _get_api_params(self, dataset_id: int, row_id: int) -> Dict[str, Any]:
        """Maps internal IDs to Hugging Face Dataset Server API parameters."""
        return {
            "dataset": "wikimedia/wikipedia" if dataset_id == 0 else "HuggingFaceFW/fineweb",
            "config": "20231101.en" if dataset_id == 0 else "CC-MAIN-2025-26",
            "split": "train", 
            "offset": int(row_id), 
            "length": 1
        }

    def _prepare_query(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms query text into both high-fidelity and binary formats."""
        with torch.no_grad():
            query_emb = self.model.encode([query], convert_to_numpy=True)
        
        # Uses the sklearn PCA model to reduce 384d -> 64d
        query_reduced = self.pca.transform(query_emb)
        query_binary = np.packbits(query_reduced > 0, axis=-1)
        return query_emb, query_binary

    def _rerank_results(self, query_emb: np.ndarray, results: List[Dict]) -> List[Dict]:
        """Reorders fetched results using cosine similarity of full embeddings."""
        if not results:
            return []
        
        texts = [r.get("text", r.get("content", "")) for r in results]
        with torch.no_grad():
            doc_embs = self.model.encode(texts, convert_to_numpy=True)
        
        scores = np.dot(doc_embs, query_emb.T).flatten()
        scored_results = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
        return [r for score, r in scored_results]

    def search(self, query: str, top_k: int = 5, rerank: bool = False) -> List[Dict]:
        """Synchronous search using thread-parallelized network requests."""
        query_emb, query_binary = self._prepare_query(query)
        _, indices = self.index.search(query_binary, top_k)
        
        def fetch_sync(idx):
            if idx == -1: return None
            d_id, r_id = self.mapping[int(idx)]
            try:
                resp = requests.get(
                    "https://datasets-server.huggingface.co/rows", 
                    params=self._get_api_params(d_id, r_id), 
                    timeout=5
                )
                return resp.json()["rows"][0]["row"] if resp.status_code == 200 else None
            except: 
                return None

        with ThreadPoolExecutor(max_workers=top_k) as executor:
            results = [r for r in executor.map(fetch_sync, indices[0]) if r is not None]
            
        return self._rerank_results(query_emb, results) if rerank else results

    async def search_async(self, query: str, top_k: int = 5, rerank: bool = False) -> List[Dict]:
        """Asynchronous search optimized for non-blocking IO."""
        query_emb, query_binary = self._prepare_query(query)
        _, indices = self.index.search(query_binary, top_k)
        
        tasks = []
        for idx in indices[0]:
            if idx == -1: continue
            d_id, r_id = self.mapping[int(idx)]
            tasks.append(self.async_client.get(
                "https://datasets-server.huggingface.co/rows", 
                params=self._get_api_params(d_id, r_id)
            ))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        results = []
        for r in responses:
            if isinstance(r, httpx.Response) and r.status_code == 200:
                results.append(r.json()["rows"][0]["row"])
        
        return self._rerank_results(query_emb, results) if rerank else results

    async def close(self):
        """Gracefully closes the persistent asynchronous HTTP client."""
        await self.async_client.aclose()