import os
import torch
import faiss
import json
import pickle
import numpy as np
import httpx
import requests
import asyncio
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, HfFileSystem, get_token
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import IncrementalPCA

class LLMIndex:
    """
    A high-performance static index for LLM Retrieval-Augmented Generation (RAG).
    Optimized for 200M+ documents using Memory-Mapped I/O and direct Parquet HTTP Range Requests.
    """
    REPO_ID = "zakerytclarke/llmindex"
    
    # Pinned Revisions to guarantee offset stability
    WIKI_REV = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
    FINEWEB_REV = "9bb295ddab0e05d785b879661af7260fed5140fc"
    
    def __init__(self, device: Optional[str] = None):
        """Initializes the index by downloading and loading assets from Hugging Face."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check for local HF token from huggingface-cli login or login()
        self.token = get_token() 
        
        self._load_models_from_hf()
        self._read_loaded_models()
        self._build_parquet_maps()
        
        # Attach token to async client if it exists
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        self.async_client = httpx.AsyncClient(timeout=10.0, headers=headers)

    def _load_models_from_hf(self):
        """Downloads required static index files and model weights."""
        self.pca_path = hf_hub_download(repo_id=self.REPO_ID, filename="pca_model_64.pkl", token=self.token)
        self.index_path = hf_hub_download(repo_id=self.REPO_ID, filename="fineweb_full_final.index", token=self.token)
        self.mapping_path = hf_hub_download(repo_id=self.REPO_ID, filename="fineweb_mapping_full.npy", token=self.token)

    def _read_loaded_models(self):
        """Loads models using memory-mapping to handle 200M rows on consumer hardware."""
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        with open(self.pca_path, "rb") as f:
            self.pca = pickle.load(f)
        
        self.index = faiss.read_index_binary(
            self.index_path, 
            faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
        )
        
        self.mapping = np.load(self.mapping_path, mmap_mode='r')

    def _build_parquet_maps(self):
        """Builds global-to-local row indices for both datasets via parquet footers."""
        # Initialize file system with the token (falls back to unauth if None)
        self.fs = HfFileSystem(token=self.token)
        self.dataset_maps = {0: [], 1: []}
        
        # Injected pinned revisions into the repository paths
        patterns = {
            0: f"datasets/wikimedia/wikipedia@{self.WIKI_REV}/**/*.parquet", 
            1: f"datasets/HuggingFaceFW/fineweb@{self.FINEWEB_REV}/data/CC-MAIN-2025-26/*.parquet"
        }
        
        for d_id, pattern in patterns.items():
            files = sorted(self.fs.glob(pattern))
            
            if d_id == 0:
                files = [f for f in files if "20231101.en" in f and "train" in f]
            
            def get_row_count(file_path):
                try:
                    with self.fs.open(file_path, mode="rb") as f:
                        return file_path, pq.read_metadata(f).num_rows
                except Exception:
                    return file_path, 0
                    
            with ThreadPoolExecutor(max_workers=20) as executor:
                results = list(executor.map(get_row_count, files))
                
            global_row_count = 0
            for file_path, count in results:
                if count > 0:
                    self.dataset_maps[d_id].append({
                        "filename": file_path,
                        "start": global_row_count,
                        "end": global_row_count + count - 1
                    })
                    global_row_count += count

    def _fetch_surgical(self, dataset_id: int, row_id: int) -> Optional[Dict[str, str]]:
        """Directly extracts a specific row group and normalizes the payload."""
        mapping = self.dataset_maps.get(dataset_id, [])
        target_file = None
        for m in mapping:
            if m["start"] <= row_id <= m["end"]:
                target_file = m
                break
                
        if not target_file:
            return None
            
        local_offset = row_id - target_file["start"]
        
        try:
            with self.fs.open(target_file["filename"], mode="rb") as f:
                pf = pq.ParquetFile(f)
                
                current_row = 0
                target_rg = 0
                rg_offset = 0
                
                for rg_idx in range(pf.metadata.num_row_groups):
                    rg = pf.metadata.row_group(rg_idx)
                    if current_row + rg.num_rows > local_offset:
                        target_rg = rg_idx
                        rg_offset = local_offset - current_row
                        break
                    current_row += rg.num_rows
                
                table = pf.read_row_group(target_rg) 
                row_data = table.slice(rg_offset, 1).to_pydict()
                
                # Parse PyArrow list structures into a flat dictionary
                parsed_row = {k: v[0] if isinstance(v, list) else v for k, v in row_data.items()}
                
                # Normalize output to strictly contain 'url' and 'text'
                return {
                    "url": str(parsed_row.get("url", "")),
                    "text": str(parsed_row.get("text", ""))
                }
        except Exception:
            return None

    def _get_api_params(self, dataset_id: int, row_id: int) -> Dict[str, Any]:
        """Kept for backward compatibility if any external code hooks into it."""
        if dataset_id == 0:
            return {
                "dataset": "wikimedia/wikipedia",
                "config": "20231101.en",
                "split": "train", 
                "revision": self.WIKI_REV,
                "offset": int(row_id), 
                "length": 1
            }
        else:
            return {
                "dataset": "HuggingFaceFW/fineweb",
                "config": "CC-MAIN-2025-26",
                "split": "train", 
                "revision": self.FINEWEB_REV,
                "offset": int(row_id), 
                "length": 1
            }

    def _prepare_query(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """Transforms query text into both high-fidelity and binary formats."""
        with torch.no_grad():
            query_emb = self.model.encode([query], convert_to_numpy=True)
        
        query_reduced = self.pca.transform(query_emb)
        query_binary = np.packbits(query_reduced > 0, axis=-1)
        return query_emb, query_binary

    def _rerank_results(self, query_emb: np.ndarray, results: List[Dict]) -> List[Dict]:
        """Reorders fetched results using cosine similarity of full embeddings."""
        if not results:
            return []
        
        texts = [r.get("text", "") for r in results]
        with torch.no_grad():
            doc_embs = self.model.encode(texts, convert_to_numpy=True)
        
        scores = np.dot(doc_embs, query_emb.T).flatten()
        scored_results = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
        return [r for score, r in scored_results]

    def search(self, query: str, top_k: int = 5, rerank: bool = False) -> List[Dict]:
        """Synchronous search using thread-parallelized surgical fetching."""
        query_emb, query_binary = self._prepare_query(query)
        _, indices = self.index.search(query_binary, top_k)
        
        def fetch_sync(idx):
            if idx == -1: return None
            d_id, r_id = self.mapping[int(idx)]
            return self._fetch_surgical(int(d_id), int(r_id))

        with ThreadPoolExecutor(max_workers=top_k) as executor:
            results = [r for r in executor.map(fetch_sync, indices[0]) if r is not None]
            
        return self._rerank_results(query_emb, results) if rerank else results

    async def search_async(self, query: str, top_k: int = 5, rerank: bool = False) -> List[Dict]:
        """Asynchronous search wrapping blocking Parquet I/O to prevent loop blocking."""
        query_emb, query_binary = self._prepare_query(query)
        _, indices = self.index.search(query_binary, top_k)
        
        async def fetch_async(idx):
            if idx == -1: return None
            d_id, r_id = self.mapping[int(idx)]
            return await asyncio.to_thread(self._fetch_surgical, int(d_id), int(r_id))
        
        tasks = [fetch_async(idx) for idx in indices[0] if idx != -1]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = [r for r in responses if r is not None and not isinstance(r, Exception)]
        
        return self._rerank_results(query_emb, results) if rerank else results

    async def close(self):
        """Gracefully closes the persistent asynchronous HTTP client."""
        await self.async_client.aclose()