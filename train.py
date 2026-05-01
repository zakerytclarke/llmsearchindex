#!/usr/bin/env python3
"""
Training script for llmindex - Trains a FAISS-based search index on Wikipedia and FineWeb.

This script streams both the Wikipedia and FineWeb datasets and trains a search model with:
- Sentence Transformers (all-MiniLM-L6-v2) for embedding
- FAISS binary index for efficient semantic search
- PCA dimensionality reduction for memory efficiency
- Checkpointing for fault tolerance

Indexing order: Wikipedia (20231101.en) → FineWeb (CC-MAIN-2025-26)

Usage:
    python train.py --save-dir ./models/

The script will save:
    - pca_model_64.pkl: Trained PCA model
    - fineweb_urls_full.txt: URLs for each indexed document
    - fineweb_full_final.index: Final FAISS binary index
"""

import os
import argparse
import time
import pickle
import torch
import faiss
import numpy as np
import psutil
from pathlib import Path
from datasets import load_dataset
from datasets import DownloadConfig
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import IncrementalPCA


class SearchIndexTrainer:
    """Trains a FAISS-based semantic search index on Wikipedia and FineWeb datasets."""
    
    # Dataset IDs
    DATASET_WIKIPEDIA = 0
    DATASET_FINEWEB = 1
    DATASET_MAP = {
        "wikimedia/wikipedia": DATASET_WIKIPEDIA,
        "HuggingFaceFW/fineweb": DATASET_FINEWEB,
    }
    
    def __init__(
        self,
        save_dir="./models",
        target_indexed_docs=None,
        pca_dim=64,
        encode_batch_size=4096,
        pca_train_target=500_000,
        checkpoint_interval=2_500_000,
    ):
        """
        Initialize the trainer.
        
        Args:
            save_dir: Directory to save models and indices
            target_indexed_docs: Total documents to index across all datasets
            pca_dim: PCA dimensionality (64 is recommended)
            encode_batch_size: Batch size for encoding
            pca_train_target: Documents to use for PCA training
            checkpoint_interval: Interval for saving checkpoints
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_indexed_docs = target_indexed_docs
        self.pca_dim = pca_dim
        self.encode_batch_size = encode_batch_size
        self.pca_train_target = pca_train_target
        self.checkpoint_interval = checkpoint_interval
        
        self.pca_path = self.save_dir / f"pca_model_{pca_dim}.pkl"
        split_tag = f"{target_indexed_docs // 1_000_000}m" if target_indexed_docs is not None else "full"
        self.mapping_path = self.save_dir / f"fineweb_mapping_{split_tag}.pkl"
        self.final_index_path = self.save_dir / f"fineweb_{split_tag}_final.index"

        self.download_config = DownloadConfig(max_retries=10)
        self.max_dataset_retries = 10
        self.dataset_retry_base = 2
        
        # Initialize model and PCA
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Initializing model on {self.device}...")
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.model.half()  # FP16 for speed on GPU
        
        # Enable torch.compile if available (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.device == "cuda":
            try:
                self.model = torch.compile(self.model)
                print("⚡ Torch compile enabled")
            except Exception:
                pass
        
        # PCA initialization
        if self.pca_path.exists():
            print(f"✅ Found pre-trained PCA model! Loading from {self.pca_path}")
            with open(self.pca_path, "rb") as f:
                self.ipca = pickle.load(f)
            self.pca_trained = True
        else:
            print("⚠️  No PCA model found. Will train a new one on the fly...")
            self.ipca = IncrementalPCA(n_components=pca_dim, batch_size=50000)
            self.pca_trained = False
        
        # FAISS index setup
        self.index = faiss.IndexBinaryFlat(pca_dim)
        
        # Mapping storage: list of (dataset_id, row_id) for each FAISS vector
        self.mapping = []
        
        # Resume from latest checkpoint when available
        checkpoint = self._find_latest_checkpoint()
        if checkpoint is not None:
            self.indexed_count = int(checkpoint.stem.split("_")[1])
            self.index = faiss.read_index_binary(str(checkpoint))
            
            # Load mapping if it exists
            if self.mapping_path.exists():
                with open(self.mapping_path, "rb") as f:
                    self.mapping = pickle.load(f)
                print(f"✅ Loaded mapping with {len(self.mapping)} entries")
            
            print(f"✅ Resuming from checkpoint {checkpoint} ({self.indexed_count:,} docs indexed)")
        else:
            self.indexed_count = 0

        # Counters
        self.training_count = 0
        self.start_time = None
    
    @staticmethod
    def get_memory_usage():
        """Get current memory usage in GB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)
    
    def train(self):
        """
        Train the search index on Wikipedia and FineWeb sequentially.
        
        Always indexes:
        1. Wikipedia (20231101.en snapshot)
        2. FineWeb (CC-MAIN-2025-26)
        
        Resumable from checkpoints.
        """
        datasets = [
            ("wikimedia/wikipedia", "20231101.en", self.DATASET_WIKIPEDIA),
            ("HuggingFaceFW/fineweb", "CC-MAIN-2025-26", self.DATASET_FINEWEB),
        ]
        
        if not self.start_time:
            self.start_time = time.time()
        
        target_msg = f"full split" if self.target_indexed_docs is None else f"{self.target_indexed_docs:,} documents"
        print(f"\n🚀 Starting training pipeline for {target_msg}...\n")
        
        # Calculate how many rows from each dataset have already been indexed
        dataset_row_counts = {self.DATASET_WIKIPEDIA: 0, self.DATASET_FINEWEB: 0}
        for dataset_id, row_id in self.mapping:
            dataset_row_counts[dataset_id] = max(dataset_row_counts[dataset_id], row_id + 1)
        
        if self.indexed_count > 0:
            print(f"Resuming from checkpoint: Wikipedia: {dataset_row_counts[self.DATASET_WIKIPEDIA]:,}, FineWeb: {dataset_row_counts[self.DATASET_FINEWEB]:,}")
        
        try:
            for dataset_name, split_name, dataset_id in datasets:
                # Calculate the starting row for this dataset
                start_row = dataset_row_counts.get(dataset_id, 0)
                
                print(f"\n📚 Streaming dataset: {dataset_name}/{split_name}...")
                if start_row > 0:
                    print(f"   Starting from row {start_row:,}")
                
                batch_text = []
                batch_metadata = []  # Store (dataset_id, row_id) for each item in batch
                
                row_counter = start_row  # Start from where we left off
                for example in self._stream_examples(dataset_name, split_name, start_offset=start_row):
                    # Extract text based on dataset type
                    if dataset_name == "wikimedia/wikipedia":
                        text = example.get('text', '')
                    else:
                        text = example.get('text', '')
                    
                    batch_text.append(text)
                    batch_metadata.append((dataset_id, row_counter))
                    row_counter += 1
                    
                    if len(batch_text) == self.encode_batch_size:
                        # 1. Encode batch on GPU
                        with torch.no_grad():
                            embeddings = self.model.encode(
                                batch_text,
                                batch_size=self.encode_batch_size,
                                convert_to_numpy=True,
                                show_progress_bar=False
                            )
                        
                        # 2. PCA Training Phase
                        if not self.pca_trained:
                            self.ipca.partial_fit(embeddings)
                            self.training_count += len(batch_text)
                            
                            if self.training_count % 100_000 < self.encode_batch_size:
                                mem = self.get_memory_usage()
                                print(f"[PCA TRAINING] {self.training_count:,} / {self.pca_train_target:,} docs | RAM: {mem:.2f}GB")
                            
                            if self.training_count >= self.pca_train_target:
                                self.pca_trained = True
                                print("\n🧠 PCA training complete! Locking model...")
                                with open(self.pca_path, "wb") as f:
                                    pickle.dump(self.ipca, f)
                                print(f"✅ PCA model saved to {self.pca_path}")
                                print("⚡ Shifting to high-speed indexing phase...\n")
                        
                        # 3. Indexing Phase
                        if self.pca_trained:
                            reduced = self.ipca.transform(embeddings)
                            binary_vectors = np.packbits(reduced > 0, axis=-1)
                            
                            self.index.add(binary_vectors)
                            self.mapping.extend(batch_metadata)
                            
                            self.indexed_count += len(batch_text)
                            
                            # Logging
                            if self.indexed_count % 100_000 < self.encode_batch_size:
                                elapsed = time.time() - self.start_time
                                rate = self.indexed_count / elapsed if elapsed > 0 else 0
                                mem = self.get_memory_usage()
                                if self.target_indexed_docs is not None and rate > 0:
                                    eta_hours = (self.target_indexed_docs - self.indexed_count) / rate / 3600
                                    print(f"[INDEXING] {self.indexed_count:,} | Rate: {rate:.0f} docs/sec | RAM: {mem:.2f}GB | ETA: {eta_hours:.1f}h")
                                else:
                                    print(f"[INDEXING] {self.indexed_count:,} | Rate: {rate:.0f} docs/sec | RAM: {mem:.2f}GB")
                            
                            # Checkpointing
                            if self.indexed_count % self.checkpoint_interval < self.encode_batch_size and self.indexed_count > 0:
                                cp_name = self.save_dir / f"checkpoint_{self.indexed_count}.index"
                                print(f"\n💾 Saving checkpoint: {cp_name}")
                                faiss.write_index_binary(self.index, str(cp_name))
                                
                                # Save mapping alongside checkpoint
                                mapping_cp = self.save_dir / f"checkpoint_{self.indexed_count}_mapping.pkl"
                                with open(mapping_cp, "wb") as f:
                                    pickle.dump(self.mapping, f)
                                
                                mem = self.get_memory_usage()
                                print(f"✅ Checkpoint secured. RAM: {mem:.2f}GB\n")
                            
                            if self.target_indexed_docs is not None and self.indexed_count >= self.target_indexed_docs:
                                print(f"\n✅ Reached target of {self.target_indexed_docs:,} documents!")
                                break
                        
                        # Clear batch
                        batch_text = []
                        batch_metadata = []
                
                if self.target_indexed_docs is not None and self.indexed_count >= self.target_indexed_docs:
                    break
        
        finally:
            pass
        
        # Save final index
        self._save_final_index()


    def _find_latest_checkpoint(self):
        """Return the most recent checkpoint file, if any."""
        checkpoints = sorted(
            [p for p in self.save_dir.glob("checkpoint_*.index") if p.name.startswith("checkpoint_")],
            key=lambda p: int(p.stem.split("_")[1])
        )
        return checkpoints[-1] if checkpoints else None

    def _load_streaming_dataset(self, dataset_name, split_name, start_offset=0):
        dataset = load_dataset(
            dataset_name,
            name=split_name,
            split="train",
            streaming=True,
            download_config=self.download_config,
        )
        if start_offset > 0:
            print(f"⏩ Skipping {start_offset:,} already indexed documents...")
            dataset = dataset.skip(start_offset)
        return dataset

    def _stream_examples(self, dataset_name, split_name, start_offset=0):
        """Yield examples from the dataset with retry support on transient stream failures."""
        retries = 0
        while True:
            if self.target_indexed_docs is not None and self.indexed_count >= self.target_indexed_docs:
                break
            try:
                dataset = self._load_streaming_dataset(dataset_name, split_name, start_offset=start_offset)
                for example in dataset:
                    yield example
                break
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                retries += 1
                if retries > self.max_dataset_retries:
                    print(f"\n❌ Failed after {self.max_dataset_retries} retries. Last error: {exc}")
                    raise
                delay = min(30, self.dataset_retry_base * (2 ** (retries - 1)))
                print(f"\n⚠️ Dataset stream error: {type(exc).__name__}: {exc}")
                print(f"Retrying in {delay}s ({retries}/{self.max_dataset_retries})...")
                time.sleep(delay)
                continue

    def _save_final_index(self):
        """Save the final trained index and mapping."""
        print(f"\n💾 Saving final index to {self.final_index_path}...")
        faiss.write_index_binary(self.index, str(self.final_index_path))
        
        # Save mapping
        with open(self.mapping_path, "wb") as f:
            pickle.dump(self.mapping, f)
        print(f"✅ Saved mapping with {len(self.mapping)} entries to {self.mapping_path}")
        
        elapsed = time.time() - self.start_time
        elapsed_hours = elapsed / 3600
        
        print(f"\n{'='*60}")
        print(f"✅ TRAINING COMPLETE!")
        print(f"{'='*60}")
        print(f"Time elapsed: {elapsed_hours:.2f} hours")
        print(f"Documents indexed: {self.indexed_count:,}")
        print(f"Index vectors: {self.index.ntotal:,}")
        print(f"Mapping entries: {len(self.mapping):,}")
        print(f"Final memory usage: {self.get_memory_usage():.2f}GB")
        print(f"\nSaved files:")
        print(f"  - {self.pca_path}")
        print(f"  - {self.mapping_path}")
        print(f"  - {self.final_index_path}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Train a FAISS-based search index on Wikipedia and FineWeb")
    parser.add_argument(
        "--target-docs",
        type=int,
        default=None,
        help="Optional cap on documents to index across all datasets. Omit to index the full splits.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./models",
        help="Directory to save models (default: ./models)",
    )
    parser.add_argument(
        "--pca-dim",
        type=int,
        default=64,
        help="PCA dimensionality (default: 64)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Encoding batch size (default: 4096)",
    )
    parser.add_argument(
        "--pca-train-target",
        type=int,
        default=500_000,
        help="Documents for PCA training (default: 500k)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=2_500_000,
        help="Save checkpoint every N docs (default: 2.5M)",
    )
    
    args = parser.parse_args()
    
    trainer = SearchIndexTrainer(
        save_dir=args.save_dir,
        target_indexed_docs=args.target_docs,
        pca_dim=args.pca_dim,
        encode_batch_size=args.batch_size,
        pca_train_target=args.pca_train_target,
        checkpoint_interval=args.checkpoint_interval,
    )
    
    # Train on Wikipedia first, then FineWeb (default behavior)
    trainer.train()


if __name__ == "__main__":
    main()
