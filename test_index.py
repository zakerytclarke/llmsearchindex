import asyncio
import time
import argparse
import json
from llmsearchindex import LLMIndex

async def run_test(query: str, top_k: int):
    # --- 1. Initialization (Cold Start) ---
    print(f"🚀 Initializing LLMIndex (Syncing with HuggingFace Hub)...")
    init_start = time.time()
    
    # This triggers the hf_hub_download and model loading
    idx = LLMIndex()
    
    init_time = time.time() - init_start
    print(f"✅ Library Ready in {init_time:.2f}s")
    print(f"📊 Index Size: {idx.index.ntotal:,} vectors")
    print("-" * 50)

    # --- 2. Search Execution (The Fast Path) ---
    print(f"🔍 Searching for: '{query}'")
    search_start = time.time()
    
    # Running the optimized async path
    results = await idx.search_async(query, top_k=top_k)
    
    search_time = time.time() - search_start

    # --- 3. Display Results ---
    if not results:
        print("❌ No results found.")
    else:
        for i, res in enumerate(results, 1):
            source = res.get('source', 'unknown')
            url = res.get('url', 'No URL')
            text = res.get('text', '')[:200].replace('\n', ' ')
            
            print(f"\n[{i}] {source.upper()} | {url}")
            print(f"    Snippet: {text}...")

    # --- 4. Latency Breakdown ---
    print("\n" + "="*40)
    print(f"{'LATENCY METRICS':<25} | {'TIME':<10}")
    print("-" * 40)
    print(f"{'Model & Index Loading':<25} | {init_time:<10.4f}s")
    print(f"{'Total Search Execution':<25} | {search_time:<10.4f}s")
    print("="*40)

    # Cleanup
    await idx.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LLMIndex Latency")
    parser.add_argument("query", type=str, help="The search query")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    
    args = parser.parse_args()

    try:
        asyncio.run(run_test(args.query, args.k))
    except KeyboardInterrupt:
        pass