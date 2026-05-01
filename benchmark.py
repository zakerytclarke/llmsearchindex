import asyncio
import time
import random
from llmsearchindex import LLMIndex

async def run_benchmark(top_k: int = 5):
    # --- 1. Define the Sliced-Food Query Pool ---
    query_pool = [
        "Who invented sliced bread?",
        "Who invented sliced ham?",
        "Who invented sliced cheese?",
        "Who invented sliced turkey?",
        "Who invented sliced beef?",
        "Who invented sliced bacon?",
        "Who invented sliced potatoes?",
        "Who invented sliced apples?",
        "Who invented sliced pickles?",
        "Who invented sliced tomatoes?"
    ]

    # --- 2. Initialization (Cold Start) ---
    print(f"🚀 Initializing LLMIndex (Syncing with HuggingFace Hub)...")
    init_start = time.time()
    
    # This triggers the hf_hub_download, model loading, and map building
    idx = LLMIndex()
    
    init_time = time.time() - init_start
    print(f"✅ Library Ready in {init_time:.2f}s")
    print(f"📊 Index Size: {idx.index.ntotal:,} vectors")
    print("-" * 50)

    # --- 3. Load Testing Loop (500 Queries) ---
    print(f"\n🚀 COMMENCING 500-QUERY BENCHMARK")
    print(f"Sampling from {len(query_pool)} 'sliced' queries | top_k: {top_k}")
    print("-" * 50)
    
    successes = 0
    failures = 0
    total_results = 0
    total_search_time = 0.0
    
    load_test_start = time.time()

    for i in range(1, 501):
        # Pick a random query from the pool for this iteration
        current_query = random.choice(query_pool)
        
        iter_start = time.time()
        try:
            results = await idx.search_async(current_query, top_k=top_k)
            iter_time = time.time() - iter_start
            total_search_time += iter_time
            
            num_results = len(results) if results else 0
            total_results += num_results
            
            if num_results > 0:
                successes += 1
                # Truncate the query for clean terminal output if it's long
                display_query = (current_query[:30] + '...') if len(current_query) > 30 else current_query
                print(f"[{i:03d}/500] ✅ {iter_time:.2f}s | Found {num_results} | Q: '{display_query}'")
            else:
                failures += 1
                print(f"[{i:03d}/500] ⚠️ {iter_time:.2f}s | 0 results   | Q: '{current_query}'")
                
        except Exception as e:
            iter_time = time.time() - iter_start
            failures += 1
            print(f"[{i:03d}/500] ❌ {iter_time:.2f}s | FAILED: {str(e)}")

    load_test_time = time.time() - load_test_start

    # --- 4. Latency & Performance Breakdown ---
    print("\n" + "="*40)
    print("BENCHMARK COMPLETE")
    print("="*40)
    print(f"{'Total Queries':<25} | 500")
    print(f"{'Successful Queries':<25} | {successes}")
    print(f"{'Failed/Empty Queries':<25} | {failures}")
    print(f"{'Total Results Fetched':<25} | {total_results}")
    print("-" * 40)
    print(f"{'Index Initialization':<25} | {init_time:.2f}s")
    print(f"{'Total Test Duration':<25} | {load_test_time:.2f}s")
    print(f"{'Avg Query Latency':<25} | {total_search_time/500:.2f}s per query")
    print("="*40)

    # Cleanup
    await idx.close()

if __name__ == "__main__":
    try:
        # Simply run the benchmark script; no parameters required
        asyncio.run(run_benchmark(top_k=5))
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")