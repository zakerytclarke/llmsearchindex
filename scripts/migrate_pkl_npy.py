import pickle
import numpy as np

def convert_mapping(pkl_path, npy_path):
    print(f"Loading {pkl_path}... This may take a while and a lot of RAM.")
    with open(pkl_path, "rb") as f:
        # Load the original list of (dataset_id, row_id)
        mapping_list = pickle.load(f)
    
    print(f"Converting to NumPy array...")
    # Convert to a structured uint32 array to save space
    # dataset_id fits in uint8, row_id needs uint32
    # But for simplicity, a standard int64 array works fine
    mapping_array = np.array(mapping_list, dtype=np.int64)
    
    print(f"Saving to {npy_path}...")
    np.save(npy_path, mapping_array)
    print("Done!")

# Run the conversion
convert_mapping("./models/fineweb_mapping_full.pkl", "./models/fineweb_mapping_full.npy")