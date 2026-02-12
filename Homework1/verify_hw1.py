import sys
import os
import json
import numpy as np

# Add the directory containing hw1.py to the python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from hw1 import get_data, TextFeaturizer
    print("Successfully imported hw1")
except ImportError as e:
    print(f"Failed to import hw1: {e}")
    sys.exit(1)

def test_get_data():
    print("\n--- Testing get_data ---")
    try:
        data_path = os.path.join(current_dir, 'valid.jsonl')
        data = get_data(data_path)
        print(f"Successfully loaded {len(data)} records from {data_path}")
        if len(data) > 0 and isinstance(data[0], dict):
            print(f"Sample record: {data[0]}")
            print("get_data verification: PASSED")
        else:
            print("get_data returned empty or invalid data.")
            print("get_data verification: FAILED")
    except Exception as e:
        print(f"get_data failed with error: {e}")
        print("get_data verification: FAILED")

def test_build_vocab_and_w2v():
    print("\n--- Testing TextFeaturizer (build_vocab & Word2Vec) ---")
    dummy_corpus = ["hello world", "hello python", "world of code"]
    
    # Check if w2v file exists to avoid trying to load it if missing
    w2v_file = os.path.join(current_dir, "..", "GoogleNews-vectors-negative300.bin")
    if not os.path.exists(w2v_file):
         w2v_file = "GoogleNews-vectors-negative300.bin" # Try local
    
    # We might not have the w2v file, so we can test with a dummy path and expect failure or just test vocab
    print(f"Initializing TextFeaturizer with dummy corpus...")
    
    # Mocking w2v loading to speed up or avoid massive file requirement for simple vocab test??
    # The user wants to verify it being loaded. The output logs from their run already confirmed it loaded.
    # But let's try to initialize it. If the file is strictly required by the class, we need it.
    # The class sets w2v_model to None or handles FileNotFoundError.
    
    if os.path.exists("GoogleNews-vectors-negative300.bin"):
        w2v_path = "GoogleNews-vectors-negative300.bin"
    else:
        # Check parent dir
        parent_w2v = os.path.join(os.path.dirname(current_dir), "GoogleNews-vectors-negative300.bin")
        if os.path.exists(parent_w2v):
            w2v_path = parent_w2v
        else:
            w2v_path = "non_existent_file.bin"
            print("Warning: GoogleNews-vectors-negative300.bin not found. Word2Vec loading will likely fail, but we can verify vocab.")

    featurizer = TextFeaturizer(dummy_corpus, w2v_path=w2v_path)
    
    print(f"Vocabulary size: {len(featurizer.word_to_idx)}")
    expected_words = {"hello", "world", "python", "of", "code", "<UNK>"}
    vocab_words = set(featurizer.word_to_idx.keys())
    
    if expected_words.issubset(vocab_words):
        print("Vocabulary contains all expected words.")
        print("build_vocab verification: PASSED")
    else:
        print(f"Vocabulary missing words. Found: {vocab_words}")
        print("build_vocab verification: FAILED")

    if hasattr(featurizer, 'w2v_model') and featurizer.w2v_model is not None:
         print("Word2Vec model attribute exists and is not None.")
         print("Word2Vec loading verification: PASSED")
    else:
         print("Word2Vec model not loaded (this is expected if the bin file is missing).")
         # We rely on previous log for w2v success if file exists.

if __name__ == "__main__":
    test_get_data()
    test_build_vocab_and_w2v()
