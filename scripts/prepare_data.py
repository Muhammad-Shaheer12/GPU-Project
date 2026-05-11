import json
import numpy as np
import os
from tokenizer import SimpleTokenizer

def prepare_data(json_path, output_name="yelp_tokenized", max_length=128):
    tokenizer = SimpleTokenizer()
    
    # Step 1: Build Vocabulary from first 500k samples
    print(f"Building Vocabulary from {json_path}...")
    sample_texts = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                sample_texts.append(json.loads(line).get('text', ''))
                if i >= 500000: break 
            except: continue
    tokenizer.build_vocab(sample_texts)
    print(f"Vocabulary Size: {tokenizer.vocab_size}")

    # Step 2: Tokenize full dataset
    print("Tokenizing full dataset...")
    X_all, y_all = [], []
    with open(json_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                review = json.loads(line)
                X_all.append(tokenizer.encode(review.get('text', ''), max_length))
                y_all.append(int(review.get('stars', 3)))
                if i % 200000 == 0: print(f"Processed {i} reviews...")
            except: continue

    # Step 3: Save binary file
    output_path = f"{output_name}.npz"
    np.savez_compressed(output_path, X=np.array(X_all, dtype=np.int32), y=np.array(y_all, dtype=np.int32))
    print(f"Done! Created {output_path}")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    JSON_PATH = os.path.join(PROJECT_ROOT, "dataset", "yelp_academic_dataset_review.json")
    if os.path.exists(JSON_PATH):
        prepare_data(JSON_PATH)
    else:
        print(f"Error: {JSON_PATH} not found.")
