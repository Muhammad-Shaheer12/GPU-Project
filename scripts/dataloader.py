import json
import numpy as np

class OfflinePreProcessor:
    def __init__(self, json_path, output_name, tokenizer, max_length=128):
        self.json_path = json_path
        self.output_name = output_name
        self.tokenizer = tokenizer
        self.max_length = max_length

    def run(self, build_vocab_samples=500000):
        X_all = []
        y_all = []
        
        print(f"Step 1: Building Vocabulary from {self.json_path}...")
        # (Optional: Read a subset to build vocab fast)
        with open(self.json_path, 'r', encoding='utf-8') as f:
            sample_texts = []
            for i, line in enumerate(f):
                review = json.loads(line)
                sample_texts.append(review.get('text', ''))
                if i >= build_vocab_samples: break 
            self.tokenizer.build_vocab(sample_texts)
        
        print(f"Vocabulary Size: {self.tokenizer.vocab_size}")

        print("Step 2: Tokenizing full dataset...")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    review = json.loads(line)
                    text = review.get('text', '')
                    # Yelp ratings are 1-5.
                    rating = int(review.get('stars', 3))
                    
                    X_all.append(self.tokenizer.encode(text, self.max_length))
                    y_all.append(rating)
                    
                    if i % 100000 == 0:
                        print(f"Processed {i} reviews...")
                except:
                    continue

        print("Step 3: Saving binary file...")
        # Save the binary file right here in the main/ directory.
        output_path = f"{self.output_name}.npz"
        np.savez_compressed(output_path, X=np.array(X_all, dtype=np.int32), y=np.array(y_all, dtype=np.int32))
        print(f"Done! Created {output_path} inside the main/ directory.")