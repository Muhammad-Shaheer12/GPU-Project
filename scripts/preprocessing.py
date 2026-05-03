import os
import sys

from tokenizer import SimpleTokenizer
from dataloader import OfflinePreProcessor

if __name__ == "__main__":
    # We find the main path, then navigate to the dataset path.
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
    JSON_FILE = "yelp_academic_dataset_review.json"
    INPUT_PATH = os.path.join(DATA_DIR, JSON_FILE)
    
    # 2. Verify the path exists (always good practice)
    if not os.path.exists(INPUT_PATH):
        print(f"Error: The file {INPUT_PATH} was not found.")
        print("Make sure you have extracted yelp_dataset.tar into the 'dataset' folder.")
        sys.exit(1)

    # 3. Create the necessary objects
    tokenizer = SimpleTokenizer()
    
    # 4. Set up the offline pre-processor
    processor = OfflinePreProcessor(json_path=INPUT_PATH,output_name="yelp_tokenized",tokenizer=tokenizer,max_length=128)

    # 5. Execute the process
    print("Starting offline tokenization pipeline...")
    processor.run(build_vocab_samples=500000)
    print("Tokenization pipeline complete.")