import torch
import numpy as np
import json
import os
from pyModel import ControlledModel # Import the architecture you built

NUM_SAMPLES = 5

def load_review_texts(json_path, count):
    """Load the first `count` raw review texts from the original Yelp JSON."""
    texts = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= count:
                break
            review = json.loads(line)
            texts.append(review.get('text', ''))
    return texts

def run_inference():
    # 1. Setup Paths and Device
    script_dir = os.path.dirname(os.path.abspath(__file__))
    WEIGHTS_PATH = os.path.join(script_dir, "..", "weights", "controlled_model_weights.pth")
    DATA_PATH = os.path.join(script_dir, "..", "scripts", "yelp_tokenized.npz")
    JSON_PATH = os.path.join(script_dir, "..", "dataset", "yelp_academic_dataset_review.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Weights not found at {WEIGHTS_PATH}")
        return

    # 2. Load the Model and Weights
    data = np.load(DATA_PATH)
    vocab_size = int(np.max(data['X'])) + 1
    
    model = ControlledModel(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval() # Set to evaluation mode (disables Dropout/Batchnorm training)
    print("Model and weights loaded successfully.")

    # 3. Load the original review texts from the Yelp JSON
    review_texts = []
    if os.path.exists(JSON_PATH):
        print("Loading original review texts...")
        review_texts = load_review_texts(JSON_PATH, NUM_SAMPLES)
    else:
        print(f"Warning: Original JSON not found at {JSON_PATH}. Showing predictions without text.")

    # 4. Grab a few samples from your tokenized file
    X_samples = data['X'][:NUM_SAMPLES]
    y_ground_truth = data['y'][:NUM_SAMPLES]
    
    inputs = torch.tensor(X_samples, dtype=torch.long).to(device)
    
    # Simulate finding the true lengths of the sentences before they were padded
    # (Since our data is already padded, we'll just say they are length 128 for the sake of the kernel test)
    lengths = torch.tensor([128]*NUM_SAMPLES, dtype=torch.int32).to(device)

    # 5. Run the Forward Pass (The custom kernels are called inside `model()`)
    print("Running forward pass (which will execute the custom CUDA kernels)...")
    with torch.no_grad():
        probs = model(inputs, lengths=lengths)
        
        # ---------------------------------------------------------
        # K15: CUSTOM CUDA KERNEL - argmax
        # ---------------------------------------------------------
        try:
            import custom_cuda_ops
            # custom_cuda_ops.argmax returns 0-indexed values
            predictions = custom_cuda_ops.argmax(probs.contiguous())
            predictions = predictions.to(torch.long) + 1
        except (ImportError, AttributeError):
            # Fallback to standard PyTorch
            predictions = torch.argmax(probs, dim=1) + 1 

    # 6. Show Results
    print("\n--- Sentiment Analysis Results ---")
    for i in range(len(X_samples)):
        print(f"\nReview {i+1}:")
        if i < len(review_texts):
            # Show a preview of the review (first 200 chars)
            preview = review_texts[i][:200]
            if len(review_texts[i]) > 200:
                preview += "..."
            print(f"  Text: \"{preview}\"")
        print(f"  -> Predicted Rating: {predictions[i].item()} stars")
        print(f"  -> Actual Rating:    {y_ground_truth[i]} stars")
        print("-" * 60)

if __name__ == "__main__":
    run_inference()
