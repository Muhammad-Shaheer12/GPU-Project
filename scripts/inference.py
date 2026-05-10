import torch
import numpy as np
import os
from pyModel import ControlledModel # Import the architecture you built

def run_inference():
    # 1. Setup Paths and Device
    # Look for the weights in the root directory as we discussed
    WEIGHTS_PATH = "../weights/controlled_model_weights.pth"
    DATA_PATH = "yelp_tokenized.npz"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Weights not found at {WEIGHTS_PATH}")
        return

    # 2. Load the Model and Weights
    # We need to detect the vocab size from the data again to initialize
    data = np.load(DATA_PATH)
    vocab_size = int(np.max(data['X'])) + 1
    
    model = ControlledModel(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval() # Set to evaluation mode (disables Dropout/Batchnorm training)
    print("Model and weights loaded successfully.")

    # 3. Grab a few samples from your tokenized file
    X_samples = data['X'][:5]  # Take first 5 reviews
    y_ground_truth = data['y'][:5]
    
    inputs = torch.tensor(X_samples, dtype=torch.long).to(device)

    # 4. Run the Forward Pass
    with torch.no_grad():
        probs = model(inputs)
        # argmax determines the final 1-5 star prediction
        predictions = torch.argmax(probs, dim=1) + 1 

    # 5. Show Results
    print("\n--- Sentiment Analysis Results ---")
    for i in range(len(X_samples)):
        print(f"Review {i+1}:")
        print(f"  -> Predicted Rating: {predictions[i].item()} stars")
        print(f"  -> Actual Rating:    {y_ground_truth[i]} stars")
        print("-" * 30)

if __name__ == "__main__":
    run_inference()