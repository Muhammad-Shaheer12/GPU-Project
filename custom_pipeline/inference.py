import torch
import os
import sys

# Setup paths for Windows DLL loading and local extension
ext_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'custom_ext'))
sys.path.append(ext_path)

if sys.platform == 'win32':
    # Add torch lib for DLLs like torch_python.dll and cublas
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.exists(torch_lib_path):
        os.add_dll_directory(torch_lib_path)
    # Add extension path for the .pyd itself
    os.add_dll_directory(ext_path)

import numpy as np
from pyModel import ControlledModel

def run_inference():
    # 1. Setup Paths and Device
    script_dir = os.path.dirname(os.path.abspath(__file__))
    WEIGHTS_PATH = os.path.join(script_dir, "..", "weights", "controlled_model_weights.pth")
    DATA_PATH = os.path.join(script_dir, "..", "scripts", "yelp_tokenized.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Weights not found at {WEIGHTS_PATH}")
        return

    # 2. Load the Model and Weights
    data = np.load(DATA_PATH)
    vocab_size = int(np.max(data['X'])) + 1
    
    model = ControlledModel(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    print("Model and weights loaded successfully.")

    # 3. Grab samples from pre-tokenized data
    num_samples = 10
    X_samples = torch.tensor(data['X'][:num_samples], dtype=torch.long).to(device)
    y_ground_truth = data['y'][:num_samples]
    lengths = torch.full((num_samples,), 128, dtype=torch.int32).to(device)

    # 4. Run Inference
    print("Running optimized CUDA inference...")
    with torch.no_grad():
        # return_argmax=True uses the custom argmax kernel internally
        predictions = model(X_samples, lengths=lengths, return_argmax=True)

    # 5. Show Results
    print("\n--- Sentiment Analysis Results ---")
    for i in range(num_samples):
        print(f"Review {i+1}:")
        print(f"  -> Predicted Rating: {predictions[i].item()} stars")
        print(f"  -> Actual Rating:    {y_ground_truth[i]} stars")
        print("-" * 40)

if __name__ == "__main__":
    run_inference()
