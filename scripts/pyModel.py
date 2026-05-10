import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time

class ControlledModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, max_len=128, hidden_dim=128):
        super().__init__()
        # Data Preparation: Embeddings & Positional Encoding
        self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        
        # Feature Extraction: Weighted Mean Pooling
        # A learnable weight matrix to compress sequences into vectors
        self.pool_weight = nn.Parameter(torch.ones(max_len, 1) / max_len)
        
        # Feature Extraction: Neural Layers
        self.fc1 = nn.Linear(embed_dim, hidden_dim) # GEMM Kernel
        self.bn1 = nn.BatchNorm1d(hidden_dim)       # Batch Norm Kernel
        self.act = nn.LeakyReLU(0.01)               # Leaky ReLU Kernel
        
        # Classification
        self.fc2 = nn.Linear(hidden_dim, 5)         # GEMV Kernel (5 classes for 1-5 stars)
        self.softmax = nn.Softmax(dim=1)            # Softmax Pipeline Kernel
        
    def forward(self, x, extract_intermediates=False):
        intermediates = {}
        
        # 1. Padding & Positional Encoding
        seq_length = x.size(1)
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0).expand_as(x)
        
        w_emb = self.word_embed(x)
        p_emb = self.pos_embed(positions)
        x_emb = w_emb + p_emb # Element-wise addition
        if extract_intermediates: intermediates['01_embedding_out'] = x_emb.detach().cpu().numpy()
        
        # 2. Weighted Mean Pooling
        pooled = torch.sum(x_emb * self.pool_weight, dim=1)
        if extract_intermediates: intermediates['02_pooled_out'] = pooled.detach().cpu().numpy()
        
        # 3. Hidden Layer (GEMM -> BN -> Activation)
        hidden = self.fc1(pooled)
        if extract_intermediates: intermediates['03_fc1_out'] = hidden.detach().cpu().numpy()
        
        hidden_bn = self.bn1(hidden)
        if extract_intermediates: intermediates['04_bn_out'] = hidden_bn.detach().cpu().numpy()
        
        act_out = self.act(hidden_bn)
        if extract_intermediates: intermediates['05_act_out'] = act_out.detach().cpu().numpy()
        
        # 4. Classification & Softmax
        logits = self.fc2(act_out)
        if extract_intermediates: intermediates['06_logits'] = logits.detach().cpu().numpy()
        
        probs = self.softmax(logits)
        if extract_intermediates: intermediates['07_probs'] = probs.detach().cpu().numpy()
        
        if extract_intermediates:
            return probs, intermediates
        return probs

# 2. Execution & Training Pipeline
def run_training():
    print("Loading binary dataset into System RAM...")
    data = np.load("yelp_tokenized.npz")
    X_np = data['X']
    y_np = data['y']
    
    # CRITICAL: Yelp ratings are 1-5. PyTorch cross-entropy requires classes 0-4.
    y_np = y_np - 1 
    
    # Dynamic vocab size calculation (max token ID + 1)
    vocab_size = int(np.max(X_np)) + 1
    print(f"Detected Vocab Size: {vocab_size}")
    
    # Convert to PyTorch Tensors
    X_tensor = torch.tensor(X_np, dtype=torch.long)
    y_tensor = torch.tensor(y_np, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=2048, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = ControlledModel(vocab_size=vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Starting Training on {len(X_np)} records...")
    model.train()
    start_time = time.time()
    
    # We will train for just 1 Epoch (7 million rows is enough for the baseline)
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")
            
    print(f"Training Complete in {(time.time() - start_time)/60:.2f} minutes.")
    
    # Save the trained model weights
    torch.save(model.state_dict(), "../weights/controlled_model_weights.pth")
    print("Saved Controlled Model weights.")

    # 3. Extract Intermediate Tensors for C++ Debugging
    print("Extracting intermediate tensors for CUDA verification...")
    model.eval()
    
    # Grab just ONE single batch to use as your C++ test case
    test_inputs, test_targets = next(iter(DataLoader(dataset, batch_size=2048, shuffle=False)))
    test_inputs = test_inputs.to(device)
    
    with torch.no_grad():
        _, intermediates = model(test_inputs, extract_intermediates=True)
        
    # Save the input batch and all intermediate math to a new .npz file
    save_dict = {
        'input_tokens': test_inputs.cpu().numpy(),
        'expected_labels': test_targets.cpu().numpy()
    }
    save_dict.update(intermediates)
    
    np.savez_compressed("cuda_debug_tensors.npz", **save_dict)
    print("Saved 'cuda_debug_tensors.npz'. Your C++ benchmark is ready!")

if __name__ == "__main__":
    run_training()