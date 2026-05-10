import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
import os
import sys

# Add custom_ext to path if needed, but it should be installed in site-packages
try:
    import custom_cuda_ops
except ImportError:
    print("Warning: custom_cuda_ops not found. Please install the custom PyTorch extension first.")
    custom_cuda_ops = None

class ControlledModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, max_len=128, hidden_dim=128):
        super().__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
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
        
    def forward(self, x, lengths=None, extract_intermediates=False):
        intermediates = {}
        use_custom = custom_cuda_ops is not None
        
        # K1: CUSTOM CUDA KERNEL - pad_truncate
        if lengths is not None and use_custom:
            x = custom_cuda_ops.pad_truncate(
                x.to(torch.int32), 
                lengths.to(torch.int32), 
                self.max_len, 
                0 # pad_token
            )
            x = x.to(torch.long)
        
        if use_custom:
            # K2: CUSTOM CUDA KERNEL - embedding_lookup (word embeddings)
            batch = x.size(0)
            seq_len = x.size(1)
            
            # Flatten tokens for the kernel, then reshape back
            flat_tokens = x.contiguous().view(-1).to(torch.int32)
            w_emb = custom_cuda_ops.embedding_lookup(
                flat_tokens,
                self.word_embed.weight.data,
                1  # unk_id
            )
            w_emb = w_emb.view(batch, seq_len, self.embed_dim)
            
            # K2 again: embedding_lookup (positional embeddings)
            pos_ids = torch.arange(seq_len, device=x.device).to(torch.int32)
            p_emb_flat = custom_cuda_ops.embedding_lookup(
                pos_ids,
                self.pos_embed.weight.data,
                0
            )
            # Broadcast: [seq_len, dim] -> [batch, seq_len, dim]
            p_emb = p_emb_flat.unsqueeze(0).expand(batch, -1, -1)
            
            # Element-wise addition
            x_emb = w_emb + p_emb
            if extract_intermediates: intermediates['01_embedding_out'] = x_emb.detach().cpu().numpy()
            
            # K4: CUSTOM CUDA KERNEL - weighted_mean_pooling
            # pool_weight is [max_len, 1] - broadcast to [batch, seq_len]
            pw = self.pool_weight.data.squeeze(-1)  # [max_len]
            pw_batch = pw.unsqueeze(0).expand(batch, -1).contiguous()  # [batch, seq_len]
            
            pooled = custom_cuda_ops.weighted_mean_pooling(
                x_emb.contiguous(),
                pw_batch
            )
            if extract_intermediates: intermediates['02_pooled_out'] = pooled.detach().cpu().numpy()
            
            # K10 + K5: CUSTOM CUDA KERNELS - gemm_tiled + bias_add (fc1)
            # nn.Linear computes: output = input @ weight.T + bias
            # Our GEMM kernel does C = A * B, so we need weight transposed
            fc1_weight_t = self.fc1.weight.data.t().contiguous()  # [embed_dim, hidden_dim]
            hidden = custom_cuda_ops.gemm_tiled(pooled.contiguous(), fc1_weight_t)
            hidden = custom_cuda_ops.bias_add(hidden, self.fc1.bias.data)
            if extract_intermediates: intermediates['03_fc1_out'] = hidden.detach().cpu().numpy()
            
            # K7 + K8 + K9: CUSTOM CUDA KERNELS - batchnorm pipeline
            bn_mean = custom_cuda_ops.batchnorm_mean(hidden.contiguous())
            bn_var = custom_cuda_ops.batchnorm_var(hidden.contiguous(), bn_mean)
            hidden_bn = custom_cuda_ops.batchnorm_apply(
                hidden.contiguous(),
                bn_mean,
                bn_var,
                self.bn1.weight.data,   # gamma
                self.bn1.bias.data,     # beta
                1e-5                     # eps
            )
            if extract_intermediates: intermediates['04_bn_out'] = hidden_bn.detach().cpu().numpy()
            
            # K6: CUSTOM CUDA KERNEL - leaky_relu
            act_out = custom_cuda_ops.leaky_relu(hidden_bn.contiguous(), 0.01)
            if extract_intermediates: intermediates['05_act_out'] = act_out.detach().cpu().numpy()
            
            # K11: CUSTOM CUDA KERNEL - logit_projection (fc2)
            # logit_projection expects weights as [hidden x classes]
            fc2_weight_t = self.fc2.weight.data.t().contiguous()  # [hidden_dim, 5]
            logits = custom_cuda_ops.logit_projection(
                act_out.contiguous(),
                fc2_weight_t
            )
            # Add fc2 bias manually
            logits = custom_cuda_ops.bias_add(logits, self.fc2.bias.data)
            if extract_intermediates: intermediates['06_logits'] = logits.detach().cpu().numpy()
            
            # K12 + K13 + K14: CUSTOM CUDA KERNELS - softmax pipeline
            row_max = custom_cuda_ops.softmax_row_max(logits.contiguous())
            row_sum = custom_cuda_ops.softmax_row_sum(logits.contiguous(), row_max)
            probs = custom_cuda_ops.softmax_normalize(logits.contiguous(), row_max, row_sum)
            if extract_intermediates: intermediates['07_probs'] = probs.detach().cpu().numpy()
            
        else:
            # ==================== FALLBACK: Standard PyTorch ====================
            seq_length = x.size(1)
            positions = torch.arange(seq_length, device=x.device).unsqueeze(0).expand_as(x)
            
            w_emb = self.word_embed(x)
            p_emb = self.pos_embed(positions)
            x_emb = w_emb + p_emb
            if extract_intermediates: intermediates['01_embedding_out'] = x_emb.detach().cpu().numpy()
            
            pooled = torch.sum(x_emb * self.pool_weight, dim=1)
            if extract_intermediates: intermediates['02_pooled_out'] = pooled.detach().cpu().numpy()
            
            hidden = self.fc1(pooled)
            if extract_intermediates: intermediates['03_fc1_out'] = hidden.detach().cpu().numpy()
            
            hidden_bn = self.bn1(hidden)
            if extract_intermediates: intermediates['04_bn_out'] = hidden_bn.detach().cpu().numpy()
            
            act_out = self.act(hidden_bn)
            if extract_intermediates: intermediates['05_act_out'] = act_out.detach().cpu().numpy()
            
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "scripts", "yelp_tokenized.npz")
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}")
        return
        
    data = np.load(data_path)
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
    os.makedirs(os.path.join(script_dir, "..", "weights"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(script_dir, "..", "weights", "controlled_model_weights.pth"))
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
