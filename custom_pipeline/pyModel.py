import torch
import torch.nn as nn
import os

try:
    import custom_cuda_ops
except ImportError:
    custom_cuda_ops = None

class ControlledModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, max_len=128, hidden_dim=128):
        super().__init__()
        self.max_len = max_len
        self.embed_dim = embed_dim
        
        # 1. Layers
        self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.pool_weight = nn.Parameter(torch.ones(max_len, 1) / max_len)
        self.pool_bias = nn.Parameter(torch.zeros(embed_dim))
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.act = nn.LeakyReLU(0.01)
        self.fc1 = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, 5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, lengths=None, return_argmax=False):
        use_custom = custom_cuda_ops is not None and not self.training
        
        if use_custom:
            # --- CUSTOM CUDA PIPELINE ---
            # K1: pad_truncate
            if lengths is not None:
                x = custom_cuda_ops.pad_truncate(x.to(torch.int32), lengths.to(torch.int32), self.max_len, 0).to(torch.long)
            
            batch, seq_len = x.size(0), x.size(1)
            
            # K2: embedding_lookup (Word + Position)
            w_emb = custom_cuda_ops.embedding_lookup(x.contiguous().view(-1).to(torch.int32), self.word_embed.weight.data, 1)
            w_emb = w_emb.view(batch, seq_len, self.embed_dim)
            
            pos_ids = torch.arange(seq_len, device=x.device).to(torch.int32)
            p_emb = custom_cuda_ops.embedding_lookup(pos_ids, self.pos_embed.weight.data, 0).unsqueeze(0)
            
            # K4: pooling
            pw = self.pool_weight.data.squeeze(-1).unsqueeze(0).expand(batch, -1).contiguous()
            pooled = custom_cuda_ops.weighted_mean_pooling((w_emb + p_emb).contiguous(), pw)
            
            # K16: Fused Bias + ReLU
            activated = custom_cuda_ops.fused_bias_leaky_relu(pooled.contiguous(), self.pool_bias.data, 0.01)
            
            # K7-9: BatchNorm
            mean = custom_cuda_ops.batchnorm_mean(activated.contiguous())
            var = custom_cuda_ops.batchnorm_var(activated.contiguous(), mean)
            bn_out = custom_cuda_ops.batchnorm_apply(activated.contiguous(), mean, var, self.bn1.weight.data, self.bn1.bias.data, 1e-5)
            
            # K10: Tiled GEMM
            hidden = custom_cuda_ops.gemm_tiled(bn_out.contiguous(), self.fc1.weight.data.t().contiguous())
            
            # K11: Projection + Bias
            logits = custom_cuda_ops.logit_projection(hidden.contiguous(), self.fc2.weight.data.t().contiguous())
            logits = custom_cuda_ops.bias_add(logits, self.fc2.bias.data)
            
            # K17: Fused Softmax
            probs = custom_cuda_ops.fused_softmax(logits.contiguous())
        else:
            # --- STANDARD PYTORCH FALLBACK ---
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
            x_emb = self.word_embed(x) + self.pos_embed(positions)
            pooled = torch.sum(x_emb * self.pool_weight, dim=1)
            activated = self.act(pooled + self.pool_bias)
            bn_out = self.bn1(activated)
            hidden = self.fc1(bn_out)
            logits = self.fc2(hidden)
            probs = self.softmax(logits)

        if return_argmax:
            if use_custom:
                return custom_cuda_ops.argmax(probs.contiguous()).to(torch.long) + 1
            return torch.argmax(probs, dim=1) + 1
            
        return probs
