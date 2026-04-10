import torch
import torch.nn as nn


class PositionEmbeddingSine3D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, embedding_size, temperature=10000, n_points=1):
        super(PositionEmbeddingSine3D, self).__init__()
        self.embedding_size = embedding_size
        self.temperature = temperature
        self.n_points = n_points

    def forward(self, x):
        num_feats = self.embedding_size // (3 * self.n_points)

        if self.n_points > 1:
            x = x.flatten(1,2)
        B, N, _ = x.shape

        dim_t = torch.arange(num_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_feats)
    
        pos_x = x[:, :, 0][..., None] / dim_t
        pos_y = x[:, :, 1][..., None] / dim_t
        pos_z = x[:, :, 2][..., None] / dim_t
        if num_feats % 2 != 0:
            pos_x = torch.cat([pos_x, torch.zeros(B, N, 1).to(pos_x.device)], dim=-1)
            pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)[..., :-1]
            pos_y = torch.cat([pos_y, torch.zeros(B, N, 1).to(pos_y.device)], dim=-1)
            pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)[..., :-1]
            pos_z = torch.cat([pos_z, torch.zeros(B, N, 1).to(pos_z.device)], dim=-1)
            pos_z = torch.stack((pos_z[:, :, 0::2].sin(), pos_z[:, :, 1::2].cos()), dim=3).flatten(2)[..., :-1]
        else:
            pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
            pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
            pos_z = torch.stack((pos_z[:, :, 0::2].sin(), pos_z[:, :, 1::2].cos()), dim=3).flatten(2)

        pos  = torch.cat([pos_x, pos_y, pos_z], dim=2)
        if self.n_points > 1:
            pos = pos.view(B, N // self.n_points, self.n_points * 3 * num_feats)

        out = torch.zeros((B, N // self.n_points, self.embedding_size), dtype=x.dtype, device=x.device)
        out[:, :, :pos.shape[2]] = pos

        return out

class RoPE(nn.Module):
    """
    Implements Rotary Positional Encoding (RoPE) for Transformers.
    RoPE is a positional encoding that incorporates rotation-invariance.
    """

    def __init__(self, dim, max_len=512):
        super(RoPE, self).__init__()
        self.dim = dim
        self.max_len = max_len
        self.inv_freq = 1. / (10000**(torch.arange(0, dim, 2).float() / dim))
        
        # Precompute the RoPE matrix for position encodings
        self.position = torch.arange(0, max_len).unsqueeze(1)  # Shape: (max_len, 1)
        self.freqs = self.position * self.inv_freq  # Shape: (max_len, dim/2)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, dim)
        """
        # Extract the even and odd dimension indices
        seq_len = x.size(1)
        
        # Compute sin and cos for the frequencies
        freqs = self.freqs[:seq_len]  # Shape: (seq_len, dim/2)
        sin_freqs = torch.sin(freqs)  # Shape: (seq_len, dim/2)
        cos_freqs = torch.cos(freqs)  # Shape: (seq_len, dim/2)

        # Expand to match the batch size and the model dimensions
        sin_freqs = sin_freqs.unsqueeze(0).expand(x.size(0), -1, -1)  # Shape: (batch_size, seq_len, dim/2)
        cos_freqs = cos_freqs.unsqueeze(0).expand(x.size(0), -1, -1)  # Shape: (batch_size, seq_len, dim/2)

        # Create the full sin and cos tensors
        x_sin = x[:, :, ::2] * sin_freqs  # Apply sin to even dimensions
        x_cos = x[:, :, 1::2] * cos_freqs  # Apply cos to odd dimensions

        # Combine the results
        x_rotary = torch.cat([x_sin, x_cos], dim=-1)  # Shape: (batch_size, seq_len, dim)

        return x_rotary



class PositionEmbeddingMLP(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, embedding_size, temperature=10000, n_points=1):
        super(PositionEmbeddingMLP, self).__init__()
        self.embedding_size = embedding_size
        self.temperature = temperature
        self.n_points = n_points
    def forward(self, x,mlp):
        num_feats = self.embedding_size // (3 * self.n_points)
        if self.n_points > 1:
            x = x.flatten(1,2)
        B, N, _ = x.shape
        assert mlp is not None, "mlp_pe should not be None"
        pos = mlp(x) 
        if self.n_points > 1:
            pos = pos.view(B, N // self.n_points, self.n_points * 3 * num_feats)

        out = torch.zeros((B, N // self.n_points, self.embedding_size), dtype=x.dtype, device=x.device)
        out[:, :, :pos.shape[2]] = pos

        return out

