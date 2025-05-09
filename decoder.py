import torch.nn as nn
import torch

# Learnable Positional Encoding
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.position_embeddings = nn.Parameter(torch.randn(max_len, d_model))
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        position_embeddings = self.position_embeddings[:seq_len].unsqueeze(0)  # [1, seq_len, d_model]
        return x + position_embeddings

# Define a single decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def create_mask(self, sz, device):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)
    
    def forward(self, x, key_padding_mask=None):
        attn_mask = self.create_mask(x.size(1), x.device)
        residual1 = x

        attn_output, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask  # <-- new
        )
        x = self.norm1(residual1 + self.dropout(attn_output))
        
        # Second residual connection: skip feed-forward network
        residual2 = x
        
        # Feed-forward network
        ff_output = self.feed_forward(x)
        x = self.norm2(residual2 + self.dropout(ff_output))
        
        return x

# Define the decoder with residual connections and linear projection

class ImageCaptionDecoder(nn.Module):
    def __init__(self, embedding_dim=512, num_heads=8, vocab_size=32000, num_decoder_layers=6, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        
        # Positional encoding
        self.pos_encoder = LearnablePositionalEncoding(embedding_dim)
        
        # Stack of decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(embedding_dim, num_heads, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Linear projection to vocabulary size
        self.projection = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, combined_embeddings, image_patch_len, padding_mask=None):
        x = self.pos_encoder(combined_embeddings)

        for layer in self.decoder_layers:
            x = layer(x, key_padding_mask=padding_mask)

        caption_tokens = x[:, image_patch_len:]
        logits = self.projection(caption_tokens)
        return logits
