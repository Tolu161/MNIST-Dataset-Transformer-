import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return x + self.positional_embedding[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.q_linear(q).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(output)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.ffn(src)
        src = src + self.dropout(src2)
        return self.norm2(src)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.cross_attn(tgt, memory, memory)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout(tgt2)
        return self.norm3(tgt)

class TransformerEncoderDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super().__init__()
        self.embedding = nn.Linear(14 * 14, d_model)
        self.pos_encoder = LearnablePositionalEncoding(max_len=16, d_model=d_model)
        self.encoder = nn.ModuleList([TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_encoder_layers)])
        self.decoder = nn.ModuleList([TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_decoder_layers)])
        self.fc_out = nn.Linear(d_model, sp.get_piece_size())

    def forward(self, src, tgt, tgt_mask=None):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        for layer in self.encoder:
            src = layer(src)
        tgt = self.pos_encoder(tgt)
        for layer in self.decoder:
            tgt = layer(tgt, src, tgt_mask)
        return self.fc_out(tgt)