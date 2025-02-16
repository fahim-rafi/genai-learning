import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.utils.data import Dataset
import requests


class CharacterTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        return ''.join([self.idx_to_char[idx] for idx in indices])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=512):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return torch.matmul(attention, V)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        Q = self.W_q(x).view(batch_size, -1, self.num_heads,
                             self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads,
                             self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads,
                             self.d_k).transpose(1, 2)
        x = self.scaled_dot_product_attention(Q, K, V, mask)
        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.num_heads * self.d_k)
        return self.W_o(x)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x, mask=None):
        x = self.norm1(x + self.attention(x, mask))
        x = self.norm2(x + self.feed_forward(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=4, num_layers=4):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model)
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.token_embedding(x)
        x = self.position_encoding(x)
        for layer in self.transformer_layers:
            x = layer(x, mask)
        return self.fc_out(x)


class TrainingConfig:
    def __init__(self):
        self.batch_size = 64
        self.seq_length = 64
        self.learning_rate = 3e-4
        self.epochs = 10

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.5)
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"Memory limit set to 50% of GPU memory")
        else:
            print("WARNING: CUDA not available, using CPU")


def get_shakespeare_data():
    """Download and prepare Shakespeare text"""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    r = requests.get(url)
    return r.text


class ShakespeareDataset(Dataset):
    def __init__(self, text, seq_length=64):
        self.text = text
        self.seq_length = seq_length
        self.tokenizer = CharacterTokenizer(text)
        self.data = self.tokenizer.encode(text)

    def __len__(self):
        return len(self.data) - self.seq_length - 1

    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.seq_length]
        target = self.data[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(sequence), torch.tensor(target)
