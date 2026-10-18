import torch
import torch.nn as nn
import torch.nn.functional as F

from model.decoder_block import DecoderBlock


class MiniGPT(nn.Module):
    '''
    Token Embedding
    +
    Position Embedding
    +
    多个 DecoderBlock
    +
    Final LayerNorm
    +
    LM Head
    '''
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        d_ff: int,
        num_layers: int,
        dropout = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_layers = num_layers

        # TokenPositionEmbedding:
        self.token_embedding = nn.Embedding(vocab_size,d_model)
        self.position_embedding = nn.Embedding(max_seq_len,d_model)
        self.dropout = nn.Dropout(dropout)

        # DecoderBlock * num_layers:
        self.net = nn.ModuleList([DecoderBlock(d_model, d_ff, dropout) for _ in range(num_layers)])
        self.final_layernorm = nn.LayerNorm(d_model)
        self.LMHead = nn.Linear(d_model,vocab_size) ## LM Head 讲了吗？

    def forward(self, input_ids):
        L = input_ids.size(1)
        if L > self.max_seq_len:
            raise ValueError("L is greater than max_seq_len!")
        
        positions = torch.arange(L,device=input_ids.device)

        x = self.dropout(self.token_embedding(input_ids) + self.position_embedding(positions))
        for i in range(self.num_layers):
            x,_ = self.net[i](x)
        return self.LMHead(self.final_layernorm(x))

