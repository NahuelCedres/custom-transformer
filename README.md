WELCOME! 🤖
===========

Welcome to my repository! Here, you can explore some of the projects I have developed.
If you have any questions, feel free to send an email to me at: nahuelcedres18@gmail.com


Index:
------

* [Transformer](#transformer)


# Transformer

I designed a custom version of a Transformer model based on the ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) paper. The implementation is organized into three main files:

* low.py: Contains the implementations of `PositionalEncoding`, `Head`, `MultiHeadAttention`, and `FeedForward` modules.
    
    ```python
    ### Libraries ###
    import torch
    import torch.nn as nn
    import math

    ### Positional Encoding ###
    class PositionalEncoding(nn.Module):
        def __init__(self, dmodel, max_seq_length):
            super(PositionalEncoding, self).__init__()
            ...
    ```
    
* mid.py: Contains the implementations of the `Encoder` and `Decoder` modules.
    
    ```python
    import torch.nn as nn
    from low import PositionalEncoding, Head, FeedForward, MultiHeadAttention

    class Encoder(nn.Module):
        def __init__(self, dmodel, num_head, dropout):
            super(Encoder, self).__init__()
            
            self.mha = MultiHeadAttention(dmodel, num_head, dropout)
            self.ff = FeedForward(dmodel, dropout)
            self.norm1 = nn.LayerNorm(dmodel)
            self.norm2 = nn.LayerNorm(dmodel)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask):       
            x_out = self.mha(x, x, x, mask)
            x = self.norm1(x + self.dropout(x_out))
            x_out = self.ff(x)
            x = self.norm2(x + self.dropout(x_out))
            
            return x

    class Decoder(nn.Module):
        def __init__(self, dmodel, num_head, dropout):
            super(Decoder, self).__init__()

            self.mha_mask = MultiHeadAttention(dmodel, num_head, dropout)
            self.mha_ = MultiHeadAttention(dmodel, num_head, dropout)
            self.ff_d = FeedForward(dmodel, dropout)
            self.norm1_d = nn.LayerNorm(dmodel)
            self.norm2_d = nn.LayerNorm(dmodel)
            self.norm3_d = nn.LayerNorm(dmodel)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, enc_out, src_mask, tgt_mask):
            x_out = self.mha_mask(x, x, x, tgt_mask)
            x = self.norm1_d(x + self.dropout(x_out))

            x_out = self.mha_(x, enc_out, enc_out, src_mask)
            x = self.norm2_d(x + self.dropout(x_out))

            x_out = self.ff_d(x)
            x = self.norm3_d(x + self.dropout(x_out))

            return x
    ```
    
* high.py: Contains the complete Transformer model.
    
    ```python
    class Transformer(nn.Module):
        
        def __init__(self, 
                     dmodel, 
                     src_vocab_size, 
                     tgt_vocab_size, 
                     seq_length, 
                     dec_seq_length, 
                     num_head,
                     num_layers, 
                     dropout, 
                     device,
                     src_pad_idx=0, tgt_pad_idx=0, tgt_sos_idx=0):
            super(Transformer, self).__init__()
            
            # GPU or CPU
            self.device = device
            
            # Embedding + Positional Encoding
            self.embd_enc = nn.Embedding(src_vocab_size, dmodel)
            self.embd_dec = nn.Embedding(tgt_vocab_size, dmodel)
            self.positional = PositionalEncoding(dmodel, seq_length)

            # Filters
            self.src_pad_idx = src_pad_idx
            self.tgt_pad_idx = tgt_pad_idx
            self.tgt_sos_idx = tgt_sos_idx
            
            # Encoder
            self.enc = nn.ModuleList([Encoder(dmodel, num_head, dropout) for _ in range(num_layers)]) 
            
            # Decoder
            self.dec = nn.ModuleList([Decoder(dmodel, num_head, dropout) for _ in range(num_layers)])

            # Output
            self.linear = nn.Linear(dmodel, tgt_vocab_size)
            self.dropout = nn.Dropout(dropout)
            
            self.apply(self._init_weights)
        ...
    ```
    
Feel free to replicate or modify the code for your personal projects!


