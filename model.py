
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import math 
import matplotlib.pyplot as plt
import numpy as np 
import random
from math import sqrt

import matplotlib.pyplot as plt
import importlib 
import layers
importlib.reload(layers)
from layers import RelativeGlobalAttention, DynamicPositionEmbedding


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed
        


class DecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, dim_feedforward, activation, dropout=0.1, seq_length=1024, bidirectional=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.mha = RelativeGlobalAttention(d=d_model, h=num_heads, max_seq=seq_length+1, bidirectional=bidirectional, rel_attn_bias=False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation(),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        self.layer_norm1 = RMSNorm(d_model)
        self.layer_norm2 = RMSNorm(d_model)
        self.dropout_attn = nn.Dropout(dropout)
   


    def forward(self, x, mask=None, first_style=True, kv_cache=False):
  
        mha_in = self.layer_norm1(x)
        mha_out, _ = self.mha(mha_in, mask=mask, cache_enabled=kv_cache) 
        
        mha_out = self.dropout_attn(mha_out)
        mha_out = x + mha_out

        ff_out = self.layer_norm2(mha_out)
        ff_out = self.ff(ff_out)
    
        return ff_out + mha_out



    
class MusicTransformer(nn.Module):
    
    def __init__(self, d_model=256, num_layers=8, num_cond_layers=4, nhead=4, forward_exp=4, dropout=0.1, seq_length=512, tokenizer=None):
        super(MusicTransformer, self).__init__()
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self.seq_length = seq_length
        self.d_model = d_model
        
        vocab_size = len(tokenizer.vocab)
        
        dim_feedforward = forward_exp * d_model
        self.embedding = nn.Sequential(nn.Embedding(vocab_size, d_model), nn.Dropout(dropout))
        self.style_embedding = nn.Sequential(nn.Embedding(vocab_size, d_model), nn.Dropout(dropout))

        self.pos_encoding = DynamicPositionEmbedding(embedding_dim=d_model, max_seq=seq_length)
        
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, nhead, dim_feedforward, activation=nn.ReLU, dropout=dropout, seq_length=seq_length, bidirectional=False) for i in range(num_layers)])
        self.style_encoder_layers = nn.ModuleList([DecoderLayer(d_model, nhead, dim_feedforward, activation=nn.ReLU, dropout=dropout, seq_length=seq_length, bidirectional=True) for i in range(num_cond_layers)])

        self.style_transform = nn.Linear(d_model, d_model)
        
        self.style_token_base = nn.Parameter(torch.randn(1,1,d_model))
                                                      
        self.final = nn.Sequential(RMSNorm(d_model), nn.Linear(d_model, vocab_size))
        
        
        self.to(self.device)

    def forward(self, vocab_indices, style_tokens=None, longer_sequence=False):

        assert (not vocab_indices.shape[1] > self.seq_length) or longer_sequence, "Sequence length greater than self.seq_length, not good"
        
        tokens = self.embedding(vocab_indices) * sqrt(self.d_model)
        tokens = self.pos_encoding(tokens)
        
        has_style = style_tokens is not None
        if has_style:
            tokens = torch.cat((style_tokens, tokens), dim=1) 
        seq_len = tokens.shape[1]    
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(tokens.device)

        x = tokens
        for layer in self.decoder_layers:
            x = layer(x, mask, first_style=has_style, kv_cache= not self.training )

        if has_style: #remove style tokens
            x = x[:, 1:]
            
        logits = self.final(x)

        return logits

    def compute_style_tokens(self, sequences):
        """
        sequences: tokenized style sequences of shape [N, S]
        Returns [N, 1, E]
        """
        self.eval()
        assert sequences is None or len(sequences.shape) == 2, "Style sequences must be tokenized batch of shape [N, S]"
        sequences.to(self.device)
        sequences = self.style_embedding(sequences) * sqrt(self.d_model)
        sequences = self.pos_encoding(sequences)
        
        N=sequences.shape[0]
        sequences = torch.cat((self.style_token_base.repeat(N, 1, 1), sequences), dim=1)
        x = sequences
        
        for layer in self.style_encoder_layers:
            x = layer(x, mask=None, first_style=True) #only mask padding tokens
        return self.style_transform(x[:, :1])
        
    def get_loss(self, vocab_indices, style_indices=None, criterion=None):
        global idx

        
        input_sequence = vocab_indices[:,:-1]
        target_sequence = vocab_indices[:,1:]
        
        if style_indices == None:
            style_indices = vocab_indices[:,:-1]
            
        style_tokens = self.compute_style_tokens(style_indices)
        
        logits = self.forward(input_sequence, style_tokens)
        
        logits = logits.view(-1, logits.size(-1))
        target_sequence = target_sequence.reshape(-1)

        loss = criterion(logits, target_sequence, reduction='none')
        loss[target_sequence == 0] = 0
        loss = loss.mean()

        return loss

        
        
    def train_batch(self, vocab_indices, style_indices=None, optimizer=None, criterion=None, sw=None, do_step=True, scheduler=None):
        """
        vocab_indices: a batch of tokenized sequences
        """
        assert vocab_indices.shape[1] == self.seq_length+1, "Training sequences must be seq_length+1 long"
        
        self.train()
        optimizer.zero_grad()

            
        loss = self.get_loss(vocab_indices,style_indices,criterion)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
        
        if do_step:
            optimizer.step()
            if scheduler:
                scheduler.step()
                
        return loss.cpu().item()
        
    def reset_kv_cache(self):
        
        for layer in self.decoder_layers:
            for module in layer.modules():
                if isinstance(module, RelativeGlobalAttention):
                    module.kv_cache = None
        
    @torch.inference_mode()
    def generate(self, style_sequence=None, start_with=None, temperature=1.0, top_k=20, max_length=512):
        self.eval()
    
        if style_sequence is not None:
            if type(style_sequence) == list:
                style_token = self.compute_style_tokens(style_sequence[0].unsqueeze(0).to(self.device))
                for i in range(1, len(style_sequence)):
                    style_token+=self.compute_style_tokens(style_sequence[i].unsqueeze(0).to(self.device))
                style_token/=len(style_sequence)
            else:
                if style_sequence.shape[0] > self.seq_length:
                    style_sequence=style_sequence[-self.seq_length:]
                style_token = self.compute_style_tokens(style_sequence.unsqueeze(0).to(self.device))
        else:
            style_token = None
    
   
        if start_with is not None:
            x = start_with.unsqueeze(0).to(self.device)
        else:
            x = torch.tensor([self.tokenizer.vocab.index('<start>')]).unsqueeze(0).to(self.device) #start token
    
        start_idx = start_with.shape[0]-1 if start_with is not None else 0


        def update_cache(x):
            self.reset_kv_cache()
            cache_update_sequence = x[:, -3*self.seq_length//4:-1]
            self.forward(cache_update_sequence, style_tokens=style_token) #shift context, liberating 1/4th
            cache_size = cache_update_sequence.shape[1]
            return cache_size
            
        cache_size=update_cache(x)
        
        for i in range(start_idx, max_length):
            
            if cache_size >= self.seq_length: #handles long sequences
                cache_size=update_cache(x)


            logits = self.forward(x[:, -1:], style_tokens=None)[0, -1, :]
            cache_size+=1
            
            logits_remove_mask = torch.zeros_like(logits, dtype=torch.bool).to(logits.device)
    
            logits_remove_mask[[0, -1, -2]] = 1

            logits /= temperature
                    
            logits[logits_remove_mask] = -float('inf')
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('inf')
            
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1)
    
            x = torch.cat((x, torch.tensor([next_token]).unsqueeze(0).to(self.device)), dim=-1) 
            yield
    
        self.train()
        self.generation_result = x
        return




        