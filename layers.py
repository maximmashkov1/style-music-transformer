import torch
from torch import nn
import math
import torch.nn.functional as F
import numpy as np

class RelativeGlobalAttention(nn.Module):
    """
    Concept from https://arxiv.org/pdf/1809.04281.pdf
    Also supports unmasked modeling, bias and kv caching.
    """
    def __init__(self, h=4, d=256, max_seq=2048, bidirectional=False, rel_attn_bias=True):
        super().__init__()
        self.max_seq = max_seq
        self.bidirectional = bidirectional
        self.rel_attn_bias = rel_attn_bias
        self.h = h
        self.d = d
        self.dh = d // h
        self.qkv = nn.Linear(d, 3*d)
        self.fc = nn.Linear(d, d)
        
        self.scale = 1/math.sqrt(self.dh)

        self.E = nn.Parameter(torch.randn(h, self.max_seq, self.dh)) #Srel weights, one sequence of vectors per head
        if self.bidirectional:
            self.Ef= nn.Parameter(torch.randn(h, self.max_seq, self.dh)) #future Srel weights, one sequence of vectors per head
        if rel_attn_bias:
            self.Eb=nn.Parameter(torch.zeros(1, h, 1, self.max_seq)).to(torch.float32) #Srel biases, one sequence of scalars per head
            if self.bidirectional:
                self.Ebf=nn.Parameter(torch.zeros(1, h, 1, self.max_seq)).to(torch.float32) #future Srel biases, one sequence of scalars per head

        self.kv_cache=None #erased externally

    def forward(self, x, mask=None, cache_enabled=False):
        """
        x: sequence of tokens [N, S, E]
        mask: either [S, S] or [N, 1, S, S]
        cache_enabled: true during inference

        if cache_enabled:
            if no cache yet - evaluate sequence as normal
            if has cache - only expect 1 q per sequence
        """
        assert not cache_enabled or x.shape[1] == 1 or self.kv_cache is None, "KV cache is enabled, not empty but input has more than one element in sequence"
        B, S, E = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.h, E // self.h).permute(2, 0, 3, 1, 4) #qkv, N, heads, S, E
        q, k, v = qkv  
        if cache_enabled:
            assert not self.bidirectional, "Can't do KV caching in bidirectional transformer"
            if self.kv_cache!=None:

                c_k, c_v = self.kv_cache
                if c_k.shape[-2] >= self.max_seq:
                    style_k=c_k[..., :1, :]
                    style_v=c_v[..., :1, :]
                    k = torch.cat((style_k, c_k[..., -self.max_seq+2:, :], k), dim=-2)
                    v = torch.cat((style_v, c_v[..., -self.max_seq+2:, :], v), dim=-2) 
                else:
                    k = torch.cat((c_k, k), dim=-2)
                    v = torch.cat((c_v, v), dim=-2) 

                Srel = self._compute_single_srel(q, k.shape[-2])
            else:
                Srel = self._compute_srel(q)
            self.kv_cache = [k, v]
        else:    
            Srel = self._compute_srel(q)

        
        QKt = torch.matmul(q, k.transpose(-1, -2))
        
        logits = (QKt + Srel) * self.scale

        if not self.bidirectional and not cache_enabled:
            logits.masked_fill_(mask, float('-1e9'))

        attention_weights = F.softmax(logits, -1)
        attention = torch.matmul(attention_weights, v)

        out = attention.permute(0, 2, 1, 3).reshape(B, S, E)
        out = self.fc(out)

        return out, attention_weights
    
    def _compute_single_srel(self, q, len):
        """
        Optimized _compute_srel that only computes relative attention for one query when using kv cache
        """
        
        SrelW = self.E[:, -len:]
        Srel = torch.matmul(q, SrelW.transpose(-1, -2))
        if self.rel_attn_bias:
            Srel += self.Eb[:, -len:]
        return Srel
        
    def _compute_srel(self, q):
        """
        To compute Srel, we learn max_length linear transformations for each head (max_length*2 if bidirectional). To obtain an entry i, j in Srel we apply [i-j]th linear transformation ([j-i]th if bidirectional and j>i) to Qi. Naturally, this limits maximum sequence length.
        """
        len_q = q.size(2)

        e = self._get_past_embedding(len_q)
        QE = torch.matmul(q, e.transpose(-1, -2)) 
        if self.rel_attn_bias:
            QE += self._get_past_bias(len_q)

        QE = self._qe_masking(QE)
        Srel = self._skewing(QE)

        if self.bidirectional: 
            ef = self._get_future_embedding(len_q)
            QEf = torch.matmul(torch.flip(q, dims=[2]), ef.transpose(-1, -2))
            if self.rel_attn_bias:
                QEf += self._get_future_bias(len_q)

            QEf = self._qe_masking(QEf)
            Srel_f = self._skewing(QEf)
            Srel += Srel_f.transpose(-1, -2)
        return Srel

            
    def _get_past_bias(self, len_q):
        starting_point = max(0,self.max_seq-len_q)
        eb = self.Eb[:, :, :, starting_point:]
        return eb.expand(-1, -1, len_q, -1)
    
    def _get_future_bias(self, len_q):
        starting_point = max(0,self.max_seq-len_q)
        eb = self.Ebf[:, :, :, starting_point:]
        return eb.expand(-1, -1, len_q, -1)

    def _get_past_embedding(self, len_q):
        starting_point = max(0,self.max_seq-len_q)
        e = self.E[:, starting_point:]
        return e

    def _get_future_embedding(self, len_q):
        starting_point = max(0,self.max_seq-len_q)
        e = self.Ef[:, starting_point:]
        return e


    def _skewing(self, t):

        padded = F.pad(t, [1, 0])
        Srel = padded.reshape(-1, t.shape[-1] + 1, t.shape[-2])
        Srel = Srel[:, 1:]       
        Srel = Srel.reshape(*t.shape)
        return Srel


    @staticmethod
    def _qe_masking(qe):
        mask = sequence_mask(
            torch.arange(qe.size()[-1] - 1, qe.size()[-1] - qe.size()[-2] - 1, -1).to(qe.device),
            qe.size()[-1])
        mask = ~mask.to(mask.device)
        return mask.to(qe.dtype) * qe


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)



class DynamicPositionEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim, max_seq=2048):
        super().__init__()
        embed_sinusoid_list = np.array([[
            [
                math.sin(
                    pos * math.exp(-math.log(10000) * i/embedding_dim) *
                    math.exp(math.log(10000)/embedding_dim * (i % 2)) + 0.5 * math.pi * (i % 2)
                )
                for i in range(embedding_dim)
            ]
            for pos in range(max_seq)
        ]])
        self.positional_embedding = embed_sinusoid_list

    def forward(self, x):
        x = x + torch.from_numpy(self.positional_embedding[:, :x.size(1), :]).to(x.device, dtype=x.dtype)
        return x

if __name__=='__main__':
    """
    Runs tests, diagrams should appear as on page 4 of https://arxiv.org/pdf/1809.04281
    """
    import torch

    attn=RelativeGlobalAttention(h=1,max_seq=3)
    
    import matplotlib.pyplot as plt
    
    attn.len_q, attn.len_k = 3, 3
    QE = torch.tensor([[[4,3,2,1],[4,3,5,1],[4,3,2,1],[4,3,2,1]],[[10,3,2,1],[10,3,5,1],[10,3,2,1],[10,3,2,1]]]).unsqueeze(0)
    QE = attn._qe_masking(QE)
    
    tensor_np =QE.clone().squeeze(0)[1].detach().numpy() 
    plt.imshow(tensor_np, cmap='viridis', interpolation='none')
    plt.colorbar() 
    plt.title("QE")
    plt.show()

    Srel = attn._skewing(QE)
    tensor_np =  Srel.squeeze(0)[1].detach().numpy()
    plt.imshow(tensor_np, cmap='viridis', interpolation='none')
    plt.colorbar() 
    plt.title("QE")
    plt.show()

    Srel = torch.transpose(Srel, 2, 3)
    tensor_np =  Srel.squeeze(0)[1].detach().numpy()
    plt.imshow(tensor_np, cmap='viridis', interpolation='none')
    plt.colorbar() 
    plt.title("QE")
    plt.show()
