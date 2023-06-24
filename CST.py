import torch
import numpy
import math
import qst

class EncoderLayer(torch.nn.Module):
    ''' Self-attention encoder layer.
        
        Parameters:
        embed_dim: int - dimensionality of the embeddings
        num_heads: int - number of attention heads
        dropout: float - probability of dropout after applying the MLP
        
        Attributes:
        norm1, norm2: nn.LayerNorm - layer norms.
        attn: nn.MultiHeadAttention - attention module.
        mlp: nn.Sequential - multilayer preceptron. '''
    def __init__(self, embed_dim=64, num_heads=16, dropout=0.1, **kwargs):
        super(type(self), self).__init__()
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True) 
        # 输入数据维度:[Number of batch, Seqence length, Embedding dimension]
        # MultiheadAttention层: Multihead(Q,K,V)=Concat(head_1,...,head_h)W
        # 单个Attention: head=Attention(Q,K,V)=softmax(QK^{T}/\sqrt{d})W.
        # Concat(head_1,...,head_h)代表把head_1,...,head_h从左往右拼成一个新的矩阵
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 4*embed_dim),
            torch.nn.GELU(), # GELU(x)=x*Phi(x),这里Phi(.)标准正态分布的CDF
            torch.nn.Dropout(dropout),
            torch.nn.Linear(4*embed_dim, embed_dim)) # 多层感知器单元, 输入和输出层单元数皆为embed_dim
        
    def forward(self, src):
        batch_size, n_tokens, embed_dim = src.shape # src代表encoder输入张量, 维度[N,S,E]
        mem = src # (batch_size, n_tokens, embed_dim)
        src = self.norm1(src) # (batch_size, n_tokens, embed_dim)
        mem = mem + self.attn(src, src, src,
                              need_weights=False)[0] # (batch_size, n_tokens, embed_dim)
        mem = mem + self.mlp(self.norm2(mem)) # (batch_size, n_tokens, embed_dim)
        return mem # (batch_size, n_tokens, embed_dim)

class Encoder(torch.nn.Module):
    ''' Transformer encoder.
        
        Parameters:
        n_layers: int - number of layers.
        
        Attributes:
        layers: nn.ModuleList - hosting encoder layers. '''
    def __init__(self, n_layers=1, **kwargs): # **kwargs: 发送一个键值对的可变长度的参数列表给函数
        super(type(self), self).__init__()
        self.layers = torch.nn.ModuleList(
            [EncoderLayer(**kwargs) for _ in range(n_layers)])
        
    def forward(self, src):
        mem = src # (batch_size, n_tokens, embed_dim)
        for layer in self.layers:
            mem = layer(mem) # (batch_size, n_tokens, embed_dim)
        return mem # (batch_size, n_tokens, embed_dim)
    
class DecoderLayer(torch.nn.Module):
    ''' Self-attention decoder layer.
        
        Parameters:
        embed_dim: int - dimensionality of the embeddings
        num_heads: int - number of attention heads
        dropout: float - probability of dropout after applying the MLP
        n_tokens_max: int - maximum number of tokens.
        
        Attributes:
        norm1, norm2: nn.LayerNorm - layer norms.
        attn: nn.MultiHeadAttention - attention module.
        mlp: nn.Sequential - multilayer preceptron. '''
    def __init__(self, embed_dim=64, num_heads=16, dropout=0.1, n_token_max=10, **kwargs):
        super(type(self), self).__init__()
        self.n_token_max = n_token_max
        self.register_buffer('attn_mask', 
            (1-torch.tril(torch.ones(n_token_max, n_token_max))).to(dtype=torch.bool))
        # ones函数生成矩阵元全是1的矩阵
        # tril函数把目标矩阵改成下三角型
        # self.register_buffer: mask矩阵的参数存在模型中,但不参与optimize
        # mask的作用: 预测序列的第i+1个词的时候只能看到第1-第i个词,看不到第i+1个词之后的信息
        self.norm1 = torch.nn.LayerNorm(embed_dim)
        self.norm2 = torch.nn.LayerNorm(embed_dim)
        self.norm3 = torch.nn.LayerNorm(embed_dim)
        self.self_attn = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True)
        self.cross_attn = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, 4*embed_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(4*embed_dim, embed_dim))
        self.register_buffer('tgt_cache', None)
        
    def reset_cache(self):
        self.tgt_cache = None
        
    def forward(self, tgt, mem, cache=False):
        batch_size, n_tokens, embed_dim = tgt.shape
        assert n_tokens < self.n_token_max, 'Number tokens {} exceeds the maximum limit {}, please increase n_token_max parameter.'.format(n_tokens, self.n_token_max)
        # assert函数:条件为true时继续运行,条件为false时直接中断程序.
        out = tgt # (batch_size, n_tokens, embed_dim)
        if cache: # use cache
            if self.tgt_cache is None:
                self.tgt_cache = tgt # (batch_size, 1, embed_dim)
            else:
                self.tgt_cache = torch.cat([self.tgt_cache, tgt], -2) # (batch_size, n_tokens, embed_dim)
            tgt = self.norm1(self.tgt_cache) # (batch_size, n_tokens, embed_dim)
            out = out + self.self_attn(tgt[:,-1:,:], tgt, tgt,
                                       need_weights=False)[0] # (batch_size, 1, embed_dim)
        else: # no cache
            attn_mask = self.attn_mask[:n_tokens, :n_tokens] # (n_token, n_token)
            tgt = self.norm1(tgt) # (batch_size, n_tokens, embed_dim)
            out = out + self.self_attn(tgt, tgt, tgt,
                                       attn_mask=attn_mask,
                                       need_weights=False)[0] # (batch_size, n_tokens, embed_dim)
        out = out + self.cross_attn(self.norm2(out), mem, mem,
                                    need_weights=False)[0] # (batch_size, n_tokens, embed_dim)
        out = out + self.mlp(self.norm3(out)) # (batch_size, n_tokens, embed_dim)
        return out
    
class Decoder(torch.nn.Module):
    ''' Transformer decoder.
        
        Parameters:
        n_layers: int - number of layers.
        
        Attributes:
        layers: nn.ModuleList - hosting encoder layers. '''
    def __init__(self, n_layers=1, **kwargs): # **kwargs: 发送一个键值对的可变长度的参数列表给函数
        super(type(self), self).__init__()
        self.layers = torch.nn.ModuleList(
            [DecoderLayer(**kwargs) for _ in range(n_layers)])
        
    def reset_cache(self):
        for layer in self.layers:
            layer.reset_cache()
        
    def forward(self, tgt, mem, cache=False):
        out = tgt # (batch_size, n_tokens, embed_dim)
        for layer in self.layers:
            out = layer(out, mem, cache=cache)
        return out # (batch_size, n_tokens, embed_dim)

class Randomizer(torch.nn.Module):
    ''' Randomizer to impose information bottleneck

        Parameters:
        embed_dim: int - dimensionality of the embeddings
    '''
    def __init__(self, embed_dim=64, **kwargs):
        super(type(self), self).__init__()
        self.logvar = torch.nn.Parameter(torch.zeros(embed_dim))

    def forward(self, avg):
        std = torch.exp(0.5 * self.logvar)
        eps = torch.randn_like(avg)
        return avg + std * eps

    def kld(self, avg):
        return (avg**2 + self.logvar.exp() - self.logvar - 1) / 2

class Transformer(torch.nn.Module):
    ''' Transformer model.
        
        Parameters:
        vocab_size: int - number of tokens in the vocabulary.
        outtk_size: int - number of tokens valid for output.
        n_tokens_max: int - maximum number of tokens.
        n_layers: int - number of layers.
        embed_dim: int - dimensionality of the embeddings
        num_heads: int - number of attention heads
        dropout: float - probability of dropout after applying the MLP
        symmetric: bool - whether to respect permutation symmetry
        
        Attributes:
        tokeembed_dim:  nn.Embedding - token embedding
        posit_embd: nn.Embedding - positional embedding
        encode: nn.Module - transformer encoder
        repara: nn.Module - reparameterization layer
        decode: nn.Module - transformer decoder
        project: nn.Sequential - output projector (map to logits) '''
    def __init__(self, embed_dim=64, vocab_size=6, outtk_size=2, n_token_max=10, **kwargs):
        super(type(self), self).__init__()
        self.token_embd = torch.nn.Embedding(vocab_size+1, embed_dim)
        self.posit_embd = torch.nn.Embedding(n_token_max, embed_dim)
        self.encode = Encoder(embed_dim=embed_dim, **kwargs)
        self.repara = Randomizer(embed_dim=embed_dim)
        self.decode = Decoder(embed_dim=embed_dim, n_token_max=n_token_max, **kwargs)
        self.project = torch.nn.Sequential(
            torch.nn.LayerNorm(embed_dim),
            torch.nn.Linear(embed_dim, outtk_size, bias=False))
    
    def embed(self, seq):
        # embed sequence to vectors (token + positional embedding)
        batch_size, n_tokens = seq.shape
        out = self.token_embd(seq) # (batch_size, n_tokens, embed_dim)
        posit = torch.arange(n_tokens, device=seq.device) # (n_tokens,)
        out = out + self.posit_embd(posit).unsqueeze(0) # (1, n_tokens, embed_dim)
        return out # (batch_size, n_tokens, embed_dim)
    
    def seq_roll(self, seq):
        seq_most = seq[:,:-1] # (batch_size, n_tokens-1)
        seq_null = torch.zeros_like(seq[:,-1:]) # (batch_size, 1)
        return torch.cat([seq_null, seq_most], -1) # (batch_size, n_tokens)
    
    def forward(self, src_seq, tgt_seq):
        src = self.embed(src_seq) # (batch_size, src_n_tokens, embed_dim)
        mem = self.encode(src)    # (batch_size, src_n_tokens, embed_dim)
        tgt = self.embed(self.seq_roll(tgt_seq)) # (batch_size, tgt_n_tokens, embed_dim)
        out = self.decode(tgt, mem) # (batch_size, tgt_n_tokens, embed_dim)
        logit = self.project(out) # (batch_size, tgt_n_tokens, outtk_size)
        return logit
    
    def logprob(self, src_seq, tgt_seq): # 根据obs翻译出out的概率对数, i.e. 概率表log(P(b|S))
        logit = self(src_seq, tgt_seq) # (batch_size, tgt_n_tokens, outtk_size)
        logprob = torch.log_softmax(logit, -1) # (batch_size, tgt_n_tokens, outtk_size)
        tgt_idx = tgt_seq.unsqueeze(-1) -1 # (batch_size, tgt_n_tokens, 1), shift back tokens by 1 for indexing 
        logprob = torch.gather(logprob, -1, tgt_idx).squeeze(-1) # (batch_size, tgt_n_tokens)
        logprob = logprob.sum(-1) # (batch_size)
        return logprob

    def loss(self, src_seq, tgt_seq, beta=1.):
        src = self.embed(src_seq) # (batch_size, src_n_tokens, embed_dim)
        mem = self.encode(src)    # (batch_size, src_n_tokens, embed_dim)
        kld = self.repara.kld(mem).mean((-2,-1)) # (batch_size)
        mem = self.repara(mem)
        tgt = self.embed(self.seq_roll(tgt_seq)) # (batch_size, tgt_n_tokens, embed_dim)
        out = self.decode(tgt, mem) # (batch_size, tgt_n_tokens, embed_dim)
        logit = self.project(out) # (batch_size, tgt_n_tokens, outtk_size)
        logprob = torch.log_softmax(logit, -1) # (batch_size, tgt_n_tokens, outtk_size)
        tgt_idx = tgt_seq.unsqueeze(-1) -1 # (batch_size, tgt_n_tokens, 1), shift back tokens by 1 for indexing 
        logprob = torch.gather(logprob, -1, tgt_idx).squeeze(-1) # (batch_size, tgt_n_tokens)
        logprob = logprob.sum(-1) # (batch_size)
        loss = - logprob + beta * kld # (batch_size,)
        return loss.mean(), logprob.mean(), kld.mean()
    
    def sample(self, src_seq, n_tokens=None, tgt_seq=None, need_logprob=False):
        if need_logprob:
            logprob = 0.
        with torch.no_grad():
            src = self.embed(src_seq) # (batch_size, src_n_tokens, embed_dim)
            mem = self.encode(src) # (batch_size, src_n_tokens, embed_dim)
            if tgt_seq is None:
                tgt_seq = torch.zeros_like(src_seq[:,:1]) # (batch_size, 1) at initialization
            else:
                seq_null = torch.zeros_like(tgt_seq[:,-1:]) # (batch_size, 1)
                tgt_seq = torch.cat([seq_null, tgt_seq], -1) # (batch_size, tgt_n_tokens + 1)
            n_tokens = src_seq.shape[-1]-tgt_seq.shape[-1] if n_tokens is None else n_tokens
            for i in range(n_tokens):
                tgt = self.embed(tgt_seq) # (batch_size, tgt_n_tokens, embed_dim)
                out = self.decode(tgt[:,-1:], mem, cache=True) # (batch_size, 1, embed_dim)
                logit = self.project(out) # (batch_size, 1, outtk_size)
                sampler = torch.distributions.Categorical(logits=logit)
                tgt_new = sampler.sample() # (batch_size, 1)
                if need_logprob:
                    logprob += sampler.log_prob(tgt_new).squeeze(-1) # (batch_size,)
                # shift all generated tokens by 1 to avoid 0 token
                tgt_new = tgt_new + 1 # (batch_size, 1)
                tgt_seq = torch.cat([tgt_seq, tgt_new], -1) # (batch_size, tgt_n_tokens+1)
        self.decode.reset_cache() # reset decoder cache
        # generated seq: rest in tgt_seq
        tgt_seq = tgt_seq[:,1:] # (batch_size, n_tokens)
        if need_logprob:
            return tgt_seq, logprob
        else:
            return tgt_seq

class Operator():
    ''' Represent a quantum operator 
        as a liner superposition of Pauli strings.
        
        Parameters:
        paulis: torch.Tensor - a list of Pauli strings 
        coeffs: torch.Tensor - a list of coefficients (generally complex) '''
    index_rule = torch.tensor(
        [[0,1,2,3],
         [1,0,3,2],
         [2,3,0,1],
         [3,2,1,0]]).flatten()
    coeff_rule = torch.tensor(
        [[1,  1,  1,  1],
         [1,  1, 1j,-1j],
         [1,-1j,  1, 1j],
         [1, 1j,-1j,  1]]).flatten()
    def __init__(self, paulis, coeffs):
        super(type(self), self).__init__()
        self.paulis = paulis
        self.coeffs = coeffs
        
    @property
    def N(self):
        return self.paulis.shape[1]

    @property
    def requires_grad(self):
        return self.coeffs.requires_grad
    
    def requires_grad_(self, requires_grad=True):
        self.coeffs.requires_grad_(requires_grad=requires_grad)
        return self
    
    def to(self, device='cpu'):
        return Operator(self.paulis.to(device=device), 
                        self.coeffs.to(device=device))
    
    @property
    def grad(self):
        grad = self.coeffs.grad
        if grad is None:
            return None
        else:
            return Operator(self.paulis, grad)
        
    def __repr__(self, max_terms=16):
        expr = ''
        dots = ''
        paulis = self.paulis
        coeffs = self.coeffs
        if paulis.shape[0] > max_terms:
            paulis = paulis[:max_terms]
            coeffs = coeffs[:max_terms]
            dots = ' ...'
        for pauli, coeff in zip(paulis, coeffs):
            if coeff != 0:
                try:
                    if coeff.imag == 0.:
                        coeff = coeff.real
                        if coeff%1 == 0:
                            if coeff == 1:
                                term = ''
                            elif coeff == -1:
                                term = '- '
                            else:
                                term = '{:d} '.format(int(coeff))
                        else:
                            term = '{:.2f} '.format(coeff)
                    else:
                        if coeff == 1j:
                            term = 'i '
                        elif coeff == -1j:
                            term = '-i '
                        else:
                            term = '({:.2f}) '.format(coeff).replace('j','i')
                except:
                    if coeff%1 == 0:
                        if coeff == 1:
                            term = ''
                        elif coeff == -1:
                            term = '- '
                        else:
                            term = '{:d} '.format(int(coeff))
                    else:
                        term = '{:.2f} '.format(coeff)
                for p in pauli:
                    if p == 0:
                        term += 'I'
                    elif p == 1:
                        term += 'X'
                    elif p == 2:
                        term += 'Y'
                    elif p == 3:
                        term += 'Z'
                term = (' ' if term[0] == '-' else ' + ') + term
                expr += term
        expr = expr.strip() + dots
        if expr == '':
            expr = '0'
        elif expr[0] == '+':
            expr = expr[1:].strip()
        if self.coeffs.grad_fn is not None:
            expr += ' (grad_fn={})'.format(type(self.coeffs.grad_fn))
        elif self.coeffs.requires_grad:
            expr += ' (requires_grad=True)'
        return expr
    
    def __neg__(self):
        return Operator(self.paulis, - self.coeffs)
    
    def __rmul__(self, other):
        return Operator(self.paulis, other * self.coeffs)
    
    def __truediv__(self, other):
        return Operator(self.paulis, self.coeffs / other)
    
    def __add__(self, other):
        if isinstance(other, Operator):
            paulis = torch.cat([self.paulis, other.paulis])
            coeffs = torch.cat([self.coeffs, other.coeffs])
            return Operator(paulis, coeffs).reduce()
        else:
            result = self + other * identity(self.N)
            return result.reduce()

    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __matmul__(self, other):
        ''' define: A @ B = A B '''
        n1, N = self.paulis.shape
        n2, N = other.paulis.shape
        paulis_prod = 4 * self.paulis.unsqueeze(1) + other.paulis.unsqueeze(0) # (n1, n2, N)
        paulis = self.index_rule.to(paulis_prod.device)[paulis_prod].view(-1, N) # (n1*n2, N)
        phases = self.coeff_rule.to(paulis_prod.device)[paulis_prod].prod(-1).view(-1) # (n1*n2, )
        coeffs = self.coeffs.unsqueeze(1) * other.coeffs.unsqueeze(0) # (n1, n2)
        coeffs = coeffs.view(-1) # (n1*n2, )
        coeffs = coeffs * phases
        return Operator(paulis, coeffs).reduce()

    def reduce(self, tol=1e-10):
        ''' Reduce the operator by:
            1. combine similar terms
            2. drop terms that are too small (abs < tol) '''
        paulis, inv_indx = torch.unique(self.paulis, dim=0, return_inverse=True)
        coeffs = torch.zeros(paulis.shape[0]).to(self.coeffs)
        coeffs.scatter_add_(0, inv_indx, self.coeffs)
        mask = coeffs.abs() > tol
        return Operator(paulis[mask], coeffs[mask])
    
    def trace(self):
        ''' compute Tr O '''
        mask = torch.all(self.paulis == 0, -1)
        return self.coeffs[mask].sum()
    
    @property
    def H(self):
        # Hermitian conjugation of this operator O -> O^†
        return Operator(self.paulis, self.coeffs.conj())
    
    def norm(self):
        return (self.H @ self).trace().real

    @property
    def operator_space(self):
        ''' my corresponding operator space is assumed to be
            spanned by the identity operator and all the Pauli
            operators in myself. '''
        return span([identity(self.N), self])

    def single_shots(self, shadow):
        ''' Estimate the single-shot expectation on a shadow

            Input:
            shadow: Shadow - the shadow dataset, with respect to which
                the operator expectation value is evaluated

            Output:
            [o_i, ...]: torch.Tensor - single-shot shadow estimations 
                of operator expectation values, each element o_i is
                    o_i = Tr (M_inv[sigma_i] O)
                operator expectation value is the mean of them
        '''
        shadow_map = shadow.shadow_map(self.paulis) # (n_sample, n_pauli)
        single_shots = shadow_map.to(self.coeffs) @ self.coeffs # (n_sample,)
        return single_shots # (n_sample,)

    def expectation(self, shadow, weights=None, batch_size=None):
        ''' Estimate the operator expectation value
            (the median of means method can be used) 

            Input:
            shadow: Shadow - the shadow dataset, with respect to which
                the operator expectation value is evaluated
            batch_size: int - batch size for median of means
            weights: torch.Tensor - a vector to reweight single-shot estimations

            Output:
            <O>: torch.Tensor - scalar tensor whose value is
                    <O> = avg_i Tr (M_inv[sigma_i] O)  '''
        single_shots = self.single_shots(shadow) # (n_sample,)
        if weights is not None:
            single_shots = weights * single_shots
        if batch_size is None:
            return single_shots.mean()
        else:
            n_sample = len(single_shots)
            pad = int(numpy.ceil(n_sample / batch_size)) * batch_size - n_sample
            single_shots = torch.nn.functional.pad(single_shots, (0, pad)) # (n_batch * batch_size,)
            single_shots = single_shots.view(-1, batch_size) # (n_batch, batch_size)
            means = single_shots.mean(-1) # (n_batch,)
            return means.real.median() + 1j * means.imag.median()
    
    def matrix_form(self):
        pauli_mats = torch.tensor([
            [[1,0],[0,1]],
            [[0,1],[1,0]],
            [[0,-1j],[1j,0]],
            [[1,0],[0,-1]]], device=self.coeffs.device)
        loc_mats = pauli_mats[self.paulis]
        mats = loc_mats[:,0,...]
        for i in range(1,loc_mats.shape[1]):
            mats = torch.einsum('ijk,ilm -> ijlkm', mats, loc_mats[:,i,...]).view(-1,2**(i+1),2**(i+1))
        return torch.einsum('i,ijk->jk', self.coeffs.to(mats), mats)

def pauli(obj, N=None):
    if isinstance(obj, torch.Tensor):
        paulis = obj.view(1,-1)
    else:
        if isinstance(obj, (tuple, list)):
            N = len(obj)
            inds = enumerate(obj)
        elif isinstance(obj, dict):
            if N is None:
                raise ValueError('pauli(inds, N) must specify qubit number N when inds is dict.')
            inds = obj.items()
        elif isinstance(obj, str):
            return pauli(list(obj))
        else:
            raise TypeError('pauli(obj) recieves obj of type {}, which is not implemented.'.format(type(obj).__name__))
        paulis = torch.zeros(1, N, dtype=torch.long)
        for i, p in inds:
            assert i < N, 'Index {} out of bound {}'.format(i, N)
            if p == 'I':
                p = 0
            elif p == 'X':
                p = 1
            elif p == 'Y':
                p = 2
            elif p == 'Z':
                p = 3
            paulis[0, i] = p 
    coeffs = torch.ones(1, dtype=torch.cfloat)
    return Operator(paulis, coeffs)

class Shadow():
    ''' Represent a classical shadow dataset.
    
    Parameters:
    obs: torch.Tensor - a batch of sets of local (single-Pauli) observables 
         (encoding: 0 = I, 1 = X, 2 = Y, 3 = Z)
    out: torch.Tensor - a batch of sets of measurement outcomes
         (encoding: 0 = +, 1 = -) '''
    pauli_map = torch.tensor([3,4,5,6])
    bit_map = torch.tensor([1,2])
    matching_tab = torch.tensor(
        [[False,False,False,False],
         [ True, True,False,False],
         [ True,False, True,False],
         [ True,False,False, True]]).flatten()
    def __init__(self, obs, out):
        self.obs = obs # (n_sample, N)
        self.out = out # (n_sample, N)

    @property
    def n_sample(self):
        return self.obs.shape[0]
    
    def to(self, device='cpu'):
        return Shadow(self.obs.to(device=device),
                      self.out.to(device=device))
        
    def __repr__(self, max_recs=16):
        expr = ''
        dots = ''
        obs = self.obs
        out = self.out
        if obs.shape[0] > max_recs:
            obs = obs[:max_recs]
            out = out[:max_recs]
            dots = '...'
        for s, b in zip(obs, out):
            term = ''
            for si, bi in zip(s, b):
                if bi == 0:
                    term += ' +'
                elif bi == 1:
                    term += ' -'
                if si == 0:
                    term += 'I'
                elif si == 1:
                    term += 'X'
                elif si == 2:
                    term += 'Y'
                elif si == 3:
                    term += 'Z'
            expr += '| ' + term.strip() + ' |\n'
        if dots == '':
            expr = expr[:-1]
        else:
            expr += dots
        return expr
    
    def tokenize(self):
        # export shadow data to language data
        src_seq = self.pauli_map.to(self.obs.device)[self.obs] # (n_sample, N = n_tokens)
        tgt_seq = self.bit_map.to(self.out.device)[self.out] # (n_sample, N = n_tokens)
        return src_seq, tgt_seq

    def shadow_map(self, paulis):
        ''' Shadow map is a linear map from operator to its 
            corresponding single-shot estimate, it can be 
            represented as a matrix
                R_{ij} = Tr (M_inv[sigma_i] O_j) 
            given the shadow data {sigma_i} and a set of Pauli
            basis {O_j}, the shadow map is determined. 

            Input:
            paulis: torch.Tensor - a list of Pauli strings.

            Output:
            shadow_map: torch.Tensor - R_{ij} matrix. '''
        match_idx = 4 * self.obs.unsqueeze(1) + paulis.unsqueeze(0) # (n_sample, n_paulis, N)
        match = torch.all(self.matching_tab.to(match_idx.device)[match_idx], -1) # (n_sample, n_paulis)
        pauli_support = (paulis != 0) # (n_pauli, N)
        shadow_weight = 3**pauli_support.sum(-1) # (n_pauli,)
        match_support = match.unsqueeze(-1) * pauli_support.unsqueeze(0) # (n_sample, n_pauli, N)
        masked_outcome = self.out.unsqueeze(1) * match_support # (n_sample, n_pauli, N)
        value = 1 - 2 * (masked_outcome.sum(-1) % 2) # (n_sample, n_pauli)
        masked_value = value * match # (n_sample, n_pauli)
        shadow_map = masked_value * shadow_weight.unsqueeze(0) # (n_sample, n_pauli)
        return shadow_map # (n_sample, n_pauli)
    
# collect shadow data
def ghz_shadow(n_qubit, n_sample):
    ''' Collect classical shadow on GHZ state by Pauli measurements
        
        Input:
        n_qubit: int - number of qubits
        n_sample: int - number of samples
        
        Output:
        shd: Shadow - classical shadow dataset '''
    rho = qst.ghz_state(n_qubit)
    obs = []
    out = []
    for _ in range(n_sample):
        sigma = qst.random_pauli_state(n_qubit)
        bit = rho.copy().measure(sigma)[0]
        tok = sigma.tokenize()
        obs.append(tok[:n_qubit,:n_qubit].diagonal())
        out.append((tok[:,-1]+bit)%2)
    obs = torch.tensor(numpy.stack(obs))
    out = torch.tensor(numpy.stack(out))
    return Shadow(obs, out)

# scramble shadow, add a random unitary layer before GHZ shadow
def scrambleghz_shadow(n_qubit, n_sample):
    ''' Collect classical shadow on scrambled GHZ state by Pauli measurements
        
        Input:
        n_qubit: int - number of qubits
        n_sample: int - number of samples
        
        Output:
        shd: Shadow - classical shadow dataset '''
    rho = qst.stabilizer_state(qst.paulis(['+ZIZY', '+YIXI', '-IXXZ', '-XZYI']))
    '''Original GHZ: stabilized by ZZII, IZZI, IIZZ, XXXX
       CliffordMap used for GHZ here is
       X0-> -ZZZZ
       Z0-> +IXII
       X1-> +IXZX
       Z1-> +ZXZY
       X2-> +XXXY
       Z2-> +XXYY
       X3-> -ZIZI
       Z3-> -XIZX'''
    obs = []
    out = []
    for _ in range(n_sample):
        sigma = qst.random_pauli_state(n_qubit)
        bit = rho.copy().measure(sigma)[0]
        tok = sigma.tokenize()
        obs.append(tok[:n_qubit,:n_qubit].diagonal())
        out.append((tok[:,-1]+bit)%2)
    obs = torch.tensor(numpy.stack(obs))
    out = torch.tensor(numpy.stack(out))
    return Shadow(obs, out)

# pauli shadow, add a random pauli layer before GHZ shadow
def paulighz_shadow(n_qubit, n_sample):
    ''' Collect classical shadow on scrambled GHZ state by Pauli measurements
        
        Input:
        n_qubit: int - number of qubits
        n_sample: int - number of samples
        
        Output:
        shd: Shadow - classical shadow dataset '''
    rho = qst.stabilizer_state(qst.paulis(['+ZYII', '-IYZI', 'IIZZ', '-YZYY']))
    '''Original GHZ: stabilized by ZZII, IZZI, IIZZ, XXXX
       Pauli Map used for GHZ here is
       X0-> +YIII
       Z0-> -ZIII
       X1-> +IZII
       Z1-> -IYII
       X2-> +IIYI
       Z2-> +IIZI
       X3-> -IIIY
       Z3-> +IIIZ'''
    obs = []
    out = []
    for _ in range(n_sample):
        sigma = qst.random_pauli_state(n_qubit)
        bit = rho.copy().measure(sigma)[0]
        tok = sigma.tokenize()
        obs.append(tok[:n_qubit,:n_qubit].diagonal())
        out.append((tok[:,-1]+bit)%2)
    obs = torch.tensor(numpy.stack(obs))
    out = torch.tensor(numpy.stack(out))
    return Shadow(obs, out)

# collect measurement data for given Pauli observable
def state_measurement(rho, n_qubit, n_sample, string):
    ''' Collect classical shadow on GHZ state by Pauli measurements
        
        Input:
        n_qubit: int - number of qubits
        n_sample: int - number of samples
        
        Output:
        shd: Shadow - classical shadow dataset '''
    obs = []
    out = []
    sigma0 = [i*'I'+string[i]+(n_qubit-i-1)*'I' for i in range (n_qubit)]
    for _ in range(n_sample):
        sigma = qst.stabilizer_state(qst.paulis(sigma0))
        bit = rho.copy().measure(sigma)[0]
        tok = sigma.tokenize()
        obs.append(tok[:n_qubit,:n_qubit].diagonal())
        out.append((tok[:,-1]+bit)%2)
    obs = torch.tensor(numpy.stack(obs))
    out = torch.tensor(numpy.stack(out))
    return Shadow(obs, out)

from IPython.display import clear_output
import os
class ClassicalShadowTransformer(torch.nn.Module):
    ''' Classical shadow transformer.
    
        Parameters: 
        N: int - number of qubits '''
    def __init__(self, n_qubit, logbeta, state_name='GHZ', **kwargs):
        super(type(self), self).__init__()
        self.transformer = Transformer(**kwargs)
        self.optimizer = torch.optim.Adam(self.transformer.parameters())
        self.register_buffer('token_map', torch.tensor([0,0,1,0,1,2,3])) # [bos,-1,+1,I,X,Y,Z]
        self.n_qubit = n_qubit       # number of qubits
        self.logbeta = logbeta       # log2 of hyperparameter beta
        self.state_name = state_name # name of the quantum state to learn
        self.loss_history = []

    @property
    def device(self):
        return self.token_map.device

    @property
    def path(self):
        # path name convention: CST_d[embed_dim]_h[num_heads]_l[n_layers]
        name = './model/CST'
        name += f'_d{self.transformer.token_embd.embedding_dim}'
        name += f'_h{self.transformer.encode.layers[0].attn.num_heads}'
        name += f'_l{len(self.transformer.encode.layers)}'
        return name

    @property
    def file(self):
        # file name convention: [state_name]_N[n_qubit]_b[logbeta]
        return  f'{self.state_name}_N{self.n_qubit}_b{self.logbeta}'

    def shadow(self, n_sample):
        if self.state_name == 'GHZ':
            rho = qst.ghz_state(self.n_qubit)
        else:
            raise NotImplementedError
        obs = []
        out = []
        for _ in range(n_sample):
            sigma = qst.random_pauli_state(self.n_qubit)
            bit = rho.copy().measure(sigma)[0]
            tok = sigma.tokenize()
            obs.append(tok[:self.n_qubit,:self.n_qubit].diagonal())
            out.append((tok[:,-1]+bit)%2)
        obs = torch.tensor(numpy.stack(obs))
        out = torch.tensor(numpy.stack(out))
        return Shadow(obs, out)
        
    def sample(self, n_sample, need_logprob=False):
        ''' Sample a batch of classical shadows.
            n_sample: int - number of samples '''
        src_seq = torch.randint(4, 7, (n_sample, self.n_qubit), device=self.device)
        tgt_seq = self.transformer.sample(src_seq, self.n_qubit)
        obs = self.token_map[src_seq]
        out = self.token_map[tgt_seq]
        if need_logprob:
            logprob = self.transformer.logprob(src_seq, tgt_seq)
            return Shadow(obs, out), logprob 
        else:
            return Shadow(obs, out)
    
    def logprob(self, shadow):
        ''' Evaluate log probability of classical shadows
            shadow: Shadow - a set of classical shadows '''
        src_seq, tgt_seq = shadow.tokenize() # 数据格式转换
        return self.transformer.logprob(src_seq, tgt_seq)

    def loss(self, shadow):
        ''' Evaluate loss function of classical shadows
            shadow: Shadow - a set of classical shadows '''
        src_seq, tgt_seq = shadow.tokenize() # 数据格式转换
        return self.transformer.loss(src_seq, tgt_seq, beta=2.**self.logbeta)
    
    def latent(self, shadow):
        src_seq, tgt_seq = shadow.tokenize()
        src = self.transformer.embed(src_seq)
        z = self.transformer.encode(src)
        return z
    
    # reconstruct density matrix
    def rho(self):
        obs = torch.cartesian_prod(*([torch.arange(1, 4, device=self.device)]*self.n_qubit)).view(-1, self.n_qubit)
        out = torch.cartesian_prod(*([torch.arange(2, device=self.device)]*self.n_qubit)).view(-1, self.n_qubit)
        n_obs, n_out = obs.shape[0], out.shape[0]
        obs = obs.unsqueeze(1).expand(n_obs, n_out, self.n_qubit).reshape(-1, self.n_qubit)
        out = out.unsqueeze(0).expand(n_obs, n_out, self.n_qubit).reshape(-1, self.n_qubit)
        shadow = Shadow(obs, out)
        weight = self.logprob(shadow).softmax(-1)
        paulis = torch.cartesian_prod(*([torch.arange(4, device=self.device)]*self.n_qubit)).view(-1, self.n_qubit)
        shadow_map = shadow.shadow_map(paulis).to(weight)
        return Operator(paulis, weight @ shadow_map)

    def can_stop(self, nsr=2, window=100):
        if len(self.loss_history) < window:
            return False
        else:
            losses = torch.tensor(self.loss_history[-window:])
            std = losses.std()
            mean0 = losses[:(window//2)].mean()
            mean1 = losses[(-window//2):].mean()
            return abs(mean0 - mean1) < std/nsr

    def optimize(self, steps, max_steps=1000, n_sample=1000, lr=0.0001, autosave=10, **kwargs):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self.transformer.train()
        for step in range(max_steps):
            if step >= steps and self.can_stop(**kwargs):
                break
            self.optimizer.zero_grad()
            shadow = self.shadow(n_sample).to(self.device)
            loss, logprob, kld = self.loss(shadow)
            loss.backward()
            self.optimizer.step()
            self.loss_history.append(loss.item())
            clear_output(wait=True)
            print(self.path + '/' + self.file)
            print(f'{step:3d}: {loss.item():8.5f} {logprob.item():8.5f} {kld.item():8.5f} {self.transformer.repara.logvar.mean().item():8.5f}')
            if autosave!=0 and (step+1)%autosave == 0:
                self.save()
        if autosave:
            self.save()
    
    def save(self):
        state_dict = {'model_state_dict': self.transformer.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'loss_history': self.loss_history}
        os.makedirs(self.path, exist_ok = True) 
        torch.save(state_dict, self.path + '/' + self.file)
        
    def load(self, filename=None):
        if filename is None:
            filename = self.path + '/' + self.file
        if os.path.exists(filename):
            state_dict = torch.load(filename, map_location=self.device)
            self.transformer.load_state_dict(state_dict['model_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            self.loss_history = state_dict['loss_history']
        return self

class Logger():
    def __init__(self, string='{:8.5f}'):
        self.string = string
        self._reset()
        
    def _reset(self):
        self.cum = None
        self.count = 0
        
    def add(self, *rec):
        if self.cum is None:
            self.cum = list(rec)
        else:
            for i in range(len(self.cum)):
                self.cum[i] += rec[i]
        self.count += 1
    
    def pop(self):
        avgs = []
        for i in range(len(self.cum)):
            avg = self.cum[i]/self.count
            avgs.append(avg.item())
        self._reset()
        return ' '.join(self.string.format(avg) for avg in avgs)
logger = Logger()

