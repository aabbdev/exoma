from typing import Any, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from fastfeedforward import FastFeedForward
import numpy as np
from tokenizer import Tokenizer
from scipy.fft import dct


@dataclass
class ModelArgs:
    dim: int = 2048
    expand_factor: int = 2
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = 259
    norm_eps: float = 1e-06
    max_batch_size: int = 32
    max_seq_len: int = 2048

def dct_pytorch(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.rfft(v, 1, onesided=False)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V
class LRUCell(nn.Module):
  '''
  Decoupled Projected state, Reset Gate and Output Gate

  LRU is 2-dimensional, hence it has 2 inputs and 2 outputs
  
  Args:
    hidden_size: The number of features in the hidden state h (also called the size of LRU),
                also number of features of the input (both dimensions have the same input)
    bias: If `False`, then the layer does not use bias weights `b_ih` and `b_hh`. Default: True

  Inputs: (h1,h2) or (h1) or h1
    - **h1** (batch, hidden_size): tensor containing h1 features
    - **h2** (batch, hidden_size): (optional; will be set to zero if not provided) 
                                   tensor containing h2 features

  Outputs: h1, h2
    - **h1** (batch, hidden_size): tensor containing the next h1
    - **h2** (batch, hidden_size): tensor containing the next h2

  '''
  def __init__(self, hidden_size, bias=True):
    super(LRUCell, self).__init__()
    self.lin = nn.Linear(hidden_size*2, hidden_size*4, bias=bias)
    self.gate0 = nn.Linear(hidden_size*2, hidden_size)
    self.gate1 = nn.Linear(hidden_size*2, hidden_size)

    self.reset_parameters([self.lin, self.gate0, self.gate1], hidden_size)
    
  def reset_parameters(self, module_list, hidden_size):
    ## glorot initialization
    stdv = 1. / math.sqrt(hidden_size)
    for module in module_list:
      for i, param in enumerate(module.parameters()):
        param.data.uniform_(-stdv, stdv)

  def forward(self, inputs):
    if type(inputs) is not tuple:
      inputs = tuple([inputs])
    if len(inputs) == 1:
      inputs = tuple([inputs[0], Variable(inputs[0].data.new(*inputs[0].size()).zero_())])
      
    inputs_cat = torch.cat(inputs, dim=1)
    g = F.sigmoid(self.lin(inputs_cat))
    z0, z1, r, q = torch.chunk(g, chunks=4, dim=1)
    h0_cap = torch.cat([inputs[0], inputs[1]*r], dim=1)
    h1_cap = torch.cat([inputs[1], inputs[0]*q], dim=1)                   
    h0_cap = F.tanh(self.gate0(h0_cap))
    h1_cap = F.tanh(self.gate1(h1_cap))

    return z0*h1_cap + (1.-z0)*inputs[0], z1*h0_cap + (1.-z1)*inputs[1]

class FastFeedForward(nn.Module):
    def __init__(self, features: int, recursion: int = 1, init = 'hyperspherical-shell'):
        super().__init__()
        self.recursion = recursion
        self.features = features
        self.depth = int(floor(log2(features)))
        nodes = 2 ** self.depth - 1
        if init == 'gaussian':
            # This from original authors
            def create_basis_vectors_of(length, scaling):
                return nn.Parameter(torch.empty(nodes, length).uniform_(-scaling, scaling))
            self.X = create_basis_vectors_of(length=features, scaling=1/sqrt(features))
            self.Y = create_basis_vectors_of(length=features, scaling=1/sqrt(self.depth + 1))
        elif init == 'hyperspherical-shell':
            def create_random_unit_vectors_of(length):
                weights = torch.randn(nodes, length)  # Initialize weights randomly
                weights = F.normalize(weights, p=2, dim=-1)  # L2-Normalize along the last dimension
                return nn.Parameter(weights)
            self.X = create_random_unit_vectors_of(length=features)
            self.Y = create_random_unit_vectors_of(length=features)
    def forward(self, oldx: torch.Tensor):
        x = oldx.reshape(-1, self.features)
        batch_size = x.shape[0]
        current_node = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        y = torch.zeros((batch_size, self.features), dtype=torch.float, device=x.device)
        for _ in range(self.depth):
            λ = torch.einsum("b i, b i -> b", x, self.X[current_node])
            y += torch.einsum("b, b j -> b j", λ, self.Y[current_node])
            branch_choice = (λ > 0).long()
            current_node = (current_node * 2) + 1 + branch_choice
        return y.reshape_as(oldx)
    def __repr__(self):
        return f"FastFeedForward({self.X.shape[-1]}, {self.Y.shape[-1]}, depth={self.depth})"

class TemporalConv1d(nn.Module):
    def __init__(self, inner_dim, conv_dim):
        super(TemporalConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=inner_dim,
            out_channels=inner_dim,
            kernel_size=conv_dim,
            groups=inner_dim,
            padding=conv_dim - 1,
            bias=False)
    def forward(self, x):
        _, L, _ = x.shape
        x = x.transpose(1, 2)
        x = self.conv(x)[:,:,:L]
        return x.transpose(1, 2)
class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x = self._norm(x.float()).type_as(x)
        if self.add_unit_offset:
            output = x * (1 + self.weight)
        else:
            output = x * self.weight
        return output

class Sampler(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
    @torch.no_grad()
    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Select the last element for each sequence.
        # (batch_size, input_len, hidden_size) -> (batch_size, hidden_size)
        hidden_states = hidden_states.index_select(
            1, output_positions).squeeze(dim=1)
        logits = torch.matmul(hidden_states, embedding.t())
        if embedding_bias is not None:
            logits += embedding_bias

        if temperatures is None:
            return torch.argmax(logits, dim=-1).squeeze(dim=-1)

        # Apply temperature scaling.
        logits.div_(temperatures.unsqueeze(dim=1))

        # Calculate probabilities with softmax.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

        # Apply top-p, top-k.
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_ps_mask = (probs_sum - probs_sort) > top_ps.unsqueeze(dim=1)
        probs_sort = torch.where(top_ps_mask, 0, probs_sort)

        top_ks_mask = torch.arange(probs_idx.shape[-1],
                                   device=probs_idx.device)
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
        top_ks_mask = top_ks_mask >= top_ks.unsqueeze(dim=1)
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)

        # Re-normalization.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        probs = torch.gather(probs_sort,
                             dim=-1,
                             index=torch.argsort(probs_idx, dim=-1))

        next_token_ids = torch.multinomial(probs,
                                           num_samples=1,
                                           replacement=True).squeeze(dim=-1)
        return next_token_ids

class ExomaAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.hidden_size = hidden_size
        self.head_dim = head_dim

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.scaling = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(self.hidden_size, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.proj_matrix = self._build_projection()
    def _build_projection(self):
        icdf_w = torch.distributions.Normal(0, 1).icdf(torch.diag_embed(torch.diag(torch.rand(self.head_dim, self.head_dim))))
        icdf_w = torch.where(torch.isinf(icdf_w), torch.full_like(icdf_w, 0), icdf_w)
        C = dct(torch.eye(self.head_dim, self.head_dim), axis=0,norm='ortho')
        C = C.type(torch.FloatTensor)
        return nn.Parameter((C @ icdf_w).contiguous(), requires_grad=False)
    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3

        batch_size, input_len, _ = hidden_states_shape

        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        # Write new kv cache.
        # [batch_size, input_len, n_local_kv_heads, head_dim]
        k_cache, v_cache = kv_cache
        k_cache.index_copy_(1, kv_write_indices, xk)
        v_cache.index_copy_(1, kv_write_indices, xv)

        key = k_cache
        value = v_cache
        if self.num_kv_heads != self.num_heads:
            # [batch_size, max_seq_len, n_local_heads, head_dim]
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(value, self.num_queries_per_kv, dim=2)

        # [batch_size, n_local_heads, input_len, head_dim]
        q = xq.transpose(1, 2)
        # [batch_size, n_local_heads, max_seq_len, head_dim]
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        def selfAttention(q, k, v):
            # [batch_size, n_local_heads, input_len, max_seq_len]
            scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
            scores = scores + mask
            scores = F.softmax(scores.float(), dim=-1).type_as(q)
            # [batch_size, n_local_heads, input_len, head_dim]
            output = torch.matmul(scores, v)
            return output
        def DCTAttention(q, k, v):
            query = nn.functional.softmax(torch.matmul(q, self.proj_matrix), dim=-1)
            key = nn.functional.softmax(torch.matmul(k.transpose(2, 3), self.proj_matrix), dim=-1)
            scores = torch.matmul(key, v)
            return torch.matmul(scores, query)
        output = DCTAttention(q, k, v)
        # [batch_size, input_len, hidden_dim]
        return self.o_proj(output.transpose(1, 2).contiguous().view(batch_size, input_len, -1))


class ExomaAttentionDecoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = ExomaAttention(args.dim, args.n_heads, args.n_heads, args.dim // args.n_heads)
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.mlp = nn.Sequential(
            RMSNorm(args.dim, eps=args.norm_eps),
            FastFeedForward(features=args.dim)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
         # DCT Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(
            hidden_states=hidden_states,
            kv_write_indices=kv_write_indices,
            kv_cache=kv_cache,
            mask=mask,
        )
        hidden_states = residual + hidden_states
        # MLP
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class ExomaRecurrentDecoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        inner_dim = (args.expand_factor * args.dim)
        convDim = 4
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.branch_one = nn.Sequential(
            nn.Linear(args.dim, inner_dim, bias=False),
            nn.GELU()
        )
        self.branch_second = nn.Sequential(
            nn.Linear(args.dim, inner_dim, bias=False),
            TemporalConv1d(inner_dim, convDim),
            LRUCell(inner_dim, bias=False)
        )
        self.feed_forward = nn.Sequential(
            RMSNorm(args.dim, eps=args.norm_eps),
            FastFeedForward(features=args.dim)
        )
        self.tail = nn.Linear(inner_dim, args.dim, bias=False)
    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        
        x_norm = self.input_layernorm(hidden_states)

        branch_one = self.branch_one(x_norm)
        branch_second = self.branch_second(x_norm)

        h = hidden_states + self.tail(branch_second * branch_one)
        return h + self.feed_forward(h)

class ExomaModel(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        ratio = 7
        self.layers = torch.nn.ModuleList()
        for _ in range(params.n_layers // 7):
            for _ in range(math.ceil(ratio/2)):
                self.layers.append(ExomaRecurrentDecoder(args=params))
            self.layers.append(ExomaAttentionDecoder(args=params))
            for _ in range(math.floor(ratio/2)):
                self.layers.append(ExomaRecurrentDecoder(args=params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                kv_write_indices=kv_write_indices,
                kv_cache=kv_caches[i],
                mask=mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states

class ExomaForCausalLM(nn.Module):
    def __init__( self, config: ModelArgs):
        super().__init__()
        self.config = config
        assert config.dim % config.n_heads == 0
        vocab_size = config.vocab_size

        self.tokenizer = Tokenizer()
        self.embedder = nn.Embedding(vocab_size, config.dim)
        self.model = ExomaModel(config)
        self.sampler = Sampler(vocab_size)

    @torch.no_grad()
    def forward(
        self,
        input_token_ids: torch.Tensor,
        input_positions: torch.Tensor,
        kv_write_indices: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
        output_positions: torch.Tensor,
        temperatures: Union[torch.Tensor, None],
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        kv_write_indices = input_positions

        # [batch_size, input_len, hidden_size]
        hidden_states = self.embedder(input_token_ids)
        # Ex normalizes the embedding by sqrt(hidden_size).
        hidden_states = hidden_states * (self.config.dim**0.5)

        hidden_states = self.model(
            hidden_states=hidden_states,
            kv_write_indices=kv_write_indices,
            kv_caches=kv_caches,
            mask=mask,
        )
        embedder_weight = self.embedder.weight
        next_tokens = self.sampler(
            embedding=embedder_weight,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )
        return next_tokens

    def generate(
        self,
        prompts: Union[str, Sequence[str]],
        device: Any,
        output_len: int = 100,
        temperature: Union[float, None] = 0.95,
        top_p: float = 1.0,
        top_k: int = 100,
    ) -> Union[str, Sequence[str]]:
        """Generates responses for given prompts using Exoma model."""
        # If a single prompt is provided, treat it as a batch of 1.
        is_str_prompt = isinstance(prompts, str)
        if is_str_prompt:
            prompts = [prompts]

        batch_size = len(prompts)
        prompt_tokens = [self.tokenizer.encode(prompt, True, True) for prompt in prompts]
        min_prompt_len = min(len(p) for p in prompt_tokens)
        max_prompt_len = max(len(p) for p in prompt_tokens)
        max_seq_len = max_prompt_len + output_len
        assert max_seq_len <= self.config.max_seq_len

        # build KV caches
        kv_caches = []
        for _ in range(self.config.n_layers):
            size = (batch_size, max_seq_len, self.config.n_heads, self.config.dim)
            dtype = torch.float32
            k_cache = torch.zeros(size=size, dtype=dtype, device=device)
            v_cache = torch.zeros(size=size, dtype=dtype, device=device)
            kv_caches.append((k_cache, v_cache))

        # prepare inputs
        token_ids_tensor = torch.full((batch_size, max_seq_len), self.tokenizer.pad_id, dtype=torch.int64)
        input_token_ids_tensor = torch.full((batch_size, min_prompt_len), self.tokenizer.pad_id, dtype=torch.int64)
        for i, p in enumerate(prompt_tokens):
            token_ids_tensor[i, :len(p)] = torch.tensor(p)
            input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(p[:min_prompt_len])
        token_ids_tensor = token_ids_tensor.to(device)
        input_token_ids_tensor = input_token_ids_tensor.to(device)
        prompt_mask_tensor = token_ids_tensor != self.tokenizer.pad_id
        input_positions_tensor = torch.arange(0, min_prompt_len, dtype=torch.int64).to(device)
        mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len), -2.3819763e38).to(torch.float)
        mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)
        temperatures_tensor = None if not temperature else torch.FloatTensor([temperature] * batch_size).to(device)
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        output_index = torch.tensor(min_prompt_len, dtype=torch.int64).to(device)

        # Prefill up to min_prompt_len tokens, then treat other prefill as
        # decode and ignore output.
        for i in range(max_seq_len - min_prompt_len):
            next_token_ids = self(
                input_token_ids=input_token_ids_tensor,
                input_positions=input_positions_tensor,
                kv_write_indices=None,
                kv_caches=kv_caches,
                mask=curr_mask_tensor,
                output_positions=output_positions_tensor,
                temperatures=temperatures_tensor,
                top_ps=top_ps_tensor,
                top_ks=top_ks_tensor,
            )

            curr_prompt_mask = prompt_mask_tensor.index_select(1, output_index).squeeze(dim=1)
            curr_token_ids = token_ids_tensor.index_select(1, output_index).squeeze(dim=1)
            output_token_ids = torch.where(curr_prompt_mask, curr_token_ids, next_token_ids).unsqueeze(dim=1)
            token_ids_tensor.index_copy_(1, output_index, output_token_ids)

            input_token_ids_tensor = output_token_ids
            input_positions_tensor = output_index.unsqueeze(dim=-1)
            curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
            output_positions_tensor = torch.tensor(0, dtype=torch.int64).to(device)
            output_index = output_index + 1

        # Detokenization.
        token_ids = token_ids_tensor.tolist()
        results = []
        for i, tokens in enumerate(token_ids):
            trimmed_output = tokens[len(prompt_tokens[i]):len(prompt_tokens[i]) + output_len]
            if self.tokenizer.eos_id in trimmed_output:
                eos_index = trimmed_output.index(self.tokenizer.eos_id)
                trimmed_output = trimmed_output[:eos_index]
            results.append(self.tokenizer.decode(trimmed_output))

        # If a string was provided as input, return a string as output.
        return results[0] if is_str_prompt else results

    def load_weights(self, model_path: str):
        self.load_state_dict(
            torch.load(
                model_path, mmap=True, weights_only=True,
            )['model_state_dict'],
            strict=False,
        )
    def save_weights(self, model_path: str):
        torch.save(
            {
                'model_state_dict': self.state_dict(),
            },
            model_path,
        )
if __name__ == "__main__":
    model = ExomaForCausalLM(ModelArgs())
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(model)
    print("Total number of parameters :", params / 1e6, "M")