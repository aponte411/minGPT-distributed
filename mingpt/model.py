"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
3) https://github.com/subramen/minGPT-ddp/blob/master/mingpt/model.py
"""

from dataclasses import dataclass
import math
from typing import Union

import torch
import torch.nn as nn
from torch.nn import functional as F


def print_model_size(model: nn.Module):
    """Print model size by considering
    params and buffers.
    https://discuss.pytorch.org/t/finding-model-size/130275
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    total_size = (param_size + buffer_size) / 1024**2
    print(f"Model size (MB): {total_size:.3f}")


# -----------------------------------------------------------------------------
# Handy to make configuration modular when using with tools like Hydra
@dataclass
class GPTConfig:
    model_type: str = "gpt2"
    # Model configuration
    n_layer: int = None
    n_head: int = None
    n_embed: int = None
    # GPT2 values from OpenAI
    vocab_size: int = 50257
    block_size: int = 1024
    # Dropout configuration
    embed_drop: float = 0.1
    resid_drop: float = 0.1
    attn_drop: float = 0.1


@dataclass
class OptimizerConfig:
    # Optimizer configuration
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)


def create_optimizer(model: torch.nn.Module, optimizer_config: OptimizerConfig):
    """
    Separate params into those with and without weight decay.
    Without decay: bias, layernorm, embeddings
    """
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=optimizer_config.learning_rate,
        betas=optimizer_config.betas,
    )
    return optimizer

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with additional projection.
    """

    def __init__(self,
                 config: GPTConfig,
                 device: str = "cpu",
                 dtype: Union[torch.FloatTensor,
                              torch.HalfTensor] = torch.float32):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        # regularization
        self.resid_drop = nn.Dropout(config.resid_drop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size))
        # Use torch implementation
        self.attn = torch.nn.MultiHeadAttention(
            embed_dim=config.n_embed,
            num_heads=config.n_head,
            dropout=config.attn_drop,
            batch_first=True,
            device=device,
            dtype=dtype,
        )

    def forward(self, x):
        # Input (batch, seq, hidden)
        batch, seq_len, hidden = x.size()
        # Self-attention: (batch, n_head, seq, hidden) x (batch, n_head, hidden, seq) -> (batch, n_head, seq, seq)
        y = self.attn(
            x,  # query
            x,  # key
            x,  # value
            attn_mask=self.mask[0, 0, :seq_len, :seq_len],
        )[0]  # grab attn output
        # output projection
        y = self.resid_drop(self.c_proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = MultiHeadSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            nn.Linear(4 * config.n_embed, config.n_embed),
            nn.GELU(),
            nn.Dropout(config.resid_drop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


# Embeddings
class GPTEmbedding(nn.Module):

    def __init__(self,
                 config: GPTConfig,
                 device: str = "cpu",
                 dtype: Union[torch.FloatTensor,
                              torch.HalfTensor] = torch.float32):
        super().__init__()
        # Token embeddings map indices to learned embedding vectors
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embed,
            device=device,
            dtype=dtype,
        )
        # Postional embeddings map indices to learned positional vectors
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, config.block_size, config.n_embed),
            device=device,
            dtype=dtype,
        )
        self.drop = nn.Dropout(config.embed_drop)
        # Sequence length must be <= block size
        self.block_sz = config.block_size

    def forward(self, x):
        # Embedding layers is given a batch of indices
        batch, seq_len = x.size()
        assert seq_len <= self.block_size, "Sequence length must be less than or equal to block size for positional embeddings."
        token_embeddings = self.token_embedding(x)
        # Grab everything up to sequence length
        pos_embeddings = self.pos_embedding(x)[:, :seq_len, :]
        # Combine embedding vectors
        embeddings = token_embeddings + pos_embeddings
        return self.drop(embeddings)


class GPT(nn.Module):
    """ GPT Language Model """

    def __init__(self, config: GPTConfig):
        super().__init__()
        # Check configuration and populate with defaults from huggingface, openai, etc.
        config = self.set_model_config(config)
        self.block_size = config.block_size
        # Embeddings
        self.embedding = GPTEmbedding(config)
        # Transformer blocks
        self.blocks = nn.Sequential(
            *[Block(config) for layer in range(config.n_layer)])
        # Decoder head
        self.ln_ff = nn.LayerNorm(config.n_embed)
        self.head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        # Initialize weights
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            # Per GPT-2 paper, we initialize projection weights with mean=0.0, std=0.02/sqrt(2*config.n_layer)
            if name.endswith("c_proj.weight"):
                param.data.normal_(mean=0.0,
                                   std=0.02 / math.sqrt(2 * config.n_layer))
        n_total_params = sum([param.numel() for param in self.parameters()])
        print(f"Number of parameters: {n_total_params}")
        print_model_size(self)

    def _set_model_config(self, config: GPTConfig) -> GPTConfig:
        type_given = config.model_type is not None
        params_given = all([
            config.n_layer is not None, config.n_head is not None,
            config.n_embed is not None
        ])
        if type_given and params_given:
            # translate from model_type to detailed configuration
            config.__dict__.update({
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':
                dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':
                dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':
                dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
                'gpt2-large':
                dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
                'gpt2-xl':
                dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
                # Gophers
                'gopher-44m':
                dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':
                dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':
                dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':
                dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])
        print(config)
        return config

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config: OptimizerConfig):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(
                        m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(
                        m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(
            inter_params
        ) == 0, "parameters %s made it into both decay/no_decay sets!" % (
            str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0
            },
        ]
        optimizer = torch.optim.AdamW(optim_groups,
                                      lr=train_config.learning_rate,
                                      betas=train_config.betas)
        return optimizer

    def forward(self, inputs, targets=None):
        x = self.embeddings(inputs)
        x = self.blocks(x)
        logits = self.decoder_head(x)
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1),
                                   ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self,
                 idx,
                 max_new_tokens,
                 temperature=1.0,
                 do_sample=False,
                 top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(
                1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
