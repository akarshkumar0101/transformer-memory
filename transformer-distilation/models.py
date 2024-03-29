import torch
from torch import nn
import numpy as np
from einops import rearrange, repeat


class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.0, bias=True):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.0, bias=True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, bias=bias, batch_first=True)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd=n_embd, dropout=dropout, bias=bias)

    def attn_forward(self, x, mask):
        self.mask = mask
        x, self.attn_weights = self.attn(x, x, x, attn_mask=~mask, need_weights=True, average_attn_weights=False)
        return x

    def forward(self, x, mask):
        x = x + self.attn_forward(self.ln_1(x), mask=mask)
        x = x + self.mlp(self.ln_2(x))
        return x

    # def attn_forward(self, x, x_mem, mask):
    #     kv = x if x_mem is None else torch.cat([x_mem, x], dim=-2)
    #     x, self.attn_weights = self.attn(x, kv, kv, attn_mask=~mask, need_weights=True, average_attn_weights=False)
    #     self.mask = mask
    #     return x

    # def forward(self, x, x_mem, mask):
    #     x = x + self.attn_forward(self.ln_1(x), None if x_mem is None else self.ln_1(x_mem), mask=mask)
    #     x = x + self.mlp(self.ln_2(x))
    #     return x


class GPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_layer, n_head, dropout=0.0, bias=True, mask="causal"):
        super().__init__()
        self.vocab_size, self.block_size, self.n_embd, self.n_layer, self.n_head = vocab_size, block_size, n_embd, n_layer, n_head
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout, bias) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / np.sqrt(2 * n_layer))

        if mask == "causal":
            self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)))
        elif mask == "full":
            self.register_buffer("mask", torch.ones(block_size, block_size, dtype=torch.bool))
        else:
            raise ValueError(f"Unknown mask type {mask}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tok):
        b, t = tok.shape
        assert t <= self.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=tok.device)  # shape (t)
        x = self.drop(self.wte(tok) + self.wpe(pos))
        for block in self.blocks:
            x = block(x, mask=self.mask[:t, :t])
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    # def loss_fn(self, tok):
    #     b, t = tok.shape
    #     x, y = tok[:, :-1], tok[:, 1:]
    #     logits = self(x)
    #     return torch.nn.functional.cross_entropy(logits.reshape(-1, self.vocab_size), y.reshape(-1), reduction="none").reshape(b, -1)

    # def forward(self, tok, x_mems_in=None, get_mem_out=False):
    #     x_mems_in = [None] * len(self.blocks) if x_mems_in is None else x_mems_in
    #     x_mems_out = []
    #     b, t = tok.shape
    #     assert t <= self.block_size

    #     pos = torch.arange(0, t, dtype=torch.long, device=tok.device)  # shape (t)
    #     x = self.drop(self.wte(tok) + self.wpe(pos))
    #     for block, x_mem_in in zip(self.blocks, x_mems_in):
    #         x_mems_out.append(x)
    #         x = block(x, x_mem_in, mask=self.mask)
    #     x = self.ln_f(x)
    #     logits = self.lm_head(x)
    #     return (logits, x_mems_out) if get_mem_out else logits


# class CompressionGPT(GPT):
#     def __init__(self, vocab_size, block_size, n_embd, n_layer, n_head, dropout=0.0, bias=True):
#         super().__init__(vocab_size, block_size, n_embd, n_layer, n_head, dropout, bias)
#         self.half = block_size // 2
#         self.mask[self.half :, : self.half - 1] = False
#         self.mask[: self.half, : self.half] = True

#     def forward(self, tok):
#         self.mask[: self.half, : self.half] = True
#         self.mask[: self.half, : self.half] = True
#         super().forward(tok)


# class Compression2AK(nn.Module):
#     def __init__(self, vocab_size, block_size, n_embd, n_layer, n_head, dropout=0.0, bias=True):
#         super().__init__()
#         self.encoder = GPT(vocab_size, block_size, n_embd, n_layer, n_head, dropout, bias, mask="causal")
#         self.decoder = GPT(vocab_size, block_size, n_embd, n_layer, n_head, dropout, bias, mask="causal")

#     def forward(self, tok_a, tok_b):
#         b, ta = tok_a.shape
#         b, tb = tok_b.shape
#         assert ta <= self.encoder.block_size
#         self.encoder.mask = torch.tril(torch.ones(ta, ta, dtype=bool, device=tok_a.device), diagonal=0)
#         self.decoder.mask = torch.tril(torch.ones(tb, tb + 1, dtype=bool, device=tok_a.device), diagonal=1)
#         _, x_mem = self.encoder(tok_a, get_mem_out=True)
#         x_mem = [x_mem_i[:, [-1], :] for x_mem_i in x_mem]
#         logits, _ = self.decoder(tok_b, x_mems_in=x_mem, get_mem_out=True)
#         return logits

#     def loss_fn(self, tok, i=None):
#         b, t = tok.shape
#         if i is None:
#             i = np.random.randint(1, t)
#         tok_a, tok_b, y = tok[:, :i], tok[:, i:-1], tok[:, i + 1 :]
#         logits = self(tok_a, tok_b)
#         return torch.nn.functional.cross_entropy(logits.reshape(-1, self.encoder.vocab_size), y.reshape(-1), reduction="none").reshape(b, -1)


class WeirdGPT(GPT):
    def __init__(self, vocab_size, block_size, n_embd, n_layer, n_head, dropout=0.0, bias=True):
        super().__init__(vocab_size, block_size, n_embd, n_layer, n_head, dropout, bias)
        self.i_dec = block_size // 2

        self.mask = torch.tril(torch.ones(block_size, block_size, dtype=bool, device="cpu"))
        self.mask[: self.i_dec, : self.i_dec] = True
        self.mask[self.i_dec :, : self.i_dec] = False
        self.mask[:, self.i_dec - 1] = True

    def forward(self, tok):
        b, t = tok.shape
        assert t == self.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=tok.device)  # shape (t)
        x = self.drop(self.wte(tok) + self.wpe(pos))
        for block in self.blocks:
            x = block(x, mask=self.mask[:t, :t])
        x = self.ln_f(x)
        logits = self.lm_head(x)
        logits = torch.where(rearrange(pos < self.i_dec, "t -> 1 t 1"), torch.nan, logits)
        return logits


class CompressionGPT(GPT):
    def __init__(self, vocab_size, block_size, n_embd, n_layer, n_head, dropout=0.0, bias=True, mode="random"):
        super().__init__(vocab_size, block_size, n_embd, n_layer, n_head, dropout, bias)
        self.wpe_enc = nn.Embedding(block_size, n_embd)
        self.mode = mode
        self.apply(self._init_weights)

    def forward(self, tok, mode=None):
        if mode is None:
            mode = self.mode
        b, t = tok.shape
        assert t <= self.block_size
        if mode.startswith("random"):
            self.idxs_dec = torch.randint(0, t, size=(b,), device=tok.device)
        elif mode.startswith("encode"):
            self.idxs_dec = torch.full((b,), fill_value=t - 1, dtype=torch.long, device=tok.device)
        elif mode.startswith("decode"):
            self.idxs_dec = torch.full((b,), fill_value=0, dtype=torch.long, device=tok.device)
        elif mode.startswith("half"):
            self.idxs_dec = torch.full((b,), fill_value=t // 2, dtype=torch.long, device=tok.device)
        else:
            raise ValueError(f"Unknown mode {mode}")
        pos = torch.arange(0, t, dtype=torch.long, device=tok.device)  # shape (t)
        self.mask = self.create_compression_attn_mask(tok, self.idxs_dec, mode)
        self.batch_mask = self.create_compression_batch_mask(pos, self.idxs_dec)

        # problem is either in masking position encodings or masking attention because removing both: works
        # position encoding seems problematic

        x = self.drop(self.wte(tok) + torch.where(self.batch_mask[:, :, None], self.wpe_enc(pos), self.wpe(pos)))
        # x = self.drop(self.wte(tok) + self.wpe(pos))
        activations = []
        for block in self.blocks:
            activations.append(x)
            x = block(x, mask=repeat(self.mask, "b t1 t2 -> (b h) t1 t2", h=self.n_head))
            # x = block(x, mask=self.mask[0])
        activations.append(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        logits = torch.where(self.batch_mask[:, :, None], torch.nan, logits)
        if mode.startswith("encode"):
            return torch.stack(activations, dim=1)
        return logits

    def create_compression_attn_mask(self, tok, idxs_dec, mode):
        b, t = tok.shape
        mask = torch.tril(torch.ones(b, t, t, dtype=torch.bool, device=idxs_dec.device))
        for i, i_dec in enumerate(idxs_dec):
            if mode.endswith("full"):
                mask[i, :i_dec, :i_dec] = True
            mask[i, i_dec:, :i_dec] = False
            mask[i, i_dec:, torch.clamp(i_dec - 1, 0, None)] = True
        return mask  # (b, t, t)

    def create_compression_batch_mask(self, pos, idxs_dec):
        return rearrange(pos, "t -> 1 t") < rearrange(idxs_dec, "b -> b 1")  # (b, t)


class MyRNN(nn.RNN):
    def __init__(self, vocab_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.embd = nn.Embedding(vocab_size, self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, vocab_size)

    def forward(self, input, h_0=None, return_hidden_states=False):
        x = self.embd(input)
        x, h_n = super().forward(x, h_0)
        logits = self.lm_head(x)
        if return_hidden_states:
            return logits, h_n
        return logits




class OneStepMLP(nn.Module):
    def __init__(self, n_embd, dropout=0.0, bias=True):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(2*n_embd, 2 * n_embd, bias=bias),
            nn.GELU(),
            nn.Linear(2*n_embd, 2 * n_embd, bias=bias),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.seq(x)
        return x

class OneStepBlock(nn.Module):
    def __init__(self, n_embd, dropout=0.0, bias=True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(2*n_embd)
        self.mlp1 = OneStepMLP(n_embd, dropout=dropout, bias=bias)
        self.ln_2 = nn.LayerNorm(2*n_embd)
        self.mlp2 = OneStepMLP(n_embd, dropout=dropout, bias=bias)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=-1)
        x = x + self.mlp1(self.ln_1(x))
        x = x + self.mlp2(self.ln_2(x))
        x, y = x.chunk(2, dim=-1)
        return x, y

class MyOneStepRecurrentNet(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([OneStepBlock(n_embd) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, tok, mem=None):
        # tok: (bs )
        # mem: (bs, n_layer, d)
        mem_outs = []
        x = self.wte(tok)  # (bs, d)
        for block, memi in zip(self.blocks, rearrange(mem, 'b l d -> l b d')):
            x, memouti = block(x, memi)
            mem_outs.append(memouti)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, torch.stack(mem_outs, dim=1)


class MyOneStepRecurrentTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer, n_head):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.wpe_wtf = nn.Embedding(6, n_embd)

    def forward(self, tok, mem):
        # tok: (bs )
        # mem: (bs, n_layer, d)
        mask = torch.ones(6, 6, dtype=bool, device=tok.device)

        x = self.wte(tok)  # (bs, d)
        x = torch.cat([mem, x[:, None, :]], dim=1) # bs, n_layer+1, d
        y = self.wpe_wtf(torch.arange(6, device=tok.device))
        x = x + self.wpe_wtf(torch.arange(6, device=tok.device)) # (6, d)
        for block in self.blocks:
            x = block(x, mask=mask)
        # x is (bs, n_layer+1, d)
        mem, x = torch.split(x, [5, 1], dim=1)
        x = rearrange(x, 'b 1 d -> b d')
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, mem
    