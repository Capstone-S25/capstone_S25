"""CPU supervised adaptation of VadCLIP's ``CLIPVAD`` model (AAAI 2024).

Structure mirrors ``src/model.py`` from the released repo exactly:

  * a windowed temporal transformer over frame features (LGT-Adapter),
  * two graph branches -- a content-similarity adjacency (``adj4``) and a fixed
    temporal-distance adjacency (``DistanceAdj``) -- each run through two GCN layers,
  * the dual detection head: a visual classifier (``logits1``, per-frame anomaly logit)
    and a language-visual alignment score (``logits2``, cosine similarity of each frame
    to class text features, scaled by 1/0.07),
  * the two prompt mechanisms: a learnable ``text_prompt_embeddings`` template into which
    real token embeddings are spliced at a prefix/postfix offset, fed through the frozen
    CLIP text tower via the authors' modified ``encode_text(embeddings, tokens)``.

Deliberate deviations (documented in the package docstring):
  * runs on CPU -- ``DistanceAdj`` takes an explicit device instead of hard-coding CUDA;
  * supervision is frame-level binary (see train.py), not the paper's video-MIL, because
    our anomaly corpora contain no negative clips. Padding is masked throughout; this is an offline supervised adaptation, not a paper reproduction.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from .clip_b16 import CLIP, tokenize


class GraphConvolution(nn.Module):
    """Simple GCN layer (arXiv:1609.02907) with a residual connection."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        residual: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        # residual rule (matches the original): identity when width is unchanged, a
        # channel-mixing Conv1d when it changes (512 -> 256), zero if disabled.
        self._conv_residual = (in_features != out_features) and residual
        if not residual:
            self.residual = lambda x: 0
        elif in_features == out_features:
            self.residual = lambda x: x
        else:
            self.residual = nn.Conv1d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=5,
                padding=2,
            )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.1)

    def forward(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = input.matmul(self.weight)  # [B, T, out]
        output = adj.matmul(support)  # [B, T, out] (padded rows stay 0)

        if self.bias is not None:
            output = output + self.bias
        if self._conv_residual:
            inp = input.permute(0, 2, 1)  # [B, in, T]
            res = self.residual(inp).permute(0, 2, 1)
            output = output + res
        else:
            output = output + self.residual(input)
        return output


class DistanceAdj(nn.Module):
    """Fixed temporal-distance adjacency kernel (device-agnostic).

    For frame indices ``i, j`` the entry is ``exp(-|i-j| / e)`` -- a smooth decay with
    temporal separation. The original hard-codes CUDA; here the device is passed in.
    """

    def __init__(self):
        super().__init__()
        self.sigma = Parameter(torch.empty(1), requires_grad=False)
        self.sigma.data.fill_(0.1)

    def forward(
        self, batch_size: int, max_seqlen: int, device: torch.device
    ) -> torch.Tensor:
        # cityblock distance over frame indices == |i - j| (pure numpy; no scipy needed)
        idx = np.arange(max_seqlen)
        dist = np.abs(idx[:, None] - idx[None, :]).astype(np.float32)  # [L, L]
        dist = torch.from_numpy(dist).to(device)
        dist = torch.exp(-dist / torch.exp(torch.tensor(1.0)))  # exp(-|i-j|/e)
        return dist.unsqueeze(0).repeat(batch_size, 1, 1).to(device)


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Temporal-tower block (carries a padding mask through the residual stream)."""

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, padding_mask: torch.Tensor):
        padding_mask = (
            padding_mask.to(dtype=bool, device=x.device)
            if padding_mask is not None
            else None
        )
        mask = (
            self.attn_mask.to(device=x.device) if self.attn_mask is not None else None
        )
        if mask is not None:
            mask = torch.isneginf(mask)
        if padding_mask is not None:
            # Each batch has different real lengths. Give padded queries a self edge
            # to avoid entirely masked windows, then zero their output explicitly.
            combined = mask.unsqueeze(0).expand(x.shape[1], -1, -1).clone()
            combined |= padding_mask.unsqueeze(1)
            for b in range(x.shape[1]):
                padded = padding_mask[b].nonzero(as_tuple=True)[0]
                combined[b, padded, padded] = False
            mask = combined.repeat_interleave(self.attn.num_heads, dim=0)
        out = self.attn(x, x, x, need_weights=False, attn_mask=mask)[0]
        return (
            out.masked_fill(padding_mask.T.unsqueeze(-1), 0)
            if padding_mask is not None
            else out
        )

    def forward(self, x):
        x, padding_mask = x
        x = x + self.attention(self.ln_1(x), padding_mask)
        x = x + self.mlp(self.ln_2(x))
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.T.unsqueeze(-1), 0)
        return (x, padding_mask)


class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class CLIPVAD(nn.Module):
    def __init__(
        self,
        num_class: int,
        embed_dim: int = 512,
        visual_length: int = 256,
        visual_width: int = 512,
        visual_head: int = 1,
        visual_layers: int = 2,
        attn_window: int = 8,
        prompt_prefix: int = 10,
        prompt_postfix: int = 10,
        clip_model: CLIP | None = None,
        device: torch.device | str = "cpu",
        use_graph: bool = True,
    ):
        super().__init__()

        self.num_class = num_class
        self.use_graph = (
            use_graph  # ablation knob: False -> temporal-only (no GCN branches)
        )
        self.visual_length = visual_length
        self.visual_width = visual_width
        self.embed_dim = embed_dim
        self.attn_window = attn_window
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix
        self.device = torch.device(device)

        # --- LGT-Adapter temporal module -------------------------------------
        self.temporal = Transformer(
            width=visual_width,
            layers=visual_layers,
            heads=visual_head,
            attn_mask=self.build_attention_mask(attn_window),
        )

        width = int(visual_width / 2)
        self.gc1 = GraphConvolution(visual_width, width, residual=True)
        self.gc2 = GraphConvolution(width, width, residual=True)
        self.gc3 = GraphConvolution(visual_width, width, residual=True)
        self.gc4 = GraphConvolution(width, width, residual=True)
        self.disAdj = DistanceAdj()
        self.linear = nn.Linear(visual_width, visual_width)
        self.gelu = QuickGELU()

        # --- dual detection head ---------------------------------------------
        self.mlp1 = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(visual_width, visual_width * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(visual_width * 4, visual_width)),
                ]
            )
        )
        self.mlp2 = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(visual_width, visual_width * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(visual_width * 4, visual_width)),
                ]
            )
        )
        self.classifier = nn.Linear(visual_width, 1)

        # --- frozen CLIP backbone (injected to avoid a second download) -------
        assert clip_model is not None, "pass a loaded CLIP module via clip_model="
        self.clipmodel = clip_model
        for p in self.clipmodel.parameters():
            p.requires_grad_(False)

        # --- learnable prompt / position embeddings --------------------------
        self.frame_position_embeddings = nn.Embedding(visual_length, visual_width)
        self.text_prompt_embeddings = nn.Embedding(77, embed_dim)
        self.initialize_parameters()
        if not use_graph:
            for layer in (self.gc1, self.gc2, self.gc3, self.gc4, self.disAdj):
                for parameter in layer.parameters():
                    parameter.requires_grad_(False)

    def initialize_parameters(self):
        nn.init.normal_(self.text_prompt_embeddings.weight, std=0.01)
        nn.init.normal_(self.frame_position_embeddings.weight, std=0.01)

    def build_attention_mask(self, attn_window: int) -> torch.Tensor:
        """Block-diagonal windowed mask: each frame attends only within its window."""
        L = self.visual_length
        mask = torch.empty(L, L)
        mask.fill_(float("-inf"))
        if attn_window < 1 or L < 1:
            raise ValueError("Visual length and attention window must be positive")
        for start in range(0, L, attn_window):
            stop = min(start + attn_window, L)
            mask[start:stop, start:stop] = 0
        return mask

    def adj4(self, x: torch.Tensor, seq_len):
        """Content-similarity adjacency: cosine sim thresholded at 0.7 + row softmax."""
        soft = nn.Softmax(1)
        x2 = x.matmul(x.permute(0, 2, 1))  # [B, T, T]
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # [B, T, 1]
        x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))
        x2 = x2 / (x_norm_x + 1e-20)
        output = torch.zeros_like(x2)
        if seq_len is None:
            for i in range(x.shape[0]):
                adj2 = F.threshold(x2[i], 0.7, 0)
                output[i] = soft(adj2)
        else:
            for i in range(len(seq_len)):
                tmp = x2[i, : seq_len[i], : seq_len[i]]
                adj2 = F.threshold(tmp, 0.7, 0)
                output[i, : seq_len[i], : seq_len[i]] = soft(adj2)
        return output

    def encode_video(self, images: torch.Tensor, padding_mask, lengths):
        """images: [B, T, W] raw CLIP frame features -> [B, T, W] temporal features."""
        images = images.to(torch.float)
        lengths = lengths.to(images.device)
        if (lengths < 1).any() or (lengths > images.shape[1]).any():
            raise ValueError("Invalid real-frame lengths")
        padding_mask = (
            torch.arange(images.shape[1], device=images.device)[None, :]
            >= lengths[:, None]
        )
        images = images.masked_fill(padding_mask.unsqueeze(-1), 0)
        position_ids = (
            torch.arange(self.visual_length, device=self.device)
            .unsqueeze(0)
            .expand(images.shape[0], -1)
        )
        frame_position_embeddings = self.frame_position_embeddings(
            position_ids
        ).permute(1, 0, 2)
        images = images.permute(1, 0, 2) + frame_position_embeddings  # [T, B, W]

        images = images.masked_fill(padding_mask.T.unsqueeze(-1), 0)
        x, _ = self.temporal((images, padding_mask))
        x = x.permute(1, 0, 2)  # -> [B, T, W]

        if not self.use_graph:
            # temporal-only ablation: skip both GCN branches, project the temporal features directly
            return self.linear(x).masked_fill(padding_mask.unsqueeze(-1), 0)

        adj = self.adj4(x, lengths)
        disadj = self.disAdj(x.shape[0], x.shape[1], x.device)
        edge_mask = (~padding_mask).unsqueeze(1) & (~padding_mask).unsqueeze(2)
        disadj = disadj.masked_fill(~edge_mask, 0)
        x1_h = self.gelu(self.gc1(x, adj)).masked_fill(padding_mask.unsqueeze(-1), 0)
        x2_h = self.gelu(self.gc3(x, disadj)).masked_fill(padding_mask.unsqueeze(-1), 0)
        x1 = self.gelu(self.gc2(x1_h, adj))
        x2 = self.gelu(self.gc4(x2_h, disadj))

        x = torch.cat((x1, x2), 2)  # [B, T, 2W]
        x = self.linear(x)  # -> [B, T, W]
        return x.masked_fill(padding_mask.unsqueeze(-1), 0)

    def encode_textprompt(self, text):
        """Splice real token embeddings into the learnable template at a prefix/postfix offset."""
        word_tokens = tokenize(text).to(self.device)  # [C, 77]
        word_embedding = self.clipmodel.encode_token(word_tokens)  # [C, 77, W]
        text_embeddings = (
            self.text_prompt_embeddings(torch.arange(77, device=self.device))
            .unsqueeze(0)
            .repeat(len(text), 1, 1)
        )  # [C, 77, W] learnable template
        text_tokens = torch.zeros(len(text), 77, dtype=torch.long, device=self.device)

        for i in range(len(text)):
            ind = int(torch.argmax(word_tokens[i], -1))  # eot position (first max id)
            assert self.prompt_prefix + ind + self.prompt_postfix < 77, (
                f"prompt too long: eot at {ind} exceeds template with prefix/postfix offsets"
            )
            text_embeddings[i, 0] = word_embedding[i, 0]  # real SOT
            text_embeddings[i, self.prompt_prefix + 1 : self.prompt_prefix + ind] = (
                word_embedding[i, 1:ind]
            )
            text_embeddings[i, self.prompt_prefix + ind + self.prompt_postfix] = (
                word_embedding[i, ind]
            )  # real EOT
            text_tokens[i, self.prompt_prefix + ind + self.prompt_postfix] = (
                word_tokens[i, ind]
            )

        return self.clipmodel.encode_text(
            text_embeddings, text_tokens
        )  # [C, embed_dim]

    def forward(self, visual: torch.Tensor, padding_mask, text, lengths):
        """visual: [B, T, W]; text: list of num_class prompt strings; lengths: [B] real frame counts.

        Returns (text_features_ori [C, W], logits1 [B, T, 1], logits2 [B, T, C] scaled by /0.07,
                 logits2_raw [B, T, C] unscaled cosine in [-1,1]). Training uses stable log-softmax of the scaled logits; inference uses float64 softmax
                 to preserve continuous-score ordering. Raw cosine is also returned for inspection.
        """
        visual_features = self.encode_video(visual, padding_mask, lengths)  # [B, T, W]
        logits1 = self.classifier(visual_features + self.mlp2(visual_features))

        text_features_ori = self.encode_textprompt(text)  # [C, W]

        # attention-weighted representative visual feature (weighted by the visual branch score)
        valid = (
            torch.arange(visual.shape[1], device=visual.device)[None, :]
            < lengths.to(visual.device)[:, None]
        )
        logits1 = logits1.masked_fill(~valid.unsqueeze(-1), 0)
        logits_attn = logits1.permute(0, 2, 1)  # [B, 1, T]
        visual_attn = logits_attn @ visual_features  # [B, 1, W]
        visual_attn = visual_attn / visual_attn.norm(dim=-1, keepdim=True).clamp_min(
            1e-8
        )
        visual_attn = visual_attn.expand(
            visual_attn.shape[0], text_features_ori.shape[0], visual_attn.shape[2]
        )

        text_features = text_features_ori.unsqueeze(0).expand(
            visual_attn.shape[0], -1, -1
        )
        text_features = text_features + visual_attn
        text_features = text_features + self.mlp1(text_features)  # [B, C, W]

        visual_features_norm = visual_features / visual_features.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        text_features_norm = (
            text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        ).permute(0, 2, 1)
        logits2_raw = visual_features_norm @ text_features_norm.type(
            visual_features_norm.dtype
        )  # [B,T,C] raw cosine in [-1,1]
        logits2 = (
            logits2_raw / 0.07
        )  # author-scaled alignment score used for training and evaluation

        return text_features_ori, logits1, logits2, logits2_raw
