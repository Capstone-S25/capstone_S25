"""Small causal GRU heads over frozen visual features and optional tracks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class TemporalHead(nn.Module):
    def __init__(self, visual_dimension, geometry_dimension=0):
        super().__init__()
        self.visual_dimension = visual_dimension
        self.geometry_dimension = geometry_dimension
        self.project = nn.Sequential(
            nn.LayerNorm(visual_dimension),
            nn.Linear(visual_dimension, 128),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.gru = nn.GRU(128 + geometry_dimension + 2, 64, batch_first=True)
        self.classifier = nn.Linear(64, 1)

    def forward(self, visual, geometry, timing, lengths):
        pieces = [self.project(visual)]
        if self.geometry_dimension:
            pieces.append(geometry)
        pieces.append(timing)
        sequence = torch.cat(pieces, dim=-1)
        packed = pack_padded_sequence(
            sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return self.classifier(hidden[-1])[:, 0]
