"""CPU supervised smoking adaptation of the VadCLIP architecture (AAAI 2024).

This implements a frozen OpenAI CLIP-B/16 encoder, temporal transformer/graphs,
learnable text prompts and dual detection branches. It differs from published
weakly supervised video MIL: training uses binary frame annotations and soft
block targets; evaluation restores original binary frames. Padded positions are
masked throughout, thresholds are selected on dev, and test groups are separate.

The whole sparse camera/date session can provide future context. It is an
OFFLINE adaptation, not a causal streaming detector or a reproduction of the
published XD-Violence/UCF-Crime numbers. Historical inspection of this small
corpus also limits claims about previously unseen data.
"""
