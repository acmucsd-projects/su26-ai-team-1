"""
SUMMARY
-------
PosFormer additions layered on top of the plain baseline in baseline_decoder.py:

  1. Position forest auxiliary task: a second decoder that sees each token's
     position in the expression's structure instead of its identity, trained
     jointly with two extra heads (nesting depth, position-within-parent).
  2. Implicit attention correction (ARM): accumulated cross-attention
     coverage is subtracted from the raw attention logits so the decoder stops
     re-attending to strokes it already consumed.

This is an add on to the baseline decoder

Differences from the official repo
-------------------------------------
- Official PosFormer trains bidirectionally (l2r + r2l concatenated into a 2b
  batch); this is unidirectional, matching our baseline. Some of their
  reported gain comes from that, so expect less than the paper's numbers.
- Official uses a DenseNet encoder; we use MobileNetV3. Their dc/nhead
  defaults were tuned against DenseNet features.
- Their ARM hardcodes `2 * nhead` when building the vocab mask, which is wrong
  when only one of cross/self coverage is on; this uses the real channel count.
- Their label parser infinite-loops on unbalanced input (see _find_matching).

WIP
------
- No Dataset/DataLoader
- Position forest labels are parsed in Python per batch. Fine relative to a
  forward pass, but precompute them in Dataset.__getitem__ if the loader
  becomes the bottleneck.
- The aux task has never been trained on real data -- the loss weights
  (0.25/0.25) are the paper's, not tuned on MathWriting.
"""

import functools
import json
import math
from pathlib import Path
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline_decoder import (
    BOS_IDX,
    EOS_IDX,
    MAX_LEN,
    PAD_IDX,
    UNK_IDX,
    ImagePositionalEncoding,
    LatexDecoder,
    WordPositionalEncoding,
    build_optimizer,
    generate_causal_mask,
    greedy_decode,
    beam_search_batch,
    latex_cross_entropy,
    shift_target_for_teacher_forcing,
    widths_to_memory_padding_mask,
)


# ===========================================================================
# Vocab: resolving structure tokens by string instead of hardcoded ids
# ===========================================================================
# The position forest parses TOKEN IDS, so it has to know which id means "^",
# "\frac", "{" and so on. The official repo hardcodes CROHME ids (^=82,
# \frac=53, {=110 ...) which are meaningless against our vocab.
#
# Our preprocessing notebook writes processed/vocab.json as a plain
# {token_string: id} map, and its tokenizer emits exactly the strings we need
# (r'\frac{1}{2}' -> ['\\frac', '{', '1', '}', '{', '2', '}']), so we can look
# them up by name and never hardcode an integer.

# Special tokens, in the order the notebook's build_vocab assigns them.
SPECIAL_TOKEN_STRINGS = ("<PAD>", "<BOS>", "<EOS>", "<UNK>")

# Structure tokens the forest parser recognizes -> the literal strings the
# tokenizer produces for them.
STRUCTURE_TOKEN_STRINGS = {
    "sup": "^",
    "sub": "_",
    "frac": "\\frac",
    "sqrt": "\\sqrt",
    "lbrace": "{",
    "rbrace": "}",
    "lbracket": "[",
    "rbracket": "]",
}

# Sentinel for a structure token that isn't in the vocab. Real ids are >= 0, so
# nothing ever equals this and the parser simply never takes that branch.
ABSENT = -1


class StructureTokens:
    """
    Token ids the position-forest parser and the ARM need to recognize.

    Build these with from_vocab() rather than by hand. Every id defaults to
    ABSENT: a wrong id doesn't crash, it silently produces structureless
    labels and the auxiliary task teaches nothing, so "absent and inert" is a
    much safer default than "plausible but wrong".
    """

    def __init__(self, sup=ABSENT, sub=ABSENT, frac=ABSENT, sqrt=ABSENT,
                 lbrace=ABSENT, rbrace=ABSENT, lbracket=ABSENT, rbracket=ABSENT,
                 pad_idx=PAD_IDX, bos_idx=BOS_IDX, eos_idx=EOS_IDX,
                 missing=()):
        self.sup, self.sub, self.frac, self.sqrt = sup, sub, frac, sqrt
        self.lbrace, self.rbrace = lbrace, rbrace
        self.lbracket, self.rbracket = lbracket, rbracket
        self.pad_idx, self.bos_idx, self.eos_idx = pad_idx, bos_idx, eos_idx
        # Names that weren't found in the vocab, for reporting.
        self.missing = tuple(missing)

    @classmethod
    def from_vocab(cls, vocab, pad_idx=PAD_IDX, bos_idx=BOS_IDX, eos_idx=EOS_IDX):
        """
        Resolve every structure token by string from a {token: id} vocab.

        Tokens absent from the vocab are left ABSENT and listed in .missing --
        that's expected for a small excerpt dataset that happens to contain no
        \\sqrt, and it degrades gracefully rather than mislabeling.
        """
        ids, missing = {}, []
        for name, token in STRUCTURE_TOKEN_STRINGS.items():
            if token in vocab:
                ids[name] = int(vocab[token])
            else:
                ids[name] = ABSENT
                missing.append(f"{name} ({token!r})")
        return cls(pad_idx=pad_idx, bos_idx=bos_idx, eos_idx=eos_idx,
                   missing=missing, **ids)

    @property
    def non_visual(self):
        """
        Tokens the ARM excludes from coverage: layout commands with no ink of
        their own. "{" and "^" don't correspond to any pixels, so letting them
        accumulate coverage would penalize image regions the decoder hasn't
        actually consumed yet. (The official repo masks {, ^, _.)
        """
        return tuple(i for i in (self.lbrace, self.sup, self.sub) if i != ABSENT)

    def __repr__(self):
        shown = {n: getattr(self, n) for n in STRUCTURE_TOKEN_STRINGS}
        return f"StructureTokens({shown}, missing={list(self.missing)})"


class VocabConfig(NamedTuple):
    vocab_size: int
    structure_tokens: StructureTokens
    vocab: dict


def load_vocab_config(vocab_path="processed/vocab.json",
                      pad_idx=PAD_IDX, bos_idx=BOS_IDX, eos_idx=EOS_IDX,
                      unk_idx=UNK_IDX):
    """
    Read processed/vocab.json and derive everything the model needs from it:
    the real vocab_size and the structure token ids, resolved by string.

    This is the intended way to construct a PosFormerDecoder -- it removes both
    the "what is vocab_size?" unknown and the hardcoded-token-id hazard in one
    step.

    Raises if the special tokens aren't at the ids this codebase assumes,
    because baseline_decoder's PAD_IDX/BOS_IDX/EOS_IDX are module constants;
    if the vocab builder ever reorders them, silently proceeding would corrupt
    both the loss mask and the decode loop.
    """
    path = Path(vocab_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate it by running the preprocessing "
            f"notebook (mathwriting_preprocessing.ipynb) on the "
            f"data-preprocessing branch -- it writes processed/vocab.json "
            f"along with processed/labels/*.jsonl."
        )
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    expected = dict(zip(SPECIAL_TOKEN_STRINGS, (pad_idx, bos_idx, eos_idx, unk_idx)))
    for token, want in expected.items():
        if token not in vocab:
            raise ValueError(
                f"{path} is missing the special token {token!r}. This codebase "
                f"assumes the notebook's SPECIAL_TOKENS order "
                f"{list(SPECIAL_TOKEN_STRINGS)}."
            )
        if int(vocab[token]) != want:
            raise ValueError(
                f"{token!r} has id {vocab[token]} in {path}, but this codebase "
                f"assumes {want}. Either the vocab builder changed order or the "
                f"constants in baseline_decoder.py need updating -- do not "
                f"ignore this, it would corrupt the padding mask and decoding."
            )

    return VocabConfig(
        vocab_size=len(vocab),
        structure_tokens=StructureTokens.from_vocab(vocab, pad_idx, bos_idx, eos_idx),
        vocab=vocab,
    )


# ===========================================================================
# Position forest labels
# ===========================================================================
# Flag vocabulary for a node's position within its parent, from
# label_make_muti.py's `flag = [0, 1, 2, 3, 4, 5]`:
#   0 = PAD, 1 = BOS, 2 = EOS
#   3 = same level as parent (ordinary symbol, or a structural marker)
#   4 = "upper" branch: superscript / fraction numerator / sqrt index
#   5 = "lower" branch: subscript / fraction denominator / sqrt radicand
NUM_POS_CLASSES = 6      # values a final-position label can take
NUM_LAYER_CLASSES = 5    # nesting depth is clamped to 0..4
FOREST_PATH_LEN = 5      # each token's root->node path, padded/truncated to 5


def _find_matching(indices, start, end, open_id, close_id):
    """
    Index of the closer matching the opener at `start`, or -1 if `start` isn't
    that opener or the group never closes.

    Returns -1 (not 0) on unbalanced input. The official find_end_bigbracket
    returns 0, and its caller's `i = end1 + 1` then jumps the cursor backward
    and loops forever.
    The `end1 < 0` guards below depend on this; don't "simplify" them away.
    """
    if start >= end or start < 0 or indices[start] != open_id:
        return -1
    count = 1
    i = start + 1
    while count > 0 and i < end:
        if indices[i] == open_id:
            count += 1
        elif indices[i] == close_id:
            count -= 1
        i += 1
    return i - 1 if count == 0 else -1


def _forest_helper(indices, start, end, result, st):
    """
    Recursive descent over one expression, appending a level flag to every
    token's path. Port of `helper` in label_make_muti.py.

    `result[i]` accumulates the path from root to token i, so len(result[i])-1
    is the token's nesting depth and result[i][-1] is its position in its
    immediate parent. This is a "forest", not one tree: a flat expression like
    "a b c" is a sequence of depth-0 roots.
    """
    i = start + 1
    while i < end:
        tok = indices[i]

        if tok != ABSENT and (tok == st.sup or tok == st.sub):
            # ^{...} / _{...}: the brace group after the marker is one branch.
            branch = 4 if tok == st.sup else 5
            end1 = _find_matching(indices, i + 1, end, st.lbrace, st.rbrace)
            if end1 < 0:                       # malformed -> ordinary symbol
                result[i].append(3)
                i += 1
                continue
            # The marker and its braces sit at the PARENT's level (flag 3),
            # only the contents descend.
            result[i].append(3)
            result[i + 1].append(3)
            result[end1].append(3)
            for j in range(i + 2, end1):
                result[j].append(branch)
            _forest_helper(indices, i + 1, end1, result, st)
            i = end1 + 1

        elif tok != ABSENT and tok == st.frac:
            # \frac{num}{den} -> numerator is the upper branch, denom lower.
            end1 = _find_matching(indices, i + 1, end, st.lbrace, st.rbrace)
            end2 = _find_matching(indices, end1 + 1, end, st.lbrace, st.rbrace) if end1 >= 0 else -1
            if end1 < 0 or end2 < 0:
                result[i].append(3)
                i += 1
                continue
            result[i].append(3)
            for j in range(i + 2, end1):
                result[j].append(4)
            for j in range(end1 + 2, end2):
                result[j].append(5)
            result[i + 1].append(3)
            result[end1].append(3)
            result[end1 + 1].append(3)
            result[end2].append(3)
            _forest_helper(indices, i + 1, end1, result, st)
            _forest_helper(indices, end1 + 1, end2, result, st)
            i = end2 + 1

        elif tok != ABSENT and tok == st.sqrt:
            if i + 1 < end and indices[i + 1] == st.lbracket:
                # \sqrt[n]{x}: index n is the upper branch, radicand lower.
                end1 = _find_matching(indices, i + 1, end, st.lbracket, st.rbracket)
                end2 = _find_matching(indices, end1 + 1, end, st.lbrace, st.rbrace) if end1 >= 0 else -1
                if end1 < 0 or end2 < 0:
                    result[i].append(3)
                    i += 1
                    continue
                result[i].append(3)
                for j in range(i + 2, end1):
                    result[j].append(4)
                for j in range(end1 + 2, end2):
                    result[j].append(5)
                result[i + 1].append(3)
                result[end1].append(3)
                result[end1 + 1].append(3)
                result[end2].append(3)
                _forest_helper(indices, i + 1, end1, result, st)
                _forest_helper(indices, end1 + 1, end2, result, st)
                i = end2 + 1
            else:
                # \sqrt{x}: radicand only.
                end1 = _find_matching(indices, i + 1, end, st.lbrace, st.rbrace)
                if end1 < 0:
                    result[i].append(3)
                    i += 1
                    continue
                result[i].append(3)
                for j in range(i + 2, end1):
                    result[j].append(5)
                result[i + 1].append(3)
                result[end1].append(3)
                _forest_helper(indices, i + 1, end1, result, st)
                i = end1 + 1

        elif tok == st.pad_idx:
            result[i].append(0)
            i += 1
        elif tok == st.bos_idx:
            result[i].append(1)
            i += 1
        elif tok == st.eos_idx:
            result[i].append(2)
            i += 1
        else:
            result[i].append(3)
            i += 1

    return result


def indices_to_forest_paths(indices, st):
    """One sequence of token ids -> list of variable-length path lists."""
    result = [[] for _ in indices]
    return _forest_helper(list(indices), -1, len(indices), result, st)


def forest_paths_to_tensors(tokens, st, device=None):
    """
    Build everything the aux task needs from a padded batch of token ids.

    tokens: [batch, seq_len] LongTensor (the FULL sequence, incl. BOS/EOS)
    st:     StructureTokens, ideally from StructureTokens.from_vocab()

    Returns:
        path_feat:  [batch, seq_len, 5] float -- the position decoder's INPUT.
                    Ports `tgt2muti_label`: each token's root->node path,
                    zero-padded to 5. Fed through Linear(5, d_model) in place of
                    a token embedding, which is the whole trick -- the position
                    decoder never sees symbol identities, only where each symbol
                    sits in the structure.
        layer_num:  [batch, seq_len] long, nesting depth clamped to 0..4
        final_pos:  [batch, seq_len] long, position-in-parent class 0..5

    layer_num/final_pos port `tgt2layernum_and_pos`. The official repo's `out2*`
    variants rotate by one to align with the shifted output; here the caller
    does that with a plain slice, which is equivalent and easier to follow.
    """
    tok_list = tokens.detach().cpu().tolist()

    path_feat, layer_num, final_pos = [], [], []
    for seq in tok_list:
        paths = indices_to_forest_paths(seq, st)

        row_feat, row_layer, row_pos = [], [], []
        for path in paths:
            padded = (path + [0] * FOREST_PATH_LEN)[:FOREST_PATH_LEN]
            row_feat.append([float(v) for v in padded])

            if len(path) == 0:
                # Unreachable for well-formed input, but an unbalanced brace can
                # leave a token unvisited -- treat as depth 0 / PAD class.
                row_layer.append(0)
                row_pos.append(0)
            elif len(path) == 1:
                row_layer.append(0)
                row_pos.append(path[0])
            elif len(path) <= FOREST_PATH_LEN:
                row_layer.append(len(path) - 1)
                row_pos.append(path[-2])
            else:
                row_layer.append(NUM_LAYER_CLASSES - 1)  # clamp deep nesting
                row_pos.append(path[3])

        path_feat.append(row_feat)
        layer_num.append(row_layer)
        final_pos.append(row_pos)

    device = device or tokens.device
    return (
        torch.tensor(path_feat, dtype=torch.float, device=device),
        torch.tensor(layer_num, dtype=torch.long, device=device),
        torch.tensor(final_pos, dtype=torch.long, device=device),
    )


# ===========================================================================
# Implicit attention correction (ARM)
# ===========================================================================


class MaskBatchNorm2d(nn.Module):
    """BatchNorm over only the non-padded spatial positions. Port of arm.py."""

    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x, mask):
        # x: [b, d, h, w]; mask: [b, 1, h, w] True where padded
        x = x.permute(0, 2, 3, 1)               # [b, h, w, d]
        valid = ~mask.squeeze(1)
        out = torch.zeros_like(x)
        if valid.any():
            # The official version writes normalized values back in place and
            # leaves padded positions at their pre-norm value; zeros here is
            # equivalent downstream, because those positions get -inf'd by the
            # key padding mask in the re-softmax anyway.
            out[valid] = self.bn(x[valid])
        return out.permute(0, 3, 1, 2)


class AttentionRefinementModule(nn.Module):
    """
    PosFormer's implicit attention correction, ported from
    Pos_Former/model/transformer/arm.py.

    What it does: accumulate how much attention each image region has already
    received across previous decoding steps ("coverage"), run that coverage map
    through a small conv net, and SUBTRACT the result from the raw pre-softmax
    cross-attention logits. Regions already consumed get pushed down, so the
    decoder stops re-attending to the same stroke and dropping or duplicating
    symbols -- the classic HMER failure mode.

    The "implicit" part is the vocab masking: structural tokens ({, ^, _)
    contribute no coverage, because they have no ink to consume.
    """

    non_visual_ids: torch.Tensor  # tells the type checker this buffer is a Tensor

    def __init__(self, nhead, dc=32, cross_coverage=True, self_coverage=True,
                 non_visual_ids=()):
        super().__init__()
        assert cross_coverage or self_coverage, "ARM needs at least one coverage source"
        self.nhead = nhead
        self.cross_coverage = cross_coverage
        self.self_coverage = self_coverage
        self.register_buffer(
            "non_visual_ids", torch.tensor(list(non_visual_ids), dtype=torch.long)
        )

        in_chs = nhead * (int(cross_coverage) + int(self_coverage))
        self.conv = nn.Conv2d(in_chs, dc, kernel_size=5, padding=2)
        self.act = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(dc, nhead, kernel_size=1, bias=False)
        self.post_norm = MaskBatchNorm2d(nhead)

    def forward(self, prev_attn, key_padding_mask, h, curr_attn, tgt_tokens):
        """
        prev_attn / curr_attn: [(batch*nhead), tgt_len, H*W] post-softmax
            attention -- prev from the previous decoder layer (cross coverage),
            curr from this layer's first softmax pass (self coverage).
        key_padding_mask: [batch, H*W] bool or None
        h: feature-map height, needed to un-flatten H*W back to a grid (4 for
            our encoder -- see baseline_decoder's CONFIRMED block)
        tgt_tokens: [batch, tgt_len] token ids, for the non-visual mask

        Returns [(batch*nhead), tgt_len, H*W] to subtract from attn logits.
        """
        bn, t, l = curr_attn.shape
        b = bn // self.nhead
        w = l // h

        if key_padding_mask is None:
            key_padding_mask = torch.zeros(b, l, dtype=torch.bool, device=curr_attn.device)
        # [b, l] -> [(b*t), 1, h, w], one spatial mask per decoding step
        mask = (key_padding_mask.view(b, 1, h, w)
                .expand(b, t, h, w)
                .reshape(b * t, 1, h, w))

        curr = curr_attn.view(b, self.nhead, t, l)
        parts = []
        if self.cross_coverage:
            parts.append(prev_attn.view(b, self.nhead, t, l))
        if self.self_coverage:
            parts.append(curr)
        attns = torch.cat(parts, dim=1)                      # [b, n_in, t, l]
        n_in = attns.size(1)

        if self.non_visual_ids.numel() > 0:
            keep = ~torch.isin(tgt_tokens, self.non_visual_ids)   # [b, t]
            keep = keep.unsqueeze(1).unsqueeze(-1).expand(b, n_in, t, l)
            attns = attns * keep.float()

        # Exclusive cumulative sum over decoding steps: at step t this holds
        # attention accumulated over steps 0..t-1, never including t itself.
        # That exclusivity is what keeps the model causal.
        attns = attns.cumsum(dim=2) - attns

        attns = attns.permute(0, 2, 1, 3).reshape(b * t, n_in, h, w)
        cov = self.act(self.conv(attns))
        cov = cov.masked_fill(mask, 0.0)
        cov = self.proj(cov)
        cov = self.post_norm(cov, mask)
        # [(b*t), nhead, h, w] -> [(b*nhead), t, l]
        cov = cov.view(b, t, self.nhead, l).permute(0, 2, 1, 3).reshape(b * self.nhead, t, l)
        return cov


class _MultiheadAttentionARM(nn.Module):
    """
    Minimal batch-first MHA that exposes pre-softmax logits so the ARM can be
    subtracted from them. nn.MultiheadAttention cannot do this -- it only
    returns post-softmax weights -- which is why the official repo vendors its
    own copy of F.multi_head_attention_forward, and why this port needs one too.

    Verified to match nn.MultiheadAttention exactly when arm_fn is None.
    """

    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def _shape(self, x, b):
        # [b, len, d] -> [(b*nhead), len, head_dim]
        return (x.view(b, -1, self.nhead, self.head_dim)
                 .permute(0, 2, 1, 3)
                 .reshape(b * self.nhead, -1, self.head_dim))

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, arm_fn=None):
        b, t, d = query.shape
        s = key.size(1)

        q = self._shape(self.q_proj(query), b) / math.sqrt(self.head_dim)
        k = self._shape(self.k_proj(key), b)
        v = self._shape(self.v_proj(value), b)

        logits = torch.bmm(q, k.transpose(1, 2))          # [(b*nhead), t, s]

        def masked_softmax(scores):
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    scores = scores.masked_fill(attn_mask, float("-inf"))
                else:
                    scores = scores + attn_mask
            if key_padding_mask is not None:
                scores = scores.view(b, self.nhead, t, s).masked_fill(
                    key_padding_mask.view(b, 1, 1, s), float("-inf")
                ).view(b * self.nhead, t, s)
            attn = F.softmax(scores, dim=-1)
            return F.dropout(attn, p=self.dropout, training=self.training)

        attn = masked_softmax(logits)
        if arm_fn is not None:
            # Refine once: subtract coverage from the RAW logits, re-softmax.
            logits = logits - arm_fn(attn)
            attn = masked_softmax(logits)

        out = torch.bmm(attn, v)
        out = (out.view(b, self.nhead, t, self.head_dim)
                  .permute(0, 2, 1, 3)
                  .reshape(b, t, d))
        return self.out_proj(out), attn


class ARMDecoderLayer(nn.Module):
    """Post-LN decoder layer matching nn.TransformerDecoderLayer, but its
    cross-attention accepts an ARM callback and returns its attention map."""

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.self_attn = _MultiheadAttentionARM(d_model, nhead, dropout=dropout)
        self.cross_attn = _MultiheadAttentionARM(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, arm_fn=None, tgt_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        tgt2, attn = self.cross_attn(tgt, memory, memory, arm_fn=arm_fn,
                                     key_padding_mask=memory_key_padding_mask)
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt, attn


class ARMDecoder(nn.Module):
    """
    Stack of ARMDecoderLayers sharing ONE AttentionRefinementModule.

    Wiring detail from transformer_decoder.py that's easy to get wrong: layer 0
    runs with arm_fn=None (there is no previous attention yet), and each layer's
    attention map is what refines the NEXT layer. So with 3 layers the ARM fires
    on layers 1 and 2 only, and the module is shared, not per-layer.
    """

    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout, arm=None):
        super().__init__()
        self.layers = nn.ModuleList([
            ARMDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.arm = arm

    def forward(self, tgt, memory, memory_height, tgt_tokens, tgt_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        out = tgt
        arm_fn = None
        attn = None
        for i, layer in enumerate(self.layers):
            out, attn = layer(out, memory, arm_fn=arm_fn, tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
            if self.arm is not None and i != len(self.layers) - 1:
                arm_fn = functools.partial(
                    self.arm, attn, memory_key_padding_mask, memory_height,
                    tgt_tokens=tgt_tokens,
                )
        return out, attn


# ===========================================================================
# The toggleable PosFormer decoder
# ===========================================================================


class PosFormerOutput(NamedTuple):
    logits: torch.Tensor                         # [b, l, vocab_size]
    layer_logits: Optional[torch.Tensor] = None  # [b, l, 5]  nesting depth
    pos_logits: Optional[torch.Tensor] = None    # [b, l, 6]  position in parent


class PosFormerDecoder(LatexDecoder):
    """
    Baseline decoder + the two PosFormer additions, each individually
    toggleable. With both flags False this is exactly LatexDecoder.

    forward() keeps LatexDecoder's signature and return type (plain logits), so
    greedy_decode / beam_search_* from baseline_decoder work against it
    unchanged -- pass memory_height=4 through their **model_kwargs when use_arm
    is True. The auxiliary heads are only reachable via forward_with_aux(),
    which is training-only; PosFormer doesn't use the position decoder at
    inference either.

    Construct it from the real vocab:
        cfg = load_vocab_config("processed/vocab.json")
        model = PosFormerDecoder(cfg.vocab_size,
                                 structure_tokens=cfg.structure_tokens)
    """

    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=3,
                 dim_feedforward=1024, dropout=0.1, max_len=MAX_LEN,
                 use_arm=True, use_position_forest=True,
                 dc=32, cross_coverage=True, self_coverage=True,
                 structure_tokens=None):
        super().__init__(vocab_size, d_model, nhead, num_layers,
                         dim_feedforward, dropout, max_len)
        self.use_arm = use_arm
        self.use_position_forest = use_position_forest
        self.structure_tokens = structure_tokens or StructureTokens()

        if (use_arm or use_position_forest) and structure_tokens is None:
            raise ValueError(
                "PosFormerDecoder needs structure_tokens when use_arm or "
                "use_position_forest is on. Build them from the real vocab:\n"
                "    cfg = load_vocab_config('processed/vocab.json')\n"
                "    PosFormerDecoder(cfg.vocab_size, "
                "structure_tokens=cfg.structure_tokens)\n"
                "Without them the forest labels carry no structure and the "
                "auxiliary task trains on nothing -- silently."
            )

        def make_arm():
            return AttentionRefinementModule(
                nhead, dc=dc, cross_coverage=cross_coverage,
                self_coverage=self_coverage,
                non_visual_ids=self.structure_tokens.non_visual,
            )

        if use_arm:
            # Replace the stock stack: nn.TransformerDecoderLayer cannot expose
            # pre-softmax logits, so the ARM can't be bolted onto it.
            self.decoder = ARMDecoder(d_model, nhead, num_layers,
                                      dim_feedforward, dropout, arm=make_arm())

        if use_position_forest:
            # A second decoder over the SAME image memory. Its "embedding" is a
            # Linear(5, d) over the root->node path rather than a token
            # embedding -- it sees structure, not symbols. Roughly doubles
            # decoder params; training-only, so inference cost is unchanged,
            # but it does grow checkpoints.
            self.pos_embed = nn.Sequential(
                nn.Linear(FOREST_PATH_LEN, d_model), nn.GELU(), nn.LayerNorm(d_model)
            )
            self.pos_word_pos_enc = WordPositionalEncoding(d_model, max_len, dropout)
            self.pos_norm = nn.LayerNorm(d_model)
            if use_arm:
                self.pos_decoder = ARMDecoder(d_model, nhead, num_layers,
                                              dim_feedforward, dropout, arm=make_arm())
            else:
                layer = nn.TransformerDecoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                    dropout=dropout, batch_first=True,
                )
                self.pos_decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
            self.layer_head = nn.Sequential(
                nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, NUM_LAYER_CLASSES)
            )
            self.pos_head = nn.Sequential(
                nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, NUM_POS_CLASSES)
            )

    def forward(self, tgt_tokens, memory, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, memory_height=None):
        if not self.use_arm:
            return super().forward(tgt_tokens, memory, tgt_key_padding_mask,
                                   memory_key_padding_mask)

        if memory_height is None:
            raise ValueError(
                "use_arm=True needs memory_height (the feature map's H) to "
                "un-flatten memory back to a spatial grid. For our encoder "
                "that is 4 (64px images at stride 16). When calling "
                "greedy_decode/beam_search_batch, pass memory_height=4 -- they "
                "forward **model_kwargs straight through."
            )

        seq_len = tgt_tokens.size(1)
        causal_mask = generate_causal_mask(seq_len, tgt_tokens.device)

        x = self.token_embed(tgt_tokens) * math.sqrt(self.d_model)
        x = self.word_pos_enc(x)

        out, _ = self.decoder(
            x, memory, memory_height, tgt_tokens,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_proj(out)

    def forward_with_aux(self, tgt_tokens, memory, forest_paths,
                         tgt_key_padding_mask=None, memory_key_padding_mask=None,
                         memory_height=None):
        """
        Training-time forward: main logits plus the two auxiliary heads.

        forest_paths: [b, l, 5] float from forest_paths_to_tensors(...)[0],
        sliced to match tgt_tokens.
        """
        logits = self.forward(tgt_tokens, memory, tgt_key_padding_mask,
                              memory_key_padding_mask, memory_height=memory_height)
        if not self.use_position_forest:
            return PosFormerOutput(logits=logits)

        p = self.pos_norm(self.pos_word_pos_enc(self.pos_embed(forest_paths)))
        causal_mask = generate_causal_mask(tgt_tokens.size(1), tgt_tokens.device)

        if self.use_arm:
            pos_out, _ = self.pos_decoder(
                p, memory, memory_height, tgt_tokens,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        else:
            pos_out = self.pos_decoder(
                tgt=p, memory=memory, tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return PosFormerOutput(
            logits=logits,
            layer_logits=self.layer_head(pos_out),
            pos_logits=self.pos_head(pos_out),
        )


def aux_position_losses(out, labels, layer_target, pos_target, pad_idx=PAD_IDX):
    """
    The two auxiliary losses, factored out so posformer_train_step (which takes
    a precomputed feature map) and hmer_model.hmer_posformer_train_step (which
    takes raw images and so owns encoder gradients) cannot drift apart.

    out:           PosFormerOutput with layer_logits/pos_logits populated
    labels:        [b, l] the main CE targets, i.e. tokens[:, 1:]. Used only for
                   its pad mask, so both objectives mask exactly the same
                   positions.
    layer_target,
    pos_target:    [b, l+1] FULL-sequence targets from forest_paths_to_tensors;
                   sliced [:, 1:] here to line up with labels.

    Masked to non-pad positions rather than ignore_index, because class 0 is a
    MEANINGFUL label for both heads (depth 0 / PAD-position) -- ignore_index=0
    would silently drop every top-level symbol from the loss.
    """
    non_pad = labels.ne(pad_idx).reshape(-1)
    layer_loss = F.cross_entropy(
        out.layer_logits.reshape(-1, NUM_LAYER_CLASSES),
        layer_target[:, 1:].reshape(-1), reduction="none",
    )[non_pad].mean()
    pos_loss = F.cross_entropy(
        out.pos_logits.reshape(-1, NUM_POS_CLASSES),
        pos_target[:, 1:].reshape(-1), reduction="none",
    )[non_pad].mean()
    return layer_loss, pos_loss


def posformer_train_step(model, img_pos_enc, batch, optimizer, scheduler=None,
                         pad_idx=PAD_IDX, label_smoothing=0.0, grad_clip=1.0,
                         layer_weight=0.25, pos_weight=0.25):
    """
    The PosFormer counterpart to baseline_decoder.train_step. Kept separate
    rather than branching inside that function, so the baseline path stays
    untouched if you have to fall back to it.

    batch is the same dict train_step takes:
      "feat_map": encoder output, [batch, H*W, d_model] (or a 4D grid)
      "tokens":   [batch, seq_len] padded ids, already incl. BOS/EOS
      "memory_key_padding_mask": [batch, H*W] bool, True where width padding
      "feat_h":   optional; only needed if img_pos_enc has no feat_h set

    Loss mix is the official one (lit_posformer.py:90):
        (ce + 0.25 * layer_loss + 0.25 * pos_loss) / 1.5
    The /1.5 renormalizes so the total stays comparable in magnitude to plain
    CE, which means you don't have to retune the learning rate when flipping
    the aux task on. Keep that division if you change the weights.

    Auxiliary losses are masked to non-pad positions rather than using
    ignore_index, because class 0 is a MEANINGFUL label for both heads (depth 0
    / PAD-position) -- ignore_index=0 would silently drop every top-level
    symbol from the loss.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    feat_map = batch["feat_map"]
    feat_h = batch.get("feat_h")
    # Reuse the baseline's shape resolution so both files agree about what the
    # encoder's flattened output means, and so the ARM's memory_height comes
    # from the same place the positional encoding does.
    memory_height, _ = img_pos_enc.infer_hw(feat_map, feat_h)
    memory = img_pos_enc(feat_map, feat_h=feat_h)

    memory_key_padding_mask = batch.get("memory_key_padding_mask")
    if memory_key_padding_mask is None:
        raise ValueError(
            "batch['memory_key_padding_mask'] is missing. Images are padded to "
            "the batch's max width, so cross-attention needs to know which "
            "feature columns are padding. Build it with "
            "widths_to_memory_padding_mask(). If you are deliberately testing "
            "with uniform-width images, pass an all-False mask explicitly."
        )

    tokens = batch["tokens"]
    decoder_in, labels, tgt_key_padding_mask = shift_target_for_teacher_forcing(
        tokens, pad_idx=pad_idx
    )

    # Parse once over the full sequence, then slice: paths[:, :-1] lines up with
    # decoder_in, and the targets at [:, 1:] line up with labels. This is what
    # the repo's tgt2muti_label / out2layernum_and_pos pair does by rotating.
    paths, layer_target, pos_target = forest_paths_to_tensors(
        tokens, model.structure_tokens, device=tokens.device
    )
    out = model.forward_with_aux(
        decoder_in, memory, paths[:, :-1],
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask,
        memory_height=memory_height,
    )

    ce = latex_cross_entropy(out.logits, labels, pad_idx=pad_idx,
                             label_smoothing=label_smoothing)

    stats = {"ce_loss": ce.item()}
    loss = ce
    if model.use_position_forest:
        layer_loss, pos_loss = aux_position_losses(
            out, labels, layer_target, pos_target, pad_idx=pad_idx
        )
        loss = (ce + layer_weight * layer_loss + pos_weight * pos_loss) / (
            1.0 + layer_weight + pos_weight
        )
        stats["layer_loss"] = layer_loss.item()
        stats["pos_loss"] = pos_loss.item()

    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    with torch.no_grad():
        keep = labels.ne(pad_idx)
        correct = out.logits.argmax(-1).eq(labels) & keep
        stats["token_acc"] = correct.sum().item() / max(1, keep.sum().item())
    stats["loss"] = loss.item()
    stats["lr"] = optimizer.param_groups[0]["lr"]
    return stats


if __name__ == "__main__":
    # End-to-end sanity check. Shapes match the real encoder:
    # MobileNetV3 at stride 16 on 64px-tall images -> [batch, 4*(W//16), 256].
    batch, seq_len = 2, 12
    d_model = 256
    stride = 16
    feat_h, feat_w = 4, 12          # feat_h = 64 // 16, CONFIRMED

    # --- vocab: normally load_vocab_config("processed/vocab.json"), but that
    # file only exists after someone runs the preprocessing notebook. Build an
    # equivalent stand-in so this check runs on a clean checkout.
    demo_vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKEN_STRINGS)}
    for tok in ["\\frac", "\\sqrt", "^", "_", "{", "}", "[", "]",
                "a", "b", "c", "1", "2", "3", "x", "y", "+", "="]:
        demo_vocab[tok] = len(demo_vocab)
    vocab_size = len(demo_vocab)
    st = StructureTokens.from_vocab(demo_vocab)
    print("vocab_size:        ", vocab_size)
    print("structure tokens:  ",
          {n: getattr(st, n) for n in STRUCTURE_TOKEN_STRINGS},
          "| missing:", list(st.missing) or "none")

    # --- position forest: \frac { a } { b } ^ { 2 } wrapped in BOS/EOS
    V = demo_vocab
    demo = torch.tensor([[BOS_IDX, V["\\frac"], V["{"], V["a"], V["}"],
                          V["{"], V["b"], V["}"], V["^"], V["{"], V["2"],
                          V["}"], EOS_IDX]])
    paths, layer, pos = forest_paths_to_tensors(demo, st)
    print("\nforest for \\frac{a}{b}^{2}:")
    print("  layer_num:", layer[0].tolist())   # nesting depth per token
    print("  final_pos:", pos[0].tolist())     # 4 = upper branch, 5 = lower
    print("  -> 'a' depth/pos:", layer[0, 3].item(), pos[0, 3].item(), "(numerator)")
    print("  -> 'b' depth/pos:", layer[0, 6].item(), pos[0, 6].item(), "(denominator)")
    print("  -> '2' depth/pos:", layer[0, 10].item(), pos[0, 10].item(), "(superscript)")

    # --- the encoder hands back a flattened sequence, not a 4D grid
    img_pos_enc = ImagePositionalEncoding(d_model, feat_h=feat_h)
    encoder_out = torch.randn(batch, feat_h * feat_w, d_model)
    memory = img_pos_enc(encoder_out)

    true_widths = torch.tensor([180, 64])
    mem_pad_mask = widths_to_memory_padding_mask(true_widths, feat_h, feat_w, stride)

    tokens = torch.randint(4, vocab_size, (batch, seq_len))
    tokens[:, 0] = BOS_IDX
    tokens[0, -2] = EOS_IDX
    tokens[0, -1] = PAD_IDX
    tokens[1, -1] = EOS_IDX

    print("\ntoggle matrix (all four combinations must train and decode):")
    for use_arm, use_forest in [(True, True), (True, False), (False, True), (False, False)]:
        model = PosFormerDecoder(
            vocab_size=vocab_size, d_model=d_model, num_layers=3,
            use_arm=use_arm, use_position_forest=use_forest, structure_tokens=st,
        )
        opt, sched = build_optimizer(model, total_steps=1000)
        s = posformer_train_step(
            model, img_pos_enc,
            {"feat_map": encoder_out, "tokens": tokens,
             "memory_key_padding_mask": mem_pad_mask},
            opt, sched,
        )
        # memory_height is only consumed when use_arm=True; harmless otherwise.
        ids, _ = beam_search_batch(model, memory, memory_key_padding_mask=mem_pad_mask,
                                   beam_width=3, max_len=8, memory_height=feat_h)[0]
        greedy = greedy_decode(model, memory, memory_key_padding_mask=mem_pad_mask,
                               max_len=8, memory_height=feat_h)
        print(f"  arm={use_arm!s:<5} forest={use_forest!s:<5} "
              f"loss={s['loss']:.4f} "
              f"aux={'-' if 'pos_loss' not in s else round(s['pos_loss'], 3)} "
              f"beam={len(ids)} greedy={[len(g) for g in greedy]}")

    # --- both flags off must reduce to the plain baseline, exactly
    torch.manual_seed(0)
    base = LatexDecoder(vocab_size, d_model, num_layers=3, dropout=0.0).eval()
    off = PosFormerDecoder(vocab_size, d_model, num_layers=3, dropout=0.0,
                           use_arm=False, use_position_forest=False,
                           structure_tokens=st).eval()
    off.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        a = base(tokens, memory, memory_key_padding_mask=mem_pad_mask)
        b = off(tokens, memory, memory_key_padding_mask=mem_pad_mask)
    print(f"\nboth flags off == baseline: {torch.equal(a, b)} "
          f"| max|diff| = {(a - b).abs().max().item()}")
