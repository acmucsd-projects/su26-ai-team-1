"""
SUMMARY
-------
Decoder half of an image-to-LaTeX model: forward pass, training (loss + one
optimization step), and inference (greedy + beam search). No PosFormer
additions -- plain baseline.
 
CONFIRMED (from the team's `data-preprocessing` branch)
---------------------------------------------------------
- Special token ids: PAD=0, BOS=1, EOS=2, UNK=3. Real vocab ids start at 4.
  Labels in processed/labels/*.jsonl already come as [BOS, ...tokens, EOS].
- Image height is fixed at 64px; width varies per sample, padded per-batch.
  This is why memory_key_padding_mask is required everywhere below -- without
  it, cross-attention treats padded columns as real image content.

CONFIRMED (from the team's `model-encoder` branch, mobilenet_encoder.py)
------------------------------------------------------------------------
- The encoder returns a FLATTENED [batch, H*W, d_model] sequence, not a 4D
  grid. ImagePositionalEncoding accepts both, but needs feat_h for this form.
- Stride is 16 (MobileNetV3-Large truncated at features[:13]). With the fixed
  64px height that makes feat_h = 64 // 16 = 4, so seq_len = 4 * (W // 16).
- It flattens ROW-MAJOR (`features.flatten(2).permute(0, 2, 1)`), which is
  what this file's positional encoding and padding mask both assume.
- d_model=256 matches on both sides.
- The encoder does NOT return a padding mask, so building it from each
  sample's true width stays this file's job.

ASSUMPTIONS (still open)
-------------------------
- vocab_size=150 in the sanity check is a placeholder. The preprocessing
  notebook computes len(vocab) at runtime and writes processed/vocab.json;
  its stored outputs are cleared, so nobody has the real number yet.
- Architecture sizes (d_model=256, nhead=8, num_layers=3, dim_feedforward=1024)
  match BTTR/CoMER/PosFormer's published HMER config, not tuned on our data.
- max_len=200 is an arbitrary cap, not derived from real label lengths yet.
 
WIP -- NOT YET DONE
--------------------
- No Dataset/DataLoader wired to processed/labels/*.jsonl.
- No PosFormer additions in THIS file, by design -- they live in
  latex_decoder.py, which imports from here and adds them behind toggles.
  This file stays the plain baseline we can ship on its own.
- No KV cache for inference -- decoding is O(L^2) per sequence.
"""

import math
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
# Special token ids -- CONFIRMED against the preprocessing branch's vocab
# builder, which assigns these four first, before any real token.
# Passed as arguments (defaulting to these) rather than hardcoded in function
# bodies, so a vocab change means editing only these lines.
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3
 
# ASSUMPTION: max sequence length for the sinusoidal PE table and generation
# loops. Not yet derived from real MathWriting label lengths -- the
# preprocessing notebook reports token-length percentiles, so set this from
# that output (comfortably above p99) once the notebook has been run.
MAX_LEN = 200
 
 
class WordPositionalEncoding(nn.Module):
    """Standard 1D sinusoidal positional encoding for target LaTeX token embeddings."""
 
    pe: torch.Tensor  # tells the type checker this buffer is a Tensor, not a Module
 
    def __init__(self, d_model, max_len=MAX_LEN, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]
 
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return self.dropout(x + self.pe[:, : x.size(1)])
 
 
class ImagePositionalEncoding(nn.Module):
    """
    2D positional encoding for the MobileNet feature map. Half the channels
    encode row position, half encode column position, then the grid is
    flattened into a sequence for the decoder to cross-attend over.

    Accepts EITHER encoder output layout:
      - [batch, d_model, H, W]  -- a 4D grid; H/W come from the shape.
      - [batch, H*W, d_model]   -- already flattened (this is what
        encoder returns). H is NOT recoverable from this shape on its own, so
        it must come from `feat_h` (constructor or per-call argument).

    Why feat_h is knowable at all: input images are a fixed 64px tall (see the
    CONFIRMED block), so the encoder's output height is the constant
    64 // stride, and the variable width is then just (H*W) // feat_h.

    This module NEVER silently skips positional encoding. If it cannot resolve
    H it raises -- a decoder that cross-attends to an unordered bag of image
    features still trains and still emits plausible LaTeX, it just quietly
    loses all spatial structure, which is exactly the failure you would not
    catch from a loss curve.

    max_h only needs to cover the encoder's output height (4, confirmed --
    see the CONFIRMED block), but max_w must cover the WIDEST batch: images are
    ~64px tall with a median aspect ratio near 3:1 and p99 widths around
    420px, so at stride 16 that is ~27 columns. 64 leaves headroom.
    """

    pe: torch.Tensor  # tells the type checker this buffer is a Tensor, not a Module

    def __init__(self, d_model, max_h=64, max_w=64, dropout=0.1, feat_h=None):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even to split across H/W"
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.max_h = max_h
        self.max_w = max_w
        # CONFIRMED as 4 for our encoder: images are a fixed 64px tall and
        # mobilenet_encoder.py cuts MobileNetV3-Large at features[:13] (stride
        # 16), so feat_h = 64 // 16 = 4. Still defaults to None rather than 4 --
        # a wrong feat_h reshapes the grid and corrupts the encoding silently,
        # so a caller on a different stride should have to say so explicitly.
        self.feat_h = feat_h

        pe = torch.zeros(d_model, max_h, max_w)
        d_half = d_model // 2
        div_term = torch.exp(torch.arange(0, d_half, 2).float() * (-math.log(10000.0) / d_half))
 
        pos_h = torch.arange(0, max_h).unsqueeze(1).float()
        pe_h = torch.zeros(d_half, max_h)
        pe_h[0::2, :] = torch.sin(pos_h * div_term).T
        pe_h[1::2, :] = torch.cos(pos_h * div_term).T
 
        pos_w = torch.arange(0, max_w).unsqueeze(1).float()
        pe_w = torch.zeros(d_half, max_w)
        pe_w[0::2, :] = torch.sin(pos_w * div_term).T
        pe_w[1::2, :] = torch.cos(pos_w * div_term).T
 
        pe[:d_half, :, :] = pe_h.unsqueeze(2).expand(-1, -1, max_w)
        pe[d_half:, :, :] = pe_w.unsqueeze(1).expand(-1, max_h, -1)
        self.register_buffer("pe", pe)  # [d_model, max_h, max_w]
 
    def infer_hw(self, feat, feat_h=None):
        """
        Resolve (H, W) for either supported layout. Raises rather than guessing.

        Also returned for callers that need the grid height downstream -- the
        PosFormer attention-correction module needs it to un-flatten memory.
        """
        h = feat_h if feat_h is not None else self.feat_h

        if feat.dim() == 4:
            # [batch, d_model, H, W]
            _, c, grid_h, grid_w = feat.shape
            if c != self.d_model:
                raise ValueError(
                    f"4D input should be [batch, d_model, H, W] with "
                    f"d_model={self.d_model}, got channel dim {c}. If your "
                    f"encoder returns [batch, H, W, d_model], permute it first."
                )
            if h is not None and grid_h != h:
                raise ValueError(
                    f"feat_h={h} disagrees with the actual feature-map height "
                    f"{grid_h}. One of them is wrong -- most likely the encoder "
                    f"stride is not what feat_h assumes (feat_h should be "
                    f"64 // stride)."
                )
            h, w = grid_h, grid_w

        elif feat.dim() == 3:
            # [batch, H*W, d_model] -- encoder output.
            _, length, d = feat.shape
            if d != self.d_model:
                raise ValueError(
                    f"3D input should be [batch, H*W, d_model] with "
                    f"d_model={self.d_model}, got last dim {d}. If your encoder "
                    f"returns [batch, d_model, H*W], transpose the last two dims."
                )
            if h is None:
                raise ValueError(
                    "Cannot add 2D positional encoding to a flattened "
                    f"[batch, {length}, {self.d_model}] feature sequence: H and W "
                    "are not recoverable from H*W alone.\n"
                    "Fix by telling this module the feature-map height, either\n"
                    "  ImagePositionalEncoding(d_model, feat_h=64 // stride)\n"
                    "or per call\n"
                    "  img_pos_enc(feat, feat_h=64 // stride)\n"
                    "Images are a fixed 64px tall, so feat_h is a constant: 4 at "
                    "stride 16, 2 at stride 32. Do NOT work around this by "
                    "skipping the encoding -- the decoder would lose all spatial "
                    "information and fail silently."
                )
            if length % h != 0:
                raise ValueError(
                    f"Sequence length {length} is not divisible by feat_h={h}, so "
                    f"it cannot be a {h}xW grid. Either feat_h is wrong (should be "
                    f"64 // stride) or the encoder is not flattening a full grid."
                )
            w = length // h

        else:
            raise ValueError(
                f"Expected [batch, d_model, H, W] or [batch, H*W, d_model], "
                f"got a {feat.dim()}D tensor of shape {tuple(feat.shape)}."
            )

        if h > self.max_h or w > self.max_w:
            raise ValueError(
                f"Feature grid {h}x{w} exceeds the precomputed PE table "
                f"{self.max_h}x{self.max_w}. Rebuild with larger max_h/max_w -- "
                f"max_w must cover the widest image in any batch."
            )
        return h, w

    def forward(self, feat, feat_h=None):
        """
        feat: [batch, d_model, H, W] or [batch, H*W, d_model]
        feat_h: feature-map height; required for the flattened form unless it
            was set on the constructor.
        Returns: [batch, H*W, d_model]

        CONFIRMED: the flattened path assumes ROW-MAJOR ordering (row 0 cols
        0..W-1, then row 1, ...), and mobilenet_encoder.py on the model-encoder
        branch does exactly `features.flatten(2).permute(0, 2, 1)`. That also
        matches the 4D path here and the ordering widths_to_memory_padding_mask()
        builds, so all three agree.

        Worth re-checking if the encoder is ever rewritten: a column-major
        flatten puts every position code on the wrong cell and nothing errors --
        the model just trains worse. It is the one part of this interface a
        shape check cannot catch.
        """
        h, w = self.infer_hw(feat, feat_h)
        pe = self.pe[:, :h, :w]                                  # [d_model, h, w]

        if feat.dim() == 4:
            out = (feat + pe.unsqueeze(0)).flatten(2).permute(0, 2, 1)
        else:
            # Flatten the PE the same row-major way so both paths agree exactly.
            out = feat + pe.flatten(1).permute(1, 0).unsqueeze(0)  # [1, h*w, d]

        return self.dropout(out)                                 # [batch, H*W, d_model]

 
def widths_to_memory_padding_mask(true_widths, feat_h, feat_w, stride, device=None):
    """
    Build the cross-attention padding mask from each sample's true (unpadded)
    image width, which processed/labels/*.jsonl stores per record.
 
    Images are padded to the batch's max width, so without this the decoder
    attends to blank padded columns as if they were real ink.
 
    true_widths: [batch] ints, each sample's real pixel width before padding
    Returns: [batch, feat_h * feat_w] bool, True = "ignore this position",
             matching nn.Transformer's memory_key_padding_mask convention.
    """
    true_widths = torch.as_tensor(true_widths, device=device)
    # How many feature columns each sample actually occupies.
    valid_cols = torch.ceil(true_widths.float() / stride).long().clamp(max=feat_w)
    col_idx = torch.arange(feat_w, device=true_widths.device)
    # [batch, feat_w] -> True where the column is padding
    col_mask = col_idx.unsqueeze(0) >= valid_cols.unsqueeze(1)
    # Same mask applies to every row, then flattened to match memory's H*W
    # ordering (flatten(2) above is row-major: row 0 cols 0..W-1, row 1, ...).
    return col_mask.unsqueeze(1).expand(-1, feat_h, -1).reshape(true_widths.size(0), -1)
 
 
def generate_causal_mask(size, device):
    """
    Blocks each decoding step from attending to future tokens.
 
    Returns a BOOL mask (True = blocked). PyTorch deprecates mixing a float
    attn_mask with a bool key_padding_mask -- since the padding masks here are
    bool, this must be bool too or every forward pass emits warnings.
    """
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)
 
 
class LatexDecoder(nn.Module):
    """
    Autoregressive decoder: predicts the next LaTeX token from previous tokens
    (causal self-attention) plus the encoder's feature map (cross-attention).
    """
 
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=3,
                 dim_feedforward=1024, dropout=0.1, max_len=MAX_LEN):
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.word_pos_enc = WordPositionalEncoding(d_model, max_len, dropout)
 
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)
 
    def forward(self, tgt_tokens, memory, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """
        tgt_tokens: [batch, seq_len] input LaTeX token ids (teacher forcing, shifted right)
        memory: [batch, H*W, d_model] encoder features, already through ImagePositionalEncoding
        memory_key_padding_mask: [batch, H*W] bool -- REQUIRED for real batches,
            since MathWriting images are padded to the batch's max width.
        """
        seq_len = tgt_tokens.size(1)
        causal_mask = generate_causal_mask(seq_len, tgt_tokens.device)
 
        x = self.token_embed(tgt_tokens) * math.sqrt(self.d_model)
        x = self.word_pos_enc(x)
 
        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_proj(out)  # [batch, seq_len, vocab_size]
 
 
# ===========================================================================
# Training: loss, teacher forcing, one optimization step
# ===========================================================================
 
 
def shift_target_for_teacher_forcing(tgt, pad_idx=PAD_IDX):
    """
    Split a full target sequence into (decoder input, labels).
 
    CONFIRMED: processed/labels/*.jsonl already stores token_ids wrapped as
    [BOS, t1, ..., tn, EOS], so the collate function only pads to the batch's
    max length -- it should NOT add BOS/EOS again.
 
    Returns:
        decoder_in: [batch, seq_len-1]  -- [BOS, t1, ..., tn]      (shifted right)
        labels:     [batch, seq_len-1]  -- [t1,  t2, ..., EOS]     (un-shifted)
        tgt_key_padding_mask: [batch, seq_len-1] bool, True where padded
    """
    decoder_in = tgt[:, :-1]
    labels = tgt[:, 1:]
    # nn.Transformer expects True = "ignore this position".
    tgt_key_padding_mask = decoder_in.eq(pad_idx)
    return decoder_in, labels, tgt_key_padding_mask
 
 
def latex_cross_entropy(logits, labels, pad_idx=PAD_IDX, label_smoothing=0.0):
    """
    Cross-entropy over the vocab dimension, ignoring padded label positions.
 
    logits: [batch, seq_len, vocab_size]
    labels: [batch, seq_len]
 
    label_smoothing defaults to 0.0 to keep the baseline simple. HMER papers
    commonly use ~0.1: LaTeX has many near-interchangeable tokens and
    smoothing keeps the model from over-committing. Worth an ablation once
    the baseline trains.
 
    NOTE: UNK (id 3) is NOT ignored here -- it's a real prediction target. The
    preprocessing notebook reports an OOV rate per split; if valid/test OOV is
    non-trivial, decide as a team whether the model should learn to emit UNK
    or whether those samples should be excluded from the metric.
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=pad_idx,
        label_smoothing=label_smoothing,
    )
 
 
def build_optimizer(model, lr=3e-4, weight_decay=1e-4, warmup_steps=500,
                    total_steps=50_000, lr_by_prefix=None):
    """
    Reasonable defaults, not tuned. AdamW + linear warmup into cosine decay.

    Warmup matters more than the exact LR here: post-LN transformers (which is
    what nn.TransformerDecoderLayer defaults to) are unstable in the first few
    hundred steps without it. Scheduler steps PER BATCH, not per epoch.

    total_steps should be roughly (num_epochs * batches_per_epoch); the cosine
    curve is wrong if it doesn't match the real schedule length.

    lr_by_prefix: optional {parameter-name-prefix: lr} for per-submodule
        learning rates, e.g. {"encoder.": 1e-4} to fine-tune a pretrained
        encoder more gently than a randomly-initialized decoder. First matching
        prefix wins; anything unmatched uses `lr`. Kept generic on purpose --
        this file knows nothing about encoders (see hmer_model.py, which uses
        it). LambdaLR scales each group's own initial_lr, so the warmup/cosine
        shape applies to every group while the ratio between them is preserved.
    """
    # No weight decay on biases/norms -- standard practice, small but free win.
    # Params are bucketed by (lr, decay?) so each distinct lr becomes its own
    # pair of AdamW groups.
    buckets = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group_lr = lr
        for prefix, prefix_lr in (lr_by_prefix or {}).items():
            if name.startswith(prefix):
                group_lr = prefix_lr
                break
        skip_decay = param.ndim == 1 or name.endswith(".bias")
        buckets.setdefault((group_lr, skip_decay), []).append(param)

    optimizer = torch.optim.AdamW(
        [{"params": params, "lr": group_lr,
          "weight_decay": 0.0 if skip_decay else weight_decay}
         for (group_lr, skip_decay), params in buckets.items()],
        lr=lr, betas=(0.9, 0.98), eps=1e-9,
    )

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
 
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler
 
 
def train_step(model, img_pos_enc, batch, optimizer, scheduler=None,
               pad_idx=PAD_IDX, label_smoothing=0.0, grad_clip=1.0):
    """
    One optimization step. Wire this into your own epoch loop.
 
    batch is a dict with:
      "feat_map": encoder output BEFORE ImagePositionalEncoding -- this
                  function applies it. Either [batch, d_model, H, W] or the
                  flattened [batch, H*W, d_model] that the encoder returns.
      "feat_h":   optional int, the feature-map height. Only needed for the
                  flattened form, and only if img_pos_enc was constructed
                  without feat_h. See ImagePositionalEncoding for why.
      "tokens":   [batch, seq_len] padded target ids, already incl. BOS/EOS
      "memory_key_padding_mask": [batch, H*W] bool, True where the feature
                  column comes from width padding. Build it with
                  widths_to_memory_padding_mask() from each sample's true
                  width in processed/labels/*.jsonl.
 
    ASSUMPTION: the encoder is a separate module owned by whoever's building
    it. If the encoder should be trained jointly (it usually should be, at
    least fine-tuned), run it here and include its parameters in the
    optimizer -- right now this function takes features that are already
    computed, so the encoder gets no gradient.
 
    Returns a dict of scalars for logging.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
 
    memory = img_pos_enc(batch["feat_map"], feat_h=batch.get("feat_h"))
    decoder_in, labels, tgt_key_padding_mask = shift_target_for_teacher_forcing(
        batch["tokens"], pad_idx=pad_idx
    )
 
    # MathWriting images have variable width, padded per-batch -- so this mask
    # is required, not optional. Missing it silently degrades training rather
    # than erroring: the model wastes attention on blank padded columns and
    # nothing in the loss curve reveals it.
    memory_key_padding_mask = batch.get("memory_key_padding_mask")
    if memory_key_padding_mask is None:
        raise ValueError(
            "batch['memory_key_padding_mask'] is missing. MathWriting images are "
            "padded to the batch's max width, so cross-attention needs to know "
            "which feature columns are padding. Build it with "
            "widths_to_memory_padding_mask(). If you are deliberately testing "
            "with uniform-width images, pass an all-False mask explicitly."
        )
 
    logits = model(decoder_in, memory,
                   tgt_key_padding_mask=tgt_key_padding_mask,
                   memory_key_padding_mask=memory_key_padding_mask)
 
    loss = latex_cross_entropy(logits, labels, pad_idx=pad_idx,
                               label_smoothing=label_smoothing)
    loss.backward()
 
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
 
    with torch.no_grad():
        non_pad = labels.ne(pad_idx)
        correct = logits.argmax(-1).eq(labels) & non_pad
        # Token-level accuracy, NOT expression-level. HMER is normally scored
        # on exact-match ExpRate (whole expression correct); this is just a
        # cheap training signal. Compute ExpRate separately on the val set
        # using the beam search below.
        token_acc = correct.sum().item() / max(1, non_pad.sum().item())
 
    return {
        "loss": loss.item(),
        "token_acc": token_acc,
        "lr": optimizer.param_groups[0]["lr"],
    }
 
 
# ===========================================================================
# Inference: greedy + beam search
# ===========================================================================
# Kept deliberately separate from LatexDecoder.forward, which stays a pure
# teacher-forced training pass. Everything here is @torch.no_grad and rebuilds
# the full prefix each step.
#
# NOTE on speed: there is no KV cache, so decoding step t re-runs attention
# over all t previous tokens -- O(L^2) work per sequence. Fine for validation
# on a few thousand expressions; if inference latency matters later, adding
# incremental caching is the first optimization to make.
 
 
@torch.no_grad()
def greedy_decode(model, memory, memory_key_padding_mask=None,
                  bos_idx=BOS_IDX, eos_idx=EOS_IDX, max_len=MAX_LEN, **model_kwargs):
    """
    Batched greedy decoding -- useful as a fast sanity check during training.
 
    memory: [batch, H*W, d_model] (already through ImagePositionalEncoding)
    memory_key_padding_mask: [batch, H*W] bool -- pass the same mask used in
        training, or predictions get polluted by padded columns.
    Returns: list of token id lists, BOS and EOS stripped.
    """
    model.eval()
    device = memory.device
    batch = memory.size(0)
 
    tokens = torch.full((batch, 1), bos_idx, dtype=torch.long, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
 
    for _ in range(max_len - 1):
        logits = model(tokens, memory,
                       memory_key_padding_mask=memory_key_padding_mask,
                       **model_kwargs)          # [batch, cur_len, vocab]
        next_token = logits[:, -1].argmax(-1)   # only the last step matters
        # Once a sequence has emitted EOS, keep feeding it EOS so its shape
        # stays aligned with the rest of the batch; trimmed out below.
        next_token = torch.where(finished, torch.full_like(next_token, eos_idx), next_token)
        tokens = torch.cat([tokens, next_token.unsqueeze(1)], dim=1)
        finished |= next_token.eq(eos_idx)
        if finished.all():
            break
 
    results = []
    for row in tokens[:, 1:].tolist():          # drop BOS
        if eos_idx in row:
            row = row[: row.index(eos_idx)]
        results.append(row)
    return results
 
 
@torch.no_grad()
def beam_search_single(model, memory, memory_key_padding_mask=None, beam_width=5,
                       bos_idx=BOS_IDX, eos_idx=EOS_IDX, max_len=MAX_LEN,
                       length_penalty=1.0, **model_kwargs):
    """
    Beam search for ONE image. memory: [1, H*W, d_model].
 
    Deliberately single-image: batched beam search needs careful index
    bookkeeping across (batch x beam) and is where most bugs hide. Validation
    sets here are small enough that looping is fine -- see beam_search_batch.
 
    memory_key_padding_mask is an explicit parameter (not passed via
    **model_kwargs) because it must be expanded across beams alongside
    memory -- a [1, H*W] mask against [beam_width, H*W, d] memory is a
    shape error waiting to happen.
 
    length_penalty (the alpha in GNMT-style normalization): scores are divided
    by len**alpha. alpha=1.0 is plain mean log-prob. This matters more in HMER
    than in translation because LaTeX ground truth lengths vary a lot (a single
    digit vs. a nested fraction), and un-normalized beam search is biased
    toward short outputs -- it will happily drop a trailing "}".
    Tune on the val set; 0.6-1.0 is the usual range.
 
    Returns (token_ids, score) for the best hypothesis, BOS/EOS stripped.
    """
    assert memory.size(0) == 1, "beam_search_single expects a single image; use beam_search_batch"
    model.eval()
    device = memory.device
 
    # Replicate memory (and its mask) across beams so all hypotheses decode in
    # one forward pass.
    beam_memory = memory.expand(beam_width, -1, -1)
    beam_mask = (memory_key_padding_mask.expand(beam_width, -1)
                 if memory_key_padding_mask is not None else None)
 
    tokens = torch.full((1, 1), bos_idx, dtype=torch.long, device=device)
    scores = torch.zeros(1, device=device)  # cumulative log-prob per live beam
    finished = []                           # (token_list, normalized_score)
 
    for step in range(max_len - 1):
        num_beams = tokens.size(0)
        logits = model(tokens, beam_memory[:num_beams],
                       memory_key_padding_mask=(beam_mask[:num_beams]
                                                if beam_mask is not None else None),
                       **model_kwargs)
        log_probs = F.log_softmax(logits[:, -1], dim=-1)      # [num_beams, vocab]
 
        # Total score of every (existing beam, next token) pair, then take the
        # global top-k across that flattened grid.
        cand_scores = scores.unsqueeze(1) + log_probs          # [num_beams, vocab]
        flat = cand_scores.view(-1)
        top_scores, top_idx = flat.topk(min(beam_width, flat.size(0)))
 
        vocab_size = log_probs.size(-1)
        beam_idx = torch.div(top_idx, vocab_size, rounding_mode="floor")
        token_idx = top_idx % vocab_size
 
        tokens = torch.cat([tokens[beam_idx], token_idx.unsqueeze(1)], dim=1)
        scores = top_scores
 
        # Retire any beam that just emitted EOS; the rest continue.
        keep = []
        for i in range(tokens.size(0)):
            if token_idx[i].item() == eos_idx:
                seq = tokens[i, 1:-1].tolist()  # strip BOS and EOS
                norm = scores[i].item() / (max(1, len(seq)) ** length_penalty)
                finished.append((seq, norm))
            else:
                keep.append(i)
 
        if not keep or len(finished) >= beam_width:
            break
        keep = torch.tensor(keep, device=device)
        tokens, scores = tokens[keep], scores[keep]
 
    if not finished:
        # Hit max_len without any beam emitting EOS -- return the best partial
        # hypothesis rather than nothing. Frequent early in training; if it
        # persists, MAX_LEN is probably too small for your labels.
        # int(...) because Tensor.item() is typed as Number (int|float|bool),
        # which type checkers reject as a tensor index even though argmax
        # always yields an integer at runtime.
        best = int(scores.argmax().item())
        seq = tokens[best, 1:].tolist()
        return seq, scores[best].item() / (max(1, len(seq)) ** length_penalty)
 
    return max(finished, key=lambda pair: pair[1])
 
 
@torch.no_grad()
def beam_search_batch(model, memory, memory_key_padding_mask=None, beam_width=5,
                      bos_idx=BOS_IDX, eos_idx=EOS_IDX, max_len=MAX_LEN,
                      length_penalty=1.0, **model_kwargs):
    """Run beam_search_single over each image in a batch. Returns list of (ids, score)."""
    return [
        beam_search_single(
            model, memory[i : i + 1],
            memory_key_padding_mask=(memory_key_padding_mask[i : i + 1]
                                     if memory_key_padding_mask is not None else None),
            beam_width=beam_width, bos_idx=bos_idx, eos_idx=eos_idx,
            max_len=max_len, length_penalty=length_penalty, **model_kwargs)
        for i in range(memory.size(0))
    ]
 
 
if __name__ == "__main__":
    # End-to-end sanity check on dummy data.
    # vocab_size and the feature-map dims are still placeholders -- see the
    # CONFIRMED block at the top.
    batch, seq_len, vocab_size = 2, 10, 150
    d_model = 256
    stride = 16          # CONFIRMED: MobileNetV3-Large cut at features[:13]
    h, w = 4, 12         # h = 64 // stride = 4 (fixed); w varies per batch
 
    # feat_h is what makes the flattened encoder output usable at all. Set it
    # from the real stride once confirmed it.
    img_pos_enc = ImagePositionalEncoding(d_model, feat_h=h)
    dummy_feat_map = torch.randn(batch, d_model, h, w)
    memory = img_pos_enc(dummy_feat_map)

    # Encoder hands back [batch, H*W, d_model] instead of a 4D grid.
    # Both layouts must produce identical memory, or the two halves of the
    # pipeline disagree about where each feature sits in the image.
    encoder_style = dummy_feat_map.flatten(2).permute(0, 2, 1)   # [batch, H*W, d_model]
    img_pos_enc.eval()  # dropout off, otherwise the two draws differ
    with torch.no_grad():
        from_grid = img_pos_enc(dummy_feat_map)
        from_flat = img_pos_enc(encoder_style)
    print("4D vs flattened agree:", torch.allclose(from_grid, from_flat, atol=1e-6),
          "| max|diff| =", (from_grid - from_flat).abs().max().item())
    img_pos_enc.train()

    # The flattened path must REFUSE to run rather than silently drop the
    # positional encoding when it can't work out the grid height.
    try:
        ImagePositionalEncoding(d_model)(encoder_style)
    except ValueError as e:
        print("no feat_h -> raises: ", str(e).splitlines()[0])
    try:
        img_pos_enc(encoder_style, feat_h=5)   # 48 not divisible by 5
    except ValueError as e:
        print("bad feat_h -> raises:", str(e).splitlines()[0])

    # Two images of different true widths, padded to the same batch width --
    # exactly the situation the preprocessing pipeline produces.
    true_widths = torch.tensor([180, 64])
    mem_pad_mask = widths_to_memory_padding_mask(true_widths, h, w, stride)
    print("memory shape:      ", tuple(memory.shape))
    print("mem pad mask shape:", tuple(mem_pad_mask.shape),
          "| padded positions per sample:", mem_pad_mask.sum(1).tolist())
 
    model = LatexDecoder(vocab_size=vocab_size, d_model=d_model)
    optimizer, scheduler = build_optimizer(model, total_steps=1000)
 
    # Labels arrive already wrapped as [BOS, ...tokens..., EOS] then padded.
    dummy_tokens = torch.randint(4, vocab_size, (batch, seq_len))  # real ids start at 4
    dummy_tokens[:, 0] = BOS_IDX
    dummy_tokens[0, -2] = EOS_IDX
    dummy_tokens[0, -1] = PAD_IDX
    dummy_tokens[1, -1] = EOS_IDX
 
    stats = train_step(
        model, img_pos_enc,
        # Feeding the flattened form here, since that is what the real
        # encoder produces.
        {"feat_map": encoder_style, "tokens": dummy_tokens,
         "memory_key_padding_mask": mem_pad_mask},
        optimizer, scheduler,
    )
    print("train_step:        ", {k: round(v, 4) for k, v in stats.items()})
 
    greedy = greedy_decode(model, memory, memory_key_padding_mask=mem_pad_mask, max_len=12)
    print("greedy lengths:    ", [len(s) for s in greedy])
 
    beams = beam_search_batch(model, memory, memory_key_padding_mask=mem_pad_mask,
                              beam_width=3, max_len=12)
    print("beam best:         ", [(len(ids), round(score, 3)) for ids, score in beams])
 