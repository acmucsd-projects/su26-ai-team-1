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
- Still unconfirmed: real vocab_size.
 
ASSUMPTIONS (still open)
-------------------------
- vocab_size=150 and h/w in the sanity check are placeholders.
- Architecture sizes (d_model=256, nhead=8, num_layers=3, dim_feedforward=1024)
  match BTTR/CoMER/PosFormer's published HMER config, not tuned on our data.
- max_len=200 is an arbitrary cap, not derived from real label lengths yet.
 
WIP -- NOT YET DONE
--------------------
- No Dataset/DataLoader wired to processed/labels/*.jsonl.
- No PosFormer additions (position forest, attention correction).
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
 
    max_h only needs to cover the encoder's output height (4 or 2 -- see the
    stride question above), but max_w must cover the WIDEST batch: images are
    ~64px tall with a median aspect ratio near 3:1 and p99 widths around
    420px, so at stride 16 that is ~27 columns. 64 leaves headroom.
    """
 
    pe: torch.Tensor  # tells the type checker this buffer is a Tensor, not a Module
 
    def __init__(self, d_model, max_h=64, max_w=64, dropout=0.1):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even to split across H/W"
        self.dropout = nn.Dropout(dropout)
 
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
 
    def forward(self, feat):
        # feat: [batch, d_model, H, W]  (raw MobileNet output)
        b, c, h, w = feat.shape
        feat = feat + self.pe[:, :h, :w].unsqueeze(0)
        return self.dropout(feat.flatten(2).permute(0, 2, 1))  # -> [batch, H*W, d_model]
 
 
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
 
 
def build_optimizer(model, lr=3e-4, weight_decay=1e-4, warmup_steps=500, total_steps=50_000):
    """
    Reasonable defaults, not tuned. AdamW + linear warmup into cosine decay.
 
    Warmup matters more than the exact LR here: post-LN transformers (which is
    what nn.TransformerDecoderLayer defaults to) are unstable in the first few
    hundred steps without it. Scheduler steps PER BATCH, not per epoch.
 
    total_steps should be roughly (num_epochs * batches_per_epoch); the cosine
    curve is wrong if it doesn't match the real schedule length.
    """
    # No weight decay on biases/norms -- standard practice, small but free win.
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
 
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
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
      "feat_map": [batch, d_model, H, W] encoder output (raw MobileNet features,
                  BEFORE ImagePositionalEncoding -- this function applies it)
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
 
    memory = img_pos_enc(batch["feat_map"])
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
    stride = 16          # UNRESOLVED: 16 vs 32, pending Yuki
    h, w = 4, 12         # h = 64 // stride; w varies per batch
 
    img_pos_enc = ImagePositionalEncoding(d_model)
    dummy_feat_map = torch.randn(batch, d_model, h, w)
    memory = img_pos_enc(dummy_feat_map)
 
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
        {"feat_map": dummy_feat_map, "tokens": dummy_tokens,
         "memory_key_padding_mask": mem_pad_mask},
        optimizer, scheduler,
    )
    print("train_step:        ", {k: round(v, 4) for k, v in stats.items()})
 
    greedy = greedy_decode(model, memory, memory_key_padding_mask=mem_pad_mask, max_len=12)
    print("greedy lengths:    ", [len(s) for s in greedy])
 
    beams = beam_search_batch(model, memory, memory_key_padding_mask=mem_pad_mask,
                              beam_width=3, max_len=12)
    print("beam best:         ", [(len(ids), round(score, 3)) for ids, score in beams])
 