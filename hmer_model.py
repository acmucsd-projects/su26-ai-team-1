"""
SUMMARY
-------
The full image -> LaTeX model: MobileNet encoder + Transformer decoder in one
nn.Module, so a single forward call goes from a batch of images to token logits
and a single optimizer covers both halves.

This is the only file that knows both halves exist. baseline_decoder.py and
latex_decoder.py stay encoder-agnostic (they take `memory`, not images), which
is what keeps the plain-baseline fallback intact.

WHAT THIS FILE SOLVES
---------------------
1. Connects encoder -> decoder, including the two interface details neither
   side owned: expanding 1-channel grayscale to the 3 channels MobileNet's
   ImageNet weights expect, and building the cross-attention padding mask from
   each sample's true (pre-padding) pixel width.
2. Puts the encoder in the optimizer, with a lower learning rate than the
   decoder -- see build_hmer_optimizer for why that matters.

LAYOUT-AGNOSTIC BY DESIGN
--------------------------
mobilenet_encoder.py currently returns a flattened [B, H*W, d_model], and is
being changed to return a 2D [B, d_model, 4, W] grid. This wrapper works with
both without modification, because ImagePositionalEncoding accepts either and
produces identical memory. Nothing here needs to change when that lands. But could
remove redundant code after the encoder is 2D-only.

STATUS
------
Verified end to end on dummy images (real MobileNetV3 weights, real shapes),
NOT on real data and never trained. Needs Ryan's Dataset/collate_fn to supply
`images` and `true_widths` before it can train on anything.
"""

import torch
import torch.nn as nn

from can_counting import CountingModule, counting_loss
from baseline_decoder import (
    BOS_IDX,
    EOS_IDX,
    MAX_LEN,
    PAD_IDX,
    ImagePositionalEncoding,
    beam_search_batch,
    build_optimizer,
    greedy_decode,
    latex_cross_entropy,
    shift_target_for_teacher_forcing,
    widths_to_memory_padding_mask,
)
from latex_decoder import PosFormerDecoder
from mobilenet_encoder import MobileNetEncoder

# Fixed by the preprocessing + encoder contract: images are exactly 64px tall
# and MobileNetV3 is cut at stride 16, so the feature grid is always 4 rows.
# Width is whatever the batch's padded width gives, and is never assumed.
IMAGE_HEIGHT = 64
ENCODER_STRIDE = 16
FEAT_H = IMAGE_HEIGHT // ENCODER_STRIDE     # == 4


class HMERModel(nn.Module):
    """
    encoder -> 2D positional encoding -> decoder, as one module.

    Construct it from the real vocab so nothing is hardcoded:

        from latex_decoder import load_vocab_config
        cfg = load_vocab_config("processed/vocab.json")
        model = HMERModel(cfg.vocab_size, structure_tokens=cfg.structure_tokens)

    Set use_arm=False, use_position_forest=False to get the plain baseline --
    that path is bit-identical to LatexDecoder, so it is a real fallback.
    """

    def __init__(self, vocab_size, structure_tokens=None, d_model=256,
                 nhead=8, num_layers=3, dim_feedforward=1024, dropout=0.1,
                 max_len=MAX_LEN, use_arm=True, use_position_forest=True,
                 stride=ENCODER_STRIDE, feat_h=FEAT_H, max_w=64):
        super().__init__()
        self.stride = stride
        self.feat_h = feat_h

        self.encoder = MobileNetEncoder(d_model=d_model)
        # Auxiliary CAN branch. It reads the same MobileNet features as the transformer path, but predicts a bag of symbol counts independently.
        self.counting_module = CountingModule(
            d_model=d_model, vocab_size=vocab_size, dropout=dropout
        )
        # feat_h is passed so the flattened encoder output can be un-flattened.
        # It is ignored (but cross-checked) when the encoder returns a 4D grid.
        self.img_pos_enc = ImagePositionalEncoding(
            d_model, max_h=max(8, feat_h), max_w=max_w, dropout=dropout,
            feat_h=feat_h,
        )
        self.decoder = PosFormerDecoder(
            vocab_size, d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout, max_len=max_len,
            use_arm=use_arm, use_position_forest=use_position_forest,
            structure_tokens=structure_tokens,
        )

    @staticmethod
    def _to_three_channels(images):
        """
        MobileNetV3's ImageNet weights expect 3 channels, but the preprocessing
        notebook renders single-channel ("L" mode) grayscale PNGs. Repeating the
        grey channel three times is the standard bridge -- it keeps the
        pretrained filters meaningful, where a 1-channel first conv would throw
        them away.

        expand() is a stride-0 view, so this costs no extra memory.
        """
        c = images.size(1)
        if c == 1:
            return images.expand(-1, 3, -1, -1)
        if c == 3:
            return images
        raise ValueError(
            f"Expected images with 1 (grayscale) or 3 (RGB) channels, got {c}. "
            f"Shape was {tuple(images.shape)} -- images should be "
            f"[batch, channels, 64, width]."
        )

    def encode(self, images, true_widths=None, return_features=False):
        """
        images: [batch, 1 or 3, 64, W] -- W is the batch's padded width
        true_widths: [batch] each sample's real pixel width BEFORE padding.
            Comes from the "width" field in processed/labels/*.jsonl. Optional
            only so shape-checking works without it; always pass it in training.

        Returns (memory, memory_key_padding_mask, feat_h).

        Kept separate from forward() because inference encodes ONCE and then
        decodes many steps against the same memory -- beam search would be
        pointlessly expensive if the image were re-encoded per token.
        """
        if images.size(2) != IMAGE_HEIGHT:
            raise ValueError(
                f"Expected images {IMAGE_HEIGHT}px tall, got {images.size(2)}. "
                f"feat_h={self.feat_h} is derived from that fixed height, so a "
                f"different height silently misaligns every position code."
            )

        features = self.encoder(self._to_three_channels(images))
        # Works for both encoder output layouts; see the module docstring.
        h, w = self.img_pos_enc.infer_hw(features, self.feat_h)
        memory = self.img_pos_enc(features, feat_h=self.feat_h)

        memory_key_padding_mask = None
        if true_widths is not None:
            memory_key_padding_mask = widths_to_memory_padding_mask(
                true_widths, h, w, self.stride, device=images.device
            )
        if return_features:
            return memory, memory_key_padding_mask, h, features
        return memory, memory_key_padding_mask, h

    def forward(self, images, tgt_tokens, true_widths=None,
                tgt_key_padding_mask=None, return_counts=False):
        """
        Teacher-forced training pass: images + shifted target -> token logits.
        tgt_tokens is the decoder INPUT (already shifted right).
        """
        encoded = self.encode(
            images, true_widths, return_features=return_counts
        )
        memory, memory_mask, h = encoded[:3]
        logits = self.decoder(
            tgt_tokens, memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_mask,
            memory_height=h,
        )
        if not return_counts:
            return logits

        count_predictions = self.counting_module(encoded[3], memory_mask)
        return logits, count_predictions

    def predict_counts(self, images, true_widths=None):
        """Return auxiliary symbol counts without running the decoder."""
        _, memory_mask, _, features = self.encode(
            images, true_widths, return_features=True
        )
        return self.counting_module(features, memory_mask)

    @torch.no_grad()
    def predict(self, images, true_widths=None, beam_width=1, max_len=MAX_LEN):
        """
        Images -> token id lists. beam_width=1 uses greedy (much faster, for
        monitoring during training); >1 runs beam search, which is what you want
        for a real ExpRate number.
        """
        self.eval()
        memory, memory_mask, h = self.encode(images, true_widths)
        if beam_width <= 1:
            return greedy_decode(self.decoder, memory,
                                 memory_key_padding_mask=memory_mask,
                                 max_len=max_len, memory_height=h)
        beams = beam_search_batch(self.decoder, memory,
                                  memory_key_padding_mask=memory_mask,
                                  beam_width=beam_width, max_len=max_len,
                                  memory_height=h)
        return [ids for ids, _score in beams]


def build_hmer_optimizer(model, encoder_lr=1e-4, decoder_lr=3e-4,
                         weight_decay=1e-4, warmup_steps=500,
                         total_steps=50_000):
    """
    One optimizer over BOTH halves, with a lower learning rate on the encoder.

    Why two rates rather than one: the encoder starts from ImageNet weights and
    only needs nudging, while the decoder starts from random init and needs to
    learn from scratch. A single LR either wrecks the pretrained features (too
    high) or starves the decoder (too low). ~3x apart is a common starting
    point; it is a starting point, not a tuned value.

    To FREEZE the encoder instead (a reasonable first experiment -- it isolates
    whether the decoder works before adding encoder gradients to the mix):
        for p in model.encoder.parameters():
            p.requires_grad = False
    build_optimizer skips params with requires_grad=False, so nothing else
    needs to change.
    """
    return build_optimizer(
        model, lr=decoder_lr, weight_decay=weight_decay,
        warmup_steps=warmup_steps, total_steps=total_steps,
        lr_by_prefix={"encoder.": encoder_lr},
    )


def hmer_train_step(model, batch, optimizer, scheduler=None, pad_idx=PAD_IDX,
                    label_smoothing=0.0, grad_clip=1.0,
                    counting_weight=0.1):
    """
    One optimization step over the full model. The baseline's train_step takes
    precomputed features (so the encoder gets no gradient); this one starts from
    images, so gradients flow all the way back through MobileNet.

    batch:
      "images":      [batch, 1 or 3, 64, W] padded to the batch's max width
      "tokens":      [batch, seq_len] padded ids, already [BOS ... EOS]
      "true_widths": [batch] real pixel widths before padding

    The total objective is sequence_loss + counting_weight * counting_loss.
    The separate PosFormer position-forest objective is not included here.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)

    true_widths = batch.get("true_widths")
    if true_widths is None:
        raise ValueError(
            "batch['true_widths'] is missing. Images are padded to the batch's "
            "max width, so cross-attention needs to know which feature columns "
            "are padding, and the true width is NOT recoverable after padding. "
            "The Dataset/collate_fn must carry it through from the 'width' "
            "field in processed/labels/*.jsonl."
        )

    decoder_in, labels, tgt_key_padding_mask = shift_target_for_teacher_forcing(
        batch["tokens"], pad_idx=pad_idx
    )
    logits, predicted_counts = model(
        batch["images"], decoder_in, true_widths=true_widths,
        tgt_key_padding_mask=tgt_key_padding_mask, return_counts=True,
    )

    sequence_loss = latex_cross_entropy(
        logits, labels, pad_idx=pad_idx, label_smoothing=label_smoothing
    )
    count_loss = counting_loss(predicted_counts, batch["tokens"])
    loss = sequence_loss + counting_weight * count_loss
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    with torch.no_grad():
        keep = labels.ne(pad_idx)
        correct = logits.argmax(-1).eq(labels) & keep
        token_acc = correct.sum().item() / max(1, keep.sum().item())

    return {"loss": loss.item(), "sequence_loss": sequence_loss.item(),
            "counting_loss": count_loss.item(), "token_acc": token_acc,
            "lr": optimizer.param_groups[0]["lr"]}


if __name__ == "__main__":
    # End-to-end check on dummy images: real MobileNetV3 weights, real shapes.
    # This is the first point where both halves run in one process.
    import math

    from latex_decoder import StructureTokens

    torch.manual_seed(0)
    vocab_size, seq_len = 40, 9
    # Stand-in for load_vocab_config("processed/vocab.json").
    st = StructureTokens(sup=6, sub=7, frac=4, sqrt=5, lbrace=8, rbrace=9,
                         lbracket=10, rbracket=11)

    model = HMERModel(vocab_size, structure_tokens=st, num_layers=2)
    optimizer, scheduler = build_hmer_optimizer(model, total_steps=100)

    print("param counts:")
    enc = sum(p.numel() for p in model.encoder.parameters())
    dec = sum(p.numel() for p in model.decoder.parameters())
    count = sum(p.numel() for p in model.counting_module.parameters())
    print(f"  encoder {enc/1e6:.2f}M | decoder {dec/1e6:.2f}M | "
          f"counting {count/1e6:.2f}M")
    in_opt = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
    print(f"  in optimizer: {in_opt/1e6:.2f}M "
          f"({'ALL' if in_opt == enc + dec + count else 'MISSING SOME'})")
    # Read initial_lr, not lr: LambdaLR multiplies by the warmup factor, which
    # is 0 at step 0, so every group's live `lr` starts at 0.0 by design.
    print(f"  configured LRs: {sorted({g['initial_lr'] for g in optimizer.param_groups})}"
          f"  (encoder lower than decoder)")

    print("\nend-to-end forward on dummy grayscale images:")
    for px in ([96, 64], [420, 300, 180]):
        b = len(px)
        padded_w = max(px)
        images = torch.randn(b, 1, 64, padded_w)          # 1-channel, as rendered
        widths = torch.tensor(px)
        tokens = torch.randint(4, vocab_size, (b, seq_len))
        tokens[:, 0] = BOS_IDX
        tokens[:, -1] = EOS_IDX

        memory, mask, h = model.encode(images, widths)
        stats = hmer_train_step(
            model, {"images": images, "tokens": tokens, "true_widths": widths},
            optimizer, scheduler,
        )
        expected_w = math.ceil(padded_w / ENCODER_STRIDE)
        print(f"  img {tuple(images.shape)} -> memory {tuple(memory.shape)} "
              f"(grid {h}x{memory.shape[1]//h}, expected 4x{expected_w}) "
              f"| loss {stats['loss']:.3f}")

    print("\ngradients reach the encoder (this is what task #3 is about):")
    enc_grads = [p.grad.abs().sum().item() for p in model.encoder.parameters()
                 if p.grad is not None]
    print(f"  {len(enc_grads)} encoder tensors have grads, "
          f"total magnitude {sum(enc_grads):.1f}")
    # The two rates must stay proportional as the schedule advances, otherwise
    # per-group LRs would be silently flattened by the scheduler.
    live = sorted({round(g["lr"], 8) for g in optimizer.param_groups})
    print(f"  LRs after {scheduler.last_epoch} steps: {live} "
          f"(ratio {live[1]/live[0]:.1f}x preserved)" if len(live) > 1 else "")

    print("\ninference (max_len capped -- an untrained model never emits EOS):")
    images = torch.randn(2, 1, 64, 256)
    widths = torch.tensor([256, 128])
    print("  greedy:", [len(s) for s in model.predict(images, widths, max_len=8)])
    print("  beam-3:", [len(s) for s in
                        model.predict(images, widths, beam_width=3, max_len=8)])
