"""
SUMMARY
-------
Epoch-level training and validation for HMERModel: the loop around a
per-batch train step, plus a validation pass that reports the metric HMER is
actually scored on (ExpRate), not just token accuracy.

  fit()          -- multi-epoch driver, checkpoints the best model
  train_epoch()  -- one pass over the training set

Which objective gets trained is the step_fn argument on fit()/train_epoch():
hmer_train_step (plain cross-entropy, the default) or hmer_posformer_train_step
(cross-entropy + the PosFormer auxiliary losses). Both live in hmer_model.py
because both need gradients to reach the encoder.
  validate()     -- loss + ExpRate on the validation set
  expression_rate() -- exact-match and error-tolerant rates

WHY ExpRate AND NOT token accuracy
-----------------------------------
train_step reports token_acc, which counts individual tokens under teacher
forcing -- the model is handed the correct prefix at every step. That number
looks good long before the model can produce a whole correct expression on its
own. HMER papers report ExpRate: the fraction of expressions predicted
*exactly* right, generated autoregressively with no teacher forcing. It is the
only number comparable to published results, and it is much harsher.
We also report <=1 and <=2 token error rates, as CoMER/PosFormer do.

STATUS
------
Runs end to end against the dummy Dataset at the bottom of this file. Has NOT
been run against real data -- that needs the processed/ directory built by
mathwriting_preprocessing.ipynb. The real Dataset and the batch contract now
live in dataset.py (MathWritingDataset + collate_fn), which this file imports;
_DummyDataset below is only a random-data stand-in so this file stays runnable
without processed/.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset

from baseline_decoder import (
    BOS_IDX,
    EOS_IDX,
    MAX_LEN,
    PAD_IDX,
    latex_cross_entropy,
    shift_target_for_teacher_forcing,
)
from hmer_model import (
    HMERModel,
    build_hmer_optimizer,
    hmer_posformer_train_step,  # noqa: F401  -- re-exported for fit(step_fn=...)
    hmer_train_step,
)


# ===========================================================================
# Metrics
# ===========================================================================


def strip_special(ids, pad_idx=PAD_IDX, bos_idx=BOS_IDX, eos_idx=EOS_IDX):
    """
    Target ids -> the bare token sequence, for comparison against a prediction.

    Labels are stored as [BOS, ...tokens, EOS] then right-padded, so this drops
    BOS, truncates at the first EOS, and removes padding. greedy_decode and
    beam_search already return predictions in this stripped form, so both sides
    of the comparison line up.
    """
    out = []
    for i in ids:
        if i == eos_idx:
            break
        if i in (pad_idx, bos_idx):
            continue
        out.append(i)
    return out


def _levenshtein(a, b):
    """Token-level edit distance, for the <=1 / <=2 error tolerances."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1,          # deletion
                            curr[j - 1] + 1,      # insertion
                            prev[j - 1] + (ca != cb)))   # substitution
        prev = curr
    return prev[-1]


def expression_rate(preds, targets, tolerances=(0, 1, 2)):
    """
    ExpRate family. preds/targets are lists of token-id lists.

    tolerance 0 = exact match (the headline ExpRate number);
    1 and 2 allow that many token edits, which is what papers report alongside
    it to show how close the near-misses are.
    """
    if not targets:
        return {f"exprate_leq{t}" if t else "exprate": 0.0 for t in tolerances}
    dists = [_levenshtein(p, t) for p, t in zip(preds, targets)]
    return {
        (f"exprate_leq{t}" if t else "exprate"):
            sum(d <= t for d in dists) / len(dists)
        for t in tolerances
    }


# ===========================================================================
# Loops
# ===========================================================================


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def train_epoch(model, loader, optimizer, scheduler=None, device="cpu",
                log_every=0, step_fn=hmer_train_step, **step_kwargs):
    """
    One pass over the training set. Returns averaged stats.

    step_fn selects the objective. It must have hmer_train_step's signature --
    (model, batch, optimizer, scheduler, **kwargs) -> stats dict containing at
    least "loss" and "lr". The two that exist:

        hmer_train_step            plain cross-entropy (default)
        hmer_posformer_train_step  + the PosFormer auxiliary losses

    Anything extra the step reports (ce_loss / layer_loss / pos_loss) is
    averaged and returned alongside, so no key list is hardcoded here.

    The scheduler steps PER BATCH inside the step function, not once per epoch
    -- that is why fit() sets total_steps = epochs * len(loader). Getting that
    wrong doesn't error, it just makes the cosine curve finish at the wrong
    time (too early = the LR sits at ~0 for the rest of training).
    """
    model.train()
    totals, n = defaultdict(float), 0
    for i, batch in enumerate(loader):
        stats = step_fn(model, _to_device(batch, device), optimizer,
                        scheduler, **step_kwargs)
        for k, v in stats.items():
            totals[k] += v
        n += 1
        if log_every and (i + 1) % log_every == 0:
            print(f"    step {i+1}/{len(loader)}  loss {stats['loss']:.4f}  "
                  f"lr {stats['lr']:.2e}")
    return {k: v / max(1, n) for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, device="cpu", beam_width=1, max_len=MAX_LEN,
             pad_idx=PAD_IDX, max_batches=None):
    """
    Validation loss + ExpRate.

    Two different things happen per batch, deliberately:
      1. A teacher-forced forward pass for val_loss -- directly comparable to
         train_loss, so the two curves diverging tells you about overfitting.
      2. Free-running generation for ExpRate -- no teacher forcing, which is
         the only honest measure of what the model can actually produce.

    beam_width=1 (greedy) is the default because generation dominates
    validation time. Use beam_width=5 for a number you would report; running
    beam search every epoch on a large val set is usually not worth the wall
    clock. max_batches caps the ExpRate sample if you want it cheaper still.
    """
    model.eval()
    losses, preds, targets = [], [], []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = _to_device(batch, device)
        tokens, widths = batch["tokens"], batch["true_widths"]

        decoder_in, labels, tgt_key_padding_mask = shift_target_for_teacher_forcing(
            tokens, pad_idx=pad_idx
        )
        logits = model(batch["images"], decoder_in, true_widths=widths,
                       tgt_key_padding_mask=tgt_key_padding_mask)
        losses.append(latex_cross_entropy(logits, labels, pad_idx=pad_idx).item())

        preds.extend(model.predict(batch["images"], widths,
                                   beam_width=beam_width, max_len=max_len))
        targets.extend(strip_special(row, pad_idx=pad_idx) for row in tokens.tolist())

    out = {"val_loss": sum(losses) / max(1, len(losses))}
    out.update(expression_rate(preds, targets))
    return out


def fit(model, train_loader, val_loader, epochs=10, device="cpu",
        encoder_lr=1e-4, decoder_lr=3e-4, warmup_steps=500,
        checkpoint_path="best_model.pt", val_beam_width=1,
        val_max_batches=None, log_every=0,
        step_fn=hmer_train_step, **step_kwargs):
    """
    Multi-epoch driver. Checkpoints on best ExpRate, not best val_loss --
    they don't always move together, and ExpRate is what we actually care about.

    step_fn picks the training objective and is passed straight to
    train_epoch(). Default is the plain baseline; to train the PosFormer
    auxiliary task instead:

        from hmer_model import hmer_posformer_train_step
        fit(model, train_loader, val_loader, step_fn=hmer_posformer_train_step)

    (the model must have been built with use_position_forest=True). Validation
    is unaffected either way -- the aux heads are training-only, so ExpRate
    stays the same free-running metric and the two runs are comparable.

    val_max_batches caps how many validation batches each epoch's validate()
    call consumes. None (the default) means uncapped -- the whole split, which
    is the previous behaviour exactly.

    Returns the per-epoch history so you can plot it.
    """
    model.to(device)
    total_steps = epochs * len(train_loader)
    optimizer, scheduler = build_hmer_optimizer(
        model, encoder_lr=encoder_lr, decoder_lr=decoder_lr,
        warmup_steps=min(warmup_steps, max(1, total_steps // 10)),
        total_steps=total_steps,
    )

    history, best = [], -1.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, scheduler,
                         device=device, log_every=log_every, step_fn=step_fn,
                         **step_kwargs)
        va = validate(model, val_loader, device=device, beam_width=val_beam_width,
                      max_batches=val_max_batches)

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **va,
               "secs": time.time() - t0}
        history.append(row)
        # Only printed when the step reports them, so the baseline line is
        # character-for-character what it was before step_fn existed.
        aux = ""
        if "layer_loss" in tr:
            aux = (f"  aux(layer {tr['layer_loss']:.3f} / "
                   f"pos {tr['pos_loss']:.3f})")
        print(f"epoch {epoch:>3}  train_loss {tr['loss']:.4f}  "
              f"val_loss {va['val_loss']:.4f}  ExpRate {va['exprate']:.3f}  "
              f"(<=1 {va['exprate_leq1']:.3f})  {row['secs']:.1f}s{aux}")

        if va["exprate"] > best:
            best = va["exprate"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "exprate": best}, checkpoint_path)
            print(f"           new best ExpRate {best:.3f} -> {checkpoint_path}")

    return history


# ===========================================================================
# Dummy data -- stand-in for Ryan's Dataset, and a spec of the batch contract
# ===========================================================================


class _DummyDataset(Dataset):
    """
    Random images/labels so this file is runnable before the real Dataset
    lands. The REAL Dataset must yield these three things per sample:

        "image":  [1, 64, width]  float, height exactly 64, unpadded
        "tokens": LongTensor      already [BOS, ...ids, EOS], unpadded
        "width":  int             the true pixel width of this image

    and read them from processed/images/{split}/ + processed/labels/{split}.jsonl.
    """

    def __init__(self, n=32, vocab_size=40, min_w=64, max_w=320, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.samples = []
        for _ in range(n):
            w = int(torch.randint(min_w, max_w + 1, (1,), generator=g))
            length = int(torch.randint(3, 12, (1,), generator=g))
            ids = torch.randint(4, vocab_size, (length,), generator=g)
            tokens = torch.cat([torch.tensor([BOS_IDX]), ids, torch.tensor([EOS_IDX])])
            self.samples.append({
                "image": torch.rand(1, 64, w, generator=g),
                "tokens": tokens,
                "width": w,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


# collate_fn lives in dataset.py -- it is the single source of truth for the
# batch contract, and the real MathWritingDataset ships alongside it. Imported
# here (rather than duplicated) so the dummy self-test below exercises exactly
# the same padding code path that real training does.
from dataset import collate_fn  # noqa: E402


if __name__ == "__main__":
    from latex_decoder import StructureTokens

    torch.manual_seed(0)
    vocab_size = 40
    # Stand-in for load_vocab_config("processed/vocab.json").
    st = StructureTokens(sup=6, sub=7, frac=4, sqrt=5, lbrace=8, rbrace=9,
                         lbracket=10, rbracket=11)

    train_loader = DataLoader(_DummyDataset(n=24, vocab_size=vocab_size, seed=0),
                              batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(_DummyDataset(n=8, vocab_size=vocab_size, seed=1),
                            batch_size=4, collate_fn=collate_fn)

    model = HMERModel(vocab_size, structure_tokens=st, num_layers=2)

    print("batch contract produced by collate_fn:")
    b = next(iter(train_loader))
    for k, v in b.items():
        print(f"  {k:<12} {tuple(v.shape)}  {v.dtype}")

    print("\nmetric sanity (perfect / one-token-off / empty predictions):")
    tgt = [[5, 6, 7], [8, 9], [10]]
    print("  exact      ", expression_rate(tgt, tgt))
    print("  1 tok off  ", expression_rate([[5, 6, 99], [8, 9], [10]], tgt))
    print("  empty      ", expression_rate([[], [], []], tgt))

    print("\n3 epochs on dummy data (ExpRate stays 0 -- random labels are")
    print("unlearnable by construction; this checks the loop, not learning):")
    history = fit(model, train_loader, val_loader, epochs=3,
                  checkpoint_path="/tmp/hmer_dummy_ckpt.pt")

    print(f"\nhistory rows: {len(history)}  keys: {sorted(history[0])}")
