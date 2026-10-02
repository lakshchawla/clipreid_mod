# Methodology: Part-Prompt CLIP-ReID with Language-Guided Part Interaction

Backbone: CLIP RN50 (ModifiedResNet), input 256×128, stride 16 (feature grid 16×8). Primary dataset: Market-1501
(751 train identities, 12,936 train images); MSMT17 and DukeMTMC-reID are registered in `datasets/part_datasets.py`.
Code: `processor/train_part_prompts_stage1.py`, `processor/train_part_prompts_stage2.py`.

## 1. Overview

The method extends CLIP-ReID's two-stage recipe from one identity prompt to **K+1 = 6 identity prompts per person**:
one global slot and five body-part slots (head, upper-arms/torso, lower-arms/torso, legs, feet).

- **Stage 1 (prompt learning).** CLIP's image and text encoders are frozen. Only per-(identity, slot) context
  tokens are learned, by contrasting each slot's image embedding with its prompt. An auxiliary Text Attention
  Block (TAB) adds an image-conditioned pooling of the prompt tokens.
- **Stage 2 (image-side fine-tuning).** The learned prompts are frozen and encoded once into targets. The image
  encoder is fine-tuned with the CLIP-ReID baseline losses plus a Language-Guided Part Interaction Module (LPIM)
  that yields a global-semantic token and K part tokens. The retrieval feature is a holistic concatenation of the
  baseline and LPIM features.

## 2. Part supervision: PifPaf masks (both stages)

1. PifPaf produces 36 confidence fields (17 keypoints + 19 joints) per image. They are pre-computed per image
   (BPBreID `pifpaf_maskrcnn_filtering` variant).
2. Fields are grouped into five vertical parts (BPBreID `five_v`), each as the clamped max over its members.
   A background channel is set where no part reaches 0.5. A softmax with weight 15 over the six channels gives a
   soft part distribution `[K+1, h, w]`.
3. Visibility of part k = it wins the argmax at ≥ 1 grid location (BPBreID rule). The global slot is always visible.
4. Stage 2 interpolates masks to 256×128 and applies the same flip, pad+crop and random erasing as the image.
   Padded and erased regions become background.

## 3. Stage 1: per-part prompt learning

### 3.1 Frozen image side (cached once)

For every training image, with its mask:
- Slot 0: `xproj[0]`, the stock CLIP-ReID global feature (1024-d).
- Slot k: `GWAP(x4, mask_k)` followed by a frozen `Linear(2048→1024)` equal to CLIP's attention-pool value and
  output projection, `W = c_proj·v_proj`. This is the attention pool with the attention weights replaced by the mask.

Features `[N, 6, 1024]` and visibility `[N, 6]` are cached, as in CLIP-ReID stage 1.

### 3.2 Text side

- One frozen CLIP text encoder, shared across slots.
- Templates: `A photo of a X X X X person.` and `A photo of the <part> of a X X X X person.`
- Learnable parameters: `cls_ctx[C, 6, 4, 512]`, which is 4 context tokens for every (identity, slot), initialised
  N(0, 0.02). On Market this is 751×6×4×512 ≈ 9.2M parameters. Slots differ by template words and by their own ctx.
- The prompt feature is the EOT token state projected by `text_projection`.

### 3.3 Contrastive objective with full-pool negatives (`FULL_POOL_NEGATIVES`)

For each slot s, over the batch images with part s visible (images with the part invisible are dropped from that
slot's loss), both directions are L2-normalised and scored at temperature 0.01:

- **i2t:** each image against **all C identity prompts**. The prompt bank is detached and the in-batch identities are
  spliced in with gradient. After each step the fresh prompts are written back into the bank. The whole bank is
  rebuilt every epoch.
- **t2i:** each in-batch prompt against **every dataset image visible in slot s** (static, since the image side is
  frozen). Positives are the identity's in-batch images (CLIP-ReID's positive set). The identity's other images are
  masked out of the softmax, so they are neither positives nor false negatives.

Each slot is back-propagated on its own (same gradient, 1/6 of the memory).

### 3.3.1 Why batch positives

Averaging over all of an identity's ~17 images at a sharp temperature pulled prompts toward outlier images and did
not converge.

### 3.4 Text Attention Block (TAB, enabled)

TAB gives an image-conditioned pooling of the *same* prompt, using the token states that EOT pooling discards.

- Query: `W_q x_part` (the slot's image embedding). Keys and values: `W_k`, `W_v` of the 77 projected token states
  of the slot's prompt. Tokens after EOT are masked. d = 256, 4 heads.
- Output: `t_hat = t_eot + W_o · Attn(q, K, V)`, with `W_o` zero-initialised. TAB therefore starts as the identity
  on the EOT prompt, and the first step equals the plain-EOT contrast.
- It is trained with the same two-direction contrast against the same pools. Keys and values come from the bank,
  with in-batch identities spliced in with gradient. The t2i side uses the batch positives plus 2,048 sampled
  negative images, each column scored with the prompt as conditioned by that image.
- Total slot loss: `L = L_i2t + L_t2i + 0.5 · L_TAB`.
- TAB is used **only as a stage-1 auxiliary loss**. It cannot bypass the prompts, because template tokens are
  identical across identities and every identity signal in K/V comes from `cls_ctx`. TAB can only help by making
  `cls_ctx` better under EOT pooling. The pass/fail metric is the plain-EOT image→text top-1.
  The TAB-conditioned top-1 is logged as a diagnostic.

### 3.5 Optimisation

Adam, lr 5e-4, weight decay 1e-4, 60 epochs, warmup-cosine (5 warmup epochs, from 1e-5 to 1e-6), batch 64, seed 1234.
Checkpoints store `cls_ctx`, TAB, optimiser and the knobs that stage 2 asserts against (H, W, stride, dataset, N_CTX).

### 3.6 Stage-1 results (Market-1501, 60 epochs, training identities)

Image→text top-1 per slot: global 0.958, head 0.816, upper-arms/torso 0.912, lower-arms/torso 0.962, legs 0.845,
feet 0.682 (chance 0.0013). This is a train-set alignment metric. Head and feet are the weakest and have the most
invisible parts (feet ~22–31% invisible at test).

## 4. Stage 2: image-side fine-tuning

### 4.1 Targets

The stage-1 prompts are encoded once, frozen, into `text_all[C, 6, 1024]`. Only the EOT snapshot is used, not TAB.

### 4.2 Baseline branch (CLIP-ReID stage 2, verbatim)

`gap3 = GAP(x3)`, `gap4 = GAP(x4)` with a BNNeck ID head, and `g = xproj[0]` with a BNNeck ID head. The losses are
label-smoothed ID CE on gap4 and g, batch-hard triplet (margin 0.3) on gap3, gap4 and g, and i2t
`CE(g · text_global^T)`.

### 4.3 LPIM branch (after PromptSG, CVPR'24)

- Six identity-agnostic text queries (`A photo of a person.` and `A photo of the <part> of a person.`) go through the
  frozen CLIP text encoder once. They cross-attend over the 128 x4 locations. K/V/output projections and the
  positional embedding are initialised from CLIP's attention pool. `q_proj` is new, and a residual FFN with a
  zero-initialised output follows. There is no query-to-query interaction by default.
- Outputs: `z0` (global-semantic token) and `z1..zK` (part tokens).
- `pbar` is the attentive pooling of the part tokens, `alpha = softmax(w·z_k)`, `pbar = Σ alpha_k z_k`.
- A visibility head on each part token predicts presence, so no mask is needed at test time.
- The head-averaged attention map of each part query is supervised by its PifPaf mask (KL, present parts only).

### 4.4 Loss

```
L = 1·ID + 1·TRI + 1·i2t                       # CLIP-ReID baseline branch
  + 1·LPIM_ID(z0, pbar) + 1·LPIM_TRI(z0, pbar) # BNNeck ID + triplet on LPIM outputs
  + 1·PART_TRI(z1..zK)                         # visibility-masked LSE triplet
  + 0.5·SUPCON                                 # per-slot symmetric SupCon(z_s, text_all[:, s])
  + 1·ATTN_KL + 0.1·VIS_BCE
```

- **Part triplet.** Per-part cosine distances combine over mutually visible parts: visibility-weighted mean until
  epoch 40, then log-sum-exp with γ ramped linearly to 5 by epoch 80. Pairs sharing no visible part are excluded.
- **Random text erasing.** Per (sample, slot) the identity-prompt anchor is dropped from SupCon with p = 0.1, and from
  i2t for slot 0. A per-slot Bernoulli dropout (p = 0.1) over text dimensions is shared across identities. The aim
  is that image slots learn to stand on their own when their text anchor, or the part itself, is missing.
- **Optional.** An XBM cross-batch memory for triplet mining (`BANK_SIZE`, off by default), and a self-attention
  block after the cross-attention (`MIM_SELF_LAYERS`, 0).

### 4.5 Optimisation

Adam, lr 3.5e-4 (bias lr ×2), weight decay 5e-4, 120 epochs, warmup (10 epochs, linear from ×0.01) then ×0.1 at
epochs 40 and 70. Batch 64 with 4 instances per identity (PK sampling), AMP. Augmentations are flip, pad 10 + random
crop, and random erasing with p = 0.5.

### 4.6 Inference

Three retrieval vectors (pre-BNNeck, as in CLIP-ReID), each slot L2-normalised then concatenated and normalised once:

| Row | Vector |
|---|---|
| `clipreid_baseline` | `cat(gap4, g)` |
| `lpim` | `cat(z0, pbar)` |
| `holistic` | `cat(gap4, g, z0, pbar)` |
| `part_lse` | LSE (γ = 5) of per-part cosine distance over mutually visible parts (visibility = `vis_head > 0`) |

Optional test-time only: k-reciprocal re-ranking (k1=50, k2=15, λ=0.3), plain or with the part-LSE distance as
`local_distmat` (`<row>_rr`, `<row>_rr_lse`).

## 5. Evaluation protocol

Standard single-query mAP / CMC on query and gallery, for each row above. Market-1501 is the reference. The CLIP-ReID
baseline reproduces at 89.3 mAP (paper: 89.8). Logged at each evaluation: per-part invisible rate, and the share of
query-gallery pairs with an unmatched or no shared part.

## 6. Planned ablations

1. TAB on/off in stage 1, judged on plain-EOT top-1 and on final stage-2 mAP.
2. Full-pool vs batch negatives; batch-positive t2i vs all-positive.
3. Text erasing p ∈ {0, 0.1, 0.2, 0.3} and text dropout.
4. Holistic vs baseline vs LPIM vs `part_lse`, with and without re-ranking.
5. LSE γ schedule, XBM bank, and LPIM self-attention layers.
6. Cross-dataset (MSMT17, DukeMTMC) with a per-dataset stage-1 checkpoint (asserted at load).

## 7. Known limitations (from code audit)

These are open issues in the current implementation, listed so results are not over-read.

1. **TAB does not reach stage 2.** It shapes `cls_ctx` only. Its benefit is unproven until the on/off ablation
   (§6.1). So far only a 1-epoch smoke run exists.
2. **The holistic vector concatenates `z0` and `pbar`; there is no joint attention pool over global and parts.**
   `alpha` is not masked by visibility, so occluded parts contribute to `pbar`.
3. **Part discrimination has no per-part ID classifier.** It rests on the part triplet and a batch-only SupCon
   (16 identities, with duplicated text columns giving a constant ≈ ln 4 floor per direction).
4. **Visibility differs between train and test.** Ground-truth `present` is used in training and `vis_head` at test.
   Visibility-head accuracy is not logged.
5. **Stage-1 anchors are noisier for head and feet.** The prompts were aligned to frozen GWAP→linear features, which
   lack q/k and positional terms, while stage 2 aligns LPIM tokens to them.
6. **Observed gap.** In the server logs the holistic row (89.1–89.2) matches the baseline (89.2–89.3). `part_lse`
   alone is lower (84.6–85.1). Whether the part branch gives a measurable gain over CLIP-ReID is still open.
7. **Minor.** One TAB shared across slots with no slot embedding. The TAB t2i term ignores
   `T2I_BATCH_POSITIVES=False`. TAB's loss is inactive without full-pool negatives. TAB weights in the bank are
   stale between epoch rebuilds. fp16 cosines at temperature 0.01 under AMP are noisy.
