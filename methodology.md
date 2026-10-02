# Methodology: Part-Prompt CLIP-ReID with Per-Part Matching

Backbone: CLIP RN50 (ModifiedResNet), input 256×128, stride 16 (feature grid 16×8). Datasets: Market-1501 (751 train
identities), DukeMTMC-reID, MSMT17 (`datasets/part_datasets.py`).
Code: `processor/train_part_prompts_stage1.py`, `processor/train_part_prompts_stage2.py`, `deploy/export_stage2_onnx.py`.

## 1. Overview

CLIP-ReID's single identity prompt becomes **K+1 = 6 prompts per identity**: one global slot and five body-part slots
(head, upper-arms/torso, lower-arms/torso, legs, feet). The aim is that two people who differ in a single part (for
example their shoes) are not matched, which a global average embedding cannot guarantee.

- **Stage 1 (prompt learning).** CLIP's encoders are frozen. Only per-(identity, slot) context tokens are learned.
- **Stage 2 (image-side fine-tuning).** Prompts are frozen. The image encoder is fine-tuned with the CLIP-ReID losses
  plus a part branch that gives every part its own 256-d vector, its own ID/triplet/prompt-alignment losses, and a
  visibility-aware matching rule.

## 2. Part supervision: PifPaf masks

1. PifPaf gives 36 confidence fields per image (BPBreID `pifpaf_maskrcnn_filtering`, pre-computed per image).
2. Fields are grouped into five vertical parts (BPBreID `five_v`, identical grouping) plus a background channel
   (threshold 0.5); a softmax with weight 15 gives a soft part distribution.
3. A part is **present** when it wins the argmax at ≥ 1 grid location (BPBreID rule).
4. Stage 2 applies the same flip, pad+crop and random erasing to image and mask; erased/padded regions become background.

## 3. Stage 1: per-part prompt learning (batch negatives only)

**Frozen image side (cached once):** slot 0 = `xproj[0]` (CLIP-ReID's global feature); slot k = `GWAP(x4, mask_k)` through
a frozen `Linear(2048→1024)` equal to CLIP's attention-pool value/output projection.

**Text side:** one frozen CLIP text encoder; templates `A photo of a X X X X person.` and
`A photo of the <part> of a X X X X person.`; learnable `cls_ctx[C, 6, 4, 512]` (4 context tokens per identity and slot).

**Loss per slot** (over batch images whose part is visible), CLIP-ReID's SupConLoss (raw dot product, temperature 1),
both directions, **in-batch negatives only**. The earlier dataset-wide negative pool (all prompts / all images) was
removed: it needed temperature, positive-set and bank-staleness patches and lowered accuracy.

**Cross-part negatives** (`CROSS_PART_NEG`): for a part slot, the same batch identities' prompts of the *other part
slots* are extra negatives for the image (i2t), and the other parts' image embeddings are extra negatives for the prompt
(t2i). The head prompt must prefer head evidence over torso/legs evidence. Part slots only; the global slot is excluded.

**Text Attention Block (TAB), one per slot:** the image is the query, the 77 token states of the slot's prompt are keys
and values, `t_hat = t_eot + W_o·Attn(q, K, V)` with `W_o` zero-initialised (so it starts equal to the plain contrast).
An auxiliary in-batch contrast (weight 0.5) scores image j against the prompt of every batch identity as conditioned
by image j. Because TAB is per slot, part k's cross-attention only reads part k's prompt tokens. TAB is training-only:
stage 2 consumes the plain EOT prompt vectors, so TAB can only help by making `cls_ctx` better under EOT pooling.

**Optimisation:** Adam, lr 5e-4, weight decay 1e-4, 60 epochs, warmup-cosine, batch 64.
**Diagnostics:** per-slot image→text top-1, TAB-conditioned top-1, and a cross-part confusion rate (how often a part
image is closer to its own identity's prompt of another part).

## 4. Stage 2: part-aware fine-tuning

### 4.1 Architecture

- **Baseline branch (CLIP-ReID verbatim):** `gap3 = GAP(x3)`, `gap4 = GAP(x4)`, `g = xproj[0]`, BNNeck ID heads, triplet
  on all three, i2t `CE(g · text_global^T)`. Test feature `cat(gap4, g)`.
- **LPIM (image-side cross-attention from prompts):** six identity-agnostic text queries (`A photo of the <part> of a
  person.`) cross-attend over the x4 grid, K/V/output initialised from CLIP's attention pool. Pure cross-attention (parts
  never mix) gives tokens `z0` (global-semantic) and `z1..z5` (1024-d). The part-k attention map is supervised by the
  PifPaf mask (KL), and a visibility head per part token predicts presence, so no mask is needed at test time.
- **Per-part heads:** `h_k = Linear_k(z_k)`, 256-d, each with its own BNNeck classifier. This is the part's retrieval vector.
- **Text adapters:** `A_k: 1024→256` maps the frozen stage-1 prompts into part k's 256-d space (training only).
- **Self-attended fused vector:** a learnable CLS token attends over `[z0, z1..z5]` (+ slot embedding) for one layer,
  with the keys of invisible parts masked out, then `Linear(1024→1280)`. By default the part tokens enter detached
  (`FUSE_DETACH`) so the fused loss cannot make a part token carry whole-body identity.

### 4.2 Alignment of prompts and part vectors

For each part k and each batch, the symmetric contrast is computed between `h_k` and `A_k(prompts of the batch
identities)`. The positive is (own identity, part k). With `CROSS_PART_NEG`, the same identities' prompts of the other
parts, seen through the same adapter, are negatives. So part 1's vector aligns with part 1's prompts only, in its own
256-d space. Random text erasing (a prompt anchor dropped with p = 0.1) and text-dimension dropout keep the part
vector able to stand on its own when its anchor, or the part itself, is missing.

### 4.3 Where each loss acts

| Loss | Acts on | Notes |
|---|---|---|
| ID, triplet on `gap3/gap4/g`; i2t on `g` | holistic | CLIP-ReID verbatim |
| ID + triplet on fused vector | holistic over parts | triplet on the L2-normalised vector (it is retrieved by cosine); part tokens detached by default |
| per-part ID on `h_k` (`PART_ID_W`) | part | visible parts only; BPBreID keeps this at 0, so it is ablated |
| part triplet on `h_k` | part | batch-hard, visibility-masked mean distance, LSE worst-part ramp (epochs 40→80) |
| per-part individual triplet (`PART_INDIV_TRI_W`) | part | each part mined on its own distance, both images show it |
| part↔prompt contrast (`PART_ALIGN_W`) | part | §4.2 |
| attention KL, visibility BCE | part | PifPaf masks as where-to-look and presence supervision |

Batches can be built around confusable identities per part (`PartHardPKSampler`, `--hard-sampling`): an anchor identity
and slot are drawn, half of the other identities come from the anchor's nearest identities in that slot (Jaccard +
cosine ranking on the model's own prototypes, refreshed every 10 epochs, from epoch 10). No dataset-wide pool or
memory is used (`BANK_SIZE = 0`).

### 4.4 Effect of part visibility on the embedding

Visibility is the PifPaf presence in training and the visibility head's prediction at test. An invisible part's
256-d block is zeroed, its token is masked out of the fused vector's attention, and it is excluded from every part loss
and from matching. Two concatenated vectors are never compared directly: matching is block by block over the parts
visible in both images.

### 4.5 Optimisation

Adam, lr 3.5e-4 (bias ×2), weight decay 5e-4, 120 epochs, warmup then ×0.1 at epochs 40 and 70, PK batches of 64
(16 identities × 4), AMP, flip + pad/crop + random erasing.

## 5. Evaluation

All distances are cosine distances. In the combined rows every distance is first divided by its own mean (mean-ratio),
so the weights compare like with like; the global fallback of the part rows is rescaled to the part scale. Rows (mAP / Rank-1/5/10):

| Row | Definition |
|---|---|
| `clipreid_global` | `cat(gap4, g)` — CLIP-ReID's own feature |
| `parts_selfattn` | the self-attended fused vector |
| `parts_matching` | per-part cosine over parts visible in both images, mean (BPBreID-style); fewer than 2 shared parts → global distance |
| `parts_matching_lse` | the same with a worst-part (log-sum-exp, γ = 5) combination |
| `holistic` | `cat(gap4, g, fused)` as one vector |
| `global+parts`, `global+parts_lse`, `all` | `clipreid_global` + weighted matching rows (+ self-attended row) |
| `part_<name>` | each part alone (where both images show it) |

Per-part invisible rates and the share of query–gallery pairs with unmatched parts are logged, and a results json is
written per evaluation. Optional k-reciprocal re-ranking (`--rerank`) can use the part distance as its local matrix.

## 6. Planned ablations (Market-1501, then Duke)

A. BPBreID-like (`--part-id-w 0 --part-indiv-w 0 --no-cross-part-neg`); B. + part ID; C. + part↔prompt cross-part
contrast; D. + individual triplet; E. + hard-negative sampling; F. fusion detach on/off; G. stage-1 cross-part
negatives and TAB on/off, judged on plain-EOT top-1 and on final stage-2 rows.

## 7. Known limitations

1. **Not yet trained or compared.** The pipeline is implemented and smoke-tested only; no result here shows it
   beating CLIP-ReID. The `clipreid_global` row inside the same run is the reference.
2. **Per-part ID loss goes against BPBreID's finding** that a part is rarely unique to one identity; kept modest and ablated.
3. **Visibility head reliability.** Matching and the fused vector depend on the predicted visibility at test. After a
   very short run the head predicts every part as visible; watch `vis_acc` and the per-part invisible rates.
4. **Prompt adapters can co-adapt** with the part heads, making the alignment weak; ID and triplet losses also
   constrain the heads, and the in-batch alignment top-1 is logged.
5. **Part tokens are tied to PifPaf quality.** Head and feet are the weakest and most often invisible parts, and the
   feet are the part that matters for the shoe case; the per-part rows show how good each one is.
6. **Checkpoints from the previous architecture are not loadable** (stage 1 and stage 2 must be re-run).
7. **Part embeddings are not spatially pure.** The part masks barely overlap (IoU <= 0.22), but the frozen part
   embeddings of adjacent parts are very similar (cosine 0.81 for upper vs lower torso, legs vs lower torso), because
   the RN50 features have a wide receptive field. Cross-part negatives between adjacent parts are therefore hard, and
   may need to be restricted or ablated (`--no-cross-part-neg`).
8. **Evaluation protocol.** The train/query/gallery paths are disjoint on Market-1501, DukeMTMC-reID and MSMT17, no
   mask or label is read at test time, and the stage-1 prompts exist only for training identities. But "best so far"
   is selected on the test set every 2 epochs; report the last epoch (or a held-out split) for a clean number.
   Stage-1 top-1 is measured on training identities, so it shows fit, not generalisation.
9. **Train/test visibility mismatch.** Training feeds the PifPaf presence into the fused vector; testing uses the
   predicted visibility, which is only as good as the visibility head (`vis_acc` is logged).
