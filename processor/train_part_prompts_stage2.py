"""CLIP-ReID stage 2 (image-side fine-tuning) with learnt per-part prompts on PartAwareModifiedResNet (RN50).

Run from the repo root:
  python processor/train_part_prompts_stage2.py --stage1-ckpt work_dirs/market1501/part_prompts_stage1/RN50_part_prompts_stage1_60.pth

What stage 2 does here
* Baseline branch = CLIP-ReID RN50 stage 2, verbatim: gap3 = GAP(x3), gap4 = GAP(x4) with a BNNeck ID head,
  g = xproj[0] with a BNNeck ID head; triplet on all three; i2t = CE(g @ text_global.T); test feature
  cat(gap4, g) = the `clipreid_global` row.
* Part branch. LanguageGuidedPartInteraction (LPIM, after PromptSG CVPR'24): K+1 identity-agnostic text queries
  ("A photo of a person", "A photo of the <part> of a person") cross-attend over the x4 locations (pure
  cross-attention: parts never mix) -> tokens z0 (global-semantic) and z1..zK. Every part owns
    * a 256-d head h_k = Linear_k(z_k) (the part's retrieval vector) with its own BNNeck ID classifier,
    * a text adapter A_k: 1024 -> 256 that maps the frozen stage-1 prompts into part k's space, so the prompt of
      part k is contrasted with exactly that part's 256-d vector (own part = positive, other parts = negatives).
  A learnable CLS token self-attends over [z0, visible parts] (key padding mask = part visibility) -> fused vector
  [FUSED_DIM] (`parts_selfattn`). Part visibility is the PifPaf presence in training and the visibility head's
  prediction at test, so no mask is needed at test time.
* Losses (each is holistic, part-based or both - see Stage2Loss): CLIP-ReID ID/triplet/i2t on the baseline branch;
  ID + triplet on the fused vector; per-part ID; part triplet (visibility-masked mean -> LSE worst-part ramp) and a
  per-part individual triplet on h_k; part<->prompt contrast with cross-part negatives; attention KL; visibility BCE.
* Text side (frozen): the stage-1 prompts are encoded once into text_all [C, K+1, 1024]; the K+1 semantic queries are
  encoded once from fixed templates. No dataset-wide pool or memory is used (BANK_SIZE = 0).
* Evaluation (every EVAL_PERIOD epochs), mAP / R1 / R5 / R10 for: clipreid_global (cat(gap4, g)), parts_selfattn
  (fused), parts_matching (per-part cosine, masked mean over mutually visible parts; global fallback below
  MIN_SHARED_PARTS), parts_matching_lse (worst-part), holistic, the combinations global+parts / global+parts_lse / all,
  and each part alone. Per-part invisible rate and unmatched-pair statistics are logged and a results json is written.
* --rerank adds two test-time rows (no training change): <row>_rr = k-reciprocal re-ranking (Zhong et al.
  CVPR'17, utils/reranking.py) on RERANK_FEATURE, and <row>_rr_lse = the same with the all-pairs part-LSE
  distance passed as `local_distmat`. See rerank_rows / part_lse_all_pairs.
* Optimiser / schedule / AMP = SOLVER.STAGE2 of configs/person/cnn_clipreid.yml, 256x128 as in the recipe.
"""
import os
import sys
import math
import time
import random
import json
import resource
import argparse
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

from processor.train_part_prompts_stage1 import (PartPromptLearner, encode_text, supcon_from_logits, pifpaf_to_masks,
                                                  CONTRAST_TEMP, PART_NAMES, K, S, SLOT_NAMES)
from model.make_model_clipreid import load_clip_to_cpu, TextEncoder, weights_init_kaiming, weights_init_classifier
from datasets.part_datasets import DATASETS, MASK_SUFFIX, build_dataset, mask_path, resolve_masks
from datasets.sampler import PartHardPKSampler
from datasets.bases import ImageDataset, read_image
from datasets.make_dataloader_clipreid import val_collate_fn
from loss.softmax_loss import CrossEntropyLabelSmooth
from loss.triplet_loss import euclidean_dist
from solver.lr_scheduler import WarmupMultiStepLR
from utils.metrics import eval_func
from utils.reranking import re_ranking
from utils.logger import setup_logger
from utils.meter import AverageMeter

# ----------------------------------------------------------------------------- knobs
DATA_ROOT = '../../datasets'
DATASET = 'market1501'             # market1501 | msmt17 | dukemtmc (datasets/part_datasets.py); --dataset overrides,
                                   # and it must match the dataset the stage-1 prompts were learnt on. Train-split
                                   # PifPaf masks are resolved from the name (see stage 1); query/gallery need none.
MASKS_VARIANT = 'pifpaf_maskrcnn_filtering'    # pre-saved BPBreID mask set to read; 'pifpaf' is the unfiltered one
MASKS_DIR = None                   # None = <dataset_dir>/masks/<MASKS_VARIANT>; --masks-dir points elsewhere
OUTPUT_DIR = './work_dirs/{dataset}/part_prompts_stage2'
STAGE1_CKPT = './work_dirs/{dataset}/part_prompts_stage1/RN50_part_prompts_stage1_60.pth'

BACKBONE = 'RN50'
H, W = 256, 128                    # CLIP-ReID RN50 recipe (cnn_clipreid.yml); must match the stage-1 checkpoint
STRIDE = 16
PIXEL_MEAN, PIXEL_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
PADDING, FLIP_PROB, RE_PROB = 10, 0.5, 0.5

IMS_PER_BATCH = 64                 # SOLVER.STAGE2 of configs/person/cnn_clipreid.yml (32 fits an 8 GB GPU)
NUM_INSTANCE = 4
TEST_BATCH = 128
NUM_WORKERS = 8
MAX_EPOCHS = 120
BASE_LR = 3.5e-4
WEIGHT_DECAY, WEIGHT_DECAY_BIAS, BIAS_LR_FACTOR = 5e-4, 5e-4, 2
STEPS, GAMMA = (40, 70), 0.1
WARMUP_FACTOR, WARMUP_ITERS, WARMUP_METHOD = 0.01, 10, 'linear'
USE_AMP = True

ID_W, TRI_W, I2T_W = 1.0, 1.0, 1.0            # CLIP-ReID stage-2 weights on the baseline branch (GAP(x4), GAP(x3), xproj)
FUSED_ID_W, FUSED_TRI_W = 1.0, 1.0            # ID + triplet on the self-attended fused vector (holistic over parts)
PART_ID_W = 0.3                               # ID CE (own BNNeck classifier per part) on h_1..h_K, visible parts only. BPBreID
                                              # (GiLt) keeps this at 0 - a part such as black trousers is not unique to one
                                              # identity - so it is modest and ablated ({0, 0.3, 0.5}).
PART_TRI_W = 1.0                              # part triplet on h_k: visibility-masked mean distance, LSE worst-part ramp
PART_INDIV_TRI_W = 0.5                        # per-part individual triplet: every part mined on its own distance, both visible
PART_ALIGN_W = 0.5                            # part<->prompt contrast: h_k vs the stage-1 prompts through the adapter A_k
ATTN_W, VIS_W = 1.0, 0.1                      # attention-map KL to the PifPaf masks; part-presence BCE for the visibility head
PART_DIM = 256                                # per-part retrieval vector (BPBreID's part dimension)
FUSED_DIM = 1280                              # self-attended fused vector; = 5 x 256 only to weigh equally with the part block
FUSE_LAYERS = 1                               # self-attention layers of the fusion (CLS + [z0, parts])
FUSED_TRI_NORMALIZE = True                    # triplet on the L2-normalised fused vector: it is retrieved by cosine, and its raw norm
                                              # (~20-30) would make the 0.3 margin ~1% of the distances. The CLIP-ReID branch
                                              # (gap3/gap4/g) stays on raw features, verbatim.
FUSE_DETACH = True                            # part tokens enter the fusion detached, so the fused loss cannot make a part token
                                              # carry whole-body identity (z0 and the fusion itself still train). Ablate False.
CROSS_PART_NEG = True                         # part<->prompt contrast: the batch identities' prompts of the OTHER parts are
                                              # negatives for part k's vector (part 1 <-> ctx 1 only)
LPIM_LEARN_QUERY = False                      # zero-init learnable offset on each LPIM part query (optional)
TEXT_ERASE_PROB = 0.1              # random text erasing: per (sample, slot), drop that slot's identity prompt from the
                                   # part<->prompt contrast, and for slot 0 drop the sample from i2t. The text side of Random
                                   # Erasing (RE_PROB on images): the image slot must stand on its own when its text
                                   # anchor is missing, which is the occluded-part case. 0 = off; at 0.1, 0.9^6 = 53%
                                   # of samples keep all six anchors. Ablate {0, 0.1, 0.2, 0.3}.
TEXT_DROPOUT = 0.1                 # Bernoulli mask over the 1024 text dimensions, rescaled by 1/(1-p), fresh every
                                   # step and shared across identities within a slot (so every comparison in that slot
                                   # stays in one sub-space and logits remain comparable across classes). Applies to
                                   # both text consumers, i2t and the part<->prompt contrast. The LPIM semantic queries are never touched:
                                   # they are part of the test-time path. 0 = off.
HARD_SAMPLING = False              # per-part hard-negative PK batches (PartHardPKSampler): each batch is built around one
                                   # anchor identity and one slot, HARD_FRAC of the other identities being that anchor's
                                   # nearest identities in that slot. Off = RandomIdentitySampler.
HARD_FRAC = 0.5                    # share of the P-1 other identities drawn from the anchor's neighbours
NBR_K = 15                         # neighbours kept per (slot, identity); identity level (751 on Market), not image level
NBR_JACCARD = True                 # rank neighbours by (1-NBR_LAMBDA)*Jaccard(top-NBR_K1 sets) + NBR_LAMBDA*cosine distance
NBR_K1, NBR_LAMBDA = 30, 0.3
HARD_START_EPOCH = 10              # random PK until then (features and parts settle first)
NBR_REFRESH = 10                   # epochs between neighbour-table rebuilds from the model's own per-identity prototypes (g, h_k)
                                   # (0 = keep the table built from the stage-1 prompts text_all)
MARGIN = 0.3
LSE_GAMMA = 5.0                    # soft-max sharpness over parts (-> max distance as gamma grows)
LSE_GAMMA_EPOCHS = (40, 80)        # part triplet uses the visibility-weighted mean until epoch 40, then gamma ramps
                                   # linearly to LSE_GAMMA by epoch 80 (chasing the worst part only once parts are trained)
MIM_SELF_LAYERS = 0                # self-attention blocks inside LPIM. Keep 0: they would mix the part tokens before the
                                   # per-part heads; the self-attended vector is built separately (PartFusion).

BANK_SIZE = 0                      # cross-batch memory for triplet mining; 0 = batch-only (CLIP-ReID / BPBreID
                                   # baseline behaviour), 8192 = XBM ablation
BANK_START_EPOCH = 5               # epochs of batch-only mining before the bank is used (features settle first)

EVAL_SLOTWISE_NORM = True          # holistic vector: L2-normalise each slot before concatenating (see
                                   # concat_feature). False restores the CLIP-ReID convention of a single
                                   # normalisation over the concatenation. Eval-only, no retraining needed.
EVAL_W_SELF, EVAL_W_PARTS = 1.0, 1.0   # weights of the self-attended / part-matching distances in the combined rows
MIN_SHARED_PARTS = 2               # a pair with fewer mutually visible parts falls back to the global distance
EVAL_SOFT_VIS = False              # weight parts by their visibility probability (0 below 0.5) instead of a 0/1 mask
EVAL_PART_ROWS = True              # also report every part alone (where both images show it)
EVAL_CHUNK = 2048
GRID = ((H - 16) // STRIDE + 1, (W - 16) // STRIDE + 1)      # x4 feature grid (attention_targets / training-time visibility)

RERANK = False                     # k-reciprocal re-ranking (Zhong et al. CVPR'17, utils/reranking.py). Test-time
                                   # only: it changes no gradient and no checkpoint, so --rerank can be added to
                                   # any --eval-only run over a finished stage-2 model.
RERANK_FEATURE = 'holistic'        # eval row to re-rank: 'holistic' | 'parts_selfattn' | 'clipreid_global'
RERANK_K1, RERANK_K2, RERANK_LAMBDA = 50, 15, 0.3     # utils/metrics.py:126 (the CVPR'17 paper uses 20, 6, 0.3)
RERANK_LOCAL_W = 1.0               # weight of the part-LSE distance inside the k-reciprocal neighbourhood, after
                                   # mean-ratio scaling (see rerank_rows). 0 reproduces the plain row.
RERANK_ONLY_LOCAL = False          # extra row: k-reciprocal on the part-LSE distance alone (free once it is computed)
RERANK_EVERY_EVAL = False          # False = only the last epoch / --eval-only. Measured on Market (N = 19,281):
                                   # ~2 min per re_ranking call, ~11 GB peak RSS and 2 GB VRAM for the block
RERANK_CHUNK = 512                 # gallery chunk of the all-pairs part distance (EVAL_CHUNK needs ~0.8 GB of VRAM)
RERANK_DEVICE = None               # None = DEVICE; 'cpu' when the [N,N] matmul does not fit next to the model

CHECKPOINT_PERIOD, EVAL_PERIOD, LOG_PERIOD = 20, 2, 50
SEED = 1234
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ----------------------------------------------------------------------------- data
class PartImageDataset(Dataset):
    """Training samples (img, mask [K+1,H,W], pid, camid) with joint image/mask augmentation.

    Mirrors the stage-2 train transforms of make_dataloader_clipreid.py (resize, random flip, pad + random
    crop, normalise, random erasing) and applies the spatial ones to the PifPaf masks as well; padded and
    erased regions become background in the mask (BPBreID mask_fill_value=0 -> argmax = background).
    """

    def __init__(self, dataset, dataset_dir, masks_dir, variant=None):
        self.dataset = dataset
        self.dataset_dir, self.masks_dir, self.variant = dataset_dir, masks_dir, variant or MASKS_VARIANT
        self.normalize = T.Normalize(PIXEL_MEAN, PIXEL_STD)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _background(mask, region):
        mask[:, region[0]:region[1], region[2]:region[3]] = 0
        mask[0, region[0]:region[1], region[2]:region[3]] = 1

    def _random_erase(self, img, mask):
        area = img.shape[1] * img.shape[2]
        for _ in range(10):
            target_area = random.uniform(0.02, 1 / 3) * area
            ratio = math.exp(random.uniform(math.log(0.3), math.log(1 / 0.3)))
            h = int(round(math.sqrt(target_area * ratio)))
            w = int(round(math.sqrt(target_area / ratio)))
            if w < img.shape[2] and h < img.shape[1]:
                top = random.randint(0, img.shape[1] - h)
                left = random.randint(0, img.shape[2] - w)
                img[:, top:top + h, left:left + w] = torch.randn(3, h, w)
                self._background(mask, (top, top + h, left, left + w))
                return

    def __getitem__(self, index):
        img_path, pid, camid, _ = self.dataset[index]
        img = TF.to_tensor(TF.resize(read_image(img_path), [H, W], interpolation=T.InterpolationMode.BICUBIC))
        mask = pifpaf_to_masks(np.load(mask_path(img_path, self.dataset_dir, self.masks_dir, self.variant)))
        mask = F.interpolate(mask[None], (H, W), mode='bilinear', align_corners=True)[0]
        if random.random() < FLIP_PROB:
            img, mask = img.flip(-1), mask.flip(-1)
        img = F.pad(img, (PADDING,) * 4)
        mask = F.pad(mask, (PADDING,) * 4)
        mask[0] = torch.where(mask.sum(0) == 0, torch.ones_like(mask[0]), mask[0])
        top, left = random.randint(0, 2 * PADDING), random.randint(0, 2 * PADDING)
        img = img[:, top:top + H, left:left + W]
        mask = mask[:, top:top + H, left:left + W]
        img = self.normalize(img)
        if random.random() < RE_PROB:
            self._random_erase(img, mask)
        return img, mask, pid, camid


def make_loaders(dataset, dataset_dir, masks_dir, batch):
    train_set = PartImageDataset(dataset.train, dataset_dir, masks_dir, MASKS_VARIANT)
    train_loader = DataLoader(train_set, batch_size=batch,
                              sampler=PartHardPKSampler(dataset.train, batch, NUM_INSTANCE, HARD_FRAC),
                              num_workers=NUM_WORKERS, drop_last=True)
    val_transforms = T.Compose([T.Resize((H, W)), T.ToTensor(), T.Normalize(PIXEL_MEAN, PIXEL_STD)])
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(val_set, batch_size=TEST_BATCH, shuffle=False, num_workers=NUM_WORKERS, collate_fn=val_collate_fn)
    stats_loader = DataLoader(ImageDataset(dataset.train, val_transforms), batch_size=TEST_BATCH, shuffle=False,
                              num_workers=NUM_WORKERS, collate_fn=val_collate_fn)
    return train_loader, val_loader, stats_loader


# ----------------------------------------------------------------------------- model
class BNNeckHead(nn.Module):
    """CLIP-ReID BNNeck: BatchNorm1d (frozen bias) + bias-free linear classifier."""

    def __init__(self, dim, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.bn.bias.requires_grad_(False)
        self.bn.apply(weights_init_kaiming)
        self.fc = nn.Linear(dim, num_classes, bias=False)
        self.fc.apply(weights_init_classifier)

    def forward(self, x):
        feat = self.bn(x)
        return feat, self.fc(feat)


class LanguageGuidedPartInteraction(nn.Module):
    """Language-guided Part Interaction Module (LPIM): K+1 semantic text queries attend over the x4 locations.

    PromptSG (CVPR'24) shows that CLIP-ReID's remaining gap is closed by a multimodal interaction module whose
    output *is* the ReID feature: cross-attention with the prompt embedding as query and the patch tokens as
    key/value, trained with ID + triplet. Here that idea is run with K+1 queries - "A photo of a person" and
    "A photo of the <part> of a person" - so the module yields one global-semantic token and K part tokens.
    This is the image-side cross-attention *from prompts*: the part-k query reads the image only where part k is.
    * K/V/output projections start from CLIP's own attention pool (k_proj, v_proj, c_proj, positional
      embedding), so the queries start from CLIP's pooled feature and then learn *where* to look. Queries are
      identity-agnostic and identical at train and test, so no identity prompt is needed at inference.
    * Pure cross-attention: queries never attend to each other (MIM_SELF_LAYERS optional blocks follow it).
    * The head-averaged attention map of each part query is supervised by its PifPaf mask (KL), which turns the
      masks into where-to-look supervision instead of hard pooling weights; a visibility head on each part
      token predicts part presence so visibility is available at test time without masks.
    * LPIM_LEARN_QUERY adds a zero-initialised learnable offset to each query (a per-part learnt prompt).
    forward(x4) -> dict(z [N,K+1,D], attn [N,K+1,HW], vis_logit [N,K]).
    """

    def __init__(self, attnpool, text_queries, num_self_layers=None):
        super().__init__()
        num_self_layers = MIM_SELF_LAYERS if num_self_layers is None else num_self_layers
        C, D = attnpool.k_proj.in_features, attnpool.c_proj.out_features
        self.num_heads = attnpool.num_heads
        self.register_buffer('text_queries', text_queries.detach().clone())
        self.query_delta = nn.Parameter(torch.zeros_like(self.text_queries)) if LPIM_LEARN_QUERY else None
        self.pos_embed = nn.Parameter(attnpool.positional_embedding[1:].detach().clone())
        self.q_proj = nn.Linear(D, C)
        self.k_proj = nn.Linear(C, C)
        self.v_proj = nn.Linear(C, C)
        self.c_proj = nn.Linear(C, D)
        with torch.no_grad():
            for mine, theirs in [(self.k_proj, attnpool.k_proj), (self.v_proj, attnpool.v_proj), (self.c_proj, attnpool.c_proj)]:
                mine.weight.copy_(theirs.weight)
                mine.bias.copy_(theirs.bias)
            nn.init.normal_(self.q_proj.weight, std=C ** -0.5)
            nn.init.zeros_(self.q_proj.bias)
        self.norm = nn.LayerNorm(D)
        self.ffn = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        nn.init.zeros_(self.ffn[2].weight)
        nn.init.zeros_(self.ffn[2].bias)
        self.self_layers = nn.ModuleList([nn.TransformerEncoderLayer(D, 8, 4 * D, dropout=0.0, batch_first=True, norm_first=True)
                                          for _ in range(num_self_layers)])
        self.vis_head = nn.Linear(D, 1)

    def queries(self):
        return self.text_queries if self.query_delta is None else self.text_queries + self.query_delta

    def forward(self, x4):
        N, C, Hf, Wf = x4.shape
        HW, h, d = Hf * Wf, self.num_heads, C // self.num_heads
        tokens = x4.flatten(2).transpose(1, 2) + self.pos_embed[None].to(x4.dtype)
        Sq = self.text_queries.shape[0]
        Q = self.q_proj(self.queries().to(x4.dtype)).view(Sq, h, d).transpose(0, 1)
        Kt = self.k_proj(tokens).view(N, HW, h, d).permute(0, 2, 1, 3)
        V = self.v_proj(tokens).view(N, HW, h, d).permute(0, 2, 1, 3)
        attn = torch.softmax(torch.einsum('hsd,nhld->nhsl', Q, Kt).float() / math.sqrt(d), dim=-1)
        out = torch.einsum('nhsl,nhld->nhsd', attn.to(V.dtype), V).permute(0, 2, 1, 3).reshape(N, Sq, C)
        z = self.c_proj(out)
        z = z + self.ffn(self.norm(z))
        for layer in self.self_layers:
            z = layer(z)
        return dict(z=z, attn=attn.mean(1), vis_logit=self.vis_head(z[:, 1:]).squeeze(-1).float())


class PartFusion(nn.Module):
    """Self-attended fused vector: a learnable CLS token attends over [z0, z1..zK] (+ a learnt slot embedding), keys of
    invisible parts masked out, so an occluded part never enters the vector. 1 layer by default; output
    Linear(D -> FUSED_DIM). forward(z [N,K+1,D], vis [N,K] bool) -> [N, FUSED_DIM]."""

    def __init__(self, dim, out_dim=None, layers=None, heads=8):
        super().__init__()
        out_dim, layers = out_dim or FUSED_DIM, FUSE_LAYERS if layers is None else layers      # read at build time, not def time
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.slot_embed = nn.Parameter(torch.randn(1, K + 1, dim) * 0.02)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, batch_first=True, norm_first=True)
                                     for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, out_dim)

    def forward(self, z, vis):
        N = z.shape[0]
        x = torch.cat([self.cls.expand(N, -1, -1).to(z.dtype), z + self.slot_embed.to(z.dtype)], dim=1)
        keep = torch.cat([torch.ones(N, 2, dtype=torch.bool, device=z.device), vis], dim=1)      # CLS and z0 always visible
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=~keep)
        return self.out(self.norm(x[:, 0]))


class PartCLIPReID(nn.Module):
    """Trainable image side = the CLIP-ReID RN50 branch, verbatim, plus the LPIM part branch.

    Baseline branch (make_model_clipreid.py build_transformer, RN50): gap3 = GAP(x3) [1024], gap4 = GAP(x4)
    [2048] with a BNNeck ID head, g = xproj[0] [1024] with a BNNeck ID head and the i2t loss; triplet on all
    three; test feature cat(gap4, g) - the `clipreid_global` row.
    Part branch: LPIM tokens z0..zK; per-part 256-d heads h_k = part_proj[k](z_k) (own BNNeck classifier id_parts[k]);
    the fused self-attended vector (PartFusion, own BNNeck id_fused); per-part text adapters text_adapters[k] that map
    the stage-1 prompts into part k's 256-d space (training only). Visibility: `vis` [N,K] bool when given (ground-truth
    presence in training), else the visibility head's own prediction (test). It gates the fusion; the part heads
    themselves are visibility-free and the invisible blocks are zeroed / masked by whoever consumes them.
    forward(x, vis=None) -> dict(gap3, gap4, g, z, z0, zparts, h, fused, attn, vis_logit, grid[, score_gap4,
    score_g, score_fused, score_parts [N,K,C]]).
    """

    def __init__(self, visual, num_classes, text_queries):
        super().__init__()
        self.visual = visual
        self.lpim = LanguageGuidedPartInteraction(visual.attnpool, text_queries)
        D = visual.attnpool.c_proj.out_features
        self.id_gap4 = BNNeckHead(visual.attnpool.v_proj.in_features, num_classes)
        self.id_global = BNNeckHead(D, num_classes)
        self.part_proj = nn.ModuleList([nn.Linear(D, PART_DIM) for _ in range(K)])
        self.id_parts = nn.ModuleList([BNNeckHead(PART_DIM, num_classes) for _ in range(K)])
        self.fusion = PartFusion(D)
        self.id_fused = BNNeckHead(FUSED_DIM, num_classes)
        self.text_adapters = nn.ModuleList([nn.Linear(D, PART_DIM, bias=False) for _ in range(K)])

    def forward(self, x, vis=None):
        x3, x4, xproj = self.visual(x)
        gap3, gap4, g = x3.mean((2, 3)), x4.mean((2, 3)), xproj[0]
        out = self.lpim(x4)
        z = out['z']
        h = torch.stack([proj(z[:, 1 + k]) for k, proj in enumerate(self.part_proj)], dim=1)       # [N,K,PART_DIM]
        vis_used = (out['vis_logit'] > 0) if vis is None else vis
        z_fuse = torch.cat([z[:, :1], z[:, 1:].detach()], dim=1) if FUSE_DETACH else z
        fused = self.fusion(z_fuse, vis_used)
        res = dict(gap3=gap3, gap4=gap4, g=g, z=z, z0=z[:, 0], zparts=z[:, 1:], h=h, fused=fused,
                   attn=out['attn'], vis_logit=out['vis_logit'], grid=tuple(x4.shape[2:]))
        if self.training:
            res['score_gap4'] = self.id_gap4(gap4)[1]
            res['score_g'] = self.id_global(g)[1]
            res['score_fused'] = self.id_fused(fused)[1]
            res['score_parts'] = torch.stack([head(h[:, k])[1] for k, head in enumerate(self.id_parts)], dim=1)
        return res


# ----------------------------------------------------------------------------- part distances / losses
def lse_combine(d, M, gamma):
    """Combine per-part distances with a log-sum-exp soft-max over mutually visible parts.
    d, M: [K, A, B] distances and mutual-visibility (0/1, or soft weights in [0,1]). Returns D [A,B] = (1/gamma) ln sum_k w_k e^{gamma d_k}
    with w_k = M_k / sum_k M_k, valid [A,B] (>= 1 shared part; invalid entries are -1) and the number of
    unmatched parts per pair [A,B]. gamma -> 0 is the visibility-weighted mean (BPBreID)."""
    n_shared = M.sum(0)
    valid = n_shared > 0
    w = M / n_shared.clamp(min=1e-6)[None]
    if gamma < 1e-3:
        D = (w * d).sum(0)
    else:
        logits = gamma * d + torch.log(w.clamp(min=1e-12))
        logits = logits.masked_fill(M == 0, float('-inf'))
        D = torch.logsumexp(logits, dim=0) / gamma
    D = torch.where(valid, D, torch.full_like(D, -1.0))
    return D, valid, (M.shape[0] - n_shared)


def part_pairwise_distances(pa, va, pb, vb):
    """pa [A,K,D], pb [B,K,D] (L2-normalised inside), va/vb [A,K]/[B,K] bool (or soft weights) -> d, M [K,A,B]."""
    d = 1 - torch.einsum('ikd,jkd->kij', F.normalize(pa.float(), dim=-1), F.normalize(pb.float(), dim=-1))
    if va.dtype == torch.bool and vb.dtype == torch.bool:
        M = (va.t()[:, :, None] & vb.t()[:, None, :]).to(d.dtype)
    else:
        M = torch.minimum(va.t()[:, :, None].float(), vb.t()[:, None, :].float())
    return d, M


class FeatureBank:
    """Cross-batch memory of detached embeddings (XBM, Wang & al. CVPR20).

    Triplet mining inside one PK batch only sees 64 samples / 16 identities. The bank keeps the most recent
    BANK_SIZE embeddings (fused vector, part heads h_k, visibility, label), so anchors are mined against the whole dataset: with
    8192 slots and 12936 training images, a batch is compared against ~2/3 of Market-1501. Bank entries are
    detached (gradient flows only through the anchors), which is what makes the large pool affordable.
    """

    def __init__(self, size, dim, parts, device, part_dim=None):
        self.size = size
        self.g = torch.zeros(size, dim, device=device)
        self.p = torch.zeros(size, parts, part_dim or dim, device=device)
        self.vis = torch.zeros(size, parts, dtype=torch.bool, device=device)
        self.labels = torch.zeros(size, dtype=torch.long, device=device)
        self.ptr, self.filled = 0, 0

    @torch.no_grad()
    def add(self, g, p, vis, labels):
        n = g.shape[0]
        idx = (torch.arange(n, device=g.device) + self.ptr) % self.size
        self.g[idx], self.p[idx], self.vis[idx], self.labels[idx] = g.detach().float(), p.detach().float(), vis, labels
        self.ptr = int((self.ptr + n) % self.size)
        self.filled = min(self.filled + n, self.size)

    def get(self):
        n = self.filled
        return self.g[:n], self.p[:n], self.vis[:n], self.labels[:n]


def batch_hard(D, target, col_labels, valid, self_cols, ranking_loss):
    """Batch-hard mining over arbitrary columns (batch + bank). `valid` marks comparable pairs,
    `self_cols` marks the anchor's own column. Returns the margin ranking loss over usable anchors."""
    same = target[:, None] == col_labels[None, :]
    pos_mask = same & valid & ~self_cols
    neg_mask = (~same) & valid
    dist_ap = torch.where(pos_mask, D, torch.full_like(D, float('-inf'))).max(1)[0]
    dist_an = torch.where(neg_mask, D, torch.full_like(D, float('inf'))).min(1)[0]
    ok = pos_mask.any(1) & neg_mask.any(1)
    if not ok.any():
        return D.sum() * 0
    return ranking_loss(dist_an[ok], dist_ap[ok], torch.ones_like(dist_an[ok]))


def columns_with_bank(batch_tensors, bank_tensors, target):
    """Column pool of a distance matrix = batch entries (with gradient) followed by detached bank entries.
    `bank_tensors` ends with the bank labels. Returns the pooled tensors, the pooled labels and the mask of
    each anchor's own column."""
    n = target.shape[0]
    if bank_tensors is None:
        return batch_tensors, target, torch.eye(n, dtype=torch.bool, device=target.device)
    cols = tuple(torch.cat([b, k], dim=0) for b, k in zip(batch_tensors, bank_tensors[:-1]))
    col_labels = torch.cat([target, bank_tensors[-1]])
    self_cols = F.pad(torch.eye(n, dtype=torch.bool, device=target.device), (0, col_labels.shape[0] - n))
    return cols, col_labels, self_cols


class GlobalTripletLoss(nn.Module):
    """CLIP-ReID global triplet, mined against the batch plus the cross-batch memory."""

    def __init__(self, margin=MARGIN, normalize=False):
        super().__init__()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.normalize = normalize

    def forward(self, g, labels, bank=None):
        g = g.float()
        cols, col_labels, self_cols = columns_with_bank((g,), bank, labels)
        if self.normalize:                      # anchors, batch columns and memory entries all on the unit sphere
            g, cols = F.normalize(g, dim=-1), (F.normalize(cols[0], dim=-1),)
        D = euclidean_dist(g, cols[0])
        valid = torch.ones_like(D, dtype=torch.bool)
        return batch_hard(D, labels, col_labels, valid, self_cols, self.ranking_loss)


class PartLSETripletLoss(nn.Module):
    """Batch-hard triplet on the LSE-combined part distance (BPBreID part_averaged_triplet_loss with the
    mean replaced by the soft-max), mined against the batch plus the cross-batch memory; pairs without a
    shared visible part are excluded from mining."""

    def __init__(self, margin=MARGIN, gamma=LSE_GAMMA):
        super().__init__()
        self.gamma = gamma
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, p, vis, labels, bank=None):
        p = p.float()
        cols, col_labels, self_cols = columns_with_bank((p, vis), bank, labels)
        d, M = part_pairwise_distances(p, vis, cols[0], cols[1])
        D, valid, _ = lse_combine(d, M, self.gamma)
        return batch_hard(D, labels, col_labels, valid, self_cols, self.ranking_loss)


class PartIndividualTripletLoss(nn.Module):
    """Batch-hard triplet per part on that part's own cosine distance (BPBreID's part_individual_triplet_loss), mined
    against the batch plus the cross-batch memory, over pairs where both images show the part. Unlike the part-averaged
    triplet, a wrong part cannot be hidden behind the other parts: the shoes of two otherwise identical people must
    differ in the feet vector on their own."""

    def __init__(self, margin=MARGIN):
        super().__init__()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, p, vis, labels, bank=None):
        p = p.float()
        cols, col_labels, self_cols = columns_with_bank((p, vis), bank, labels)
        d, M = part_pairwise_distances(p, vis, cols[0], cols[1])
        losses = [batch_hard(d[k], labels, col_labels, M[k] > 0, self_cols, self.ranking_loss) for k in range(d.shape[0])]
        return sum(losses) / len(losses)


def lse_gamma_at(epoch):
    """0 (weighted mean) until LSE_GAMMA_EPOCHS[0], then linear to LSE_GAMMA at LSE_GAMMA_EPOCHS[1]."""
    e0, e1 = LSE_GAMMA_EPOCHS
    if epoch <= e0:
        return 0.0
    return LSE_GAMMA * min(1.0, (epoch - e0) / max(e1 - e0, 1))


def build_neighbours(feats, k=NBR_K, jaccard=None, k1=None, lam=None):
    """Per-slot confusable identities. feats = [C,S,D] or a list of S tensors [C,D_s] (slots may differ in width)
    -> nbr [S,C,k] int64 (numpy), nearest first, self excluded.
    Distance per slot = (1-lam)*Jaccard(top-k1 sets) + lam*cosine distance (cosine only if jaccard is off)."""
    jaccard, k1, lam = NBR_JACCARD if jaccard is None else jaccard, k1 or NBR_K1, NBR_LAMBDA if lam is None else lam
    slots = list(feats.unbind(1)) if torch.is_tensor(feats) else list(feats)
    C = slots[0].shape[0]
    k, k1 = min(k, C - 1), min(k1, C - 1)
    out = []
    for x in slots:
        f = F.normalize(x.float(), dim=-1)
        dist = 1 - f @ f.t()
        if jaccard:
            member = torch.zeros_like(dist).scatter_(1, dist.topk(k1 + 1, largest=False).indices, 1.0)
            inter = member @ member.t()
            union = member.sum(1)[:, None] + member.sum(1)[None] - inter
            dist = (1 - lam) * (1 - inter / union) + lam * dist
        dist.fill_diagonal_(float('inf'))
        out.append(dist.topk(k, largest=False).indices.cpu())
    return torch.stack(out).numpy()


@torch.no_grad()
def identity_slot_means(model, loader, num_classes):
    """The model's own identity prototypes over the (un-augmented) train set, in the spaces matching actually uses:
    slot 0 = mean g (CLIP global), slot k = mean h_k over the images where part k is predicted visible. Returns a list of
    S tensors [C, D_s] and the fraction of an identity's images in which each part is predicted visible, [C, S]."""
    model.eval()
    sums, seen, vis = None, torch.zeros(num_classes, device=DEVICE), None
    for img, pid, _, _, _, _ in loader:
        res = model(img.to(DEVICE))
        pid = torch.as_tensor(np.asarray(pid), device=DEVICE).long()
        v = res['vis_logit'] > 0
        slot_feats = [res['g'].float()] + [res['h'][:, k].float() * v[:, k:k + 1] for k in range(K)]
        if sums is None:
            sums = [torch.zeros(num_classes, x.shape[1], device=DEVICE) for x in slot_feats]
            vis = torch.zeros(num_classes, S, device=DEVICE)
        for acc, x in zip(sums, slot_feats):
            acc.index_add_(0, pid, x)
        vis.index_add_(0, pid, torch.cat([torch.ones_like(v[:, :1]), v], dim=1).float())
        seen.index_add_(0, pid, torch.ones_like(pid, dtype=torch.float))
    counts = vis.clamp(min=1)                                    # per (identity, slot) number of images that show it
    return [acc / counts[:, i:i + 1] for i, acc in enumerate(sums)], vis / seen.clamp(min=1)[:, None]


def attention_targets(masks, size):
    """PifPaf soft masks [N,K+1,H,W] -> per-part attention targets [N,K,HW] (sum to 1) and presence [N,K] (bool):
    a part is present when it wins the argmax somewhere on the feature grid (BPBreID's rule on the targets)."""
    m = F.interpolate(masks, size, mode='bilinear', align_corners=True)
    present = F.one_hot(m.argmax(1), m.shape[1]).permute(0, 3, 1, 2).amax((2, 3))[:, 1:].bool()
    parts = m[:, 1:].flatten(2)
    return parts / parts.sum(-1, keepdim=True).clamp(min=1e-6), present


class Stage2Loss(nn.Module):
    """Where each term acts (H = holistic/global features, P = per-part 256-d vectors):
      H   id  = CE(score_gap4) + CE(score_g);  tri = triplet(gap3) + triplet(gap4) + triplet(g);  i2t = CE(g @ text_global.T)
          (CLIP-ReID stage 2 verbatim, loss/make_loss.py + processor_clipreid_stage2.py)
      H+P fused_id / fused_tri = ID + triplet on the self-attended fused vector (gradient reaches z0 and the fusion; part
          tokens only if FUSE_DETACH is off)
      P   part_id    = ID CE on each h_k through its own BNNeck classifier, on images where part k is present
      P   part_tri   = batch-hard triplet on the visibility-masked mean of the part distances; LSE worst-part ramp (lse_gamma_at)
      P   part_indiv = batch-hard triplet per part on that part's own distance (both images show it)
      P   align      = part<->prompt contrast: h_k vs A_k(stage-1 prompts of the batch identities), symmetric; positive =
                       (own identity, part k); with CROSS_PART_NEG the same identities' prompts of the other parts are
                       extra negatives for h_k. Random text erasing (training only, erased_text): TEXT_ERASE_PROB drops a
                       (sample, slot) anchor from this term and, for slot 0, that sample from i2t; TEXT_DROPOUT masks text
                       embedding dimensions.
      P   attn       = KL(PifPaf part mask || part attention map), present parts only;  vis = BCE(vis_logit, present)
    loss = ID_W*id + TRI_W*tri + I2T_W*i2t + FUSED_ID_W*fused_id + FUSED_TRI_W*fused_tri + PART_ID_W*part_id
           + PART_TRI_W*part_tri + PART_INDIV_TRI_W*part_indiv + PART_ALIGN_W*align + ATTN_W*attn + VIS_W*vis"""

    def __init__(self, num_classes, text_all):
        super().__init__()
        self.xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        self.triplet = GlobalTripletLoss(MARGIN)
        self.fused_triplet = GlobalTripletLoss(MARGIN, normalize=FUSED_TRI_NORMALIZE)
        self.part_triplet = PartLSETripletLoss(gamma=0.0)
        self.part_indiv = PartIndividualTripletLoss(MARGIN)
        self.register_buffer('text_all', text_all)

    def erased_text(self, target):
        """Random text erasing, training only: a per-slot dimension mask on the frozen prompts (TEXT_DROPOUT) and a
        per (sample, slot) anchor-erasing mask (TEXT_ERASE_PROB). Returns the masked text_all [C, S, D] and the
        keep mask [B, S]; both are identities at eval or with the knobs at 0."""
        text_all, B = self.text_all, target.shape[0]
        if self.training and TEXT_DROPOUT > 0:
            m = (torch.rand(S, text_all.shape[-1], device=text_all.device) > TEXT_DROPOUT).float() / (1 - TEXT_DROPOUT)
            text_all = text_all * m[None]                       # shared across identities within a slot
        erase = torch.ones(B, S, dtype=torch.bool, device=target.device)
        if self.training and TEXT_ERASE_PROB > 0:
            erase = torch.rand(B, S, device=target.device) >= TEXT_ERASE_PROB
        return text_all, erase

    def part_prompt_contrast(self, h, present, target, text_all, erase, adapters):
        """h [B,K,d]; text_all [C,S,D]. Per part k, over images showing it: i2t over the columns (identity u, part k') of the
        batch identities seen through adapter k (positive = own identity at part k; other identities at k and, with
        CROSS_PART_NEG, every identity at k' != k are negatives), t2i = prompt of each sample's identity vs the part-k
        vectors of the batch. Returns the mean loss and the per-part in-batch i2t top-1 [K] (own prompt of own part)."""
        uniq, inv = torch.unique(target, return_inverse=True)
        U, dev = uniq.shape[0], target.device
        col_u = torch.arange(U, device=dev).repeat_interleave(K)
        col_s = torch.arange(K, device=dev).repeat(U)
        losses, acc = [], torch.zeros(K)
        for k in range(K):
            keep = present[:, k] & erase[:, k + 1]
            if keep.sum() < 2:
                continue
            t = F.normalize(adapters[k](text_all[uniq][:, 1:]).float(), dim=-1)                 # [U,K,d]
            x = F.normalize(h[keep, k].float(), dim=-1)                                         # [n,d]
            own = inv[keep]
            logits = x @ t.flatten(0, 1).t() / CONTRAST_TEMP                                    # [n, U*K], column = u*K + k'
            if not CROSS_PART_NEG:
                logits = logits.masked_fill((col_s != k)[None], float('-inf'))
            pos = (col_u[None] == own[:, None]) & (col_s[None] == k)
            same = target[keep][:, None] == target[keep][None]
            losses.append(supcon_from_logits(logits, pos) + supcon_from_logits(t[own, k] @ x.t() / CONTRAST_TEMP, same))
            acc[k] = (logits.argmax(1) == own * K + k).float().mean().item()
        loss = sum(losses) / len(losses) if losses else h.sum() * 0
        return loss, acc

    def forward(self, res, masks, target, bank=None, text_adapters=None):
        terms = {}
        text_all, erase = self.erased_text(target)
        terms['id'] = self.xent(res['score_gap4'], target) + self.xent(res['score_g'], target)
        terms['tri'] = self.triplet(res['gap3'], target) + self.triplet(res['gap4'], target) + self.triplet(res['g'], target)
        keep0 = erase[:, 0]
        terms['i2t'] = (self.xent(res['g'][keep0] @ text_all[:, 0].t(), target[keep0]) if keep0.any()
                        else res['g'].sum() * 0)

        f_bank = p_bank = None
        if bank is not None:
            bf, bh, bvis, blabels = bank
            f_bank, p_bank = (bf, blabels), (bh, bvis, blabels)
        terms['fused_id'] = self.xent(res['score_fused'], target)
        terms['fused_tri'] = self.fused_triplet(res['fused'], target, f_bank)

        attn_target, present = attention_targets(masks, res['grid'])
        part_id = [self.xent(res['score_parts'][present[:, k], k], target[present[:, k]]) for k in range(K) if present[:, k].any()]
        terms['part_id'] = sum(part_id) / len(part_id) if part_id else res['h'].sum() * 0
        terms['part_tri'] = self.part_triplet(res['h'], present, target, p_bank)
        terms['part_indiv'] = self.part_indiv(res['h'], present, target, p_bank)
        terms['align'], align_acc = self.part_prompt_contrast(res['h'], present, target, text_all, erase, text_adapters)
        terms['align_acc'] = align_acc.mean()
        terms.update({f'align_acc_{name}': align_acc[k] for k, name in enumerate(PART_NAMES)})

        attn = res['attn'][:, 1:].float().clamp(min=1e-8)
        kl = (attn_target * (attn_target.clamp(min=1e-8).log() - attn.log())).sum(-1)
        terms['attn'] = (kl * present).sum() / present.sum().clamp(min=1)
        terms['vis'] = F.binary_cross_entropy_with_logits(res['vis_logit'], present.float())

        total = (ID_W * terms['id'] + TRI_W * terms['tri'] + I2T_W * terms['i2t']
                 + FUSED_ID_W * terms['fused_id'] + FUSED_TRI_W * terms['fused_tri'] + PART_ID_W * terms['part_id']
                 + PART_TRI_W * terms['part_tri'] + PART_INDIV_TRI_W * terms['part_indiv'] + PART_ALIGN_W * terms['align']
                 + ATTN_W * terms['attn'] + VIS_W * terms['vis'])
        return total, terms


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def extract(model, loader):
    """Test-time vectors. Visibility = the model's own prediction (no mask needed); the invisible part blocks are zeroed
    and the visibility travels with them (`vis`: bool, or probability * (p > 0.5) with EVAL_SOFT_VIS)."""
    model.eval()
    feats = {k: [] for k in ['gap4', 'g', 'fused', 'h', 'vis']}
    pids, camids = [], []
    for img, pid, camid, _, _, _ in loader:
        res = model(img.to(DEVICE))
        prob = res['vis_logit'].sigmoid()
        vis = prob > 0.5
        for k in ['gap4', 'g', 'fused']:
            feats[k].append(res[k].float().cpu())
        feats['h'].append((res['h'].float() * vis[..., None]).cpu())
        feats['vis'].append((prob * vis if EVAL_SOFT_VIS else vis).cpu())
        pids.extend(np.asarray(pid))
        camids.extend(np.asarray(camid))
    return {k: torch.cat(v) for k, v in feats.items()}, np.asarray(pids), np.asarray(camids)


@torch.no_grad()
def part_distmats(qp, qv, gp, gv, gamma=LSE_GAMMA, chunk=EVAL_CHUNK):
    """Part matching, per-part cosine distances combined over the parts visible in BOTH images (qp [Nq,K,d], qv [Nq,K]).
    Returns (mean [Nq,Ng], LSE worst-part [Nq,Ng], shared-part count [Nq,Ng]); entries without a shared part are
    meaningless (-1) and are replaced by the caller's global-distance fallback."""
    qp, qv = qp.to(DEVICE), qv.to(DEVICE)
    mean, lse, shared = [], [], []
    for i in range(0, gp.shape[0], chunk):
        d, M = part_pairwise_distances(qp, qv, gp[i:i + chunk].to(DEVICE), gv[i:i + chunk].to(DEVICE))
        mean.append(lse_combine(d, M, 0.0)[0].cpu())
        lse.append(lse_combine(d, M, gamma)[0].cpu())
        shared.append(M.sum(0).cpu())
    return torch.cat(mean, 1).numpy(), torch.cat(lse, 1).numpy(), torch.cat(shared, 1).numpy()


@torch.no_grad()
def single_part_distmat(qh, qv, gh, gv, k, chunk=EVAL_CHUNK):
    """Cosine distance of part k alone; pairs where either image does not show it rank last (max + 1)."""
    qk = F.normalize(qh[:, k].to(DEVICE).float(), dim=-1)
    qvk = (qv[:, k] > 0).to(DEVICE)
    dist, ok = [], []
    for i in range(0, gh.shape[0], chunk):
        gk = F.normalize(gh[i:i + chunk, k].to(DEVICE).float(), dim=-1)
        dist.append((1 - qk @ gk.t()).cpu())
        ok.append((qvk[:, None] & (gv[i:i + chunk, k] > 0).to(DEVICE)[None]).cpu())
    D, valid = torch.cat(dist, 1), torch.cat(ok, 1)
    D[~valid] = (D[valid].max() + 1) if valid.any() else 1.0
    return D.numpy()


@torch.no_grad()
def part_lse_all_pairs(p, v, gamma=LSE_GAMMA, chunk=RERANK_CHUNK, device=None):
    """All-pairs part distance over query+gallery: [N, N] float16, same metric as part_distmats' LSE.

    re_ranking adds `local_distmat` to its own all-pairs `original_dist` *before* the k-reciprocal
    neighbourhood is built, so the local matrix has to span query+gallery, not the [Nq, Ng] block that
    part_distmats returns. Pairs with no mutually visible part get max + 1, as there. float16 and no
    unmatched/valid bookkeeping: on Market N = 19,281, so each [N, N] float32 array costs 1.5 GB.
    """
    device = device or RERANK_DEVICE or DEVICE
    p, v = p.to(device), v.to(device)
    cols, max_valid = [], 0.0
    for i in range(0, p.shape[0], chunk):
        d, M = part_pairwise_distances(p, v, p[i:i + chunk], v[i:i + chunk])
        D, valid, _ = lse_combine(d, M, gamma)
        if valid.any():
            max_valid = max(max_valid, D[valid].max().item())
        cols.append(torch.where(valid, D, torch.full_like(D, -1.0)).half().cpu())   # -1 = no shared part
    D = torch.cat(cols, 1)
    D[D < 0] = max_valid + 1
    return D.numpy()


def rerank_rows(feats, num_query, global_dist, logger):
    """The k-reciprocal rows: plain, and with the part-LSE distance inside the neighbourhood.

    `<feature>_rr` is re_ranking on the chosen retrieval vector. `<feature>_rr_lse` passes the all-pairs part
    distance as `local_distmat`, so the Jaccard neighbourhood is built from the global *and* the part-visibility
    distance instead of fusing two finished rankings. re_ranking sums the two raw matrices and only normalises
    afterwards, and its own distance is a squared euclidean on unit vectors (= 2 x the cosine distance of
    `global_dist`), so the local matrix is rescaled to that mean first. `global_dist` is the row's own [Nq, Ng]
    cosine distmat, already computed by evaluate().
    """
    vec = {'clipreid_global': lambda f: F.normalize(torch.cat([f['gap4'], f['g']], dim=1), dim=1),
           'parts_selfattn': lambda f: F.normalize(f['fused'], dim=1),
           'holistic': lambda f: concat_feature(f['gap4'], f['g'], f['fused'])}[RERANK_FEATURE](feats)
    device = RERANK_DEVICE or DEVICE
    qf, gf = vec[:num_query].to(device), vec[num_query:].to(device)
    rows, start = {}, time.time()
    rows[f'{RERANK_FEATURE}_rr'] = re_ranking(qf, gf, RERANK_K1, RERANK_K2, RERANK_LAMBDA)

    d_lse = part_lse_all_pairs(feats['h'], feats['vis'])
    scale = RERANK_LOCAL_W * 2 * float(global_dist.mean()) / float(d_lse.astype(np.float32).mean())
    local = d_lse.astype(np.float32) * scale
    del d_lse                                     # every [N,N] float32 array is 1.5 GB on Market
    rows[f'{RERANK_FEATURE}_rr_lse'] = re_ranking(qf, gf, RERANK_K1, RERANK_K2, RERANK_LAMBDA, local_distmat=local)
    if RERANK_ONLY_LOCAL:                         # re_ranking normalises per column, so the scale cancels here
        rows['part_lse_rr'] = re_ranking(qf, gf, RERANK_K1, RERANK_K2, RERANK_LAMBDA, local_distmat=local, only_local=True)
    del local
    logger.info('re-ranking on `{}` (k1={}, k2={}, lambda={}, local_w={} -> scale {:.3f}): {} in {:.0f}s, '
                'peak RSS {:.1f} GB'.format(RERANK_FEATURE, RERANK_K1, RERANK_K2, RERANK_LAMBDA, RERANK_LOCAL_W,
                                            scale, ' '.join(rows), time.time() - start,
                                            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20))
    return rows


def concat_feature(*slots):
    """Slots concatenated into one retrieval vector. With EVAL_SLOTWISE_NORM each slot is L2-normalised first so
    all slots weigh equally (concatenating then normalising once - CLIP-ReID's convention - weights slots by norm).
    Normalised once at the end."""
    if EVAL_SLOTWISE_NORM:
        slots = [F.normalize(x, dim=-1) for x in slots]
    return F.normalize(torch.cat(slots, dim=1), dim=1)


def evaluate(model, val_loader, num_query, logger, tag, rerank=False, out_json=None):
    """Rows (all cosine distances, so combinations add directly):
      clipreid_global    cat(gap4, g), CLIP-ReID's own feature
      parts_selfattn     the self-attended fused vector
      parts_matching     per-part cosine over the parts visible in both images, mean (BPBreID); fewer than
                         MIN_SHARED_PARTS shared -> global distance. parts_matching_lse: worst-part (LSE) version
      holistic           cat(gap4, g, fused) as one vector
      global+parts, global+parts_lse, all    clipreid_global + w * the matching rows (+ the self-attended row); every
                         distance is divided by its own mean first, so the weights compare like with like
      part_<name>        each part alone (pairs where both images show it)
    """
    f, pids, camids = extract(model, val_loader)
    q, gal = slice(0, num_query), slice(num_query, None)
    q_pids, g_pids, q_cams, g_cams = pids[q], pids[gal], camids[q], camids[gal]
    summary, base = {}, {}

    def dist(vec):
        v = F.normalize(vec, dim=1)
        return (1 - v[q].to(DEVICE) @ v[gal].to(DEVICE).t()).cpu().numpy()

    def score(name, distmat):
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_cams, g_cams)
        summary[name] = dict(mAP=mAP, R1=cmc[0], R5=cmc[4], R10=cmc[9])
        logger.info('[{:22s}] mAP: {:.1%}  Rank-1: {:.1%}  Rank-5: {:.1%}  Rank-10: {:.1%}'.format(name, mAP, cmc[0], cmc[4], cmc[9]))

    d_global = dist(torch.cat([f['gap4'], f['g']], dim=1))
    d_self = dist(f['fused'])
    d_mean, d_lse, n_shared = part_distmats(f['h'][q], f['vis'][q], f['h'][gal], f['vis'][gal])
    valid = n_shared >= MIN_SHARED_PARTS
    fb = float(d_mean[valid].mean() / d_global[valid].mean()) if valid.any() else 1.0   # global fallback, rescaled to the part scale
    fb_lse = float(d_lse[valid].mean() / d_global[valid].mean()) if valid.any() else 1.0
    d_parts, d_parts_lse = np.where(valid, d_mean, d_global * fb), np.where(valid, d_lse, d_global * fb_lse)

    logger.info(f'Validation Results - {tag}')
    logger.info('invisible rate per part | query: {} | gallery: {}'.format(
        dict(zip(PART_NAMES, (1 - (f['vis'][q] > 0).float().mean(0)).numpy().round(3).tolist())),
        dict(zip(PART_NAMES, (1 - (f['vis'][gal] > 0).float().mean(0)).numpy().round(3).tolist()))))
    logger.info('query-gallery pairs with >=1 unmatched part: {:.1%} | with < {} shared parts (global fallback): {:.2%} | '
                'mean unmatched parts: {:.2f}'.format((n_shared < K).mean(), MIN_SHARED_PARTS, (~valid).mean(), (K - n_shared).mean()))
    score('clipreid_global', d_global)
    score('parts_selfattn', d_self)
    score('parts_matching', d_parts)
    score('parts_matching_lse', d_parts_lse)
    d_hol = dist(concat_feature(f['gap4'], f['g'], f['fused']))
    score('holistic', d_hol)
    def unit(d):
        return d / d.mean()                       # mean-ratio: every distance has mean 1 before the weights apply
    ug, us, up, upl = unit(d_global), unit(d_self), unit(d_parts), unit(d_parts_lse)
    score('global+parts', ug + EVAL_W_PARTS * up)
    score('global+parts_lse', ug + EVAL_W_PARTS * upl)
    score('all', ug + EVAL_W_SELF * us + EVAL_W_PARTS * up)
    del d_mean, d_lse, d_parts, d_parts_lse, up, upl
    if EVAL_PART_ROWS:
        for k, name in enumerate(PART_NAMES):
            score(f'part_{name}', single_part_distmat(f['h'][q], f['vis'][q], f['h'][gal], f['vis'][gal], k))
    if rerank:
        ref = {'clipreid_global': d_global, 'parts_selfattn': d_self, 'holistic': d_hol}[RERANK_FEATURE]
        for name, distmat in rerank_rows(f, num_query, ref, logger).items():
            score(name, distmat)
    if out_json:
        with open(out_json, 'w') as fh:
            json.dump(dict(tag=tag, rows=summary), fh, indent=1, default=float)
        logger.info(f'results written to {out_json}')
    torch.cuda.empty_cache()
    return summary


# ----------------------------------------------------------------------------- setup
def build_text_targets(clip, num_classes, stage1_ckpt, logger):
    """Encode the stage-1 prompts once: text_all [C, K+1, 1024] (processor_clipreid_stage2.py:60-71)."""
    ckpt = torch.load(stage1_ckpt, map_location=DEVICE)
    knobs = ckpt['knobs']
    assert (knobs['H'], knobs['W'], knobs['STRIDE']) == (H, W, STRIDE), f'stage-1 knobs {knobs} != stage-2 {(H, W, STRIDE)}'
    assert knobs.get('DATASET', 'market1501') == DATASET, \
        f"stage-1 prompts are for {knobs.get('DATASET', 'market1501')}, stage 2 is running {DATASET}"
    assert ckpt['part_names'] == PART_NAMES
    prompt_learner = PartPromptLearner(num_classes, clip, PART_NAMES, n_ctx=knobs['N_CTX']).to(DEVICE)
    prompt_learner.load_state_dict(ckpt['prompt_learner'])
    text_encoder = TextEncoder(clip).to(DEVICE).eval()
    with torch.no_grad():
        text_all = torch.cat([encode_text(prompt_learner, text_encoder, torch.arange(i, min(i + 128, num_classes), device=DEVICE))
                              for i in range(0, num_classes, 128)])
    logger.info(f"text targets from {stage1_ckpt} (epoch {ckpt['epoch']}): {tuple(text_all.shape)}")
    return text_all.float()


@torch.no_grad()
def build_semantic_queries(clip, logger):
    """The K+1 identity-agnostic LPIM queries through the frozen CLIP text encoder: [K+1, 1024]."""
    from model.clip.clip import tokenize
    templates = ['A photo of a person.'] + [f"A photo of the {p.replace('_', ' ')} of a person." for p in PART_NAMES]
    tq = clip.encode_text(tokenize(templates).to(DEVICE)).float()
    logger.info('LPIM queries: ' + ' | '.join(templates))
    return tq


def make_optimizer(model):
    """make_optimizer_2stage (solver/make_optimizer_prompt.py) without the config object."""
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr, wd = BASE_LR, WEIGHT_DECAY
        if 'bias' in key:
            lr, wd = BASE_LR * BIAS_LR_FACTOR, WEIGHT_DECAY_BIAS
        params.append({'params': [value], 'lr': lr, 'weight_decay': wd})
    return torch.optim.Adam(params)


def save_checkpoint(model, optimizer, scheduler, epoch, path):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'epoch': epoch, 'knobs': dict(H=H, W=W, STRIDE=STRIDE, BACKBONE=BACKBONE, LSE_GAMMA=LSE_GAMMA,
                                              PART_NAMES=PART_NAMES, MIM_SELF_LAYERS=MIM_SELF_LAYERS, PART_DIM=PART_DIM,
                                              FUSED_DIM=FUSED_DIM, FUSE_LAYERS=FUSE_LAYERS, FUSE_DETACH=FUSE_DETACH,
                                              LPIM_LEARN_QUERY=LPIM_LEARN_QUERY, BANK_SIZE=BANK_SIZE,
                                              EVAL_SLOTWISE_NORM=EVAL_SLOTWISE_NORM)}, path)


# ----------------------------------------------------------------------------- stage 2
def update_hard_sampler(epoch, model, criterion, train_loader, stats_loader, logger):
    """HARD_SAMPLING: random PK until HARD_START_EPOCH, then batches built around confusable identities. The table
    comes from the stage-1 prompts (text_all) first and is rebuilt from the model's own prototypes every NBR_REFRESH."""
    sampler = train_loader.sampler
    if not HARD_SAMPLING or not isinstance(sampler, PartHardPKSampler):
        return
    if epoch < HARD_START_EPOCH:
        sampler.set_neighbours(None)
    elif NBR_REFRESH > 0 and (epoch - HARD_START_EPOCH) % NBR_REFRESH == 0:
        protos, vis_rate = identity_slot_means(model, stats_loader, criterion.text_all.shape[0])
        sampler.set_neighbours(build_neighbours(protos), (vis_rate >= 0.5).t().cpu().numpy())
        logger.info(f'epoch {epoch}: hard-negative table rebuilt from model prototypes (k={NBR_K}, hard_frac={HARD_FRAC}, '
                    f'anchor-eligible identities per slot: {(vis_rate >= 0.5).sum(0).tolist()})')
    elif sampler.nbr is None:
        sampler.set_neighbours(build_neighbours(criterion.text_all))
        logger.info(f'epoch {epoch}: hard-negative table built from stage-1 prompts (k={NBR_K}, hard_frac={HARD_FRAC})')


def do_train_stage2(model, criterion, train_loader, val_loader, num_query, logger, start_epoch=1, resume=None,
                    stats_loader=None):
    """processor_clipreid_stage2.py:73-137 with the part losses and the global / self-attended / part-matching evaluation."""
    optimizer = make_optimizer(model)
    scheduler = WarmupMultiStepLR(optimizer, STEPS, GAMMA, WARMUP_FACTOR, WARMUP_ITERS, WARMUP_METHOD)
    if resume is not None:
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
    scaler = amp.GradScaler(enabled=USE_AMP)
    meters = {k: AverageMeter() for k in ['loss', 'id', 'tri', 'i2t', 'fused_id', 'fused_tri', 'part_id', 'part_tri', 'part_indiv',
                                          'align', 'align_acc', 'attn', 'vis', 'vis_acc', 'acc']
              + [f'align_acc_{n}' for n in PART_NAMES]}
    best = {}
    feat_bank = FeatureBank(BANK_SIZE, FUSED_DIM, K, DEVICE, part_dim=PART_DIM) if BANK_SIZE > 0 else None
    if feat_bank is not None:
        logger.info(f'cross-batch memory for triplet mining: {BANK_SIZE} slots, used from epoch {BANK_START_EPOCH}')
    all_start = time.monotonic()
    logger.info('start training')
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        for m in meters.values():
            m.reset()
        scheduler.step()
        criterion.part_triplet.gamma = lse_gamma_at(epoch)
        update_hard_sampler(epoch, model, criterion, train_loader, stats_loader, logger)
        model.train()
        start = time.time()
        for n_iter, (img, masks, target, _) in enumerate(train_loader):
            optimizer.zero_grad()
            img, masks, target = img.to(DEVICE), masks.to(DEVICE), target.to(DEVICE)
            use_bank = feat_bank is not None and epoch >= BANK_START_EPOCH and feat_bank.filled > 0
            present = attention_targets(masks, GRID)[1]
            with amp.autocast(enabled=USE_AMP):
                res = model(img, present)
                assert tuple(res['grid']) == GRID, f"x4 grid {tuple(res['grid'])} != GRID {GRID}: visibility/attention targets use GRID"
                loss, terms = criterion(res, masks, target, feat_bank.get() if use_bank else None, model.text_adapters)
            if feat_bank is not None:
                feat_bank.add(res['fused'], res['h'], present, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            acc = (res['score_gap4'].max(1)[1] == target).float().mean()
            meters['vis_acc'].update(((res['vis_logit'] > 0) == present).float().mean().item(), 1)
            meters['loss'].update(loss.item(), img.shape[0])
            meters['acc'].update(acc.item(), 1)
            for k, v in terms.items():
                meters[k].update(v.item(), img.shape[0])
            if (n_iter + 1) % LOG_PERIOD == 0:
                m = {k: v.avg for k, v in meters.items()}
                logger.info('Epoch[{}] Iteration[{}/{}] Loss: {:.3f} (id {:.3f} tri {:.3f} i2t {:.3f} | fused_id {:.3f} fused_tri {:.3f} | '
                            'part_id {:.3f} part_tri {:.3f} part_indiv {:.3f} align {:.3f} (top1 {:.3f}) attn {:.3f} vis {:.3f} vis_acc {:.3f}) '
                            'Acc: {:.3f}, Base Lr: {:.2e}, LSE gamma: {:.2f}'
                            .format(epoch, n_iter + 1, len(train_loader), m['loss'], m['id'], m['tri'], m['i2t'], m['fused_id'],
                                    m['fused_tri'], m['part_id'], m['part_tri'], m['part_indiv'], m['align'], m['align_acc'],
                                    m['attn'], m['vis'], m['vis_acc'], m['acc'], scheduler.get_lr()[0], criterion.part_triplet.gamma))
        time_per_batch = (time.time() - start) / (n_iter + 1)
        logger.info('Epoch {} done. Loss: {:.3f} Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s] | part<->prompt top-1 {}'
                    .format(epoch, meters['loss'].avg, time_per_batch, train_loader.batch_size / time_per_batch,
                            {n: round(meters[f'align_acc_{n}'].avg, 3) for n in PART_NAMES}))
        if epoch % CHECKPOINT_PERIOD == 0 or epoch == MAX_EPOCHS:
            path = os.path.join(OUTPUT_DIR, f'{BACKBONE}_part_stage2_{epoch}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, path)
            logger.info(f'saved {path}')
        if epoch % EVAL_PERIOD == 0 or epoch == MAX_EPOCHS:
            summary = evaluate(model, val_loader, num_query, logger, f'Epoch: {epoch}',
                               rerank=RERANK and (RERANK_EVERY_EVAL or epoch == MAX_EPOCHS),
                               out_json=os.path.join(OUTPUT_DIR, f'results_epoch{epoch}.json'))
            for name, m in summary.items():
                if m['mAP'] > best.get(name, {'mAP': -1})['mAP']:
                    best[name] = dict(epoch=epoch, **m)
            logger.info('best so far: ' + ' | '.join(
                '{} mAP {:.1%} R1 {:.1%} @epoch {}'.format(n, b['mAP'], b['R1'], b['epoch']) for n, b in best.items()))
    logger.info('Total running time: {}'.format(timedelta(seconds=time.monotonic() - all_start)))


def main():
    global MAX_EPOCHS, IMS_PER_BATCH, EVAL_PERIOD, TEXT_ERASE_PROB, TEXT_DROPOUT, DATASET, OUTPUT_DIR
    global HARD_SAMPLING, HARD_FRAC, NBR_K, HARD_START_EPOCH, NBR_REFRESH, PART_ID_W, PART_INDIV_TRI_W, PART_ALIGN_W
    global CROSS_PART_NEG, FUSE_DETACH, EVAL_W_SELF, EVAL_W_PARTS
    global MASKS_VARIANT, MASKS_DIR
    global RERANK, RERANK_FEATURE, RERANK_K1, RERANK_K2, RERANK_LAMBDA, RERANK_LOCAL_W
    global RERANK_ONLY_LOCAL, RERANK_EVERY_EVAL, RERANK_DEVICE
    parser = argparse.ArgumentParser(description='CLIP-ReID stage 2 with per-part prompts (RN50)')
    parser.add_argument('--dataset', choices=list(DATASETS), default=DATASET)
    parser.add_argument('--masks-variant', choices=list(MASK_SUFFIX), default=MASKS_VARIANT)
    parser.add_argument('--masks-dir', type=str, default=MASKS_DIR, help='pre-saved masks outside the dataset dir')
    parser.add_argument('--stage1-ckpt', type=str, default='')
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--batch', type=int, default=IMS_PER_BATCH)
    parser.add_argument('--eval-period', type=int, default=EVAL_PERIOD)
    parser.add_argument('--resume', type=str, default='', help='stage-2 checkpoint to continue from')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--weights', type=str, default='', help='stage-2 checkpoint for --eval-only')
    parser.add_argument('--text-erase-prob', type=float, default=TEXT_ERASE_PROB,
                        help='random text erasing: drop a (sample, slot) identity prompt from i2t / SupCon (0 = off)')
    parser.add_argument('--text-dropout', type=float, default=TEXT_DROPOUT,
                        help='Bernoulli dropout over the text embedding dimensions (0 = off)')
    parser.add_argument('--hard-sampling', action='store_true', default=HARD_SAMPLING,
                        help='per-part hard-negative PK batches (HARD_SAMPLING)')
    parser.add_argument('--hard-frac', type=float, default=HARD_FRAC)
    parser.add_argument('--nbr-k', type=int, default=NBR_K)
    parser.add_argument('--hard-start-epoch', type=int, default=HARD_START_EPOCH)
    parser.add_argument('--nbr-refresh', type=int, default=NBR_REFRESH, help='0 = keep the stage-1 prompt table')
    parser.add_argument('--part-id-w', type=float, default=PART_ID_W, help='0 = BPBreID GiLt (no per-part ID loss)')
    parser.add_argument('--part-indiv-w', type=float, default=PART_INDIV_TRI_W, help='per-part individual triplet weight (0 = off)')
    parser.add_argument('--part-align-w', type=float, default=PART_ALIGN_W, help='part<->prompt contrast weight (0 = off)')
    parser.add_argument('--no-cross-part-neg', action='store_true', help='ablation: other parts\' prompts are not negatives')
    parser.add_argument('--no-fuse-detach', action='store_true', help='ablation: let the fused loss reach the part tokens')
    parser.add_argument('--eval-w-self', type=float, default=EVAL_W_SELF)
    parser.add_argument('--eval-w-parts', type=float, default=EVAL_W_PARTS)
    parser.add_argument('--rerank', action='store_true', help='k-reciprocal re-ranking rows (test-time only)')
    parser.add_argument('--rerank-feature', choices=['holistic', 'parts_selfattn', 'clipreid_global'], default=RERANK_FEATURE)
    parser.add_argument('--rerank-k1', type=int, default=RERANK_K1)
    parser.add_argument('--rerank-k2', type=int, default=RERANK_K2)
    parser.add_argument('--rerank-lambda', type=float, default=RERANK_LAMBDA)
    parser.add_argument('--rerank-local-w', type=float, default=RERANK_LOCAL_W, help='0 = plain k-reciprocal twice')
    parser.add_argument('--rerank-only-local', action='store_true', default=RERANK_ONLY_LOCAL)
    parser.add_argument('--rerank-every-eval', action='store_true', default=RERANK_EVERY_EVAL)
    parser.add_argument('--rerank-device', type=str, default=RERANK_DEVICE, help="'cpu' if the [N,N] matmul will not fit")
    args = parser.parse_args()
    MAX_EPOCHS, IMS_PER_BATCH, EVAL_PERIOD, DATASET = args.epochs, args.batch, args.eval_period, args.dataset
    OUTPUT_DIR = OUTPUT_DIR.format(dataset=DATASET)
    args.stage1_ckpt = args.stage1_ckpt or STAGE1_CKPT.format(dataset=DATASET)
    MASKS_VARIANT, MASKS_DIR = args.masks_variant, args.masks_dir
    TEXT_ERASE_PROB, TEXT_DROPOUT = args.text_erase_prob, args.text_dropout
    HARD_SAMPLING, HARD_FRAC, NBR_K, HARD_START_EPOCH = args.hard_sampling, args.hard_frac, args.nbr_k, args.hard_start_epoch
    NBR_REFRESH, PART_ID_W, PART_INDIV_TRI_W, PART_ALIGN_W = args.nbr_refresh, args.part_id_w, args.part_indiv_w, args.part_align_w
    CROSS_PART_NEG, FUSE_DETACH = CROSS_PART_NEG and not args.no_cross_part_neg, FUSE_DETACH and not args.no_fuse_detach
    EVAL_W_SELF, EVAL_W_PARTS = args.eval_w_self, args.eval_w_parts
    RERANK, RERANK_FEATURE, RERANK_K1, RERANK_K2 = args.rerank, args.rerank_feature, args.rerank_k1, args.rerank_k2
    RERANK_LAMBDA, RERANK_LOCAL_W = args.rerank_lambda, args.rerank_local_w
    RERANK_ONLY_LOCAL, RERANK_EVERY_EVAL, RERANK_DEVICE = args.rerank_only_local, args.rerank_every_eval, args.rerank_device

    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = setup_logger('transreid', OUTPUT_DIR, if_train=not args.eval_only)
    logger.info('knobs: ' + ', '.join(f'{k}={v}' for k, v in dict(
        DATASET=DATASET, MASKS_VARIANT=MASKS_VARIANT, MASKS_DIR=MASKS_DIR, H=H, W=W, IMS_PER_BATCH=IMS_PER_BATCH, NUM_INSTANCE=NUM_INSTANCE, MAX_EPOCHS=MAX_EPOCHS, BASE_LR=BASE_LR,
        STEPS=STEPS, ID_W=ID_W, TRI_W=TRI_W, I2T_W=I2T_W, FUSED_ID_W=FUSED_ID_W, FUSED_TRI_W=FUSED_TRI_W, PART_ID_W=PART_ID_W,
        PART_TRI_W=PART_TRI_W, PART_INDIV_TRI_W=PART_INDIV_TRI_W, PART_ALIGN_W=PART_ALIGN_W, ATTN_W=ATTN_W, VIS_W=VIS_W,
        PART_DIM=PART_DIM, FUSED_DIM=FUSED_DIM, FUSE_LAYERS=FUSE_LAYERS, FUSE_DETACH=FUSE_DETACH, CROSS_PART_NEG=CROSS_PART_NEG,
        LPIM_LEARN_QUERY=LPIM_LEARN_QUERY, TEXT_ERASE_PROB=TEXT_ERASE_PROB, TEXT_DROPOUT=TEXT_DROPOUT,
        LSE_GAMMA_EPOCHS=LSE_GAMMA_EPOCHS, MIM_SELF_LAYERS=MIM_SELF_LAYERS, MARGIN=MARGIN, LSE_GAMMA=LSE_GAMMA, USE_AMP=USE_AMP,
        BANK_SIZE=BANK_SIZE, BANK_START_EPOCH=BANK_START_EPOCH, EVAL_SLOTWISE_NORM=EVAL_SLOTWISE_NORM,
        EVAL_W_SELF=EVAL_W_SELF, EVAL_W_PARTS=EVAL_W_PARTS, MIN_SHARED_PARTS=MIN_SHARED_PARTS, EVAL_SOFT_VIS=EVAL_SOFT_VIS,
        RERANK=RERANK, RERANK_FEATURE=RERANK_FEATURE, RERANK_K1=RERANK_K1, RERANK_K2=RERANK_K2,
        RERANK_LAMBDA=RERANK_LAMBDA, RERANK_LOCAL_W=RERANK_LOCAL_W, RERANK_ONLY_LOCAL=RERANK_ONLY_LOCAL,
        RERANK_EVERY_EVAL=RERANK_EVERY_EVAL, HARD_SAMPLING=HARD_SAMPLING, HARD_FRAC=HARD_FRAC, NBR_K=NBR_K,
        NBR_JACCARD=NBR_JACCARD, HARD_START_EPOCH=HARD_START_EPOCH, NBR_REFRESH=NBR_REFRESH).items()))

    dataset, dataset_dir = build_dataset(DATASET, DATA_ROOT)
    masks_dir = resolve_masks(DATASET, dataset, dataset_dir, logger.info, require=not args.eval_only,
                              variant=MASKS_VARIANT, masks=MASKS_DIR)
    num_classes, num_query = dataset.num_train_pids, len(dataset.query)
    train_loader, val_loader, stats_loader = make_loaders(dataset, dataset_dir, masks_dir, IMS_PER_BATCH)

    h_res, w_res = (H - 16) // STRIDE + 1, (W - 16) // STRIDE + 1
    clip = load_clip_to_cpu(BACKBONE, h_res, w_res, STRIDE).to(DEVICE)
    text_all = build_text_targets(clip, num_classes, args.stage1_ckpt, logger)
    text_queries = build_semantic_queries(clip, logger)
    model = PartCLIPReID(clip.visual, num_classes, text_queries).to(DEVICE)
    del clip
    torch.cuda.empty_cache()
    criterion = Stage2Loss(num_classes, text_all).to(DEVICE)
    logger.info('trainable parameters: {:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if args.eval_only:
        ckpt = torch.load(args.weights, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        evaluate(model, val_loader, num_query, logger, f"{args.weights} (epoch {ckpt['epoch']})", rerank=RERANK,
                 out_json=os.path.join(OUTPUT_DIR, 'results_eval.json'))
        return

    resume, start_epoch = None, 1
    if args.resume:
        resume = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(resume['model'])
        start_epoch = resume['epoch'] + 1
        logger.info(f"resuming from {args.resume} (epoch {resume['epoch']})")
    do_train_stage2(model, criterion, train_loader, val_loader, num_query, logger, start_epoch, resume, stats_loader)


if __name__ == '__main__':
    main()
