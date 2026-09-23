"""CLIP-ReID stage 2 (image-side fine-tuning) with learnt per-part prompts on PartAwareModifiedResNet (RN50).

Run from the repo root:
  python processor/train_part_prompts_stage2.py --stage1-ckpt work_dirs/market1501/part_prompts_stage1/RN50_part_prompts_stage1_60.pth

What stage 2 does here
* Baseline branch = CLIP-ReID RN50 stage 2, verbatim: gap3 = GAP(x3), gap4 = GAP(x4) with a BNNeck ID head,
  g = xproj[0] with a BNNeck ID head; triplet on all three; i2t = CE(g @ text_global.T); test feature
  cat(gap4, g) = the `clipreid_baseline` eval row (paper: 89.8 mAP on Market-1501; reproduced at 89.3).
* LPIM branch (LanguageGuidedPartInteraction, after PromptSG CVPR'24): K+1 identity-agnostic text queries
  ("A photo of a person", "A photo of the <part> of a person") cross-attend over the x4 locations with
  projections initialised from CLIP's attention pool; pure cross-attention (no query-to-query term; optional
  MIM_SELF_LAYERS). Output z0 (global-semantic token) and z1..zK (parts); attentive pooling gives pbar. The
  part attention maps are supervised by the PifPaf masks (KL) and a visibility head predicts part presence,
  so nothing external is needed at test time. Losses: ID + triplet on z0 and pbar, visibility-weighted part
  triplet (LSE gamma annealed after epoch 40), symmetric SupCon between z_s and the stage-1 identity prompts,
  attention KL, visibility BCE.
* Text side (frozen): the stage-1 prompts are encoded once into text_all [C, K+1, 1024] (SupCon targets);
  the K+1 semantic queries are encoded once from fixed templates.
* Evaluation (every EVAL_PERIOD epochs), mAP / R1 / R5 / R10 for: clipreid_baseline (cat(gap4, g)), lpim
  (cat(z0, pbar)), holistic (cat(gap4, g, z0, pbar)), part_lse (LSE over mutually visible parts) and, with
  FUSE_W > 0, fused. Per-part invisible rate and unmatched-pair statistics are logged.
* --rerank adds two test-time rows (no training change): <row>_rr = k-reciprocal re-ranking (Zhong et al.
  CVPR'17, utils/reranking.py) on RERANK_FEATURE, and <row>_rr_lse = the same with the all-pairs part-LSE
  distance passed as `local_distmat`, so part visibility shapes the k-reciprocal neighbourhood itself rather
  than being fused into the final score (FUSE_W). See rerank_rows / part_lse_all_pairs.
* Optimiser / schedule / AMP = SOLVER.STAGE2 of configs/person/cnn_clipreid.yml, 256x128 as in the recipe.
"""
import os
import sys
import math
import time
import random
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

from processor.train_part_prompts_stage1 import (PartPromptLearner, encode_text, supcon, pifpaf_to_masks,
                                                  PART_NAMES, K, S, SLOT_NAMES)
from model.make_model_clipreid import load_clip_to_cpu, TextEncoder, weights_init_kaiming, weights_init_classifier
from datasets.market1501 import Market1501
from datasets.sampler import RandomIdentitySampler
from datasets.bases import ImageDataset, read_image
from datasets.make_dataloader_clipreid import val_collate_fn
from loss.softmax_loss import CrossEntropyLabelSmooth
from loss.triplet_loss import euclidean_dist
from solver.lr_scheduler import WarmupMultiStepLR
from utils.metrics import eval_func, euclidean_distance
from utils.reranking import re_ranking
from utils.logger import setup_logger
from utils.meter import AverageMeter

# ----------------------------------------------------------------------------- knobs
DATA_ROOT = '../../datasets'
MASKS_DIR = f'{DATA_ROOT}/market1501/masks/pifpaf_maskrcnn_filtering/bounding_box_train'
OUTPUT_DIR = './work_dirs/market1501/part_prompts_stage2'
STAGE1_CKPT = './work_dirs/market1501/part_prompts_stage1/RN50_part_prompts_stage1_60.pth'

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
LPIM_ID_W, LPIM_TRI_W = 1.0, 1.0              # ID + triplet on the LPIM global token z0 and the pooled parts
PART_TRI_W = 1.0                              # per-part triplet on z1..zK (visibility-weighted, LSE annealed)
SUPCON_W = 0.5                                # symmetric SupCon between z_s and the stage-1 identity prompts (PromptSG lambda)
ATTN_W, VIS_W = 1.0, 0.1                      # attention-map KL to the PifPaf masks; part-presence BCE for the visibility head
TEXT_ERASE_PROB = 0.1              # random text erasing: per (sample, slot), drop that slot's identity prompt from the
                                   # SupCon term, and for slot 0 drop the sample from i2t. The text side of Random
                                   # Erasing (RE_PROB on images): the image slot must stand on its own when its text
                                   # anchor is missing, which is the occluded-part case. 0 = off; at 0.1, 0.9^6 = 53%
                                   # of samples keep all six anchors. Ablate {0, 0.1, 0.2, 0.3}.
TEXT_DROPOUT = 0.1                 # Bernoulli mask over the 1024 text dimensions, rescaled by 1/(1-p), fresh every
                                   # step and shared across identities within a slot (so every comparison in that slot
                                   # stays in one sub-space and logits remain comparable across classes). Applies to
                                   # both text consumers, i2t and SupCon. The LPIM semantic queries are never touched:
                                   # they are part of the test-time path. 0 = off.
MARGIN = 0.3
LSE_GAMMA = 5.0                    # soft-max sharpness over parts (-> max distance as gamma grows)
LSE_GAMMA_EPOCHS = (40, 80)        # part triplet uses the visibility-weighted mean until epoch 40, then gamma ramps
                                   # linearly to LSE_GAMMA by epoch 80 (chasing the worst part only once parts are trained)
LSE_INCLUDE_GLOBAL = False         # add z0 as slot 0 of the LSE part distance
MIM_SELF_LAYERS = 0                # self-attention blocks after the cross-attention (PromptSG: +1.9 / +1.6 mAP for 1 / 2
                                   # layers on ViT). 0 = pure cross-attention, no query-to-query interaction.
FUSE_W = 0.0                       # >0: also evaluate d_holistic/mean + FUSE_W * d_lse/mean

BANK_SIZE = 0                      # cross-batch memory for triplet mining; 0 = batch-only (CLIP-ReID / BPBreID
                                   # baseline behaviour), 8192 = XBM ablation
BANK_START_EPOCH = 5               # epochs of batch-only mining before the bank is used (features settle first)

EVAL_SLOTWISE_NORM = True          # holistic vector: L2-normalise each slot before concatenating (see
                                   # holistic_vector). False restores the CLIP-ReID convention of a single
                                   # normalisation over the concatenation. Eval-only, no retraining needed.
EVAL_CHUNK = 2048

RERANK = False                     # k-reciprocal re-ranking (Zhong et al. CVPR'17, utils/reranking.py). Test-time
                                   # only: it changes no gradient and no checkpoint, so --rerank can be added to
                                   # any --eval-only run over a finished stage-2 model.
RERANK_FEATURE = 'holistic'        # eval row to re-rank: 'holistic' | 'lpim' | 'clipreid_baseline'
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

    def __init__(self, dataset, masks_dir):
        self.dataset = dataset
        self.masks_dir = masks_dir
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
        stem = os.path.splitext(os.path.basename(img_path))[0]
        img = TF.to_tensor(TF.resize(read_image(img_path), [H, W], interpolation=T.InterpolationMode.BICUBIC))
        mask = pifpaf_to_masks(np.load(f'{self.masks_dir}/{stem}.npy'))
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


def make_loaders(dataset, batch):
    train_set = PartImageDataset(dataset.train, MASKS_DIR)
    train_loader = DataLoader(train_set, batch_size=batch,
                              sampler=RandomIdentitySampler(dataset.train, batch, NUM_INSTANCE),
                              num_workers=NUM_WORKERS, drop_last=True)
    val_transforms = T.Compose([T.Resize((H, W)), T.ToTensor(), T.Normalize(PIXEL_MEAN, PIXEL_STD)])
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(val_set, batch_size=TEST_BATCH, shuffle=False, num_workers=NUM_WORKERS, collate_fn=val_collate_fn)
    return train_loader, val_loader


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
    * K/V/output projections start from CLIP's own attention pool (k_proj, v_proj, c_proj, positional
      embedding), so at initialisation, with near-uniform attention, every query returns CLIP's mean-pooled
      projected feature; the queries then learn *where* to look. Queries are identity-agnostic and identical
      at train and test, so no identity prompt or inversion network is needed at inference.
    * Pure cross-attention: queries never attend to each other (MIM_SELF_LAYERS optional blocks follow it).
    * The head-averaged attention map of each part query is supervised by its PifPaf mask (KL), which turns the
      masks into where-to-look supervision instead of hard pooling weights; a visibility head on each part
      token predicts part presence so visibility is available at test time without masks.
    * Attentive pooling over the K part tokens (alpha = softmax(w . z_k)) gives one occlusion-aware parts vector.
    forward(x4) -> dict(z [N,K+1,D], attn [N,K+1,HW], pbar [N,D], alpha [N,K], vis_logit [N,K]).
    """

    def __init__(self, attnpool, text_queries, num_self_layers=MIM_SELF_LAYERS):
        super().__init__()
        C, D = attnpool.k_proj.in_features, attnpool.c_proj.out_features
        self.num_heads = attnpool.num_heads
        self.register_buffer('text_queries', text_queries.detach().clone())
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
        self.pool_w = nn.Linear(D, 1)
        self.vis_head = nn.Linear(D, 1)

    def forward(self, x4):
        N, C, Hf, Wf = x4.shape
        HW, h, d = Hf * Wf, self.num_heads, C // self.num_heads
        tokens = x4.flatten(2).transpose(1, 2) + self.pos_embed[None].to(x4.dtype)
        Sq = self.text_queries.shape[0]
        Q = self.q_proj(self.text_queries.to(x4.dtype)).view(Sq, h, d).transpose(0, 1)
        Kt = self.k_proj(tokens).view(N, HW, h, d).permute(0, 2, 1, 3)
        V = self.v_proj(tokens).view(N, HW, h, d).permute(0, 2, 1, 3)
        attn = torch.softmax(torch.einsum('hsd,nhld->nhsl', Q, Kt).float() / math.sqrt(d), dim=-1)
        out = torch.einsum('nhsl,nhld->nhsd', attn.to(V.dtype), V).permute(0, 2, 1, 3).reshape(N, Sq, C)
        z = self.c_proj(out)
        z = z + self.ffn(self.norm(z))
        for layer in self.self_layers:
            z = layer(z)
        parts = z[:, 1:]
        alpha = torch.softmax(self.pool_w(parts).squeeze(-1).float(), dim=1)
        pbar = (alpha[..., None].to(parts.dtype) * parts).sum(1)
        return dict(z=z, attn=attn.mean(1), pbar=pbar, alpha=alpha, vis_logit=self.vis_head(parts).squeeze(-1).float())


class PartCLIPReID(nn.Module):
    """Trainable image side = the CLIP-ReID RN50 branch, verbatim, plus the LPIM part branch.

    Baseline branch (make_model_clipreid.py build_transformer, RN50): gap3 = GAP(x3) [1024], gap4 = GAP(x4)
    [2048] with a BNNeck ID head, g = xproj[0] [1024] with a BNNeck ID head and the i2t loss; triplet on all
    three; test feature cat(gap4, g) - the `clipreid_baseline` row.
    LPIM branch: z0 (global-semantic token) and pbar (attentively pooled parts), each with a BNNeck ID head;
    z1..zK for the per-part triplet / LSE matching; visibility from the LPIM head.
    forward(x) -> dict(gap3, gap4, g, z, z0, zparts, pbar, attn, alpha, vis_logit[, score_gap4, score_g,
    score_z0, score_pbar]).
    The M1 variant (GWAP parts through a linear map, BPAM masks, concat ID head) is the ablation reference:
    git show e52d763.
    """

    def __init__(self, visual, num_classes, text_queries):
        super().__init__()
        self.visual = visual
        self.lpim = LanguageGuidedPartInteraction(visual.attnpool, text_queries)
        D = visual.attnpool.c_proj.out_features
        self.id_gap4 = BNNeckHead(visual.attnpool.v_proj.in_features, num_classes)
        self.id_global = BNNeckHead(D, num_classes)
        self.id_z0 = BNNeckHead(D, num_classes)
        self.id_pbar = BNNeckHead(D, num_classes)

    def forward(self, x):
        x3, x4, xproj = self.visual(x)
        gap3, gap4, g = x3.mean((2, 3)), x4.mean((2, 3)), xproj[0]
        out = self.lpim(x4)
        res = dict(gap3=gap3, gap4=gap4, g=g, z=out['z'], z0=out['z'][:, 0], zparts=out['z'][:, 1:], pbar=out['pbar'],
                   attn=out['attn'], alpha=out['alpha'], vis_logit=out['vis_logit'], grid=tuple(x4.shape[2:]))
        if self.training:
            res['score_gap4'] = self.id_gap4(gap4)[1]
            res['score_g'] = self.id_global(g)[1]
            res['score_z0'] = self.id_z0(res['z0'])[1]
            res['score_pbar'] = self.id_pbar(res['pbar'])[1]
        return res


# ----------------------------------------------------------------------------- part distances / losses
def lse_combine(d, M, gamma):
    """Combine per-part distances with a log-sum-exp soft-max over mutually visible parts.
    d, M: [K, A, B] distances and 0/1 mutual-visibility. Returns D [A,B] = (1/gamma) ln sum_k w_k e^{gamma d_k}
    with w_k = M_k / sum_k M_k, valid [A,B] (>= 1 shared part; invalid entries are -1) and the number of
    unmatched parts per pair [A,B]. gamma -> 0 is the visibility-weighted mean (BPBreID)."""
    n_shared = M.sum(0)
    valid = n_shared > 0
    w = M / n_shared.clamp(min=1)[None]
    if gamma < 1e-3:
        D = (w * d).sum(0)
    else:
        logits = gamma * d + torch.log(w.clamp(min=1e-12))
        logits = logits.masked_fill(M == 0, float('-inf'))
        D = torch.logsumexp(logits, dim=0) / gamma
    D = torch.where(valid, D, torch.full_like(D, -1.0))
    return D, valid, (M.shape[0] - n_shared)


def part_pairwise_distances(pa, va, pb, vb):
    """pa [A,K,D], pb [B,K,D] (L2-normalised inside), va/vb [A,K]/[B,K] bool -> d, M of shape [K,A,B]."""
    d = 1 - torch.einsum('ikd,jkd->kij', F.normalize(pa.float(), dim=-1), F.normalize(pb.float(), dim=-1))
    M = (va.t()[:, :, None] & vb.t()[:, None, :]).to(d.dtype)
    return d, M


class FeatureBank:
    """Cross-batch memory of detached embeddings (XBM, Wang & al. CVPR20).

    Triplet mining inside one PK batch only sees 64 samples / 16 identities. The bank keeps the most recent
    BANK_SIZE embeddings (z0, part tokens, visibility, label), so anchors are mined against the whole dataset: with
    8192 slots and 12936 training images, a batch is compared against ~2/3 of Market-1501. Bank entries are
    detached (gradient flows only through the anchors), which is what makes the large pool affordable.
    """

    def __init__(self, size, dim, parts, device):
        self.size = size
        self.g = torch.zeros(size, dim, device=device)
        self.p = torch.zeros(size, parts, dim, device=device)
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

    def __init__(self, margin=MARGIN):
        super().__init__()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, g, labels, bank=None):
        g = g.float()
        cols, col_labels, self_cols = columns_with_bank((g,), bank, labels)
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


def part_embeddings_for_lse(z0, zparts, vis_parts):
    """[N,M,D] and [N,M] visibility used by the LSE distance; optionally includes z0 (always visible) as slot 0."""
    if LSE_INCLUDE_GLOBAL:
        return torch.cat([z0[:, None], zparts], dim=1), torch.cat([torch.ones_like(vis_parts[:, :1]), vis_parts], dim=1)
    return zparts, vis_parts


def lse_gamma_at(epoch):
    """0 (weighted mean) until LSE_GAMMA_EPOCHS[0], then linear to LSE_GAMMA at LSE_GAMMA_EPOCHS[1]."""
    e0, e1 = LSE_GAMMA_EPOCHS
    if epoch <= e0:
        return 0.0
    return LSE_GAMMA * min(1.0, (epoch - e0) / max(e1 - e0, 1))


def attention_targets(masks, size):
    """PifPaf soft masks [N,K+1,H,W] -> per-part attention targets [N,K,HW] (sum to 1) and presence [N,K] (bool):
    a part is present when it wins the argmax somewhere on the feature grid (BPBreID's rule on the targets)."""
    m = F.interpolate(masks, size, mode='bilinear', align_corners=True)
    present = F.one_hot(m.argmax(1), m.shape[1]).permute(0, 3, 1, 2).amax((2, 3))[:, 1:].bool()
    parts = m[:, 1:].flatten(2)
    return parts / parts.sum(-1, keepdim=True).clamp(min=1e-6), present


class Stage2Loss(nn.Module):
    """Baseline terms are CLIP-ReID stage 2 verbatim (loss/make_loss.py + processor_clipreid_stage2.py):
        id  = CE(score_gap4) + CE(score_g);  tri = triplet(gap3) + triplet(gap4) + triplet(g);  i2t = CE(g @ text_global.T)
    LPIM terms (all on features that are matched at test time):
        lpim_id  = CE(score_z0) + CE(score_pbar);  lpim_tri = triplet(z0) + triplet(pbar)
        part_tri = batch-hard triplet on z1..zK, visibility-weighted mean -> LSE as gamma anneals (lse_gamma_at)
        supcon   = mean over slots of the symmetric SupCon between z_s and the batch's stage-1 identity prompts
                   (PromptSG's L_SupCon, batch positives / batch negatives, normalised, CLIP temperature)
        Random text erasing (training only, erased_text): TEXT_ERASE_PROB drops a (sample, slot) prompt anchor
        from SupCon and, for slot 0, that sample from i2t; TEXT_DROPOUT masks text embedding dimensions.
        attn     = KL(PifPaf part mask || part attention map), present parts only;  vis = BCE(vis_logit, present)
    loss = ID_W*id + TRI_W*tri + I2T_W*i2t + LPIM_ID_W*lpim_id + LPIM_TRI_W*lpim_tri + PART_TRI_W*part_tri
           + SUPCON_W*supcon + ATTN_W*attn + VIS_W*vis"""

    def __init__(self, num_classes, text_all):
        super().__init__()
        self.xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        self.triplet = GlobalTripletLoss(MARGIN)
        self.part_triplet = PartLSETripletLoss(gamma=0.0)
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

    def forward(self, res, masks, target, bank=None):
        terms = {}
        text_all, erase = self.erased_text(target)
        terms['id'] = self.xent(res['score_gap4'], target) + self.xent(res['score_g'], target)
        terms['tri'] = self.triplet(res['gap3'], target) + self.triplet(res['gap4'], target) + self.triplet(res['g'], target)
        keep0 = erase[:, 0]
        terms['i2t'] = (self.xent(res['g'][keep0] @ text_all[:, 0].t(), target[keep0]) if keep0.any()
                        else res['g'].sum() * 0)

        terms['lpim_id'] = self.xent(res['score_z0'], target) + self.xent(res['score_pbar'], target)
        z0_bank = p_bank = None
        if bank is not None:
            bz0, bzp, bvis, blabels = bank
            z0_bank = (bz0, blabels)
            p_bank = part_embeddings_for_lse(bz0, bzp, bvis) + (blabels,)
        terms['lpim_tri'] = self.triplet(res['z0'], target, z0_bank) + self.triplet(res['pbar'], target)

        attn_target, present = attention_targets(masks, res['grid'])
        p_lse, v_lse = part_embeddings_for_lse(res['z0'], res['zparts'], present)
        terms['part_tri'] = self.part_triplet(p_lse, v_lse, target, p_bank)

        text_b = text_all[target]
        supcon_terms = []
        for s_ in range(S):
            keep = erase[:, s_] if s_ == 0 else present[:, s_ - 1] & erase[:, s_]
            if keep.sum() < 2:
                continue
            z_s, t_s, tgt = res['z'][keep, s_].float(), text_b[keep, s_], target[keep]
            supcon_terms.append(supcon(z_s, t_s, tgt, tgt) + supcon(t_s, z_s, tgt, tgt))
        terms['supcon'] = sum(supcon_terms) / len(supcon_terms)

        attn = res['attn'][:, 1:].float().clamp(min=1e-8)
        kl = (attn_target * (attn_target.clamp(min=1e-8).log() - attn.log())).sum(-1)
        terms['attn'] = (kl * present).sum() / present.sum().clamp(min=1)
        terms['vis'] = F.binary_cross_entropy_with_logits(res['vis_logit'], present.float())

        total = (ID_W * terms['id'] + TRI_W * terms['tri'] + I2T_W * terms['i2t']
                 + LPIM_ID_W * terms['lpim_id'] + LPIM_TRI_W * terms['lpim_tri'] + PART_TRI_W * terms['part_tri']
                 + SUPCON_W * terms['supcon'] + ATTN_W * terms['attn'] + VIS_W * terms['vis'])
        return total, terms


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def extract(model, loader):
    model.eval()
    feats = {k: [] for k in ['gap4', 'g', 'z0', 'pbar', 'zparts', 'vis']}
    pids, camids = [], []
    for img, pid, camid, _, _, _ in loader:
        res = model(img.to(DEVICE))
        for k in ['gap4', 'g', 'z0', 'pbar', 'zparts']:
            feats[k].append(res[k].float().cpu())
        feats['vis'].append((res['vis_logit'] > 0).cpu())
        pids.extend(np.asarray(pid))
        camids.extend(np.asarray(camid))
    return {k: torch.cat(v) for k, v in feats.items()}, np.asarray(pids), np.asarray(camids)


@torch.no_grad()
def part_lse_distmat(qp, qv, gp, gv, gamma=LSE_GAMMA, chunk=EVAL_CHUNK):
    """Part-based query-gallery distance 1 - S_final, S_final = 1 - (1/gamma) ln sum_k w_k e^{gamma d_k}.
    Returns distmat [Nq,Ng] (no shared part -> max + 1), unmatched-part counts [Nq,Ng], shared mask [Nq,Ng]."""
    qp, qv = qp.to(DEVICE), qv.to(DEVICE)
    D, unmatched, valid = [], [], []
    for i in range(0, gp.shape[0], chunk):
        d, M = part_pairwise_distances(qp, qv, gp[i:i + chunk].to(DEVICE), gv[i:i + chunk].to(DEVICE))
        Dc, vc, uc = lse_combine(d, M, gamma)
        D.append(Dc.cpu()); valid.append(vc.cpu()); unmatched.append(uc.cpu())
    D, valid, unmatched = torch.cat(D, 1), torch.cat(valid, 1), torch.cat(unmatched, 1)
    D[~valid] = D[valid].max() + 1
    return D.numpy(), unmatched.numpy(), valid.numpy()


@torch.no_grad()
def part_lse_all_pairs(p, v, gamma=LSE_GAMMA, chunk=RERANK_CHUNK, device=None):
    """All-pairs part distance over query+gallery: [N, N] float16, same metric as part_lse_distmat.

    re_ranking adds `local_distmat` to its own all-pairs `original_dist` *before* the k-reciprocal
    neighbourhood is built, so the local matrix has to span query+gallery, not the [Nq, Ng] block that
    part_lse_distmat returns. Pairs with no mutually visible part get max + 1, as there. float16 and no
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
    distance instead of fusing two finished rankings (which is what FUSE_W does). The local matrix is rescaled to
    the mean of the global one first: re_ranking sums the two raw matrices and only normalises afterwards, while
    the global side is a squared euclidean on unit vectors (0..4) and the part side is 1 - cos (0..2);
    `global_dist` is the row's own [Nq, Ng] distmat, already computed by evaluate().
    """
    vec = {'clipreid_baseline': lambda f: F.normalize(torch.cat([f['gap4'], f['g']], dim=1), dim=1),
           'lpim': lambda f: concat_feature(f['z0'], f['pbar']),
           'holistic': lambda f: concat_feature(f['gap4'], f['g'], f['z0'], f['pbar'])}[RERANK_FEATURE](feats)
    device = RERANK_DEVICE or DEVICE
    qf, gf = vec[:num_query].to(device), vec[num_query:].to(device)
    rows, start = {}, time.time()
    rows[f'{RERANK_FEATURE}_rr'] = re_ranking(qf, gf, RERANK_K1, RERANK_K2, RERANK_LAMBDA)

    p, v = part_embeddings_for_lse(feats['z0'], feats['zparts'], feats['vis'])
    d_lse = part_lse_all_pairs(p, v)
    scale = RERANK_LOCAL_W * float(global_dist.mean()) / float(d_lse.astype(np.float32).mean())
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
    all slots weigh equally (concatenating then normalising once - CLIP-ReID's convention - weights slots by norm:
    measured at init the global slot carried ~9% of the distance against five parts). Normalised once at the end."""
    if EVAL_SLOTWISE_NORM:
        slots = [F.normalize(x, dim=-1) for x in slots]
    return F.normalize(torch.cat(slots, dim=1), dim=1)


def evaluate(model, val_loader, num_query, logger, tag, rerank=False):
    f, pids, camids = extract(model, val_loader)
    q, gal = slice(0, num_query), slice(num_query, None)
    q_pids, g_pids, q_cams, g_cams = pids[q], pids[gal], camids[q], camids[gal]

    def dist(vec):
        return euclidean_distance(vec[q].to(DEVICE), vec[gal].to(DEVICE))

    results = {
        'clipreid_baseline': dist(F.normalize(torch.cat([f['gap4'], f['g']], dim=1), dim=1)),
        'lpim': dist(concat_feature(f['z0'], f['pbar'])),
        'holistic': dist(concat_feature(f['gap4'], f['g'], f['z0'], f['pbar'])),
    }
    p_lse, v_lse = part_embeddings_for_lse(f['z0'], f['zparts'], f['vis'])
    d_lse, unmatched, shared = part_lse_distmat(p_lse[q], v_lse[q], p_lse[gal], v_lse[gal])
    results['part_lse'] = d_lse
    if FUSE_W > 0:
        results['fused'] = results['holistic'] / results['holistic'].mean() + FUSE_W * d_lse / d_lse.mean()
    if rerank:
        results.update(rerank_rows(f, num_query, results[RERANK_FEATURE], logger))

    logger.info(f'Validation Results - {tag}')
    logger.info('invisible rate per part | query: {} | gallery: {}'.format(
        dict(zip(PART_NAMES, (1 - f['vis'][q].float().mean(0)).numpy().round(3).tolist())),
        dict(zip(PART_NAMES, (1 - f['vis'][gal].float().mean(0)).numpy().round(3).tolist()))))
    logger.info('query-gallery pairs with >=1 unmatched part: {:.1%} | with no shared part: {:.2%} | mean unmatched parts: {:.2f}'.format(
        (unmatched > 0).mean(), (~shared).mean(), unmatched.mean()))
    summary = {}
    for name, distmat in results.items():
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_cams, g_cams)
        summary[name] = dict(mAP=mAP, R1=cmc[0], R5=cmc[4], R10=cmc[9])
        logger.info('[{:17s}] mAP: {:.1%}  Rank-1: {:.1%}  Rank-5: {:.1%}  Rank-10: {:.1%}'.format(name, mAP, cmc[0], cmc[4], cmc[9]))
    torch.cuda.empty_cache()
    return summary


# ----------------------------------------------------------------------------- setup
def build_text_targets(clip, num_classes, stage1_ckpt, logger):
    """Encode the stage-1 prompts once: text_all [C, K+1, 1024] (processor_clipreid_stage2.py:60-71)."""
    ckpt = torch.load(stage1_ckpt, map_location=DEVICE)
    knobs = ckpt['knobs']
    assert (knobs['H'], knobs['W'], knobs['STRIDE']) == (H, W, STRIDE), f'stage-1 knobs {knobs} != stage-2 {(H, W, STRIDE)}'
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
                                              LSE_INCLUDE_GLOBAL=LSE_INCLUDE_GLOBAL, PART_NAMES=PART_NAMES, MIM_SELF_LAYERS=MIM_SELF_LAYERS,
                                              BANK_SIZE=BANK_SIZE, EVAL_SLOTWISE_NORM=EVAL_SLOTWISE_NORM)}, path)


# ----------------------------------------------------------------------------- stage 2
def do_train_stage2(model, criterion, train_loader, val_loader, num_query, logger, start_epoch=1, resume=None):
    """processor_clipreid_stage2.py:73-137 with the part losses and the two-distance evaluation."""
    optimizer = make_optimizer(model)
    scheduler = WarmupMultiStepLR(optimizer, STEPS, GAMMA, WARMUP_FACTOR, WARMUP_ITERS, WARMUP_METHOD)
    if resume is not None:
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
    scaler = amp.GradScaler(enabled=USE_AMP)
    meters = {k: AverageMeter() for k in ['loss', 'id', 'tri', 'i2t', 'lpim_id', 'lpim_tri', 'part_tri', 'supcon', 'attn', 'vis', 'acc']}
    best = {}
    feat_bank = FeatureBank(BANK_SIZE, model.lpim.c_proj.out_features, K, DEVICE) if BANK_SIZE > 0 else None
    if feat_bank is not None:
        logger.info(f'cross-batch memory for triplet mining: {BANK_SIZE} slots, used from epoch {BANK_START_EPOCH}')
    all_start = time.monotonic()
    logger.info('start training')
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        for m in meters.values():
            m.reset()
        scheduler.step()
        criterion.part_triplet.gamma = lse_gamma_at(epoch)
        model.train()
        start = time.time()
        for n_iter, (img, masks, target, _) in enumerate(train_loader):
            optimizer.zero_grad()
            img, masks, target = img.to(DEVICE), masks.to(DEVICE), target.to(DEVICE)
            use_bank = feat_bank is not None and epoch >= BANK_START_EPOCH and feat_bank.filled > 0
            with amp.autocast(enabled=USE_AMP):
                res = model(img)
                loss, terms = criterion(res, masks, target, feat_bank.get() if use_bank else None)
            if feat_bank is not None:
                feat_bank.add(res['z0'], res['zparts'], res['vis_logit'] > 0, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            acc = (res['score_gap4'].max(1)[1] == target).float().mean()
            meters['loss'].update(loss.item(), img.shape[0])
            meters['acc'].update(acc.item(), 1)
            for k, v in terms.items():
                meters[k].update(v.item(), img.shape[0])
            if (n_iter + 1) % LOG_PERIOD == 0:
                m = {k: v.avg for k, v in meters.items()}
                logger.info('Epoch[{}] Iteration[{}/{}] Loss: {:.3f} (id {:.3f} tri {:.3f} i2t {:.3f} | lpim_id {:.3f} lpim_tri {:.3f} '
                            'part_tri {:.3f} supcon {:.3f} attn {:.3f} vis {:.3f}) Acc: {:.3f}, Base Lr: {:.2e}, LSE gamma: {:.2f}'
                            .format(epoch, n_iter + 1, len(train_loader), m['loss'], m['id'], m['tri'], m['i2t'], m['lpim_id'],
                                    m['lpim_tri'], m['part_tri'], m['supcon'], m['attn'], m['vis'], m['acc'],
                                    scheduler.get_lr()[0], criterion.part_triplet.gamma))
        time_per_batch = (time.time() - start) / (n_iter + 1)
        logger.info('Epoch {} done. Loss: {:.3f} Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(epoch, meters['loss'].avg, time_per_batch, train_loader.batch_size / time_per_batch))
        if epoch % CHECKPOINT_PERIOD == 0 or epoch == MAX_EPOCHS:
            path = os.path.join(OUTPUT_DIR, f'{BACKBONE}_part_stage2_{epoch}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, path)
            logger.info(f'saved {path}')
        if epoch % EVAL_PERIOD == 0 or epoch == MAX_EPOCHS:
            summary = evaluate(model, val_loader, num_query, logger, f'Epoch: {epoch}',
                               rerank=RERANK and (RERANK_EVERY_EVAL or epoch == MAX_EPOCHS))
            for name, m in summary.items():
                if m['mAP'] > best.get(name, {'mAP': -1})['mAP']:
                    best[name] = dict(epoch=epoch, **m)
            logger.info('best so far: ' + ' | '.join(
                '{} mAP {:.1%} R1 {:.1%} @epoch {}'.format(n, b['mAP'], b['R1'], b['epoch']) for n, b in best.items()))
    logger.info('Total running time: {}'.format(timedelta(seconds=time.monotonic() - all_start)))


def main():
    global MAX_EPOCHS, IMS_PER_BATCH, EVAL_PERIOD, TEXT_ERASE_PROB, TEXT_DROPOUT
    global RERANK, RERANK_FEATURE, RERANK_K1, RERANK_K2, RERANK_LAMBDA, RERANK_LOCAL_W
    global RERANK_ONLY_LOCAL, RERANK_EVERY_EVAL, RERANK_DEVICE
    parser = argparse.ArgumentParser(description='CLIP-ReID stage 2 with per-part prompts (RN50)')
    parser.add_argument('--stage1-ckpt', type=str, default=STAGE1_CKPT)
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
    parser.add_argument('--rerank', action='store_true', help='k-reciprocal re-ranking rows (test-time only)')
    parser.add_argument('--rerank-feature', choices=['holistic', 'lpim', 'clipreid_baseline'], default=RERANK_FEATURE)
    parser.add_argument('--rerank-k1', type=int, default=RERANK_K1)
    parser.add_argument('--rerank-k2', type=int, default=RERANK_K2)
    parser.add_argument('--rerank-lambda', type=float, default=RERANK_LAMBDA)
    parser.add_argument('--rerank-local-w', type=float, default=RERANK_LOCAL_W, help='0 = plain k-reciprocal twice')
    parser.add_argument('--rerank-only-local', action='store_true', default=RERANK_ONLY_LOCAL)
    parser.add_argument('--rerank-every-eval', action='store_true', default=RERANK_EVERY_EVAL)
    parser.add_argument('--rerank-device', type=str, default=RERANK_DEVICE, help="'cpu' if the [N,N] matmul will not fit")
    args = parser.parse_args()
    MAX_EPOCHS, IMS_PER_BATCH, EVAL_PERIOD = args.epochs, args.batch, args.eval_period
    TEXT_ERASE_PROB, TEXT_DROPOUT = args.text_erase_prob, args.text_dropout
    RERANK, RERANK_FEATURE, RERANK_K1, RERANK_K2 = args.rerank, args.rerank_feature, args.rerank_k1, args.rerank_k2
    RERANK_LAMBDA, RERANK_LOCAL_W = args.rerank_lambda, args.rerank_local_w
    RERANK_ONLY_LOCAL, RERANK_EVERY_EVAL, RERANK_DEVICE = args.rerank_only_local, args.rerank_every_eval, args.rerank_device

    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = setup_logger('transreid', OUTPUT_DIR, if_train=not args.eval_only)
    logger.info('knobs: ' + ', '.join(f'{k}={v}' for k, v in dict(
        H=H, W=W, IMS_PER_BATCH=IMS_PER_BATCH, NUM_INSTANCE=NUM_INSTANCE, MAX_EPOCHS=MAX_EPOCHS, BASE_LR=BASE_LR,
        STEPS=STEPS, ID_W=ID_W, TRI_W=TRI_W, I2T_W=I2T_W, LPIM_ID_W=LPIM_ID_W, LPIM_TRI_W=LPIM_TRI_W, PART_TRI_W=PART_TRI_W,
        SUPCON_W=SUPCON_W, ATTN_W=ATTN_W, VIS_W=VIS_W, TEXT_ERASE_PROB=TEXT_ERASE_PROB, TEXT_DROPOUT=TEXT_DROPOUT, LSE_GAMMA_EPOCHS=LSE_GAMMA_EPOCHS, MIM_SELF_LAYERS=MIM_SELF_LAYERS,
        MARGIN=MARGIN, LSE_GAMMA=LSE_GAMMA, LSE_INCLUDE_GLOBAL=LSE_INCLUDE_GLOBAL, FUSE_W=FUSE_W, USE_AMP=USE_AMP,
        BANK_SIZE=BANK_SIZE, BANK_START_EPOCH=BANK_START_EPOCH, EVAL_SLOTWISE_NORM=EVAL_SLOTWISE_NORM,
        RERANK=RERANK, RERANK_FEATURE=RERANK_FEATURE, RERANK_K1=RERANK_K1, RERANK_K2=RERANK_K2,
        RERANK_LAMBDA=RERANK_LAMBDA, RERANK_LOCAL_W=RERANK_LOCAL_W, RERANK_ONLY_LOCAL=RERANK_ONLY_LOCAL,
        RERANK_EVERY_EVAL=RERANK_EVERY_EVAL).items()))

    dataset = Market1501(root=DATA_ROOT)
    num_classes, num_query = dataset.num_train_pids, len(dataset.query)
    train_loader, val_loader = make_loaders(dataset, IMS_PER_BATCH)

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
        evaluate(model, val_loader, num_query, logger, f"{args.weights} (epoch {ckpt['epoch']})", rerank=RERANK)
        return

    resume, start_epoch = None, 1
    if args.resume:
        resume = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(resume['model'])
        start_epoch = resume['epoch'] + 1
        logger.info(f"resuming from {args.resume} (epoch {resume['epoch']})")
    do_train_stage2(model, criterion, train_loader, val_loader, num_query, logger, start_epoch, resume)


if __name__ == '__main__':
    main()
