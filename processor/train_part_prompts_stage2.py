"""CLIP-ReID stage 2 (image-side fine-tuning) with learnt per-part prompts on PartAwareModifiedResNet (RN50).

Run from the repo root:
  python processor/train_part_prompts_stage2.py --stage1-ckpt work_dirs/market1501/part_prompts_stage1/RN50_part_prompts_stage1_60.pth

What stage 2 does here
* Image side (trainable): pretrained CLIP RN50 + BPAM pixel classifier (PartAwareModifiedResNet) + part
  projection Linear(2048->1024) (initialised from CLIP's attention pool c_proj @ v_proj) + BNNeck ID heads.
  Part pooling uses the *learned* BPAM masks (BPBreID learnable attention); PifPaf masks are only the
  supervision target of the pixel classifier, so nothing external is needed at test time.
* Per image: g = xproj[0] (global, aligned with the global prompt), p_k = part_proj(gwap(x4, mask_k))
  (aligned with the part prompts), c = concat(p_1..p_K) (global_concat_vector), h = cat(g, c) (holistic
  vector), visibility [N, K+1] (slot 0 always visible).
* Text side (frozen): the stage-1 prompts are encoded once into text_all [C, K+1, 1024].
* Losses (batch-based negatives, PK sampler): label-smoothed ID on g and c (+ optional per-part ID),
  global triplet on g (CLIP-ReID), part triplet where per-part cosine distances are combined with a
  log-sum-exp soft-max over mutually visible parts (one bad part pulls the pair distance up), per-slot
  i2t cross-entropy against text_all, and the BPBreID pixel-part cross-entropy.
* Evaluation (every EVAL_PERIOD epochs): mAP / R1 / R5 / R10 for the holistic vector h (euclidean on the
  normalised concat, CLIP-ReID style) and for the part-based LSE distance (mutual-visibility weights,
  pairs with no shared part get the worst distance); unmatched-part statistics are logged.
* Optimiser / schedule / AMP = SOLVER.STAGE2 of configs/person/cnn_clipreid.yml.
"""
import os
import sys
import math
import time
import random
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

from processor.train_part_prompts_stage1 import (PartAwareModifiedResNet, PartPromptLearner, encode_text,
                                                  pifpaf_to_masks, PART_NAMES, K, S, SLOT_NAMES)
from model.make_model_clipreid import load_clip_to_cpu, TextEncoder, weights_init_kaiming, weights_init_classifier
from datasets.market1501 import Market1501
from datasets.sampler import RandomIdentitySampler
from datasets.bases import ImageDataset, read_image
from datasets.make_dataloader_clipreid import val_collate_fn
from loss.softmax_loss import CrossEntropyLabelSmooth
from loss.triplet_loss import euclidean_dist
from solver.lr_scheduler import WarmupMultiStepLR
from utils.metrics import eval_func, euclidean_distance
from utils.logger import setup_logger
from utils.meter import AverageMeter

# ----------------------------------------------------------------------------- knobs
DATA_ROOT = '../../datasets'
MASKS_DIR = f'{DATA_ROOT}/market1501/masks/pifpaf_maskrcnn_filtering/bounding_box_train'
OUTPUT_DIR = './work_dirs/market1501/part_prompts_stage2'
STAGE1_CKPT = './work_dirs/market1501/part_prompts_stage1/RN50_part_prompts_stage1_60.pth'

BACKBONE = 'RN50'
H, W = 384, 128                    # must match the stage-1 checkpoint (checked at load time)
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

ID_W, TRI_W, PART_TRI_W, PART_ID_W, I2T_W, PIX_W = 1.0, 1.0, 1.0, 0.0, 1.0, 0.35
MARGIN = 0.3
LSE_GAMMA = 5.0                    # soft-max sharpness over parts (-> max distance as gamma grows)
LSE_INCLUDE_GLOBAL = False         # add g as slot 0 of the LSE part distance
FUSE_W = 0.0                       # >0: also evaluate d_holistic/mean + FUSE_W * d_lse/mean

BANK_SIZE = 8192                   # cross-batch memory for triplet mining (0 = batch-only negatives)
BANK_START_EPOCH = 5               # epochs of batch-only mining before the bank is used (features settle first)

EVAL_SLOTWISE_NORM = True          # holistic vector: L2-normalise each slot before concatenating (see
                                   # holistic_vector). False restores the CLIP-ReID convention of a single
                                   # normalisation over the concatenation. Eval-only, no retraining needed.
EVAL_CHUNK = 2048
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


class PartCLIPReID(nn.Module):
    """Trainable image side. forward(x) -> dict(g, p, c, vis, pixels_cls_scores, parts_masks[, score_g, score_c, score_p])."""

    def __init__(self, visual, num_classes):
        super().__init__()
        self.net = PartAwareModifiedResNet(visual)
        ap = visual.attnpool
        self.part_proj = nn.Linear(ap.v_proj.in_features, ap.c_proj.out_features)
        with torch.no_grad():
            self.part_proj.weight.copy_(ap.c_proj.weight @ ap.v_proj.weight)
            self.part_proj.bias.copy_(ap.c_proj.weight @ ap.v_proj.bias + ap.c_proj.bias)
        dim = ap.c_proj.out_features
        self.id_global = BNNeckHead(dim, num_classes)
        self.id_concat = BNNeckHead(K * dim, num_classes)
        self.id_parts = nn.ModuleList([BNNeckHead(dim, num_classes) for _ in range(K)])

    def forward(self, x):
        _, _, xproj, out = self.net(x)
        g = xproj[0]
        p = self.part_proj(out['part_emb_x4'])
        c = p.flatten(1)
        vis = out['visibility'].clone()
        vis[:, 0] = True
        res = dict(g=g, p=p, c=c, vis=vis, pixels_cls_scores=out['pixels_cls_scores'], parts_masks=out['parts_masks'])
        if self.training:
            res['score_g'] = self.id_global(g)[1]
            res['score_c'] = self.id_concat(c)[1]
            if PART_ID_W > 0:
                res['score_p'] = torch.stack([head(p[:, k])[1] for k, head in enumerate(self.id_parts)], dim=1)
        return res


# ----------------------------------------------------------------------------- part distances / losses
def lse_combine(d, M, gamma):
    """Combine per-part distances with a log-sum-exp soft-max over mutually visible parts.
    d, M: [K, A, B] distances and 0/1 mutual-visibility. Returns D [A,B] = (1/gamma) ln sum_k w_k e^{gamma d_k}
    with w_k = M_k / sum_k M_k, valid [A,B] (>= 1 shared part; invalid entries are -1) and the number of
    unmatched parts per pair [A,B]."""
    n_shared = M.sum(0)
    valid = n_shared > 0
    w = M / n_shared.clamp(min=1)[None]
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
    BANK_SIZE embeddings (g, p, visibility, label), so anchors are mined against the whole dataset: with
    8192 slots and 12936 training images, a batch is compared against ~2/3 of Market-1501. Bank entries are
    detached (gradient flows only through the anchors), which is what makes the large pool affordable.
    """

    def __init__(self, size, dim, parts, device):
        self.size = size
        self.g = torch.zeros(size, dim, device=device)
        self.p = torch.zeros(size, parts, dim, device=device)
        self.vis = torch.zeros(size, parts + 1, dtype=torch.bool, device=device)
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


def part_embeddings_for_lse(g, p, vis):
    """[N,K,D] and [N,K] visibility used by the LSE distance; optionally includes g as slot 0."""
    if LSE_INCLUDE_GLOBAL:
        return torch.cat([g[:, None], p], dim=1), vis
    return p, vis[:, 1:]


class Stage2Loss(nn.Module):
    """loss = ID_W*(ID_g + ID_c [+ PART_ID_W*ID_parts]) + TRI_W*TRI_g + PART_TRI_W*TRI_lse + I2T_W*I2T + PIX_W*PIX."""

    def __init__(self, num_classes, text_all):
        super().__init__()
        self.xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        self.triplet = GlobalTripletLoss(MARGIN)
        self.part_triplet = PartLSETripletLoss()
        self.register_buffer('text_all', text_all)

    def forward(self, res, masks, target, bank=None):
        terms = {}
        terms['id'] = self.xent(res['score_g'], target) + self.xent(res['score_c'], target)
        if PART_ID_W > 0:
            part_id = []
            for k in range(K):
                keep = res['vis'][:, k + 1]
                if keep.sum() > 1:
                    part_id.append(self.xent(res['score_p'][keep, k], target[keep]))
            terms['id'] = terms['id'] + PART_ID_W * (sum(part_id) / max(len(part_id), 1))
        g_bank = p_bank = None
        if bank is not None:
            bg, bp, bvis, blabels = bank
            g_bank = (bg, blabels)
            bp_lse, bv_lse = part_embeddings_for_lse(bg, bp, bvis)
            p_bank = (bp_lse, bv_lse, blabels)
        terms['tri'] = self.triplet(res['g'], target, g_bank)
        p_lse, v_lse = part_embeddings_for_lse(res['g'], res['p'], res['vis'])
        terms['part_tri'] = self.part_triplet(p_lse, v_lse, target, p_bank)
        i2t = []
        for s in range(S):
            keep = res['vis'][:, s]
            if keep.sum() < 2:
                continue
            img_s = res['g'] if s == 0 else res['p'][:, s - 1]
            i2t.append(self.xent(img_s[keep] @ self.text_all[:, s].t(), target[keep]))
        terms['i2t'] = sum(i2t) / len(i2t)
        scores = res['pixels_cls_scores']
        pix_targets = F.interpolate(masks, scores.shape[2:], mode='bilinear', align_corners=True).argmax(1)
        terms['pix'] = F.cross_entropy(scores.permute(0, 2, 3, 1).flatten(0, 2), pix_targets.flatten(), label_smoothing=0.1)
        total = ID_W * terms['id'] + TRI_W * terms['tri'] + PART_TRI_W * terms['part_tri'] + I2T_W * terms['i2t'] + PIX_W * terms['pix']
        return total, terms


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def extract(model, loader):
    model.eval()
    g, p, vis, pids, camids = [], [], [], [], []
    for img, pid, camid, _, _, _ in loader:
        res = model(img.to(DEVICE))
        g.append(res['g'].float().cpu())
        p.append(res['p'].float().cpu())
        vis.append(res['vis'].cpu())
        pids.extend(np.asarray(pid))
        camids.extend(np.asarray(camid))
    return torch.cat(g), torch.cat(p), torch.cat(vis), np.asarray(pids), np.asarray(camids)


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


def holistic_vector(g, p):
    """All slots in one vector, matched with a single distance (no visibility logic) - the CLIP-ReID style
    of matching applied to the part-aware features.

    Concatenating first and normalising once (CLIP-ReID's convention) makes each slot's influence
    proportional to its norm: measured at init the global slot carries ~9% of the distance and the five
    parts ~91%, even though the global slot is individually the strongest. Normalising per slot first
    gives all six an equal 1/6 share; on a 1-epoch checkpoint that was worth +0.5 mAP / +1.1 Rank-1.
    """
    if EVAL_SLOTWISE_NORM:
        h = torch.cat([F.normalize(g, dim=-1), F.normalize(p, dim=-1).flatten(1)], dim=1)
    else:
        h = torch.cat([g, p.flatten(1)], dim=1)
    return F.normalize(h, dim=1)


def evaluate(model, val_loader, num_query, logger, tag):
    g, p, vis, pids, camids = extract(model, val_loader)
    q, gal = slice(0, num_query), slice(num_query, None)
    q_pids, g_pids, q_cams, g_cams = pids[q], pids[gal], camids[q], camids[gal]

    h = holistic_vector(g, p)
    d_h = euclidean_distance(h[q].to(DEVICE), h[gal].to(DEVICE))
    p_lse, v_lse = part_embeddings_for_lse(g, p, vis)
    d_lse, unmatched, shared = part_lse_distmat(p_lse[q], v_lse[q], p_lse[gal], v_lse[gal])

    results = {'holistic': d_h, 'part_lse': d_lse}
    if FUSE_W > 0:
        results['fused'] = d_h / d_h.mean() + FUSE_W * d_lse / d_lse.mean()
    logger.info(f'Validation Results - {tag}')
    logger.info('invisible rate per slot | query: {} | gallery: {}'.format(
        dict(zip(SLOT_NAMES, (1 - vis[q].float().mean(0)).numpy().round(3).tolist())),
        dict(zip(SLOT_NAMES, (1 - vis[gal].float().mean(0)).numpy().round(3).tolist()))))
    logger.info('query-gallery pairs with >=1 unmatched part: {:.1%} | with no shared part: {:.2%} | mean unmatched parts: {:.2f}'.format(
        (unmatched > 0).mean(), (~shared).mean(), unmatched.mean()))
    summary = {}
    for name, distmat in results.items():
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_cams, g_cams)
        summary[name] = dict(mAP=mAP, R1=cmc[0], R5=cmc[4], R10=cmc[9])
        logger.info('[{:9s}] mAP: {:.1%}  Rank-1: {:.1%}  Rank-5: {:.1%}  Rank-10: {:.1%}'.format(name, mAP, cmc[0], cmc[4], cmc[9]))
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
                                              LSE_INCLUDE_GLOBAL=LSE_INCLUDE_GLOBAL, PART_NAMES=PART_NAMES,
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
    meters = {k: AverageMeter() for k in ['loss', 'id', 'tri', 'part_tri', 'i2t', 'pix', 'acc']}
    best = {}
    feat_bank = FeatureBank(BANK_SIZE, model.part_proj.out_features, K, DEVICE) if BANK_SIZE > 0 else None
    if feat_bank is not None:
        logger.info(f'cross-batch memory for triplet mining: {BANK_SIZE} slots, used from epoch {BANK_START_EPOCH}')
    all_start = time.monotonic()
    logger.info('start training')
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        for m in meters.values():
            m.reset()
        scheduler.step()
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
                feat_bank.add(res['g'], res['p'], res['vis'], target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            acc = (res['score_g'].max(1)[1] == target).float().mean()
            meters['loss'].update(loss.item(), img.shape[0])
            meters['acc'].update(acc.item(), 1)
            for k, v in terms.items():
                meters[k].update(v.item(), img.shape[0])
            if (n_iter + 1) % LOG_PERIOD == 0:
                logger.info('Epoch[{}] Iteration[{}/{}] Loss: {:.3f} (id {:.3f} tri {:.3f} part_tri {:.3f} i2t {:.3f} pix {:.3f}) Acc: {:.3f}, Base Lr: {:.2e}'
                            .format(epoch, n_iter + 1, len(train_loader), meters['loss'].avg, meters['id'].avg, meters['tri'].avg,
                                    meters['part_tri'].avg, meters['i2t'].avg, meters['pix'].avg, meters['acc'].avg, scheduler.get_lr()[0]))
        time_per_batch = (time.time() - start) / (n_iter + 1)
        logger.info('Epoch {} done. Loss: {:.3f} Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(epoch, meters['loss'].avg, time_per_batch, train_loader.batch_size / time_per_batch))
        if epoch % CHECKPOINT_PERIOD == 0 or epoch == MAX_EPOCHS:
            path = os.path.join(OUTPUT_DIR, f'{BACKBONE}_part_stage2_{epoch}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, path)
            logger.info(f'saved {path}')
        if epoch % EVAL_PERIOD == 0 or epoch == MAX_EPOCHS:
            summary = evaluate(model, val_loader, num_query, logger, f'Epoch: {epoch}')
            for name, m in summary.items():
                if m['mAP'] > best.get(name, {'mAP': -1})['mAP']:
                    best[name] = dict(epoch=epoch, **m)
            logger.info('best so far: ' + ' | '.join(
                '{} mAP {:.1%} R1 {:.1%} @epoch {}'.format(n, b['mAP'], b['R1'], b['epoch']) for n, b in best.items()))
    logger.info('Total running time: {}'.format(timedelta(seconds=time.monotonic() - all_start)))


def main():
    global MAX_EPOCHS, IMS_PER_BATCH, EVAL_PERIOD
    parser = argparse.ArgumentParser(description='CLIP-ReID stage 2 with per-part prompts (RN50)')
    parser.add_argument('--stage1-ckpt', type=str, default=STAGE1_CKPT)
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--batch', type=int, default=IMS_PER_BATCH)
    parser.add_argument('--eval-period', type=int, default=EVAL_PERIOD)
    parser.add_argument('--resume', type=str, default='', help='stage-2 checkpoint to continue from')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--weights', type=str, default='', help='stage-2 checkpoint for --eval-only')
    args = parser.parse_args()
    MAX_EPOCHS, IMS_PER_BATCH, EVAL_PERIOD = args.epochs, args.batch, args.eval_period

    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    torch.backends.cudnn.benchmark = True
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = setup_logger('transreid', OUTPUT_DIR, if_train=not args.eval_only)
    logger.info('knobs: ' + ', '.join(f'{k}={v}' for k, v in dict(
        H=H, W=W, IMS_PER_BATCH=IMS_PER_BATCH, NUM_INSTANCE=NUM_INSTANCE, MAX_EPOCHS=MAX_EPOCHS, BASE_LR=BASE_LR,
        STEPS=STEPS, ID_W=ID_W, TRI_W=TRI_W, PART_TRI_W=PART_TRI_W, PART_ID_W=PART_ID_W, I2T_W=I2T_W, PIX_W=PIX_W,
        MARGIN=MARGIN, LSE_GAMMA=LSE_GAMMA, LSE_INCLUDE_GLOBAL=LSE_INCLUDE_GLOBAL, FUSE_W=FUSE_W, USE_AMP=USE_AMP,
        BANK_SIZE=BANK_SIZE, BANK_START_EPOCH=BANK_START_EPOCH, EVAL_SLOTWISE_NORM=EVAL_SLOTWISE_NORM).items()))

    dataset = Market1501(root=DATA_ROOT)
    num_classes, num_query = dataset.num_train_pids, len(dataset.query)
    train_loader, val_loader = make_loaders(dataset, IMS_PER_BATCH)

    h_res, w_res = (H - 16) // STRIDE + 1, (W - 16) // STRIDE + 1
    clip = load_clip_to_cpu(BACKBONE, h_res, w_res, STRIDE).to(DEVICE)
    text_all = build_text_targets(clip, num_classes, args.stage1_ckpt, logger)
    model = PartCLIPReID(clip.visual, num_classes).to(DEVICE)
    del clip
    torch.cuda.empty_cache()
    criterion = Stage2Loss(num_classes, text_all).to(DEVICE)
    logger.info('trainable parameters: {:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if args.eval_only:
        ckpt = torch.load(args.weights, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        evaluate(model, val_loader, num_query, logger, f"{args.weights} (epoch {ckpt['epoch']})")
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
