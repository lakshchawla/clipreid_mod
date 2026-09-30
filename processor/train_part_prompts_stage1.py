"""CLIP-ReID stage 1 (prompt learning) with per-part prompts on top of PartAwareModifiedResNet (RN50).

Self-contained port of main.ipynb: every module used by the notebook lives in this file.
Run from the repo root:  python processor/train_part_prompts_stage1.py

What stage 1 does here
* Image side is frozen: the pretrained CLIP RN50 (ModifiedResNet) gives x4 and xproj; PifPaf masks from
  MASKS_DIR are used as the part attention (BPBreID learnable_attention_enabled=False); slot 0 = xproj[0]
  (stock CLIP-ReID global feature), slot k = GWAP(x4, mask_k) -> frozen Linear(2048->1024) derived from
  CLIP's attention pool (c_proj @ v_proj). Features and visibility are extracted once and cached, exactly
  like processor_clipreid_stage1.py does for the global feature.
* Text side: one shared frozen CLIP text encoder; PartPromptLearner owns cls_ctx[num_class, K+1, 4, 512]
  (CLIP-ReID's 4 ctx tokens, one set per slot). These are the only trained parameters.
* Loss per batch = sum over slots of SupCon(img_k, text_k) + SupCon(text_k, img_k) (repo SupConLoss),
  each slot encoded and back-propagated on its own (same gradients, 1/(K+1) of the memory); images whose
  part k is invisible are dropped from slot k's loss.
* Optimiser / schedule = SOLVER.STAGE1 of configs/person/cnn_clipreid.yml (Adam, warmup-cosine).
"""
import os
import re
import sys
import glob
import time
import argparse
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from model.make_model_clipreid import load_clip_to_cpu, TextEncoder
from model.clip.clip import tokenize
from loss.supcontrast import SupConLoss
from solver.scheduler_factory import create_scheduler
from datasets.part_datasets import DATASETS, MASK_SUFFIX, build_dataset, mask_path, resolve_masks
from utils.logger import setup_logger
from utils.meter import AverageMeter

# ----------------------------------------------------------------------------- knobs
DATA_ROOT = '../../datasets'
DATASET = 'market1501'             # market1501 | msmt17 | dukemtmc (datasets/part_datasets.py); --dataset overrides.
                                   # The train-split images and their PifPaf masks are resolved from the name: masks
                                   # live at <dataset_dir>/masks/pifpaf_maskrcnn_filtering/<image path relative to
                                   # <dataset_dir>>.npy. Only Market-1501 ships them; for the others that path is
                                   # where reid_masks/compute_masks.py writes.
MASKS_VARIANT = 'pifpaf_maskrcnn_filtering'    # pre-saved BPBreID mask set to read; 'pifpaf' is the unfiltered one
MASKS_DIR = None                   # None = <dataset_dir>/masks/<MASKS_VARIANT>; set (or --masks-dir) for a mask set
                                   # kept outside the dataset directory
OUTPUT_DIR = './work_dirs/{dataset}/part_prompts_stage1'

BACKBONE = 'RN50'
H, W = 256, 128                    # CLIP-ReID RN50 recipe (cnn_clipreid.yml); 384x128 is an ablation, stage 2 must match
STRIDE = 16                        # MODEL.STRIDE_SIZE
PIXEL_MEAN, PIXEL_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
N_CTX = 4                          # learnable ctx tokens per (identity, slot)
MASK_SOFTMAX_WEIGHT, MASK_BG_THRESHOLD = 15, 0.5

MAX_EPOCHS = 60                    # SOLVER.STAGE1 of configs/person/cnn_clipreid.yml
IMS_PER_BATCH = 64
BASE_LR = 5e-4                     # 3.5e-4 is the CLIP-ReID value. 1e-3 won short (12-epoch) sweeps but on
                                   # the 60-epoch schedule it sits near peak for ~15 epochs and the loss
                                   # rises there (server log, epochs 7-24); 5e-4 keeps the gain without that.
WARMUP_LR_INIT = 1e-5
LR_MIN = 1e-6
WARMUP_EPOCHS = 5
WEIGHT_DECAY = 1e-4
EXTRACT_BATCH = 64
CHECKPOINT_PERIOD = 10
EVAL_PERIOD = 10
LOG_PERIOD = 50
FULL_POOL_NEGATIVES = True         # contrast against the whole dataset instead of the batch (see do_train_stage1)
T2I_BATCH_POSITIVES = True         # t2i positives = the identity's in-batch images only; its other images are
                                   # masked out of the softmax (neither positive nor negative). False = SupCon
                                   # over all of the identity's images, whose floor is ln(#positives).
TEXT_BANK_REFRESH = 1              # epochs between full prompt-bank rebuilds (only used when FULL_POOL_NEGATIVES)
TEXT_BANK_MOMENTUM = 0.0           # in-place write-back of freshly encoded prompts: bank = m*old + (1-m)*new;
                                   # 0 = replace. Keeps every bank row at most ~C/ids-per-batch iterations stale.
CONTRAST_NORMALIZE = True          # L2-normalise + temperature in the full-pool contrast (see supcon)
CONTRAST_TEMP = 0.01               # CLIP's own temperature; swept against 0.03/0.07/unnormalised (see docs below)
TAB_ENABLED = False                # Text Attention Block: image-as-query cross-attention over the prompt tokens (see TextAttentionBlock)
TAB_W = 0.5                        # weight of the auxiliary TAB contrast; the plain-EOT contrast stays the primary loss
TAB_DIM, TAB_HEADS = 256, 4
TAB_T2I_POOL = 2048                # sampled negative images for the TAB t2i term (conditioning over the full pool is ~60 GFLOP/slot)
NUM_IDS = None                     # None = all identities; an int limits to the first N (quick runs)
SEED = 1234
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------------------------------------------------------------- PifPaf masks (BPBreID)
PIFPAF_KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
                    "right_knee", "left_ankle", "right_ankle"]
PIFPAF_JOINTS = ["left_ankle_to_left_knee", "left_knee_to_left_hip", "right_ankle_to_right_knee",
                 "right_knee_to_right_hip", "left_hip_to_right_hip", "left_shoulder_to_left_hip",
                 "right_shoulder_to_right_hip", "left_shoulder_to_right_shoulder", "left_shoulder_to_left_elbow",
                 "right_shoulder_to_right_elbow", "left_elbow_to_left_wrist", "right_elbow_to_right_wrist",
                 "left_eye_to_right_eye", "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear",
                 "right_eye_to_right_ear", "left_ear_to_left_shoulder", "right_ear_to_right_shoulder"]
PIFPAF_PARTS_MAP = {k: i for i, k in enumerate(PIFPAF_KEYPOINTS + PIFPAF_JOINTS)}
FIVE_V = {
    "head": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_eye_to_right_eye",
             "nose_to_left_eye", "nose_to_right_eye", "left_eye_to_left_ear", "right_eye_to_right_ear",
             "left_ear_to_left_shoulder", "right_ear_to_right_shoulder"],
    "upper_arms_torso": ["left_elbow", "right_elbow", "left_shoulder_to_left_elbow", "right_shoulder_to_right_elbow",
                         "left_shoulder", "right_shoulder", "left_shoulder_to_right_shoulder"],
    "lower_arms_torso": ["left_wrist", "right_wrist", "left_elbow_to_left_wrist", "right_elbow_to_right_wrist",
                         "left_hip", "right_hip", "right_shoulder_to_right_hip"],
    "legs": ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle_to_left_knee", "left_knee_to_left_hip",
             "right_ankle_to_right_knee", "right_knee_to_right_hip"],
    "feet": ["left_ankle", "right_ankle"],
}
PART_NAMES = list(FIVE_V.keys())
K = len(PART_NAMES)
SLOT_NAMES = ['global'] + PART_NAMES
S = K + 1


def pifpaf_to_masks(raw, softmax_weight=MASK_SOFTMAX_WEIGHT, threshold=MASK_BG_THRESHOLD):
    """[36,H,W] PifPaf confidence fields -> [K+1,H,W] soft part distribution, channel 0 = background
    (BPBreID MaskGroupingTransform 'five_v' + AddBackgroundMask('threshold'))."""
    raw = torch.as_tensor(raw, dtype=torch.float32)
    parts = torch.stack([raw[[PIFPAF_PARTS_MAP[k] for k in group]].max(0)[0].clamp(0, 1) for group in FIVE_V.values()])
    background = (parts.max(0)[0] < threshold).float()
    masks = torch.cat([background[None], parts])
    return F.softmax(masks * softmax_weight, dim=0)


# ----------------------------------------------------------------------------- BPAM blocks (BPBreID)
class PixelToPartClassifier(nn.Module):
    """Pixel -> body-part classifier (bpbreid.py:376-396): BN2d + 1x1 conv to K+1 logits (0 = background)."""
    def __init__(self, dim, parts_num):
        super().__init__()
        self.bn = nn.BatchNorm2d(dim)
        self.classifier = nn.Conv2d(dim, parts_num + 1, kernel_size=1)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
        nn.init.normal_(self.classifier.weight, 0, 0.001)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        return self.classifier(self.bn(x))


def gwap(feat, masks):
    """Global weighted average pooling (bpbreid.py:489-503). feat [N,D,H,W], masks [N,M,H,W] -> [N,M,D]."""
    num = torch.einsum('nmhw,ndhw->nmd', masks, feat)
    den = masks.sum((2, 3)).clamp(min=1e-6)[..., None]
    return num / den


def binary_visibility(probs):
    """Binary part visibility (bpbreid.py:181-192): visible iff the part wins the argmax at >= 1 location."""
    one_hot = F.one_hot(probs.argmax(1), probs.shape[1]).permute(0, 3, 1, 2)
    return one_hot.amax((2, 3)).bool()


# ----------------------------------------------------------------------------- image side
class PartAwareModifiedResNet(nn.Module):
    """CLIP-ReID RN50 image encoder + BPBreID body-part attention (BPAM); see main.ipynb.

    Returns the unchanged (x3, x4, xproj) of ModifiedResNet plus a dict with BPAM masks, visibility,
    GWAP part embeddings on x4 and part-aware attention-pooled CLIP embeddings. In stage 1 only
    part_emb_x4, visibility and xproj are consumed.
    """

    def __init__(self, backbone, parts_num=K):
        super().__init__()
        self.backbone = backbone
        self.parts_num = parts_num
        self.pixel_classifier = PixelToPartClassifier(backbone.attnpool.k_proj.in_features, parts_num)

    def part_aware_attnpool(self, x4, queries):
        ap = self.backbone.attnpool
        loc = x4.flatten(2).permute(2, 0, 1)
        pos = ap.positional_embedding[:, None, :].to(loc.dtype)
        q = queries + pos[:1]
        kv = torch.cat([q, loc + pos[1:]], dim=0)
        Q = q.shape[0]
        attn_mask = torch.zeros(Q, kv.shape[0], device=x4.device, dtype=loc.dtype)
        attn_mask[:, :Q] = float('-inf')
        attn_mask.fill_diagonal_(0)
        out, _ = F.multi_head_attention_forward(
            query=q, key=kv, value=kv,
            embed_dim_to_check=q.shape[-1],
            num_heads=ap.num_heads,
            q_proj_weight=ap.q_proj.weight,
            k_proj_weight=ap.k_proj.weight,
            v_proj_weight=ap.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([ap.q_proj.bias, ap.k_proj.bias, ap.v_proj.bias]),
            bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0,
            out_proj_weight=ap.c_proj.weight,
            out_proj_bias=ap.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
            attn_mask=attn_mask,
        )
        return out

    def forward(self, x, external_masks=None):
        b = self.backbone
        x = x.type(b.conv1.weight.dtype)
        for conv, bn in [(b.conv1, b.bn1), (b.conv2, b.bn2), (b.conv3, b.bn3)]:
            x = b.relu(bn(conv(x)))
        x = b.avgpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        x3 = b.layer3(x)
        x4 = b.layer4(x3)
        xproj = b.attnpool(x4)

        pixels_cls_scores = self.pixel_classifier(x4)
        if external_masks is None:
            probs = pixels_cls_scores.softmax(1)
        else:
            probs = F.interpolate(external_masks, x4.shape[2:], mode='bilinear', align_corners=True)
        background_mask, parts_masks = probs[:, 0], probs[:, 1:]
        foreground_mask = parts_masks.max(1)[0]
        visibility = binary_visibility(probs)

        part_emb_x4 = gwap(x4, parts_masks)
        fg_emb_x4 = gwap(x4, foreground_mask[:, None])[:, 0]
        global_x4 = x4.mean((2, 3))

        queries = torch.cat([global_x4[None], fg_emb_x4[None], part_emb_x4.permute(1, 0, 2)], dim=0)
        pooled = self.part_aware_attnpool(x4, queries)

        parts_out = dict(
            pixels_cls_scores=pixels_cls_scores,
            parts_masks=parts_masks, background_mask=background_mask, foreground_mask=foreground_mask,
            visibility=visibility,
            part_emb_x4=part_emb_x4, fg_emb_x4=fg_emb_x4,
            global_clip=pooled[0], fg_emb_clip=pooled[1], part_emb_clip=pooled[2:].permute(1, 0, 2),
        )
        return x3, x4, xproj, parts_out


class PartImageEncoder(nn.Module):
    """One 1024-d CLIP-space embedding per slot, no attention added (see main.ipynb).

    slot 0 = xproj[0]; slot k = gwap(x4, mask_k) -> frozen Linear(2048->1024) with
    W = c_proj.W @ v_proj.W, b = c_proj.W @ v_proj.b + c_proj.b (CLIP's attention pool with the attention
    weights replaced by the part mask). Everything on the image side is frozen.
    forward(x, masks) -> img_feats [N, K+1, 1024], visibility [N, K+1] (slot 0 always visible).
    """

    def __init__(self, part_aware_backbone):
        super().__init__()
        self.net = part_aware_backbone
        ap = part_aware_backbone.backbone.attnpool
        self.part_proj = nn.Linear(ap.v_proj.in_features, ap.c_proj.out_features)
        with torch.no_grad():
            self.part_proj.weight.copy_(ap.c_proj.weight @ ap.v_proj.weight)
            self.part_proj.bias.copy_(ap.c_proj.weight @ ap.v_proj.bias + ap.c_proj.bias)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x, masks):
        _, _, xproj, out = self.net(x, external_masks=masks)
        parts = self.part_proj(out['part_emb_x4'])
        img_feats = torch.cat([xproj[0][:, None], parts], dim=1)
        visibility = out['visibility'].clone()
        visibility[:, 0] = True
        return img_feats, visibility


# ----------------------------------------------------------------------------- text side
class PartPromptLearner(nn.Module):
    """Learnable prompts per (identity, slot); slot 0 = global, slots 1..K = body parts (see main.ipynb).

    One shared frozen text encoder; slots differ by template words and by their own ctx.
    cls_ctx [num_class, K+1, n_ctx, 512] are the only trained parameters.
    forward(label [B], slots=None) -> prompts [B*len(slots), 77, 512], tokenized [B*len(slots), 77].
    """

    def __init__(self, num_class, clip_model, part_names, n_ctx=N_CTX):
        super().__init__()
        templates = ['A photo of a X X X X person.'] + [f"A photo of the {p.replace('_', ' ')} of a X X X X person." for p in part_names]
        tokenized = tokenize(templates).to(clip_model.token_embedding.weight.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized).type(clip_model.dtype)
        x_id = tokenize('X')[0, 1].item()
        self.templates = templates
        self.num_slots = len(templates)
        self.n_ctx = n_ctx
        self.register_buffer('tokenized', tokenized)
        for s in range(self.num_slots):
            x0 = (tokenized[s] == x_id).nonzero()[0].item()
            self.register_buffer(f'prefix_{s}', embedding[s, :x0].clone())
            self.register_buffer(f'suffix_{s}', embedding[s, x0 + n_ctx:].clone())
        ctx = torch.empty(num_class, self.num_slots, n_ctx, embedding.shape[-1], dtype=clip_model.dtype)
        nn.init.normal_(ctx, std=0.02)
        self.cls_ctx = nn.Parameter(ctx)

    def forward(self, label, slots=None):
        slots = range(self.num_slots) if slots is None else slots
        b = label.shape[0]
        ctx = self.cls_ctx[label]
        prompts = []
        for s in slots:
            prefix = getattr(self, f'prefix_{s}')[None].expand(b, -1, -1)
            suffix = getattr(self, f'suffix_{s}')[None].expand(b, -1, -1)
            prompts.append(torch.cat([prefix, ctx[:, s], suffix], dim=1))
        prompts = torch.stack(prompts, dim=1).flatten(0, 1)
        tokenized = self.tokenized[list(slots)][None].expand(b, -1, -1).flatten(0, 1)
        return prompts, tokenized


def encode_text(prompt_learner, text_encoder, label, slots=None):
    """label [B] -> text_feats [B, len(slots), 1024]."""
    prompts, tokenized = prompt_learner(label, slots)
    return text_encoder(prompts, tokenized).view(label.shape[0], -1, text_encoder.text_projection.shape[1])


def encode_text_tokens(prompt_learner, text_encoder, label, slots):
    """Same forward as TextEncoder (make_model_clipreid.py) but keeps every token: returns the projected token
    states [B*len(slots), 77, 1024], the EOT vector [B*len(slots), 1024] (bit-identical to encode_text) and the
    key mask [B*len(slots), 77] (positions up to and including EOT; the padding after it is masked out)."""
    prompts, tokenized = prompt_learner(label, slots)
    x = prompts + text_encoder.positional_embedding.type(text_encoder.dtype)
    x = text_encoder.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
    x = text_encoder.ln_final(x).type(text_encoder.dtype)
    tokens = x @ text_encoder.text_projection
    eot_pos = tokenized.argmax(dim=-1)
    eot = tokens[torch.arange(tokens.shape[0], device=tokens.device), eot_pos]
    key_mask = torch.arange(tokens.shape[1], device=tokens.device)[None] <= eot_pos[:, None]
    return tokens, eot, key_mask


class TextAttentionBlock(nn.Module):
    """TAB: the image is the query, the prompt's token states are keys and values.

    Stage 1 normally contrasts an image slot embedding against the EOT-pooled prompt. TAB adds an
    image-conditioned pooling of the same prompt: q = W_q x_image attends over the 77 token states (keys after EOT
    masked), and the result is added as a residual on the EOT vector, t_hat = t_eot + W_o Attn(q, K, V).
    * W_o is zero-initialised, so with TAB enabled the block starts as the identity on t_eot: the first step is
      numerically the plain-EOT contrast, and every setting validated for it still holds.
    * Used only as an auxiliary loss in stage 1 (TAB_W). Stage 2 keeps consuming the per-identity EOT snapshot,
      so TAB can only help by making cls_ctx better under EOT pooling - the plain-EOT top-1 is the pass/fail metric.
    * The block cannot bypass the prompts: template tokens are identical across identities, so every identity
      signal in K/V still comes from the ctx tokens.
    Query variants (test, see query()): the part embedding alone, or the part embedding gated by the image's
    global embedding - both L2-normalised first and the product rescaled to |x_part|, so the two variants feed
    the attention at the same scale and differ only in direction (the raw product of two ~5-norm vectors has
    ~25x scale and sign flips).
    """

    def __init__(self, dim=1024, d_tab=TAB_DIM, heads=TAB_HEADS):
        super().__init__()
        self.heads = heads
        self.w_q = nn.Linear(dim, d_tab)
        self.w_k = nn.Linear(dim, d_tab)
        self.w_v = nn.Linear(dim, d_tab)
        self.w_o = nn.Linear(d_tab, dim)
        nn.init.zeros_(self.w_o.weight)
        nn.init.zeros_(self.w_o.bias)

    def query(self, x_part, x_global):
        q = self.w_q(x_part)                                                                       # q = part embedding
        # q = self.w_q(F.normalize(x_part, dim=-1) * F.normalize(x_global, dim=-1)                  # q = part (x) global (test)
        #              * x_part.shape[-1] ** 0.5 * x_part.norm(dim=-1, keepdim=True))               #   rescaled to |x_part|
        return q

    def kv(self, tokens):
        """tokens [C, L, dim] -> K, V [C, L, d_tab] (what the bank stores)."""
        return self.w_k(tokens), self.w_v(tokens)

    def forward(self, q, K, V, key_mask, t_eot):
        """q [M, d_tab]; K, V [C, L, d_tab]; key_mask [C, L]; t_eot [C, dim] -> t_hat [M, C, dim]."""
        M, (C, L, d_tab) = q.shape[0], K.shape
        h, d = self.heads, d_tab // self.heads
        qh, Kh, Vh = q.view(M, h, d), K.view(C, L, h, d).float(), V.view(C, L, h, d).float()
        scores = torch.einsum('mhd,clhd->mchl', qh, Kh) / d ** 0.5
        scores = scores.masked_fill(~key_mask[None, :, None, :], float('-inf'))
        out = torch.einsum('mchl,clhd->mchd', torch.softmax(scores, dim=-1), Vh).reshape(M, C, d_tab)
        return t_eot[None] + self.w_o(out)


# ----------------------------------------------------------------------------- data
def load_dataset(dataset_name=None, root=DATA_ROOT, num_ids=None, logger=print,
                 variant=None, masks=None):
    """Train split of the dataset with its PifPaf masks, both resolved by datasets/part_datasets.py.
    Returns image paths, relabelled ids [N], masks [N, K+1, h, w] (the grid is whatever the .npy files hold;
    every consumer interpolates, so Market's 17x9 and a 33x17 MSMT17 grid both work - but one dataset's masks
    must be one grid, since they are stacked here)."""
    dataset, dataset_dir = build_dataset(dataset_name or DATASET, root, verbose=False)
    keep = sorted({pid for _, pid, _, _ in dataset.train})
    if num_ids is not None:
        keep = keep[:num_ids]
    pid2label = {pid: i for i, pid in enumerate(keep)}
    items = [it for it in dataset.train if it[1] in pid2label]
    paths = [it[0] for it in items]
    variant, masks = variant or MASKS_VARIANT, masks or MASKS_DIR
    masks_dir = resolve_masks(dataset_name or DATASET, dataset, dataset_dir, logger, paths=paths,
                              variant=variant, masks=masks)
    labels = torch.tensor([pid2label[it[1]] for it in items])
    masks = torch.stack([pifpaf_to_masks(np.load(mask_path(p, dataset_dir, masks_dir, variant))) for p in paths])
    return paths, labels, masks


@torch.no_grad()
def extract_features(image_encoder, transform, paths, masks, logger):
    """Frozen image side, run once and cached (processor_clipreid_stage1.py:43-53)."""
    feats, vis = [], []
    t0 = time.monotonic()
    for i in range(0, len(paths), EXTRACT_BATCH):
        x = torch.stack([transform(Image.open(p).convert('RGB')) for p in paths[i:i + EXTRACT_BATCH]]).to(DEVICE)
        f, v = image_encoder(x, masks[i:i + EXTRACT_BATCH].to(DEVICE))
        feats.append(f)
        vis.append(v)
        if (i // EXTRACT_BATCH + 1) % LOG_PERIOD == 0:
            logger.info(f'extracted {i + len(x)}/{len(paths)} images')
    logger.info(f'feature extraction time: {timedelta(seconds=time.monotonic() - t0)}')
    feats, vis = torch.cat(feats), torch.cat(vis)
    torch.cuda.empty_cache()
    return feats, vis


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def all_text_feats(prompt_learner, text_encoder, batch=96):
    num_class = prompt_learner.cls_ctx.shape[0]
    out = [encode_text(prompt_learner, text_encoder, torch.arange(i, min(i + batch, num_class), device=DEVICE))
           for i in range(0, num_class, batch)]
    return torch.cat(out)


@torch.no_grad()
def slot_accuracy(img_feats, labels, vis, text_all):
    """Image -> text top-1 identity accuracy per slot (invisible parts skipped)."""
    acc = {}
    text_n = F.normalize(text_all, dim=-1)
    for s, name in enumerate(SLOT_NAMES):
        keep = vis[:, s]
        sims = F.normalize(img_feats[keep, s], dim=-1) @ text_n[:, s].T
        acc[name] = (sims.argmax(1) == labels[keep]).float().mean().item()
    return acc


# ----------------------------------------------------------------------------- stage 1
def supcon(anchors, cols, a_labels, c_labels, pos_mask=None, exclude=None):
    """SupCon (loss/supcontrast.py) over an arbitrary column pool, with optional normalisation.

    The repo's SupConLoss scores raw dot products at temperature 1. Against a 64-column batch that works,
    but the per-row logit spread of CLIP features is only ~3, so against 751 prompts / 12936 images the
    softmax stays nearly uniform and the loss cannot fall below ~ln(#columns) - the image side is frozen,
    so nothing can widen that gap. Normalising and dividing by a small temperature restores the dynamic
    range the full-dataset pool needs.

    pos_mask [A,B] overrides the default positive set (label equality); exclude [A,B] removes columns from
    the softmax altogether (used to keep an identity's out-of-batch images out of both numerator and
    denominator). cols may be [A,B,D]: one (image-conditioned) column vector per row, scored row-wise.
    """
    if cols.dim() == 3:
        logits = torch.einsum('md,mcd->mc', F.normalize(anchors, dim=-1), F.normalize(cols, dim=-1)) / CONTRAST_TEMP
    elif CONTRAST_NORMALIZE:
        logits = F.normalize(anchors, dim=-1) @ F.normalize(cols, dim=-1).t() / CONTRAST_TEMP
    else:
        logits = anchors @ cols.t()
    mask = (a_labels[:, None] == c_labels[None, :]) if pos_mask is None else pos_mask
    return supcon_from_logits(logits, mask, exclude)


def supcon_from_logits(logits, pos_mask, exclude=None):
    """SupCon on a precomputed [A,B] logit matrix (see supcon)."""
    if exclude is not None:
        logits = logits.masked_fill(exclude, float('-inf'))
    logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_log_prob = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
    return -(pos_log_prob.sum(1) / pos_mask.sum(1).clamp(min=1)).mean()


def build_slot_image_pools(img_feats, labels, vis):
    """Per slot, the dataset-wide pool of image embeddings whose part is visible:
    [(feats [Ns,D], labels [Ns], column_of_image [N] with -1 for images outside the pool, image_of_column [Ns])].
    Static because the image side is frozen and its features are cached."""
    pools = []
    for s in range(S):
        keep = vis[:, s]
        column = torch.full((labels.shape[0],), -1, dtype=torch.long, device=labels.device)
        column[keep] = torch.arange(int(keep.sum()), device=labels.device)
        pools.append((img_feats[keep, s].detach(), labels[keep], column, keep.nonzero().squeeze(1)))
    return pools


def t2i_masks(tgt, pool_labels, batch_columns):
    """Batch positives, dataset negatives. Rows = in-batch prompts, columns = the slot's dataset pool.
    Positives = the prompt's identity images that are in this batch (CLIP-ReID's positive set); the
    identity's other images are excluded from the softmax so they are neither positives nor false negatives.
    Every other-identity image in the dataset is a negative."""
    same = tgt[:, None] == pool_labels[None, :]
    in_batch = torch.zeros(pool_labels.shape[0], dtype=torch.bool, device=tgt.device)
    in_batch[batch_columns] = True
    return same & in_batch[None, :], same & ~in_batch[None, :]


def t2i_loss_floor(pools, num_class):
    """Smallest value the t2i term can reach, per slot.

    SupCon averages the log-probability over every positive column: -1/P sum_i log p_i with sum_i p_i <= 1,
    so the minimum is ln(P). With the dataset-wide pool a prompt has P = all images of its identity (~17 on
    Market-1501), hence a floor of ~ln(17) = 2.8 per slot; with the old 64-image batch a prompt usually had
    a single positive, so the floor was ~0. The full-pool loss therefore levels off well above zero by
    construction - convergence has to be read as 'loss - floor', which is what the epoch log reports.
    i2t has exactly one positive per row, so its floor is 0.
    """
    if T2I_BATCH_POSITIVES:
        return [0.0] * len(pools)
    floors = []
    for _, lbl, _, _ in pools:
        counts = torch.bincount(lbl, minlength=num_class).float()
        counts = counts[counts > 0]
        floors.append(float(torch.log(counts.mean())) if counts.numel() else 0.0)
    return floors


@torch.no_grad()
def build_text_bank(prompt_learner, text_encoder, num_class, batch=96):
    """Detached prompts of every identity and slot: [C, K+1, D]."""
    was_training = prompt_learner.training
    prompt_learner.eval()
    bank = torch.cat([encode_text(prompt_learner, text_encoder, torch.arange(i, min(i + batch, num_class), device=DEVICE))
                      for i in range(0, num_class, batch)]).detach()
    prompt_learner.train(was_training)
    return bank


@torch.no_grad()
def build_tab_bank(prompt_learner, text_encoder, tab, num_class, batch=48):
    """Detached TAB keys/values of every identity and slot: K, V [C, K+1, 77, TAB_DIM] (fp16) and key masks
    [C, K+1, 77]; the EOT bank (build_text_bank) supplies the residual."""
    Ks, Vs, Ms = [], [], []
    for i in range(0, num_class, batch):
        lab = torch.arange(i, min(i + batch, num_class), device=DEVICE)
        tokens, _, kmask = encode_text_tokens(prompt_learner, text_encoder, lab, list(range(S)))
        Kb, Vb = tab.kv(tokens)
        Ks.append(Kb.view(len(lab), S, *Kb.shape[1:]).half()); Vs.append(Vb.view(len(lab), S, *Vb.shape[1:]).half())
        Ms.append(kmask.view(len(lab), S, -1))
    return torch.cat(Ks), torch.cat(Vs), torch.cat(Ms)


def tab_losses(tab, tab_bank, text_bank_s, bank_labels, img_s, img_g, tgt, uniq, tokens_uniq, text_uniq, kmask_uniq,
               s, pool, batch_columns, img_feats, chunk=16):
    """TAB auxiliary contrast for one slot, both directions, with image-conditioned prompts.
    i2t: batch images vs all C prompts (bank K/V; in-batch identities spliced in fresh, with gradient).
    t2i: each in-batch prompt vs its batch positives + TAB_T2I_POOL sampled pool images, every column scored with
    the prompt as conditioned by that image; same-identity images outside the batch are excluded (T2I_BATCH_POSITIVES rule)."""
    Kb, Vb, Mb = tab_bank
    Ku, Vu = tab.kv(tokens_uniq)
    K_cols = Kb[:, s].float().index_copy(0, uniq, Ku)
    V_cols = Vb[:, s].float().index_copy(0, uniq, Vu)
    M_cols = Mb[:, s].index_copy(0, uniq, kmask_uniq)
    t_eot_cols = text_bank_s.index_copy(0, uniq, text_uniq)
    t_hat = tab(tab.query(img_s, img_g), K_cols, V_cols, M_cols, t_eot_cols)
    loss_i2t = supcon(img_s, t_hat, tgt, bank_labels)

    pool_feats, pool_labels, _, pool_index = pool
    neg = torch.randperm(pool_feats.shape[0], device=DEVICE)[:min(TAB_T2I_POOL, pool_feats.shape[0])]
    cols = torch.unique(torch.cat([batch_columns, neg]))
    x_cols, lab_cols = pool_feats[cols], pool_labels[cols]
    q_cols = tab.query(x_cols, img_feats[pool_index[cols], 0])
    in_batch = torch.isin(cols, batch_columns)
    logits = []
    for i in range(0, uniq.shape[0], chunk):
        t_hat_c = tab(q_cols, Ku[i:i + chunk], Vu[i:i + chunk], kmask_uniq[i:i + chunk], text_uniq[i:i + chunk])
        logits.append(torch.einsum('jd,jud->uj', F.normalize(x_cols, dim=-1), F.normalize(t_hat_c, dim=-1)) / CONTRAST_TEMP)
    logits = torch.cat(logits)
    same = uniq[:, None] == lab_cols[None, :]
    loss_t2i = supcon_from_logits(logits, same & in_batch[None], same & ~in_batch[None])
    return loss_i2t + loss_t2i


@torch.no_grad()
def slot_accuracy_tab(img_feats, labels, vis, tab, text_all, tab_bank, chunk=128):
    """Image -> text top-1 per slot with image-conditioned prompts t_hat (diagnostic; stage 2 never sees these)."""
    Kb, Vb, Mb = tab_bank
    acc = {}
    for s, name in enumerate(SLOT_NAMES):
        keep = vis[:, s].nonzero().squeeze(1)
        hits = 0
        for i in range(0, len(keep), chunk):
            idx = keep[i:i + chunk]
            q = tab.query(img_feats[idx, s], img_feats[idx, 0])
            t_hat = tab(q, Kb[:, s], Vb[:, s], Mb[:, s], text_all[:, s])
            sims = torch.einsum('md,mcd->mc', F.normalize(img_feats[idx, s], dim=-1), F.normalize(t_hat, dim=-1))
            hits += (sims.argmax(1) == labels[idx]).sum().item()
        acc[name] = hits / max(len(keep), 1)
    return acc


def do_train_stage1(prompt_learner, text_encoder, img_feats, labels, vis, logger, resume=None, tab=None):
    """CLIP-ReID stage 1 per slot (processor_clipreid_stage1.py:56-97) on cached image features.

    Negative pool (FULL_POOL_NEGATIVES): both directions contrast against the whole dataset instead of the
    64-sample batch.
    * t2i is exact: the image side is frozen and cached, so each in-batch prompt is scored against every
      image whose slot is visible (columns = the full dataset pool, no gradient needed on them). With
      T2I_BATCH_POSITIVES the positives stay CLIP-ReID's (the identity's in-batch images) and the identity's
      remaining images are masked out: averaging over all ~17 images of an identity at a sharp temperature
      pulled every prompt towards its identity's outlier images and never converged (server run: t2i flat
      at ~15 above its floor for 50 epochs, head/torso top-1 10-20 points below the batch-local run).
    * i2t uses a prompt bank: every identity's prompt is encoded without gradient and the identities
      present in the batch are spliced back in with gradient, so each image is scored against all C
      identities while only in-batch prompts receive gradient. After each step the fresh prompts are written
      back into the bank (TEXT_BANK_MOMENTUM); with a rebuild only once per epoch the bank drifted ~200
      iterations behind the trained prompts and the loss jumped +3 at every epoch boundary. The full
      rebuild every TEXT_BANK_REFRESH epochs is kept as a safety net.
    `resume` = a checkpoint dict saved by this script: training restarts at its epoch + 1 with the same
    warmup-cosine schedule; the Adam state is restored when the checkpoint has it."""
    xent = SupConLoss(DEVICE)
    params = list(prompt_learner.parameters()) + (list(tab.parameters()) if tab is not None else [])
    optimizer = torch.optim.Adam(params, lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = create_scheduler(optimizer, num_epochs=MAX_EPOCHS, lr_min=LR_MIN,
                                 warmup_lr_init=WARMUP_LR_INIT, warmup_t=WARMUP_EPOCHS, noise_range=None)
    start_epoch = 1
    if resume is not None:
        start_epoch = resume['epoch'] + 1
        if 'optimizer' in resume:
            optimizer.load_state_dict(resume['optimizer'])
        logger.info(f"resuming from epoch {resume['epoch']} (optimizer state {'restored' if 'optimizer' in resume else 'reset'})")
    loss_meter = AverageMeter()
    num_image = labels.shape[0]
    num_class = prompt_learner.cls_ctx.shape[0]
    i_ter = num_image // IMS_PER_BATCH
    slot_pools = build_slot_image_pools(img_feats, labels, vis) if FULL_POOL_NEGATIVES else None
    bank_labels = torch.arange(num_class, device=DEVICE)
    text_bank = None
    if FULL_POOL_NEGATIVES:
        logger.info('full-dataset negatives: t2i pool per slot = {} images, i2t pool = {} prompts'
                    .format([int(p[0].shape[0]) for p in slot_pools], num_class))
    all_start = time.monotonic()
    logger.info('start training')
    slot_meters = [AverageMeter() for _ in range(S)]
    i2t_meter, t2i_meter, tab_meter = AverageMeter(), AverageMeter(), AverageMeter()
    tab_bank = None
    if tab is not None:
        logger.info('TAB enabled: image-conditioned auxiliary contrast, weight {}, d={} heads={} t2i pool {} | trainable TAB params {:,}'
                    .format(TAB_W, TAB_DIM, TAB_HEADS, TAB_T2I_POOL, sum(p.numel() for p in tab.parameters())))
    floors = t2i_loss_floor(slot_pools, num_class) if FULL_POOL_NEGATIVES else [0.0] * S
    total_floor = sum(floors)
    if FULL_POOL_NEGATIVES and not T2I_BATCH_POSITIVES:
        logger.info('t2i floor ln(mean positives) per slot: {} | total {:.2f} (loss cannot go below this; '
                    'watch "above floor")'.format({n: round(f, 2) for n, f in zip(SLOT_NAMES, floors)}, total_floor))
    elif FULL_POOL_NEGATIVES:
        logger.info('t2i: batch positives, dataset negatives (out-of-batch same-identity images masked out); '
                    'floor ~0. i2t bank write-back momentum {}'.format(TEXT_BANK_MOMENTUM))
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        loss_meter.reset()
        i2t_meter.reset()
        t2i_meter.reset()
        tab_meter.reset()
        for m in slot_meters:
            m.reset()
        scheduler.step(epoch)
        prompt_learner.train()
        if FULL_POOL_NEGATIVES and (text_bank is None or (epoch - start_epoch) % TEXT_BANK_REFRESH == 0):
            text_bank = build_text_bank(prompt_learner, text_encoder, num_class)
            if tab is not None:
                tab_bank = build_tab_bank(prompt_learner, text_encoder, tab, num_class)
        iter_list = torch.randperm(num_image, device=DEVICE)
        for i in range(i_ter + 1):
            b_list = iter_list[i * IMS_PER_BATCH:(i + 1) * IMS_PER_BATCH] if i != i_ter else iter_list[i * IMS_PER_BATCH:num_image]
            if len(b_list) < 2:
                continue
            target, img_b, vis_b = labels[b_list], img_feats[b_list], vis[b_list]
            optimizer.zero_grad()
            batch_loss = 0.0
            for s in range(S):
                keep = vis_b[:, s]
                if keep.sum() < 2:
                    continue
                tgt, img_s = target[keep], img_b[keep, s]
                uniq, inv = torch.unique(tgt, return_inverse=True)
                if tab is not None:
                    tokens_uniq, text_uniq, kmask_uniq = encode_text_tokens(prompt_learner, text_encoder, uniq, slots=[s])
                else:
                    text_uniq = encode_text(prompt_learner, text_encoder, uniq, slots=[s])[:, 0]
                text_s = text_uniq[inv]
                if FULL_POOL_NEGATIVES:
                    pool_feats, pool_labels, column, pool_index = slot_pools[s]
                    text_cols = text_bank[:, s].index_copy(0, uniq, text_uniq)
                    loss_i2t = supcon(img_s, text_cols, tgt, bank_labels)
                    if T2I_BATCH_POSITIVES:
                        pos, excl = t2i_masks(tgt, pool_labels, column[b_list[keep]])
                        loss_t2i = supcon(text_s, pool_feats, tgt, pool_labels, pos_mask=pos, exclude=excl)
                    else:
                        loss_t2i = supcon(text_s, pool_feats, tgt, pool_labels)
                else:
                    loss_i2t = xent(img_s, text_s, tgt, tgt)
                    loss_t2i = xent(text_s, img_s, tgt, tgt)
                loss = loss_i2t + loss_t2i
                loss_tab = None
                if tab is not None and FULL_POOL_NEGATIVES:
                    loss_tab = tab_losses(tab, tab_bank, text_bank[:, s], bank_labels, img_s, img_b[keep, 0], tgt, uniq,
                                          tokens_uniq, text_uniq, kmask_uniq, s, slot_pools[s], column[b_list[keep]], img_feats)
                    loss = loss + TAB_W * loss_tab
                loss.backward()
                batch_loss += loss.item()
                if FULL_POOL_NEGATIVES:
                    with torch.no_grad():
                        text_bank[uniq, s] = TEXT_BANK_MOMENTUM * text_bank[uniq, s] + (1 - TEXT_BANK_MOMENTUM) * text_uniq.detach()
                        if tab is not None:
                            Kb, Vb = tab.kv(tokens_uniq.detach())
                            tab_bank[0][uniq, s] = Kb.half(); tab_bank[1][uniq, s] = Vb.half(); tab_bank[2][uniq, s] = kmask_uniq
                n_keep = int(keep.sum())
                slot_meters[s].update(loss.item(), n_keep)
                i2t_meter.update(loss_i2t.item(), n_keep)
                t2i_meter.update(loss_t2i.item(), n_keep)
                if loss_tab is not None:
                    tab_meter.update(loss_tab.item(), n_keep)
            optimizer.step()
            loss_meter.update(batch_loss, len(b_list))
            if (i + 1) % LOG_PERIOD == 0:
                logger.info('Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}'
                            .format(epoch, i + 1, i_ter + 1, loss_meter.avg, scheduler._get_lr(epoch)[0]))
        logger.info('Epoch[{}] done. Loss: {:.3f} (i2t {:.3f}, t2i {:.3f}, tab {:.3f}, above floor {:.3f}) per slot {} Base Lr: {:.2e}'
                    .format(epoch, loss_meter.avg, i2t_meter.avg * S, t2i_meter.avg * S, tab_meter.avg * S,
                            loss_meter.avg - total_floor, {n: round(m.avg, 3) for n, m in zip(SLOT_NAMES, slot_meters)},
                            scheduler._get_lr(epoch)[0]))

        if epoch % EVAL_PERIOD == 0 or epoch == MAX_EPOCHS:
            prompt_learner.eval()
            text_all = all_text_feats(prompt_learner, text_encoder)
            acc = slot_accuracy(img_feats, labels, vis, text_all)
            logger.info('Epoch[{}] image->text top-1 per slot: {}'.format(epoch, {k: round(v, 3) for k, v in acc.items()}))
            if tab is not None:
                acc_tab = slot_accuracy_tab(img_feats, labels, vis, tab, text_all, build_tab_bank(prompt_learner, text_encoder, tab, num_class))
                logger.info('Epoch[{}] image->text top-1 per slot, TAB-conditioned (diagnostic): {}'.format(epoch, {k: round(v, 3) for k, v in acc_tab.items()}))
        if epoch % CHECKPOINT_PERIOD == 0 or epoch == MAX_EPOCHS:
            path = os.path.join(OUTPUT_DIR, f'{BACKBONE}_part_prompts_stage1_{epoch}.pth')
            torch.save({'prompt_learner': prompt_learner.state_dict(), 'optimizer': optimizer.state_dict(),
                        'tab': tab.state_dict() if tab is not None else None,
                        'templates': prompt_learner.templates, 'part_names': PART_NAMES, 'epoch': epoch,
                        'knobs': dict(H=H, W=W, STRIDE=STRIDE, N_CTX=N_CTX, BACKBONE=BACKBONE,
                                      FULL_POOL_NEGATIVES=FULL_POOL_NEGATIVES, T2I_BATCH_POSITIVES=T2I_BATCH_POSITIVES,
                                      CONTRAST_NORMALIZE=CONTRAST_NORMALIZE, CONTRAST_TEMP=CONTRAST_TEMP,
                                      TEXT_BANK_MOMENTUM=TEXT_BANK_MOMENTUM, BASE_LR=BASE_LR,
                                      TAB_ENABLED=TAB_ENABLED, TAB_W=TAB_W, TAB_DIM=TAB_DIM, TAB_HEADS=TAB_HEADS,
                                      DATASET=DATASET)}, path)
            logger.info(f'saved {path}')
    logger.info('Stage1 running time: {}'.format(timedelta(seconds=time.monotonic() - all_start)))


def main():
    global NUM_IDS, MAX_EPOCHS, TAB_ENABLED, DATASET, OUTPUT_DIR, MASKS_VARIANT, MASKS_DIR
    parser = argparse.ArgumentParser(description='CLIP-ReID stage 1 with per-part prompts (RN50)')
    parser.add_argument('--dataset', choices=list(DATASETS), default=DATASET)
    parser.add_argument('--masks-variant', choices=list(MASK_SUFFIX), default=MASKS_VARIANT)
    parser.add_argument('--masks-dir', type=str, default=MASKS_DIR, help='pre-saved masks outside the dataset dir')
    parser.add_argument('--num-ids', type=int, default=NUM_IDS, help='limit to the first N identities')
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--resume', type=str, default='', help='checkpoint saved by this script to continue from')
    parser.add_argument('--tab', action='store_true', help='enable the Text Attention Block auxiliary contrast (TAB_ENABLED)')
    args = parser.parse_args()
    NUM_IDS, MAX_EPOCHS, DATASET = args.num_ids, args.epochs, args.dataset
    OUTPUT_DIR = OUTPUT_DIR.format(dataset=DATASET)
    MASKS_VARIANT, MASKS_DIR = args.masks_variant, args.masks_dir
    TAB_ENABLED = TAB_ENABLED or args.tab
    resume = torch.load(args.resume, map_location=DEVICE) if args.resume else None

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = setup_logger('transreid', OUTPUT_DIR, if_train=True)
    logger.info('knobs: ' + ', '.join(f'{k}={v}' for k, v in dict(
        H=H, W=W, STRIDE=STRIDE, N_CTX=N_CTX, MAX_EPOCHS=MAX_EPOCHS, IMS_PER_BATCH=IMS_PER_BATCH, BASE_LR=BASE_LR,
        WARMUP_LR_INIT=WARMUP_LR_INIT, LR_MIN=LR_MIN, WARMUP_EPOCHS=WARMUP_EPOCHS, WEIGHT_DECAY=WEIGHT_DECAY,
        FULL_POOL_NEGATIVES=FULL_POOL_NEGATIVES, T2I_BATCH_POSITIVES=T2I_BATCH_POSITIVES, CONTRAST_NORMALIZE=CONTRAST_NORMALIZE,
        CONTRAST_TEMP=CONTRAST_TEMP, TEXT_BANK_REFRESH=TEXT_BANK_REFRESH, TEXT_BANK_MOMENTUM=TEXT_BANK_MOMENTUM,
        TAB_ENABLED=TAB_ENABLED, TAB_W=TAB_W, TAB_DIM=TAB_DIM, TAB_HEADS=TAB_HEADS, TAB_T2I_POOL=TAB_T2I_POOL,
        NUM_IDS=NUM_IDS, DATASET=DATASET, DATA_ROOT=DATA_ROOT, MASKS_VARIANT=MASKS_VARIANT, MASKS_DIR=MASKS_DIR).items()))

    h_res, w_res = (H - 16) // STRIDE + 1, (W - 16) // STRIDE + 1
    clip = load_clip_to_cpu(BACKBONE, h_res, w_res, STRIDE).to(DEVICE).eval()
    for p in clip.parameters():
        p.requires_grad_(False)
    image_encoder = PartImageEncoder(PartAwareModifiedResNet(clip.visual)).to(DEVICE).eval()
    text_encoder = TextEncoder(clip).to(DEVICE).eval()
    transform = T.Compose([T.Resize((H, W)), T.ToTensor(), T.Normalize(PIXEL_MEAN, PIXEL_STD)])

    paths, labels, masks = load_dataset(DATASET, DATA_ROOT, NUM_IDS, logger.info, MASKS_VARIANT, MASKS_DIR)
    num_class = int(labels.max()) + 1
    logger.info(f'{len(paths)} images, {num_class} identities, slots: {SLOT_NAMES}')
    labels = labels.to(DEVICE)
    img_feats, vis = extract_features(image_encoder, transform, paths, masks, logger)
    logger.info('img_feats {} | visibility {} | visible fraction per slot: {}'.format(
        tuple(img_feats.shape), tuple(vis.shape), dict(zip(SLOT_NAMES, vis.float().mean(0).cpu().numpy().round(3).tolist()))))

    prompt_learner = PartPromptLearner(num_class, clip, PART_NAMES).to(DEVICE)
    tab = TextAttentionBlock().to(DEVICE) if TAB_ENABLED else None
    if resume is not None:
        prompt_learner.load_state_dict(resume['prompt_learner'])
        if tab is not None and resume.get('tab') is not None:
            tab.load_state_dict(resume['tab'])
        logger.info(f"loaded {args.resume} (epoch {resume['epoch']})")
    logger.info('templates:\n' + '\n'.join(prompt_learner.templates))
    logger.info('trainable parameters: {:,} (cls_ctx {})'.format(prompt_learner.cls_ctx.numel(), tuple(prompt_learner.cls_ctx.shape)))
    acc = slot_accuracy(img_feats, labels, vis, all_text_feats(prompt_learner, text_encoder))
    logger.info('before training image->text top-1 per slot: {} (chance {:.4f})'.format({k: round(v, 3) for k, v in acc.items()}, 1 / num_class))

    do_train_stage1(prompt_learner, text_encoder, img_feats, labels, vis, logger, resume, tab)


if __name__ == '__main__':
    main()
