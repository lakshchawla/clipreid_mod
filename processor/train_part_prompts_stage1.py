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
* Loss per batch = sum over slots of SupCon(img_k, text_k) + SupCon(text_k, img_k) (CLIP-ReID's SupConLoss:
  raw dot product, temperature 1, in-batch negatives only - no dataset-wide pool), each slot encoded and
  back-propagated on its own (same gradients, 1/(K+1) of the memory); images whose part k is invisible are
  dropped from slot k's loss. Part slots also see CROSS_PART_NEG: the same batch identities' prompts of the other
  parts (i2t) and the other parts' image embeddings (t2i) are extra negatives, so part k's prompt must prefer
  part k's image evidence over another part's.
* TAB (one block per slot) is an auxiliary image-conditioned contrast on the same batch (see TextAttentionBlock).
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
from solver.scheduler_factory import create_scheduler
from datasets.part_datasets import DATASETS, MASK_SUFFIX, build_dataset, mask_path, resolve_masks
from utils.logger import setup_logger
from utils.meter import AverageMeter

# ----------------------------------------------------------------------------- knobs
DATA_ROOT = '../../datasets'
DATASET = 'dukemtmc'             # market1501 | msmt17 | dukemtmc (datasets/part_datasets.py); --dataset overrides.
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
CROSS_PART_NEG = True              # part slots: other parts of the batch identities are extra negatives (see docstring)
CONTRAST_TEMP = 0.01               # temperature of the normalised `supcon` (used by stage 2); stage 1 is raw-dot like CLIP-ReID
TAB_ENABLED = True                 # Text Attention Block: image-as-query cross-attention over the prompt tokens (see TextAttentionBlock)
TAB_W = 0.5                        # weight of the auxiliary TAB contrast; the plain-EOT contrast stays the primary loss
TAB_DIM, TAB_HEADS = 256, 4        # one block per slot, so part k's cross-attention only ever reads part k's prompt tokens
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
    * One block per slot (nn.ModuleList in main). Used only as an auxiliary loss in stage 1 (TAB_W). Stage 2 keeps consuming the per-identity EOT snapshot,
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

    def query(self, x_part, x_global=None):
        q = self.w_q(x_part)                                                                       # q = part embedding
        # q = self.w_q(F.normalize(x_part, dim=-1) * F.normalize(x_global, dim=-1)                  # q = part (x) global (test)
        #              * x_part.shape[-1] ** 0.5 * x_part.norm(dim=-1, keepdim=True))               #   rescaled to |x_part|
        return q

    def kv(self, tokens):
        """tokens [C, L, dim] -> K, V [C, L, d_tab]."""
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
def supcon_from_logits(logits, pos_mask):
    """SupCon on a precomputed [A,B] logit matrix: mean log-probability of the positives (pos_mask [A,B] bool)."""
    logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_log_prob = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
    return -(pos_log_prob.sum(1) / pos_mask.sum(1).clamp(min=1)).mean()


def supcon(anchors, cols, a_labels, c_labels):
    """Normalised, temperature-scaled SupCon (CONTRAST_TEMP): used by stage 2's contrastive terms."""
    logits = F.normalize(anchors, dim=-1) @ F.normalize(cols, dim=-1).t() / CONTRAST_TEMP
    return supcon_from_logits(logits, a_labels[:, None] == c_labels[None, :])


def supcon_batch(anchors, cols, a_labels, c_labels, neg_cols=None):
    """CLIP-ReID's SupConLoss (loss/supcontrast.py: raw dot product, temperature 1; positives = equal labels among
    `cols`), plus optional extra columns `neg_cols` [Nn,D] that only enter the softmax denominator."""
    logits = anchors @ cols.t()
    pos = a_labels[:, None] == c_labels[None, :]
    if neg_cols is not None and neg_cols.shape[0] > 0:
        logits = torch.cat([logits, anchors @ neg_cols.t()], dim=1)
        pos = torch.cat([pos, torch.zeros(pos.shape[0], neg_cols.shape[0], dtype=torch.bool, device=pos.device)], dim=1)
    return supcon_from_logits(logits, pos)


def tab_loss_batch(tab, img_s, tgt, inv, tokens_uniq, text_uniq, kmask_uniq):
    """TAB auxiliary contrast for one slot, in-batch only, both directions.
    t_hat[j, u] = the prompt of in-batch identity u as conditioned by image j. Scoring image j against the prompt of
    every in-batch image m is t_hat[j, inv[m]], so one [M, M] matrix serves i2t (rows) and t2i (its transpose). Same
    raw-dot SupCon and the same columns as the base loss, so at initialisation (W_o = 0) it equals the base loss
    without cross-part negatives exactly."""
    K, V = tab.kv(tokens_uniq)
    t_hat = tab(tab.query(img_s), K, V, kmask_uniq, text_uniq)                   # [M, U, dim]
    logits = torch.einsum('jd,jmd->jm', img_s, t_hat[:, inv])                     # [M, M]
    pos = tgt[:, None] == tgt[None, :]
    return supcon_from_logits(logits, pos) + supcon_from_logits(logits.t(), pos)


@torch.no_grad()
def slot_accuracy_tab(img_feats, labels, vis, tabs, prompt_learner, text_encoder, text_all, batch=48, chunk=128):
    """Image -> text top-1 per slot with image-conditioned prompts t_hat (diagnostic; stage 2 never sees these)."""
    num_class, acc = text_all.shape[0], {}
    for s, name in enumerate(SLOT_NAMES):
        Ks, Vs, Ms = [], [], []
        for i in range(0, num_class, batch):
            lab = torch.arange(i, min(i + batch, num_class), device=DEVICE)
            tokens, _, kmask = encode_text_tokens(prompt_learner, text_encoder, lab, [s])
            Kb, Vb = tabs[s].kv(tokens)
            Ks.append(Kb.half()); Vs.append(Vb.half()); Ms.append(kmask)
        Kb, Vb, Mb = torch.cat(Ks), torch.cat(Vs), torch.cat(Ms)
        keep = vis[:, s].nonzero().squeeze(1)
        hits = 0
        for i in range(0, len(keep), chunk):
            idx = keep[i:i + chunk]
            t_hat = tabs[s](tabs[s].query(img_feats[idx, s]), Kb, Vb, Mb, text_all[:, s])
            sims = torch.einsum('md,mcd->mc', F.normalize(img_feats[idx, s], dim=-1), F.normalize(t_hat, dim=-1))
            hits += (sims.argmax(1) == labels[idx]).sum().item()
        acc[name] = hits / max(len(keep), 1)
    return acc


@torch.no_grad()
def cross_part_confusion(img_feats, labels, vis, text_all):
    """Per part slot: share of visible images whose slot embedding is closer to its own identity's prompt of ANOTHER
    part than to its own part's prompt (0 = every part prefers its own prompt)."""
    txt = F.normalize(text_all[labels], dim=-1)                                   # [N, S, D]
    out = {}
    for s in range(1, S):
        keep = vis[:, s]
        sims = torch.einsum('nd,nkd->nk', F.normalize(img_feats[keep, s], dim=-1), txt[keep][:, 1:])   # [n, K]
        own = sims[:, s - 1]
        others = torch.cat([sims[:, :s - 1], sims[:, s:]], dim=1).max(1)[0]
        out[SLOT_NAMES[s]] = (others > own).float().mean().item()
    return out


def do_train_stage1(prompt_learner, text_encoder, img_feats, labels, vis, logger, resume=None, tabs=None):
    """CLIP-ReID stage 1 per slot (processor_clipreid_stage1.py:56-97) on cached image features, batch negatives only.

    Per slot, over the batch images whose part is visible: i2t = SupCon(image, its identity's prompt), t2i =
    SupCon(prompt, images), both CLIP-ReID's raw-dot loss. With CROSS_PART_NEG a part slot also gets negative-only
    columns: the prompts of the *other parts* of the batch identities (encoded without gradient, once per batch) for
    i2t, and the other parts' image embeddings for t2i. With TAB, tab_loss_batch adds the image-conditioned contrast.
    `resume` = a checkpoint dict saved by this script: training restarts at its epoch + 1 with the same
    warmup-cosine schedule; the Adam state is restored when the checkpoint has it."""
    params = list(prompt_learner.parameters()) + (list(tabs.parameters()) if tabs is not None else [])
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
    all_start = time.monotonic()
    logger.info('start training: in-batch negatives only, cross-part negatives {}'.format('on' if CROSS_PART_NEG else 'off'))
    slot_meters = [AverageMeter() for _ in range(S)]
    i2t_meter, t2i_meter, tab_meter = AverageMeter(), AverageMeter(), AverageMeter()
    if tabs is not None:
        logger.info('TAB enabled: one block per slot, image-conditioned auxiliary contrast, weight {}, d={} heads={} | trainable TAB params {:,}'
                    .format(TAB_W, TAB_DIM, TAB_HEADS, sum(p.numel() for p in tabs.parameters())))
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        loss_meter.reset()
        i2t_meter.reset()
        t2i_meter.reset()
        tab_meter.reset()
        for m in slot_meters:
            m.reset()
        scheduler.step(epoch)
        prompt_learner.train()
        iter_list = torch.randperm(num_image, device=DEVICE)
        for i in range(i_ter + 1):
            b_list = iter_list[i * IMS_PER_BATCH:(i + 1) * IMS_PER_BATCH] if i != i_ter else iter_list[i * IMS_PER_BATCH:num_image]
            if len(b_list) < 2:
                continue
            target, img_b, vis_b = labels[b_list], img_feats[b_list], vis[b_list]
            optimizer.zero_grad()
            batch_loss = 0.0
            text_ng = None
            if CROSS_PART_NEG:
                with torch.no_grad():
                    text_ng = encode_text(prompt_learner, text_encoder, torch.unique(target))        # [Ub, S, D]
            for s in range(S):
                keep = vis_b[:, s]
                if keep.sum() < 2:
                    continue
                tgt, img_s = target[keep], img_b[keep, s]
                uniq, inv = torch.unique(tgt, return_inverse=True)
                if tabs is not None:
                    tokens_uniq, text_uniq, kmask_uniq = encode_text_tokens(prompt_learner, text_encoder, uniq, slots=[s])
                else:
                    text_uniq = encode_text(prompt_learner, text_encoder, uniq, slots=[s])[:, 0]
                text_s = text_uniq[inv]
                neg_txt = neg_img = None
                if CROSS_PART_NEG and s >= 1:
                    others = [k for k in range(1, S) if k != s]
                    neg_txt = text_ng[:, others].flatten(0, 1)                                      # [Ub*(K-1), D]
                    neg_img = torch.cat([img_b[vis_b[:, k], k] for k in others])                   # [n, D]
                loss_i2t = supcon_batch(img_s, text_s, tgt, tgt, neg_txt)
                loss_t2i = supcon_batch(text_s, img_s, tgt, tgt, neg_img)
                loss = loss_i2t + loss_t2i
                loss_tab = None
                if tabs is not None:
                    loss_tab = tab_loss_batch(tabs[s], img_s, tgt, inv, tokens_uniq, text_uniq, kmask_uniq)
                    loss = loss + TAB_W * loss_tab
                loss.backward()
                batch_loss += loss.item()
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
        logger.info('Epoch[{}] done. Loss: {:.3f} (i2t {:.3f}, t2i {:.3f}, tab {:.3f}) per slot {} Base Lr: {:.2e}'
                    .format(epoch, loss_meter.avg, i2t_meter.avg * S, t2i_meter.avg * S, tab_meter.avg * S,
                            {n: round(m.avg, 3) for n, m in zip(SLOT_NAMES, slot_meters)}, scheduler._get_lr(epoch)[0]))

        if epoch % EVAL_PERIOD == 0 or epoch == MAX_EPOCHS:
            prompt_learner.eval()
            text_all = all_text_feats(prompt_learner, text_encoder)
            acc = slot_accuracy(img_feats, labels, vis, text_all)
            logger.info('Epoch[{}] image->text top-1 per slot: {}'.format(epoch, {k: round(v, 3) for k, v in acc.items()}))
            logger.info('Epoch[{}] part image closer to another part\'s prompt (lower is better): {}'.format(
                epoch, {k: round(v, 3) for k, v in cross_part_confusion(img_feats, labels, vis, text_all).items()}))
            if tabs is not None:
                acc_tab = slot_accuracy_tab(img_feats, labels, vis, tabs, prompt_learner, text_encoder, text_all)
                logger.info('Epoch[{}] image->text top-1 per slot, TAB-conditioned (diagnostic): {}'.format(epoch, {k: round(v, 3) for k, v in acc_tab.items()}))
        if epoch % CHECKPOINT_PERIOD == 0 or epoch == MAX_EPOCHS:
            path = os.path.join(OUTPUT_DIR, f'{BACKBONE}_part_prompts_stage1_{epoch}.pth')
            torch.save({'prompt_learner': prompt_learner.state_dict(), 'optimizer': optimizer.state_dict(),
                        'tab': tabs.state_dict() if tabs is not None else None,
                        'templates': prompt_learner.templates, 'part_names': PART_NAMES, 'epoch': epoch,
                        'knobs': dict(H=H, W=W, STRIDE=STRIDE, N_CTX=N_CTX, BACKBONE=BACKBONE,
                                      CROSS_PART_NEG=CROSS_PART_NEG, BASE_LR=BASE_LR,
                                      TAB_ENABLED=TAB_ENABLED, TAB_W=TAB_W, TAB_DIM=TAB_DIM, TAB_HEADS=TAB_HEADS,
                                      DATASET=DATASET)}, path)
            logger.info(f'saved {path}')
    logger.info('Stage1 running time: {}'.format(timedelta(seconds=time.monotonic() - all_start)))


def main():
    global NUM_IDS, MAX_EPOCHS, TAB_ENABLED, DATASET, OUTPUT_DIR, MASKS_VARIANT, MASKS_DIR, CROSS_PART_NEG
    parser = argparse.ArgumentParser(description='CLIP-ReID stage 1 with per-part prompts (RN50)')
    parser.add_argument('--dataset', choices=list(DATASETS), default=DATASET)
    parser.add_argument('--masks-variant', choices=list(MASK_SUFFIX), default=MASKS_VARIANT)
    parser.add_argument('--masks-dir', type=str, default=MASKS_DIR, help='pre-saved masks outside the dataset dir')
    parser.add_argument('--num-ids', type=int, default=NUM_IDS, help='limit to the first N identities')
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--resume', type=str, default='', help='checkpoint saved by this script to continue from')
    parser.add_argument('--no-cross-part-neg', action='store_true', help='ablation: drop the other-part negatives (CROSS_PART_NEG)')
    parser.add_argument('--tab', action='store_true', help='enable the Text Attention Block auxiliary contrast (TAB_ENABLED)')
    args = parser.parse_args()
    NUM_IDS, MAX_EPOCHS, DATASET = args.num_ids, args.epochs, args.dataset
    OUTPUT_DIR = OUTPUT_DIR.format(dataset=DATASET)
    MASKS_VARIANT, MASKS_DIR = args.masks_variant, args.masks_dir
    TAB_ENABLED = TAB_ENABLED or args.tab
    CROSS_PART_NEG = CROSS_PART_NEG and not args.no_cross_part_neg
    resume = torch.load(args.resume, map_location=DEVICE) if args.resume else None

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger = setup_logger('transreid', OUTPUT_DIR, if_train=True)
    logger.info('knobs: ' + ', '.join(f'{k}={v}' for k, v in dict(
        H=H, W=W, STRIDE=STRIDE, N_CTX=N_CTX, MAX_EPOCHS=MAX_EPOCHS, IMS_PER_BATCH=IMS_PER_BATCH, BASE_LR=BASE_LR,
        WARMUP_LR_INIT=WARMUP_LR_INIT, LR_MIN=LR_MIN, WARMUP_EPOCHS=WARMUP_EPOCHS, WEIGHT_DECAY=WEIGHT_DECAY,
        CROSS_PART_NEG=CROSS_PART_NEG, TAB_ENABLED=TAB_ENABLED, TAB_W=TAB_W, TAB_DIM=TAB_DIM, TAB_HEADS=TAB_HEADS,
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
    tabs = nn.ModuleList([TextAttentionBlock() for _ in range(S)]).to(DEVICE) if TAB_ENABLED else None
    if resume is not None:
        prompt_learner.load_state_dict(resume['prompt_learner'])
        if tabs is not None and resume.get('tab') is not None:
            tabs.load_state_dict(resume['tab'])
        logger.info(f"loaded {args.resume} (epoch {resume['epoch']})")
    logger.info('templates:\n' + '\n'.join(prompt_learner.templates))
    logger.info('trainable parameters: {:,} (cls_ctx {})'.format(prompt_learner.cls_ctx.numel(), tuple(prompt_learner.cls_ctx.shape)))
    acc = slot_accuracy(img_feats, labels, vis, all_text_feats(prompt_learner, text_encoder))
    logger.info('before training image->text top-1 per slot: {} (chance {:.4f})'.format({k: round(v, 3) for k, v in acc.items()}, 1 / num_class))

    do_train_stage1(prompt_learner, text_encoder, img_feats, labels, vis, logger, resume, tabs)


if __name__ == '__main__':
    main()
