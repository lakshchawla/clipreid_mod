"""Export a trained stage-2 checkpoint to a DeepStream-ready ONNX ReID embedder (+ TRT engine, + nvinfer/tracker configs).

Run from the repo root:
  python deploy/export_stage2_onnx.py --weights work_dirs/market1501/part_prompts_stage2/RN50_part_prompts_stage2_120.pth

What comes out: one graph, one input `input` [-1, 3, 256, 128] and one output `reid_embedding` [-1, D], the same
L2-normalised retrieval vector the stage-2 `evaluate` rows are built from (FEATURE selects the row). Nothing else
from training is in the graph: no BNNeck heads (test features are the raw slots), no masks, no text encoder, no text
adapters - the LPIM text queries are a frozen buffer, so their query projection folds into a constant. Part visibility
is the model's own prediction (visibility head > 0), also inside the fused vector's attention mask.

Decisions that make the graph DeepStream/TensorRT-friendly (this is the whole point of the file):
  * Pixel normalisation is baked in (NORMALIZE_IN_GRAPH): the network takes raw RGB in [0, 255], so nvinfer runs
    with net-scale-factor=1.0 and offsets=0;0;0. nvinfer applies one scalar scale and per-channel offsets, which
    cannot express the per-channel std of the training transform ([0.229, 0.224, 0.225]); folding the Sub/Div into
    the graph is the only exact option and it costs two eltwise layers.
  * Batch is the only dynamic dimension: every reshape uses -1 in the batch slot and every other extent is a
    python int taken from the weights (the feature grid is len(positional_embedding) - 1), so the graph holds no
    Shape -> Gather -> Reshape chain. Spatial size is fixed to the training 256x128.
  * CLIP's attention pool is re-expressed for the one token that is used at test time (`xproj[0]`): explicit
    q/k/v matmuls instead of F.multi_head_attention_forward, which exports to a ~40-node subgraph with
    dynamic reshapes. Same arithmetic, asserted below against the original module (ATOL).
  * The LPIM cross-attention is re-expressed as plain MatMul/Softmax (no Einsum: TRT supports it but lowers it
    through a generic path) and its query tensor Q = q_proj(text_queries) is precomputed as a constant.
  * Output is L2-normalised inside the graph, so the tracker's own feature normalisation stays off and any
    cosine/L2 matcher is exact.
  * fp32 weights, opset ONNX_OPSET; precision (fp16/int8) is a TRT build flag, not an export flag.

Not exportable, by construction: the `parts_matching*` eval rows. They are query-gallery distances over mutually
visible parts, not embeddings, so no single-vector matcher can reproduce them. WITH_PARTS=True adds `part_embeddings`
[-1, K, 256] (L2-normalised, invisible parts zeroed) and `part_visibility` [-1, K] as extra outputs for a custom
matcher that averages the per-part cosine distances over the parts visible in both images; DeepStream's tracker
ignores extra tensors, so leave it off unless something downstream consumes them.

  python deploy/export_stage2_onnx.py --weights <ckpt> --engine --fp16     # also builds .engine with trtexec
  python deploy/export_stage2_onnx.py --random-weights --engine            # graph/TRT check without a checkpoint
"""
import argparse
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from processor.train_part_prompts_stage2 import PartCLIPReID, PIXEL_MEAN, PIXEL_STD, EVAL_SLOTWISE_NORM
from processor.train_part_prompts_stage1 import PART_NAMES
from model.make_model_clipreid import load_clip_to_cpu

# ----------------------------------------------------------------------------- knobs
OUT_DIR = './deploy/onnx'
FEATURE = 'holistic'               # retrieval vector to export, = the stage-2 eval row of the same name:
                                   #   'global'   cat(gap4, g)             3072-d, CLIP-ReID's own test feature (`clipreid_global`)
                                   #   'selfattn' fused                    1280-d, the self-attended part vector (`parts_selfattn`)
                                   #   'holistic' cat(gap4, g, fused)      4352-d, global + fused (`holistic`)
WITH_PARTS = False                 # also output part_embeddings [-1,K,256] and part_visibility [-1,K] (sigmoid)
NORMALIZE_IN_GRAPH = True          # input is raw RGB 0..255; False expects the already-normalised tensor
ONNX_OPSET = 17                    # TRT 8.5+ / DeepStream 6.2+; 16 for older DeepStream
DYNAMIC_BATCH = True
MIN_BATCH, OPT_BATCH, MAX_BATCH = 1, 16, 32    # TRT optimisation profile = DeepStream batch-size range
WORKSPACE_MB = 2048                # trtexec build workspace; DeepStream's own build uses workspace-size
ENGINE_COS = 0.999                 # min cosine(PyTorch fp32, TensorRT engine) accepted on the engine check
ATOL = 2e-4                        # max |exported - reference| accepted on the parity check
SEED = 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FEATURE_SLOTS = {'global': ('gap4', 'g'), 'selfattn': ('fused',), 'holistic': ('gap4', 'g', 'fused')}


# ----------------------------------------------------------------------------- export graph
def sample_input(embedder, batch, size):
    """An input in the range the graph expects: RGB 0..255, or already normalised when NORMALIZE_IN_GRAPH is off
    (feeding 0..255 into the pre-normalised graph inflates activations and makes the parity numbers meaningless)."""
    x = torch.rand(batch, 3, *size) * 255
    return x if embedder.normalize_in_graph else (x - embedder.pixel_mean.cpu()) / embedder.pixel_std.cpu()


def l2norm(x, dim=-1, eps=1e-12):
    """F.normalize written as a plain broadcast Div: its ONNX lowering adds a Shape -> Expand pair per call."""
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


class Stage2Embedder(nn.Module):
    """Test-time image path of PartCLIPReID as a single tensor-in / tensor-out module.

    Holds references to the trained submodules (no weight copies), and re-expresses the two attention blocks in
    ops TensorRT maps one-to-one. Reshapes use -1 in the batch slot so batch stays the only dynamic dimension.
    """

    def __init__(self, model, feature=FEATURE, with_parts=WITH_PARTS, normalize_in_graph=NORMALIZE_IN_GRAPH,
                 slotwise_norm=EVAL_SLOTWISE_NORM):
        super().__init__()
        assert feature in FEATURE_SLOTS, f'feature must be one of {list(FEATURE_SLOTS)}'
        self.visual, self.lpim, self.fusion, self.part_proj = model.visual, model.lpim, model.fusion, model.part_proj
        self.feature, self.with_parts, self.slotwise_norm = feature, with_parts, slotwise_norm
        self.attnpool = model.visual.attnpool
        self.heads, self.dim_head = self.attnpool.num_heads, self.attnpool.k_proj.in_features // self.attnpool.num_heads
        self.hw = self.attnpool.positional_embedding.shape[0] - 1        # static feature-grid length (H/32 * W/32)
        self.normalize_in_graph = normalize_in_graph
        self.register_buffer('pixel_mean', torch.tensor(PIXEL_MEAN).view(1, 3, 1, 1) * 255)
        self.register_buffer('pixel_std', torch.tensor(PIXEL_STD).view(1, 3, 1, 1) * 255)
        with torch.no_grad():                       # LPIM queries are frozen -> q_proj(text_queries) is a constant
            q = self.lpim.q_proj(self.lpim.queries().float())
        self.register_buffer('lpim_q', q.view(q.shape[0], self.lpim.num_heads, -1).transpose(0, 1).contiguous())

    def attnpool_token0(self, x4):
        """CLIP AttentionPool2d restricted to the query it is used for at test time: xproj[0] [N, D].

        The pool prepends a mean token, adds the positional embedding and runs MHA; only token 0 of the output
        is read, and a single query attends over all HW+1 keys, so heads/keys stay but the output length is 1.
        """
        h, d = self.heads, self.dim_head
        tokens = x4.flatten(2).transpose(1, 2)                                        # [N, HW, C]
        seq = torch.cat([tokens.mean(1, keepdim=True), tokens], dim=1) + self.attnpool.positional_embedding[None]
        q = self.attnpool.q_proj(seq[:, :1]).reshape(-1, 1, h, d).transpose(1, 2)     # [N, h, 1, d]
        k = self.attnpool.k_proj(seq).reshape(-1, self.hw + 1, h, d).transpose(1, 2)
        v = self.attnpool.v_proj(seq).reshape(-1, self.hw + 1, h, d).transpose(1, 2)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * d ** -0.5, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(-1, h * d)                # [N, C]
        return self.attnpool.c_proj(out)

    def lpim_forward(self, x4):
        """LanguageGuidedPartInteraction.forward with a constant query and matmuls instead of einsum."""
        m, h, d = self.lpim, self.lpim.num_heads, self.lpim.k_proj.in_features // self.lpim.num_heads
        hw, sq = self.hw, self.lpim_q.shape[1]
        tokens = x4.flatten(2).transpose(1, 2) + m.pos_embed[None]
        k = m.k_proj(tokens).reshape(-1, hw, h, d).transpose(1, 2)                    # [N, h, HW, d]
        v = m.v_proj(tokens).reshape(-1, hw, h, d).transpose(1, 2)
        attn = torch.softmax(torch.matmul(self.lpim_q[None], k.transpose(-1, -2)) / math.sqrt(d), dim=-1)
        z = m.c_proj(torch.matmul(attn, v).transpose(1, 2).reshape(-1, sq, h * d))    # [N, K+1, D]
        z = z + m.ffn(m.norm(z))
        for layer in m.self_layers:
            z = layer(z)
        return z, m.vis_head(z[:, 1:]).squeeze(-1)

    def fuse(self, z, vis):
        """PartFusion.forward with the CLS token and the always-visible mask built from tensors (no Shape ops)."""
        f = self.fusion
        x = torch.cat([torch.zeros_like(z[:, :1]) + f.cls, z + f.slot_embed], dim=1)
        always = vis[:, :1] | ~vis[:, :1]                                              # [N,1] True, batch-shaped without Shape ops
        keep = torch.cat([always, always, vis], dim=1)
        for layer in f.layers:
            x = layer(x, src_key_padding_mask=~keep)
        return f.out(f.norm(x[:, 0]))

    def parts_forward(self, x4):
        """z, per-part 256-d heads h [N,K,256], fused vector, visibility logits."""
        z, vis_logit = self.lpim_forward(x4)
        h = torch.stack([proj(z[:, 1 + k]) for k, proj in enumerate(self.part_proj)], dim=1)
        return z, h, self.fuse(z, vis_logit > 0), vis_logit

    def forward(self, x):
        if self.normalize_in_graph:
            x = (x - self.pixel_mean) / self.pixel_std
        x = self.visual.layer2(self.visual.layer1(self.visual.avgpool(self.stem(x))))
        x4 = self.visual.layer4(self.visual.layer3(x))
        gap4, g = x4.mean((2, 3)), self.attnpool_token0(x4)
        z, h, fused, vis_logit = self.parts_forward(x4)
        slots = dict(gap4=gap4, g=g, fused=fused)
        chosen = [slots[name] for name in FEATURE_SLOTS[self.feature]]
        if self.slotwise_norm and self.feature == 'holistic':     # the global row normalises once, as CLIP-ReID does
            chosen = [l2norm(s) for s in chosen]
        embedding = l2norm(torch.cat(chosen, dim=1))
        if self.with_parts:
            vis = vis_logit > 0
            return embedding, l2norm(h) * vis[..., None].float(), torch.sigmoid(vis_logit)
        return embedding

    def stem(self, x):
        v = self.visual
        for conv, bn in [(v.conv1, v.bn1), (v.conv2, v.bn2), (v.conv3, v.bn3)]:
            x = v.relu(bn(conv(x)))
        return x


# ----------------------------------------------------------------------------- build / export
def build_model(weights, random_weights, num_classes, logger=print):
    """Rebuild the stage-2 architecture from the checkpoint alone (knobs, text queries and class count are in it)."""
    import processor.train_part_prompts_stage2 as s2
    if random_weights:
        clip = load_clip_to_cpu(s2.BACKBONE, (s2.H - 16) // s2.STRIDE + 1, (s2.W - 16) // s2.STRIDE + 1, s2.STRIDE)
        queries = clip.encode_text(__import__('model.clip.clip', fromlist=['tokenize']).tokenize(
            ['A photo of a person.'] + [f"A photo of the {p.replace('_', ' ')} of a person." for p in PART_NAMES])).float()
        model = PartCLIPReID(clip.visual.float(), num_classes, queries)
        logger(f'random stage-2 weights (CLIP-pretrained visual), {num_classes} classes - graph check only')
        return model.eval(), dict(H=s2.H, W=s2.W, STRIDE=s2.STRIDE, PART_NAMES=PART_NAMES)
    ckpt = torch.load(weights, map_location='cpu', weights_only=False)
    state, knobs = ckpt['model'], ckpt['knobs']
    assert knobs['BACKBONE'] == s2.BACKBONE, f"checkpoint backbone {knobs['BACKBONE']} != {s2.BACKBONE}"
    for name in ['MIM_SELF_LAYERS', 'PART_DIM', 'FUSED_DIM', 'FUSE_LAYERS', 'LPIM_LEARN_QUERY']:
        setattr(s2, name, knobs[name])                             # module shapes must match the weights
    num_classes = state['id_gap4.fc.weight'].shape[0]
    clip = load_clip_to_cpu(knobs['BACKBONE'], (knobs['H'] - 16) // knobs['STRIDE'] + 1,
                            (knobs['W'] - 16) // knobs['STRIDE'] + 1, knobs['STRIDE'])
    model = PartCLIPReID(clip.visual.float(), num_classes, state['lpim.text_queries'].float())
    model.load_state_dict({k: v.float() for k, v in state.items()})
    logger(f"loaded {weights} (epoch {ckpt['epoch']}), {num_classes} classes, knobs: "
           + ', '.join(f'{k}={v}' for k, v in knobs.items()))
    return model.eval(), knobs


@torch.no_grad()
def check_against_reference(model, embedder, size, logger=print):
    """The exported graph must reproduce the stage-2 eval row it claims, computed by the training script itself.

    Two levels: the rewritten attention blocks against PartCLIPReID's own test slots, then the assembled
    embedding against `concat_feature` (stage 2's evaluate()) over the same slots.
    """
    from processor.train_part_prompts_stage2 import concat_feature
    device = next(model.parameters()).device
    raw = torch.rand(4, 3, *size, device=device) * 255
    net_in = (raw - embedder.pixel_mean) / embedder.pixel_std           # what the backbone sees either way
    ref = model(net_in)
    x4 = model.visual(net_in)[1]
    g = embedder.attnpool_token0(x4)
    z, h, fused, vis = embedder.parts_forward(x4)
    diffs = {'g (attnpool token 0)': (g - ref['g']).abs().max().item(),
             'z': (z - ref['z']).abs().max().item(),
             'h': (h - ref['h']).abs().max().item(),
             'fused': (fused - ref['fused']).abs().max().item(),
             'vis_logit': (vis - ref['vis_logit']).abs().max().item()}
    logger('rewritten blocks vs PartCLIPReID: ' + ', '.join(f'{k} {v:.2e}' for k, v in diffs.items()))
    assert max(diffs.values()) < ATOL, f'rewrite changes the features: {diffs}'

    slots = [ref[name] for name in FEATURE_SLOTS[embedder.feature]]
    expected = (concat_feature(*slots) if embedder.feature == 'holistic'
                else F.normalize(torch.cat(slots, dim=1), dim=1))      # the eval row, from the training script
    got = embedder(raw if embedder.normalize_in_graph else net_in)
    got = got[0] if isinstance(got, tuple) else got
    diff = (got - expected).abs().max().item()
    logger(f"`{embedder.feature}` embedding vs evaluate()'s row: max|diff| {diff:.2e}, "
           f'cos {F.cosine_similarity(got, expected).min():.6f}')
    assert diff < ATOL, f'assembled embedding differs from the {embedder.feature} eval row by {diff:.2e}'


def export(embedder, path, size, opset=ONNX_OPSET, dynamic=DYNAMIC_BATCH, logger=print):
    names = ['reid_embedding'] + (['part_embeddings', 'part_visibility'] if embedder.with_parts else [])
    axes = {n: {0: 'batch'} for n in ['input'] + names} if dynamic else None
    dummy = torch.rand(OPT_BATCH, 3, *size, device=next(embedder.parameters()).device) * 255   # fixed batch = OPT_BATCH
    torch.onnx.export(embedder, (dummy,), path, input_names=['input'], output_names=names, dynamic_axes=axes,
                      opset_version=opset, do_constant_folding=True, dynamo=False)
    import onnx
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    try:
        import onnxslim
        graph = onnxslim.slim(graph)
        onnx.save(graph, path)
    except ImportError:
        logger('onnxslim not installed - graph saved unsimplified')
    ops = sorted({n.op_type for n in graph.graph.node})
    logger(f'{path}: {len(graph.graph.node)} nodes, opset {opset}, ops: {" ".join(ops)}')
    for bad in ('NonZero', 'Loop', 'If', 'ScatterND', 'Einsum'):
        assert bad not in ops, f'{bad} in the graph - TensorRT/DeepStream will not like it'
    return graph


@torch.no_grad()
def verify_onnx(embedder, path, size, dynamic=DYNAMIC_BATCH, logger=print):
    """ONNX Runtime vs PyTorch on the same random input: max abs diff and cosine similarity per output."""
    import onnxruntime as ort
    batch = 4 if dynamic else OPT_BATCH
    x = sample_input(embedder, batch, size)
    torch_out = embedder(x.to(next(embedder.parameters()).device))
    torch_out = [torch_out] if torch.is_tensor(torch_out) else list(torch_out)
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    ort_out = sess.run(None, {'input': x.numpy()})
    for meta, t, o in zip(sess.get_outputs(), torch_out, ort_out):
        t, o = t.float().cpu().flatten(1), torch.from_numpy(o).float().flatten(1)
        logger(f'  {meta.name:16s} {tuple(meta.shape)}  max|diff| {(t - o).abs().max():.2e}  '
               f'cos {F.cosine_similarity(t, o).min():.6f}')
        assert (t - o).abs().max() < 1e-3, f'{meta.name}: ONNX Runtime disagrees with PyTorch'
    if dynamic:
        for b in (MIN_BATCH, MAX_BATCH):                   # the profile extremes must run, not just OPT_BATCH
            sess.run(None, {'input': sample_input(embedder, b, size).numpy()})
        logger(f'  dynamic batch {MIN_BATCH}..{MAX_BATCH} runs')


def build_engine(onnx_path, fp16, size, logger=print):
    """trtexec build with the DeepStream batch profile; DeepStream can also build this itself from the ONNX."""
    engine = onnx_path.replace('.onnx', f"_b{MAX_BATCH}_{'fp16' if fp16 else 'fp32'}.engine")
    def shape(b):
        return f'input:{b}x3x{size[0]}x{size[1]}'
    cmd = ['trtexec', f'--onnx={onnx_path}', f'--saveEngine={engine}', f'--memPoolSize=workspace:{WORKSPACE_MB}',
           f'--minShapes={shape(MIN_BATCH)}', f'--optShapes={shape(OPT_BATCH)}',
           f'--maxShapes={shape(MAX_BATCH)}'] + (['--fp16'] if fp16 else [])
    logger('$ ' + ' '.join(cmd))
    out = subprocess.run(cmd, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if 'Throughput' in line or 'Latency: min' in line or 'GPU Compute Time: min' in line or 'Engine built' in line:
            logger('  ' + line.split('] ')[-1])
    assert out.returncode == 0, f'trtexec failed:\n{out.stdout[-3000:]}\n{out.stderr[-2000:]}'
    logger(f'engine: {engine} ({os.path.getsize(engine) / 2 ** 20:.0f} MB)')
    return engine


@torch.no_grad()
def verify_engine(embedder, engine, size, batch=OPT_BATCH, logger=print):
    """Run the built engine on a fixed batch through trtexec and compare with PyTorch.

    fp16 engines differ from fp32 PyTorch in the last couple of digits; what matters for retrieval is that the
    direction survives, so the cosine similarity per embedding is the number to read (> ENGINE_COS).
    """
    import json
    import tempfile
    x = sample_input(embedder, batch, size)
    ref = embedder(x.to(next(embedder.parameters()).device))
    ref = (ref[0] if isinstance(ref, tuple) else ref).float().cpu()
    with tempfile.TemporaryDirectory() as tmp:
        raw, js = os.path.join(tmp, 'input.bin'), os.path.join(tmp, 'out.json')
        x.numpy().astype(np.float32).tofile(raw)
        cmd = ['trtexec', f'--loadEngine={engine}', f'--shapes=input:{batch}x3x{size[0]}x{size[1]}',
               f'--loadInputs=input:{raw}', f'--exportOutput={js}', '--iterations=1', '--warmUp=0', '--duration=0']
        run = subprocess.run(cmd, capture_output=True, text=True)
        assert run.returncode == 0, f'trtexec inference failed:\n{run.stdout[-2000:]}'
        tensors = {o['name']: torch.tensor(o['values']).view(*[int(d) for d in o['dimensions'].split('x')])
                   for o in json.load(open(js))}
    trt = tensors['reid_embedding']
    cos = F.cosine_similarity(ref, trt).min().item()
    logger(f'  engine vs PyTorch: max|diff| {(ref - trt).abs().max():.2e}  cos {cos:.6f}  (batch {batch})')
    assert cos > ENGINE_COS, f'engine embeddings diverge from PyTorch (min cos {cos:.6f})'


# ----------------------------------------------------------------------------- DeepStream configs
def write_deepstream_configs(onnx_path, dim, size, out_dir, logger=print):
    """nvinfer config (secondary embedder) and the NvDCF tracker ReID block, filled from the exported graph."""
    onnx_rel = os.path.basename(onnx_path)
    nvinfer = f"""# DeepStream nvinfer - stage-2 part-prompt ReID embedder (secondary GIE, tensor output only).
# Pixel normalisation is inside the ONNX graph, so net-scale-factor=1 and offsets=0 (nvinfer cannot apply a
# per-channel std). The engine is built on first run if model-engine-file is absent.
[property]
gpu-id=0
onnx-file={onnx_rel}
model-engine-file={onnx_rel}_b{MAX_BATCH}_gpu0_fp16.engine
infer-dims=3;{size[0]};{size[1]}
model-color-format=0
net-scale-factor=1.0
offsets=0.0;0.0;0.0
batch-size={MAX_BATCH}
network-mode=2
network-type=100
output-tensor-meta=1
output-blob-names=reid_embedding
process-mode=2
gie-unique-id=2
operate-on-gie-id=1
operate-on-class-ids=0
maintain-aspect-ratio=0
symmetric-padding=0
"""
    tracker = f"""# NvDCF / NvDeepSORT ReID block (config_tracker_NvDCF_accuracy.yml): re-identification with this embedder.
# reidFeatureSize must equal the exported embedding width; the graph already L2-normalises, so keep
# addFeatureNormalization off. netScaleFactor/offsets are 1/0 because normalisation is inside the graph.
ReID:
  reidType: 2                     # 2 = ReID feature is used for matching (0 = off, 1 = re-assoc only)
  batchSize: {MAX_BATCH}
  workspaceSize: 1000
  reidFeatureSize: {dim}
  reidHistorySize: 100
  inferDims: [3, {size[0]}, {size[1]}]
  networkMode: 1                  # 0 fp32, 1 fp16, 2 int8
  inputOrder: 0                   # NCHW
  colorFormat: 0                  # RGB
  offsets: [0.0, 0.0, 0.0]
  netScaleFactor: 1.0
  keepAspc: 0                     # training resized without keeping aspect ratio
  addFeatureNormalization: 0      # already unit-norm in the graph
  onnxFile: "{onnx_rel}"
  modelEngineFile: "{onnx_rel}_b{MAX_BATCH}_gpu0_fp16.engine"
  inputBlobName: "input"
  outputReidTensor: "reid_embedding"
"""
    stem = os.path.splitext(onnx_rel)[0]
    for name, text in [(f'config_infer_{stem}.txt', nvinfer), (f'tracker_reid_{stem}.yml', tracker)]:
        with open(os.path.join(out_dir, name), 'w') as fh:
            fh.write(text)
        logger(f'config: {os.path.join(out_dir, name)}')


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--weights', type=str, default='', help='stage-2 checkpoint (save_checkpoint format)')
    p.add_argument('--random-weights', action='store_true', help='no checkpoint: check the graph / TRT build only')
    p.add_argument('--num-classes', type=int, default=751, help='only with --random-weights')
    p.add_argument('--feature', choices=list(FEATURE_SLOTS), default=FEATURE)
    p.add_argument('--with-parts', action='store_true', default=WITH_PARTS)
    p.add_argument('--no-normalize-in-graph', dest='normalize_in_graph', action='store_false', default=NORMALIZE_IN_GRAPH)
    p.add_argument('--opset', type=int, default=ONNX_OPSET)
    p.add_argument('--static-batch', dest='dynamic', action='store_false', default=DYNAMIC_BATCH)
    p.add_argument('--out-dir', type=str, default=OUT_DIR)
    p.add_argument('--out', type=str, default='', help='onnx filename (default: stage2_<feature>_reid.onnx)')
    p.add_argument('--engine', action='store_true', help='also build a TensorRT engine with trtexec')
    p.add_argument('--fp16', action='store_true', help='fp16 engine (DeepStream network-mode=2)')
    args = p.parse_args()
    assert args.weights or args.random_weights, 'pass --weights <stage-2 checkpoint> (or --random-weights)'

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(args.out_dir, exist_ok=True)
    print(' '.join(sys.argv))

    model, knobs = build_model(args.weights, args.random_weights, args.num_classes)
    model = model.to(DEVICE).eval()
    size = (knobs['H'], knobs['W'])
    embedder = Stage2Embedder(model, args.feature, args.with_parts, args.normalize_in_graph).to(DEVICE).eval()
    check_against_reference(model, embedder, size)

    with torch.no_grad():
        dim = embedder(torch.rand(1, 3, *size, device=DEVICE) * 255)
        dim = (dim[0] if isinstance(dim, tuple) else dim).shape[1]
    path = os.path.join(args.out_dir, args.out or f'stage2_{args.feature}_reid.onnx')
    print(f'feature `{args.feature}` = cat({", ".join(FEATURE_SLOTS[args.feature])}) -> {dim}-d, input '
          f'{"RGB 0..255" if args.normalize_in_graph else "normalised"} '
          f'[{"-1" if args.dynamic else OPT_BATCH},3,{size[0]},{size[1]}]')
    export(embedder, path, size, args.opset, args.dynamic)
    verify_onnx(embedder, path, size, args.dynamic)
    write_deepstream_configs(path, dim, size, args.out_dir)
    if args.engine:
        verify_engine(embedder, build_engine(path, args.fp16, size), size, OPT_BATCH if args.dynamic else OPT_BATCH)


if __name__ == '__main__':
    main()
