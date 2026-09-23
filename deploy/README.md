# deploy — stage-2 ReID embedder for DeepStream

`export_stage2_onnx.py` turns a stage-2 checkpoint (`processor/train_part_prompts_stage2.py`) into an ONNX
embedder that DeepStream can build a TensorRT engine from, plus the two config files that consume it. The
script's docstring holds the design decisions; this file is the how-to.

## Export

```bash
# from the repo root, in the py312 env
python deploy/export_stage2_onnx.py \
    --weights work_dirs/market1501/part_prompts_stage2/RN50_part_prompts_stage2_120.pth \
    --feature holistic --engine --fp16
```

Nothing but the checkpoint is needed: the class count, the LPIM text queries and the H/W/stride knobs all come
out of it. `--random-weights` runs the same path on a CLIP-pretrained, randomly-headed model to check the graph
and the TensorRT build without a checkpoint.

Outputs in `deploy/onnx/`:

| file | what |
| --- | --- |
| `stage2_<feature>_reid.onnx` | input `input` `[-1,3,256,128]` RGB 0..255, output `reid_embedding` `[-1,D]`, unit-norm |
| `stage2_<feature>_reid_b32_fp16.engine` | only with `--engine`; TensorRT builds are host/version specific, so rebuild on the deployment machine |
| `config_infer_stage2_<feature>_reid.txt` | nvinfer secondary-GIE config |
| `tracker_reid_stage2_<feature>_reid.yml` | `ReID:` block for the NvDCF / NvDeepSORT tracker config |

| `--feature` | vector | dim | Market-1501 (M1 run, epoch 84) |
| --- | --- | --- | --- |
| `baseline` | `cat(gap4, g)` | 3072 | 89.3 mAP / 95.4 R1 |
| `lpim` | `cat(z0, pbar)` | 2048 | see the stage-2 log's `lpim` row |
| `holistic` | `cat(gap4, g, z0, pbar)` | 5120 | 89.2 mAP / 95.3 R1 |

Pick the row that wins in the stage-2 eval log of the checkpoint being exported; `holistic` is the default.
The `part_lse` row cannot be exported (it is a mutual-visibility distance, not an embedding) — `--with-parts`
adds `part_embeddings [-1,K,D]` and `part_visibility [-1,K]` for a custom matcher instead.

## Checks the script runs

1. the rewritten attention blocks vs `PartCLIPReID`'s own slots (`< 2e-4`);
2. the assembled embedding vs `evaluate()`'s row, built by `concat_feature` from the training script;
3. ONNX Runtime vs PyTorch, and the min/max batch of the profile;
4. with `--engine`: the TensorRT engine vs PyTorch (cosine `> 0.999`; fp16 costs ~2e-3 per component).

Any of them failing aborts the export.

## DeepStream

Two ways to use it; both need normalisation left at `net-scale-factor=1.0` / `offsets=0;0;0`, because the
graph does `(x - 255*mean) / (255*std)` itself (nvinfer applies one scalar scale and cannot do a per-channel std).

* **Tracker ReID** (usual case): paste the `ReID:` block into `config_tracker_NvDCF_accuracy.yml`, set
  `reidType: 2`, and check `reidFeatureSize` matches the exported dim. The embedding is already L2-normalised,
  so keep `addFeatureNormalization: 0`.
* **Secondary GIE**: `config_infer_stage2_*.txt` runs the model on the primary detector's person boxes with
  `network-type=100` (no bbox parsing) and `output-tensor-meta=1`; read the vector from
  `NvDsInferTensorMeta` in a probe.

Batching: the graph is exported with a dynamic batch dim and a `1..32` profile (`MIN/OPT/MAX_BATCH`). Keep
DeepStream's `batch-size` (or the tracker's `batchSize`) inside that range, or re-export with a wider profile.
`--static-batch` freezes the batch to `OPT_BATCH` for runtimes that reject dynamic shapes.

Engine file names differ per DeepStream version (it appends `_b<N>_gpu0_<mode>.engine` itself) — deleting the
`model-engine-file` and letting DeepStream build from the ONNX on first run is the safe path when in doubt.
