# GCSL: Graded Complementary Subspace Learning

A basic implementation of **Graded Complementary Subspace Learning** for generalizable AI-generated image and deepfake detection. Building on Effort's SVD residual adaptation, GCSL introduces multiple trainable low-rank residual groups into a frozen CLIP ViT-L/14 vision encoder and organizes them through capacity-graded cumulative-prefix supervision. The layer name `GCRLinear` follows the earlier research name, GCR.

## Method

Each Q/K/V/O projection in the visual Transformer is decomposed by SVD. The trailing seven singular components are partitioned in singular-value order into three groups with rank budgets `[4, 2, 1]`. The remaining components form the frozen principal weight:

$$W^{(m)}=W_{\mathrm{main}}+\sum_{k=1}^{m}U_k\operatorname{diag}(s_k)V_k^\top.$$

- **Multi-Subspace Residual Expansion:** optimize each group's left and right bases and scaling coefficients, together with a binary classification head. All other backbone parameters, including projection biases, remain frozen.
- **Capacity-Graded Nested Subspace Training (CG-NST):** uniformly sample one prefix depth per mini-batch and share it across all projections. Each batch uses one forward and backward pass. Cumulative rank budgets are `4 → 6 → 7`, with group activation probabilities `1`, `2/3`, and `1/3`.
- **Basis Orthogonality Regularization:** penalize basis overlap among the frozen principal subspace and active residual groups. This soft penalty does not enforce exact orthogonality or assign fixed semantic roles to individual groups.

For each projection, concatenate the principal and active residual column bases into $\widetilde U$ and $\widetilde V$, each with $q$ columns. The regularizer is

$$\ell_{\mathrm{orth}}^{(m)}=\frac{\|\widetilde U^\top\widetilde U-I_q\|_F^2+\|\widetilde V^\top\widetilde V-I_q\|_F^2}{2q^2}.$$

Average this penalty over all adapted projections. The training objective is cross-entropy plus $\lambda_{\mathrm{orth}}$ times the averaged penalty. Inference always activates the full prefix and predicts forgery probabilities in a single forward pass.

## Environment and Installation

Use Python 3.10, PyTorch 2.5.1, and Transformers 4.46.3. For GPU training, install a PyTorch build compatible with your CUDA environment before installing dependencies:

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The first run downloads `openai/clip-vit-large-patch14` from Hugging Face. For offline use, set `backbone` to a local directory containing the model and image processor files. Initialization performs SVD on 96 attention projections on CPU before moving the model to the selected device.

## Data Preparation

Inputs are extracted video frames with any required face cropping already applied. The training protocol uses FF++ c23. This repository does not distribute datasets or include face detection and video decoding. Split data by video to prevent frames from the same video appearing in both training and test sets.

Manifests are JSON arrays. Each entry contains a unique `video_id`, a binary `label` (`0` for real, `1` for fake), and a list of frame paths relative to `--data-root`. See [`examples/train.json`](examples/train.json):

```json
[
  {"video_id": "real_001", "label": 0, "frames": ["real/001/000.png", "real/001/001.png"]},
  {"video_id": "fake_001", "label": 1, "frames": ["fake/001/000.png", "fake/001/001.png"]}
]
```

List frames in temporal order. Training randomly samples up to 8 frames per video each epoch; evaluation uniformly samples up to 32. Videos with fewer frames use all available frames.

The CLIP image processor resizes, center-crops to 224 pixels, and normalizes each image. Training additionally applies horizontal flipping with probability 0.5. All frames inherit their video's label; verify frame labels when preparing partially manipulated videos.

## Training and Evaluation

Run from the repository root:

```bash
python -m experiments.gcsl.train --config configs/gcsl.yaml --train-json /path/to/train.json --data-root /path/to/frames --output outputs/gcsl
python -m experiments.gcsl.evaluate --checkpoint outputs/gcsl/epoch_10.pt --test-json /path/to/test.json --data-root /path/to/frames --output outputs/gcsl/test.json
```

Default settings are 10 epochs, batch size 64, AdamW, learning rate `2e-4`, weight decay `5e-4`, orthogonality coefficient 1, and random seed 1024. CUDA training enables mixed precision, while basis regularization remains in float32. Adjust resource-related settings in the YAML configuration and select a device with `--device cpu` or `--device cuda:0`.

Checkpoints store adapter parameters, the classification head, and the configuration. They omit the frozen pretrained backbone and optimizer state and support inference loading rather than training resumption. Reload with the same pretrained backbone and software versions. Logs are written to `logs/`; checkpoints and evaluation outputs are written to `outputs/gcsl/`.

Evaluation reports frame-level AUC in `[0, 1]`. Video-level AUC is computed after averaging frame forgery probabilities within each video. AUC is `null` for a single-class test manifest. Evaluate each target domain using its own manifest.

## Ablation Configurations

| Setting | `ranks` | `nst` | `lambda_orth` |
|---|---|---|---|
| GCSL | `[4,2,1]` | `true` | `1.0` |
| Static grouped | `[4,2,1]` | `false` | `1.0` |
| Balanced capacity | `[3,2,2]` | `true` | `1.0` |
| Reversed capacity | `[1,2,4]` | `true` | `1.0` |
| Without orthogonality | `[4,2,1]` | `true` | `0.0` |
| Single residual | `[1]` | `false` | `1.0` |

The single-residual setting is an internal control, rather than a complete reproduction of every Effort training detail. With CLIP ViT-L/14, the default 96 adapted projections and classification head contain approximately 1.38M trainable parameters. The exact count is printed at training startup.

## Project Structure and Checks

```text
gcsl/layers.py                 SVD grouping, low-rank forward, and basis regularization
gcsl/model.py                  CLIP adaptation, shared prefixes, and checkpoint interfaces
gcsl/data.py                   Video-frame manifests and sampling
experiments/gcsl/train.py      Training entry point
experiments/gcsl/evaluate.py   Full-prefix evaluation entry point
configs/gcsl.yaml              Default configuration
tests/test_layers.py          Mathematical and gradient checks
```

```bash
python -m unittest discover -s tests -v
```

Tests check full-prefix reconstruction of the initial projection, preservation of frozen weights across repeated forward passes, absence of gradients in inactive groups, and agreement between blockwise regularization and the full Gram formula. They use small random linear layers and require neither CLIP downloads nor datasets.

This release provides the core method and training and evaluation entry points. It does not include trained checkpoints or experimental results. Paper results require validation with the corresponding data splits and complete experimental setup.

## Acknowledgments

The method builds on the spectral residual adaptation idea of [Effort](https://github.com/YZY-stack/Effort-AIGI-Detection) and uses pretrained visual representations from [CLIP](https://github.com/openai/CLIP). Refer to [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) for dataset preparation and evaluation protocols. This repository implements the core method through an independent lightweight training pipeline. Upstream models and datasets remain subject to their respective licenses and terms.
