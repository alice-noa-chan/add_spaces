아래는 **새 코드에 맞게 싹 정리한 `README.md` 초안**이야.
(기존 내용 최대한 살리면서, 모델/훈련/체크포인트/하이퍼파라미터 쪽만 최신 코드 기준으로 업데이트함.)

````markdown
# Korean Spacing Model (Improved Char-CNN, PyTorch)

This repository provides a high-speed Korean spacing restoration model implemented with PyTorch.

The model:

- Learns from about 500k+ sentences (or more) where:
  - `original` contains the correctly spaced sentence.
  - `nospace` contains the same sentence with all spaces removed.
- Predicts, for each character, whether a space should follow.
- Is designed to be:
  - **Accurate enough for practical use.**
  - **Very fast on CPU.**
  - **GPU-aware** (uses CUDA automatically when available and mixed precision when possible).
- Saves:
  - Best model and latest model.
  - Quantized best model (if `torchao` is available, or a non-quantized CPU fallback).

License: **MIT**

---

## Features

- **Improved character-level CNN model**:
  - Character embedding with **learnable positional embeddings**.
  - **Multi-scale 1D convolution block** (kernel sizes e.g. 3, 5, 7).
  - Stack of **bidirectional dilated convolution blocks** with:
    - Forward & backward dilated convolutions.
    - Learnable gating between forward/backward contexts.
    - **Squeeze-and-Excitation (SE)** channel attention.
    - **LayerNorm**, **Dropout**, and **DropPath (stochastic depth)**.
  - Lightweight position-wise classifier with residual shortcut.
- **Data pipeline**:
  - Train/valid/test split with shuffling.
  - Character-level vocabulary with special tokens:
    - `<PAD>`, `<UNK>`, `<MASK>`.
- **Data augmentation** (for training):
  - Random **masking** of characters using `<MASK>` token.
  - Random **adjacent swaps** of characters (and labels) to introduce noise while preserving alignment.
  - Optional **random cropping** for sequences longer than `MAX_SEQ_LEN`.
- **Training loop** with:
  - `AdamW` optimizer.
  - L2 regularization (`weight_decay`).
  - **Focal Loss** with **label smoothing** for robust training on imbalanced labels:
    - Per-character space prediction (0: no space, 1: space).
    - `alpha` / `gamma` for hard example mining.
  - **CosineAnnealingWarmRestarts** learning rate scheduler.
  - Early stopping on validation F1.
  - Gradient clipping.
  - **Mixed precision training (AMP)** using `torch.cuda.amp` (if CUDA is available).
  - Progress bar using `tqdm`.
  - Per-epoch logging of:
    - Train loss / accuracy.
    - Validation loss / accuracy / precision / recall / F1.
    - Learning rate.
- Automatic device selection (`cuda` if available, else `cpu`).
- Model checkpoints saved under `./models`:
  - `best_model.pt`
  - `latest_model.pt`
  - `best_model_quantized.pt` (quantized or non-quantized CPU fallback)
- Optional quantization using `torchao` (if installed).

---

## Project Structure

```text
.
├─ data/
│  ├─ train_1.tsv
│  ├─ train_2.tsv
│  └─ ...
├─ models/
│  ├─ best_model.pt
│  ├─ latest_model.pt
│  └─ best_model_quantized.pt
├─ train.py
└─ README.md
````

* `data/`: Directory containing TSV files used for training.
* `models/`: Directory where trained models and quantized variants are saved.
* `train.py`: Main training script (improved Char-CNN spacing model).

---

## Data Format

Each `.tsv` file inside `./data` must:

* Be UTF-8 encoded.
* Use tab (`\t`) as delimiter.
* Contain at least the following header columns:

```tsv
original    nospace
오늘 날씨가 좋다.    오늘날씨가좋다.
이 문장은 예시 입니다.    이문장은예시입니다.
...
```

* `original`: Ground-truth sentence with correct spacing.
* `nospace`: Same sentence with all spaces removed.

During preprocessing:

* The script uses `original` to:

  * Recompute the no-space string.
  * Build the label sequence (0 or 1 at each character position).
* The `nospace` column is used as a hint only; if it disagrees with `original`,
  the `original`-based version is used.
* Samples with length mismatches or empty content are skipped, with statistics
  printed to the console.

---

## Requirements

* Python 3.13 (or compatible Python 3.x).
* PyTorch (CPU or CUDA build).
* tqdm
* Optionally:

  * `numpy` (for more stable seeding).
  * `torchao` (for quantization, optional).

Example using `pip`:

```bash
pip install torch tqdm
# Optional:
pip install numpy
pip install torchao
```

Make sure you install a PyTorch build appropriate for your environment
(CPU-only or CUDA).

---

## Training

1. Prepare the data:

   * Place all `.tsv` files under `./data`.
   * Ensure each file has `original` and `nospace` columns as described above.

2. Run training:

   ```bash
   python train.py
   ```

   The script will:

   * Detect `cuda` if available, otherwise use `cpu`.
   * Enable **mixed precision** (AMP) automatically when running on CUDA.
   * Read all `.tsv` files from `./data`.
   * Preprocess and validate samples.
   * Shuffle and split into train/valid/test.
   * Build a character vocabulary including `<PAD>`, `<UNK>`, `<MASK>`.
   * Train the improved char-CNN model with:

     * Focal loss + label smoothing.
     * Cosine annealing warm restarts scheduler.
     * Data augmentation (masking / swaps / random crop) on the training set.
   * Apply learning rate scheduling and early stopping based on validation F1.
   * Save:

     * Latest model: `./models/latest_model.pt`
     * Best model (by validation F1): `./models/best_model.pt`
     * Quantized (or CPU fallback) best model:

       * `./models/best_model_quantized.pt`

3. At the end of training, the script automatically:

   * Loads the best model (if present).
   * Evaluates on the test set.
   * Prints test metrics (loss, accuracy, precision, recall, F1).
   * Prints a few example spacing restorations from the test split.

---

## Model Files

The script saves two kinds of checkpoints:

1. **Standard checkpoints** (`best_model.pt`, `latest_model.pt`)

   ```python
   {
       "model_state_dict": ...,
       "vocab": {...},   # char2idx mapping
       "config": {...},  # model & training configuration
   }
   ```

2. **Quantized (or quantization-fallback) checkpoint** (`best_model_quantized.pt`)

   ```python
   {
       "model_state_dict": ...,
       "vocab": {...},
       "config": {...},
       "quantized": True or False,
       "quant_backend": "torchao" or None,
   }
   ```

* `quantized = True` and `quant_backend = "torchao"` when quantization succeeds.
* If quantization fails or `torchao` is not installed, the script still saves a **CPU model** under `best_model_quantized.pt` with:

  * `"quantized": False`
  * `"quant_backend": None`

You can load and use these checkpoints with standard PyTorch APIs.

---

## Inference Example

Below is an example script (`inference_example.py`) showing how to:

* Load the best model checkpoint.
* Reconstruct the same architecture.
* Restore spacing for input strings.

Run:

```bash
python inference_example.py
```

This will load `models/best_model.pt` and print restored spacing for the given example inputs.

---

## Limitations

### Invalid separation occurring inside compound nouns

Boundary prediction inside compound nouns is the weakest. This is a common
limitation of character-level CNN-based models, which can struggle to deeply
understand semantic information. In particular, technical terms are difficult
because they often lack clear, dataset-level regularities.

### Long-distance dependencies & overlapping structures

When the dependent clause is long, the correct spacing sometimes requires
deeper semantic understanding. Even with bidirectional dilated convolutions,
very long-distance dependencies remain challenging for purely char-level CNNs.

---

## Speed

Speed is fast in both general and quantized models. In addition, the loss of
accuracy from quantization is very small (virtually none in many settings).

However, actual speed depends on your hardware (CPU / GPU). Benchmarking with
100 repetitions on your own environment is recommended for precise numbers.

---

## Hyperparameters

Default hyperparameters are defined at the top of `train.py`:

* Data and model paths:

  * `DATA_DIR = "./data"`
  * `MODEL_DIR = "./models"`
* Split ratios:

  * `TRAIN_RATIO = 0.8`
  * `VALID_RATIO = 0.1`
  * `TEST_RATIO = 0.1`
* Model:

  * `MAX_SEQ_LEN = 256` (truncate or crop longer sequences)
  * `EMBED_DIM = 128`
  * `HIDDEN_DIM = 256`
  * `DROPOUT = 0.2`
  * `DROP_PATH_RATE = 0.1`
* Training:

  * `BATCH_SIZE = 64`
  * `NUM_EPOCHS = 50`
  * `LEARNING_RATE = 2e-3`
  * `WEIGHT_DECAY = 1e-2`
  * Focal loss:

    * `FOCAL_ALPHA = 0.75`
    * `FOCAL_GAMMA = 2.0`
  * Label smoothing:

    * `LABEL_SMOOTHING = 0.1`
  * Early stopping:

    * `PATIENCE = 10`
    * `MIN_DELTA = 1e-4`
  * Workers:

    * `NUM_WORKERS = 2`
* Data augmentation:

  * `AUG_MASK_PROB = 0.1`   (probability to mask a character)
  * `AUG_SWAP_PROB = 0.05`  (probability to swap adjacent characters)

You can edit these values directly in `train.py` to tune the model.

---

## Quantization

If `torchao` is installed, the script attempts to quantize the model using:

* `torchao.quantization.quantize_`
* `torchao.quantization.int8_dynamic_activation_int8_weight`

This reduces model size and can speed up CPU inference.

If `torchao` is not installed or quantization fails for any reason:

* The script still saves a (non-quantized) CPU model under the same
  `best_model_quantized.pt` file name.
* The checkpoint will include:

  * `"quantized": False`
  * `"quant_backend": None`

---

## License

This project is licensed under the **MIT License**.

You may copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, subject to the conditions of the MIT License.