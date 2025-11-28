# Korean Spacing Model (Char-CNN, PyTorch)
This repository provides a high-speed Korean spacing restoration model implemented with PyTorch.

The model:

- Learns from about 500k+ sentences (or more) where:
  - `original` contains the correctly spaced sentence.
  - `nospace` contains the same sentence with all spaces removed.
- Predicts, for each character, whether a space should follow.
- Is designed to be:
  - **Accurate enough for practical use.**
  - **Very fast on CPU.**
  - **GPU-aware** (uses CUDA automatically when available).
- Saves:
  - Best model and latest model.
  - Quantized versions (if `torchao` is available).

License: **MIT**

---

## Features

- Character-level CNN model (Embedding + 1D Conv + 1D Conv + Linear).
- Train/valid/test split with shuffling.
- Training loop with:
  - `AdamW` optimizer.
  - L2 regularization (`weight_decay`).
  - `ReduceLROnPlateau` scheduler.
  - Early stopping on validation F1.
  - Gradient clipping.
  - Progress bar using `tqdm`.
  - Per-epoch logging of:
    - Train loss / accuracy.
    - Validation loss / accuracy / precision / recall / F1.
    - Learning rate.
- Automatic device selection (`cuda` if available, else `cpu`).
- Model checkpoints saved under `./models`:
  - `best_model.pt`
  - `latest_model.pt`
  - `best_model_quantized.pt`
  - `latest_model_quantized.pt`
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
│  ├─ best_model_quantized.pt
│  └─ latest_model_quantized.pt
├─ spacing_train.py
└─ README.md
````

* `data/`: Directory containing TSV files used for training.
* `models/`: Directory where trained models and quantized variants are saved.
* `spacing_train.py`: Main training script (Char-CNN spacing model).

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
* During preprocessing:

  * The script uses `original` to:

    * Recompute the no-space string.
    * Build the label sequence (0 or 1 at each character position).
  * The `nospace` column is used as a hint only; if it disagrees with `original`,
    the `original`-based version is used.

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
   python spacing_train.py
   ```

   The script will:

   * Detect `cuda` if available, otherwise use `cpu`.
   * Read all `.tsv` files from `./data`.
   * Preprocess and validate samples.
   * Shuffle and split into train/valid/test.
   * Train the char-CNN model using AdamW.
   * Apply learning rate scheduling and early stopping.
   * Save:

     * Latest model: `./models/latest_model.pt`
     * Best model (by validation F1): `./models/best_model.pt`
     * Quantized versions:

       * `./models/latest_model_quantized.pt`
       * `./models/best_model_quantized.pt`

3. At the end of training, the script automatically:

   * Loads the best model (if present).
   * Evaluates on the test set.
   * Prints test metrics (loss, accuracy, precision, recall, F1).
   * Prints a few example spacing restorations.

---

## Model Files

Each `.pt` checkpoint saved by the script is a dictionary with keys:

* `model_state_dict`: State dict of the PyTorch model.
* `vocab`: Character-to-index mapping used during training.
* `config`: Model and training configuration dictionary.
* `quantized`: Boolean flag (`True` or `False`) for quantized checkpoints.
* `quant_backend`: Quantization backend identifier (e.g. `"torchao"` or `None`).

Example:

```python
{
    "model_state_dict": ...,
    "vocab": {...},
    "config": {...},
    "quantized": False,
    "quant_backend": None
}
```

---

## Inference Example

Below is an example script (`inference_example.py`) showing how to:

* Load the best model checkpoint.
* Reconstruct the same architecture.
* Restore spacing for input strings.

All code lines are heavily commented for clarity.

```python
import os
from typing import List, Dict, Any

import torch
from torch import nn

# Import the same model class definition used in spacing_train.py.
# If this file is separate, make sure to either:
#   1) Copy the model definition here, or
#   2) Import from the training module (e.g., `from spacing_train import CharCNNSpacingModel`)
from spacing_train import CharCNNSpacingModel, PAD_TOKEN, UNK_TOKEN


def load_model_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[nn.Module, Dict[str, int], Dict[str, Any]]:
    """
    Load a saved checkpoint file and reconstruct the model.

    Args:
        checkpoint_path: Path to the .pt file (e.g., best_model.pt).
        device:         Device to map the model to ("cpu" or "cuda").

    Returns:
        model:   Loaded PyTorch model in eval mode.
        vocab:   Character-to-index mapping (char2idx).
        config:  Configuration dictionary used during training.
    """
    # Load the checkpoint onto the desired device (map_location specifies this).
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract vocabulary and configuration information.
    char2idx: Dict[str, int] = checkpoint["vocab"]
    config: Dict[str, Any] = checkpoint["config"]

    # Create the model using the same hyperparameters used during training.
    model = CharCNNSpacingModel(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=config["num_classes"],
        pad_idx=config["pad_idx"],
        dropout=config["dropout"],
    ).to(device)

    # Load the learned parameters into the model.
    model.load_state_dict(checkpoint["model_state_dict"])

    # Set the model to evaluation mode (disables dropout, etc.).
    model.eval()

    return model, char2idx, config


def restore_spacing(
    text: str,
    model: nn.Module,
    char2idx: Dict[str, int],
    device: torch.device,
) -> str:
    """
    Restore spacing for an input string using the trained model.

    The function:
        - Removes all spaces from the input text.
        - Converts each character to an integer index.
        - Runs the model to decide where spaces should be inserted.
        - Reconstructs a spaced string according to model predictions.

    Args:
        text:     Input string (possibly without correct spaces).
        model:    Trained spacing model.
        char2idx: Character-to-index mapping.
        device:   Target device ("cpu" or "cuda").

    Returns:
        Restored text with predicted spaces inserted.
    """
    # Ensure the model is in evaluation mode.
    model.eval()

    # Remove all spaces from the input, since we want the model to decide spacing.
    no_space = text.replace(" ", "")

    # If the result is empty, simply return the original input.
    if not no_space:
        return text

    # Convert each character to its index, using UNK_TOKEN for unknown characters.
    unk_idx = char2idx.get(UNK_TOKEN)
    input_ids: List[int] = [
        char2idx.get(ch, unk_idx)
        for ch in no_space
    ]

    # Convert the list of indices into a tensor with shape [1, seq_len].
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    # Disable gradient calculation for inference.
    with torch.no_grad():
        # Forward pass: obtain logits for each position.
        logits = model(input_tensor)  # [1, seq_len, num_classes]

        # Predicted class indices at each position (0 or 1).
        preds = logits.argmax(dim=-1).squeeze(0)  # [seq_len]

    # Build the output string by inserting spaces where the model predicts label==1.
    output_chars: List[str] = []
    for ch, label in zip(no_space, preds.tolist()):
        output_chars.append(ch)
        if label == 1:
            # If the prediction is 1, insert a space after this character.
            output_chars.append(" ")

    # Join the list of characters into a single string and strip extra leading/trailing spaces.
    result = "".join(output_chars).strip()
    return result


def main() -> None:
    """
    Simple example:
        - Load the best model checkpoint.
        - Use it to restore spacing for a few example inputs.
    """
    # Decide which device to use (CUDA if available, otherwise CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using device: {device}")

    # Path to the best model checkpoint saved by spacing_train.py.
    checkpoint_path = os.path.join("models", "best_model.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            f"Train the model first with `python spacing_train.py`."
        )

    # Load model, vocabulary, and configuration.
    model, char2idx, config = load_model_checkpoint(checkpoint_path, device)

    # Some example inputs without proper spacing.
    examples = [
        "오늘날씨가좋다",
        "한국어띄어쓰기모델테스트중입니다",
        "이문장은공백이없습니다",
    ]

    print("\n[Examples] Spacing restoration")
    for text in examples:
        restored = restore_spacing(text, model, char2idx, device)
        print(f"Input : {text}")
        print(f"Output: {restored}")
        print("-" * 40)


if __name__ == "__main__":
    main()
```

Run:

```bash
python inference_example.py
```

This will load `models/best_model.pt` and print restored spacing for the given example inputs.

---

## Limit
### Invalid separation occurring inside compound nouns
Boundary prediction inside compound nouns is the weakest. Limitations that often appear in CNN-char-based models. Difficult to deeply understand semantic information. In particular, technical terms are difficult because they do not have dataset-based regularity.

### Long Relationship + Overlapping Configuration
The dependent clause is long, so the compartment is unclear without meaning information. Char-level CNNs are difficult to handle long-distance dependencies.

## Speed
Speed is fast in both general and quantized models. In addition, the loss of quantization models is very small (there is virtually no). However, it may vary depending on the PC specification, and the following figure is the average value of 100 repetitions.

### Normal Model
* Load:   0.130813s
* Processing:  0.152100s
* Unload: 0.043783s

### Quntization Model
* Load:   0.093397
* Processing:  0.018308
* Unload: 0.041244

---

## Hyperparameters

Default hyperparameters are defined at the top of `spacing_train.py`:

* Data and model paths:

  * `DATA_DIR = "./data"`
  * `MODEL_DIR = "./models"`
* Split ratios:

  * `TRAIN_RATIO = 0.8`
  * `VALID_RATIO = 0.1`
  * `TEST_RATIO = 0.1`
* Model:

  * `MAX_SEQ_LEN = 256` (truncate longer sequences)
  * `EMBED_DIM = 128`
  * `HIDDEN_DIM = 256`
  * `DROPOUT = 0.3`
* Training:

  * `BATCH_SIZE = 64`
  * `NUM_EPOCHS = 30`
  * `LEARNING_RATE = 1e-3`
  * `WEIGHT_DECAY = 1e-2`
  * Early stopping:

    * `PATIENCE = 5`
    * `MIN_DELTA = 1e-4`
  * Workers:

    * `NUM_WORKERS = 2`

You can edit these values directly in `spacing_train.py` to tune the model.

---

## Quantization

If `torchao` is installed, the script attempts to quantize the model using:

* `torchao.quantization.quantize_`
* `torchao.quantization.int8_dynamic_activation_int8_weight`

This reduces model size and can speed up CPU inference.

If `torchao` is not installed or quantization fails for any reason:

* The script still saves a non-quantized CPU model under the same `*_quantized.pt` file names.
* The checkpoint will include:

  * `"quantized": False`
  * `"quant_backend": None`

---

## License

This project is licensed under the **MIT License**.

You may copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to the conditions of the MIT License.