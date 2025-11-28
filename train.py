import os
import csv
import random
import copy
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


# ============================
# Global hyperparameters
# ============================

RANDOM_SEED: int = 42

DATA_DIR: str = "./data"
MODEL_DIR: str = "./models"

TRAIN_RATIO: float = 0.8
VALID_RATIO: float = 0.1
TEST_RATIO: float = 0.1

# If None, do not truncate sequences
MAX_SEQ_LEN: int | None = 256

BATCH_SIZE: int = 64
NUM_EPOCHS: int = 30

EMBED_DIM: int = 128
HIDDEN_DIM: int = 256
DROPOUT: float = 0.3
NUM_CLASSES: int = 2  # no-space / space

LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-2  # L2 via AdamW weight_decay

PATIENCE: int = 5
MIN_DELTA: float = 1e-4

NUM_WORKERS: int = 2

PAD_TOKEN: str = "<PAD>"
UNK_TOKEN: str = "<UNK>"


def set_seed(seed: int) -> None:
    """
    Fix random seeds for reproducibility.
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_tsv_files(data_dir: str) -> List[Tuple[str, str]]:
    """
    Read all .tsv files in data_dir and return a list of (original, nospace) pairs.
    Each TSV is expected to have 'original' and 'nospace' columns.
    """
    records: List[Tuple[str, str]] = []

    data_path = Path(data_dir)
    tsv_files = list(data_path.glob("*.tsv"))

    if not tsv_files:
        raise FileNotFoundError(f"No .tsv files found in data directory: {data_dir!r}")

    for tsv_path in tsv_files:
        with tsv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if "original" not in reader.fieldnames or "nospace" not in reader.fieldnames:
                raise ValueError(
                    f"File {tsv_path} does not contain 'original' and 'nospace' columns. "
                    f"Header: {reader.fieldnames}"
                )
            for row in reader:
                original = (row.get("original") or "").strip()
                nospace = (row.get("nospace") or "").strip()
                if not original:
                    # Skip empty lines
                    continue
                records.append((original, nospace))

    if not records:
        raise ValueError("No valid records found in TSV files.")

    return records


def build_sample_from_original(original: str) -> Tuple[str, List[int]]:
    """
    From an original spaced sentence, build a no-space string and labels indicating
    whether a space followed each character (0 or 1).
    """
    no_space_chars: List[str] = []
    labels: List[int] = []

    i = 0
    length = len(original)

    while i < length:
        ch = original[i]
        if ch == " ":
            i += 1
            continue

        no_space_chars.append(ch)

        label = 0
        if i + 1 < length and original[i + 1] == " ":
            label = 1
        labels.append(label)

        i += 1

    no_space = "".join(no_space_chars)
    return no_space, labels


def prepare_samples(raw_records: List[Tuple[str, str]]) -> List[Tuple[str, List[int]]]:
    """
    Convert (original, nospace) records into (input_text, label_sequence) samples.

    - Use 'original' as the single source of truth for labels.
    - Use nospace column only as a hint; if it disagrees with 'original',
      we trust the original-based no_space string.
    """
    samples: List[Tuple[str, List[int]]] = []
    mismatch_count = 0
    skip_count = 0

    for original, nospace in raw_records:
        no_space_from_orig, labels = build_sample_from_original(original)

        nospace_clean = nospace.replace(" ", "") if nospace else ""
        nospace_clean = nospace_clean.strip()

        if nospace_clean and nospace_clean != no_space_from_orig:
            mismatch_count += 1
            input_text = no_space_from_orig
        else:
            input_text = nospace_clean if nospace_clean else no_space_from_orig

        if not input_text:
            skip_count += 1
            continue

        if len(input_text) != len(labels):
            skip_count += 1
            continue

        samples.append((input_text, labels))

    if not samples:
        raise ValueError("No valid samples after preprocessing. Check data quality.")

    print(f"[Preprocessing] Total raw samples: {len(raw_records)}")
    print(f"[Preprocessing] Valid samples used: {len(samples)}")
    print(f"[Preprocessing] nospace mismatches corrected by original: {mismatch_count}")
    print(f"[Preprocessing] Skipped samples (length mismatch / empty): {skip_count}")

    return samples


def build_vocab(samples: List[Tuple[str, List[int]]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build character vocabulary from all samples.
    Returns char2idx and idx2char dicts including PAD and UNK tokens.
    """
    from collections import Counter

    counter = Counter()
    for text, _ in samples:
        counter.update(text)

    chars = sorted(counter.keys())

    char2idx: Dict[str, int] = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }
    for i, ch in enumerate(chars, start=2):
        char2idx[ch] = i

    idx2char: Dict[int, str] = {idx: ch for ch, idx in char2idx.items()}

    print(f"[Vocab] Unique characters: {len(chars)}")
    print(f"[Vocab] Vocab size (with specials): {len(char2idx)}")

    return char2idx, idx2char


class SpacingDataset(Dataset):
    """
    Dataset holding (input_string, label_sequence).
    """

    def __init__(
        self,
        samples: List[Tuple[str, List[int]]],
        char2idx: Dict[str, int],
        max_seq_len: int | None = None,
    ) -> None:
        self.samples = samples
        self.char2idx = char2idx
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.samples)

    def _encode_text(self, text: str) -> List[int]:
        """
        Convert text into a list of character indices.
        """
        ids: List[int] = []
        for ch in text:
            idx = self.char2idx.get(ch, self.char2idx[UNK_TOKEN])
            ids.append(idx)
        return ids

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (input_tensor, label_tensor) for one sample.
        """
        text, labels = self.samples[idx]

        if self.max_seq_len is not None and len(text) > self.max_seq_len:
            text = text[: self.max_seq_len]
            labels = labels[: self.max_seq_len]

        input_ids = self._encode_text(text)
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.long)

        return input_tensor, label_tensor


class SpacingCollator:
    """
    Collate function for DataLoader.

    Pads variable-length sequences to the max length in the batch.
    Labels at padded positions are set to -100 so that CrossEntropyLoss
    can ignore them via ignore_index.
    """

    def __init__(self, pad_idx: int) -> None:
        self.pad_idx = pad_idx

    def __call__(
        self,
        batch: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(batch)
        lengths = [len(x[0]) for x in batch]
        max_len = max(lengths)

        inputs_padded = torch.full(
            (batch_size, max_len),
            self.pad_idx,
            dtype=torch.long,
        )

        labels_padded = torch.full(
            (batch_size, max_len),
            -100,
            dtype=torch.long,
        )

        for i, (input_tensor, label_tensor) in enumerate(batch):
            seq_len = len(input_tensor)
            inputs_padded[i, :seq_len] = input_tensor
            labels_padded[i, :seq_len] = label_tensor

        lengths_tensor = torch.tensor(lengths, dtype=torch.long)

        return inputs_padded, labels_padded, lengths_tensor


class CharCNNSpacingModel(nn.Module):
    """
    Char-level CNN model for spacing:
    - Embedding
    - 1D convolution layers
    - Position-wise linear classifier
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        pad_idx: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        self.conv1 = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=hidden_dim,
            kernel_size=5,
            padding=2,
        )
        self.conv2 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=5,
            padding=2,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [batch_size, seq_len]
        returns logits: [batch_size, seq_len, num_classes]
        """
        x = self.embedding(input_ids)           # [B, L, E]
        x = x.transpose(1, 2)                   # [B, E, L]

        x = self.conv1(x)                       # [B, H, L]
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)                       # [B, H, L]
        x = torch.relu(x)
        x = self.dropout(x)

        x = x.transpose(1, 2)                   # [B, L, H]
        logits = self.fc(x)                     # [B, L, C]
        return logits


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train the model for one epoch and return (avg_loss, avg_accuracy).
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    pbar = tqdm(data_loader, desc="Train", leave=False)

    for batch in pbar:
        inputs, labels, lengths = batch

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(inputs)

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        with torch.no_grad():
            mask = labels != -100
            valid_count = mask.sum().item()

            total_loss += loss.item() * valid_count

            preds = logits.argmax(dim=-1)
            correct = (preds[mask] == labels[mask]).sum().item()
            total_correct += correct
            total_count += valid_count

            avg_loss = total_loss / max(total_count, 1)
            avg_acc = total_correct / max(total_count, 1)

            pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_loss, avg_acc


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Eval",
) -> Tuple[float, float, float, float, float]:
    """
    Evaluate the model and return:
        (avg_loss, avg_acc, precision, recall, f1)
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    tp = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        pbar = tqdm(data_loader, desc=desc, leave=False)

        for batch in pbar:
            inputs, labels, lengths = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

            mask = labels != -100
            valid_count = mask.sum().item()

            total_loss += loss.item() * valid_count

            preds = logits.argmax(dim=-1)

            mask_flat = mask.view(-1)
            true_flat = labels.view(-1)[mask_flat]
            pred_flat = preds.view(-1)[mask_flat]

            correct = (pred_flat == true_flat).sum().item()
            total_correct += correct
            total_count += valid_count

            tp += ((pred_flat == 1) & (true_flat == 1)).sum().item()
            fp += ((pred_flat == 1) & (true_flat == 0)).sum().item()
            fn += ((pred_flat == 0) & (true_flat == 1)).sum().item()

            avg_loss = total_loss / max(total_count, 1)
            avg_acc = total_correct / max(total_count, 1)

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                acc=f"{avg_acc:.4f}",
                f1=f"{f1:.4f}",
            )

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return avg_loss, avg_acc, precision, recall, f1


def save_model(
    model: nn.Module,
    save_path: str,
    char2idx: Dict[str, int],
    config: Dict[str, Any],
) -> None:
    """
    Save model state, vocabulary, and config as a checkpoint.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "vocab": char2idx,
        "config": config,
    }

    torch.save(checkpoint, save_path)
    print(f"[Save] Model saved: {save_path}")


def save_quantized_model(
    model: nn.Module,
    save_path: str,
    char2idx: Dict[str, int],
    config: Dict[str, Any],
) -> None:
    """
    Save a quantized version of the model if possible.

    Preferred path (no deprecation warning):
      - Use torchao.quantization.quantize_ if torchao is installed. :contentReference[oaicite:1]{index=1}

    If torchao is not available or quantization fails, fall back to saving
    the plain (non-quantized) CPU model under the same path.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model_cpu = copy.deepcopy(model).to("cpu")
    model_cpu.eval()

    checkpoint: Dict[str, Any]

    try:
        # New recommended quantization API via torchao (if installed)
        from torchao.quantization import (
            quantize_,
            int8_dynamic_activation_int8_weight,
        )

        q_model = quantize_(model_cpu, int8_dynamic_activation_int8_weight())
        checkpoint = {
            "model_state_dict": q_model.state_dict(),
            "vocab": char2idx,
            "config": config,
            "quantized": True,
            "quant_backend": "torchao",
        }
        torch.save(checkpoint, save_path)
        print(f"[Save] Quantized model saved (torchao) to: {save_path}")

    except Exception as e:
        # torchao not installed or quantization failed; save plain model instead.
        checkpoint = {
            "model_state_dict": model_cpu.state_dict(),
            "vocab": char2idx,
            "config": config,
            "quantized": False,
            "quant_backend": None,
        }
        torch.save(checkpoint, save_path)
        print(
            f"[Save] Could not quantize with torchao, saved non-quantized model instead: {save_path}\n"
            f"       (info: {e})"
        )


def restore_spacing(
    text: str,
    model: nn.Module,
    char2idx: Dict[str, int],
    device: torch.device,
) -> str:
    """
    Restore spacing for a given string using a trained model.
    Input text is stripped of spaces and then spaces are re-inserted
    according to model predictions.
    """
    model.eval()

    no_space = text.replace(" ", "")

    if not no_space:
        return text

    input_ids = [
        char2idx.get(ch, char2idx[UNK_TOKEN])
        for ch in no_space
    ]
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        preds = logits.argmax(dim=-1).squeeze(0)

    output_chars: List[str] = []
    for ch, label in zip(no_space, preds.tolist()):
        output_chars.append(ch)
        if label == 1:
            output_chars.append(" ")

    result = "".join(output_chars).strip()
    return result


def main() -> None:
    """
    Full training pipeline:
        1. Load & preprocess data
        2. Split into train/valid/test
        3. Build datasets and dataloaders
        4. Train with AdamW + ReduceLROnPlateau + early stopping
        5. Save best/latest + quantized versions
        6. Evaluate on test set and print examples
    """
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using device: {device}")

    raw_records = read_tsv_files(DATA_DIR)
    samples = prepare_samples(raw_records)

    char2idx, idx2char = build_vocab(samples)
    vocab_size = len(char2idx)
    pad_idx = char2idx[PAD_TOKEN]

    random.shuffle(samples)

    num_samples = len(samples)
    num_train = int(num_samples * TRAIN_RATIO)
    num_valid = int(num_samples * VALID_RATIO)
    num_test = num_samples - num_train - num_valid

    train_samples = samples[:num_train]
    valid_samples = samples[num_train: num_train + num_valid]
    test_samples = samples[num_train + num_valid:]

    print(f"[Split] train: {len(train_samples)}, valid: {len(valid_samples)}, test: {len(test_samples)}")

    train_dataset = SpacingDataset(
        train_samples,
        char2idx=char2idx,
        max_seq_len=MAX_SEQ_LEN,
    )
    valid_dataset = SpacingDataset(
        valid_samples,
        char2idx=char2idx,
        max_seq_len=MAX_SEQ_LEN,
    )
    test_dataset = SpacingDataset(
        test_samples,
        char2idx=char2idx,
        max_seq_len=MAX_SEQ_LEN,
    )

    collate_fn = SpacingCollator(pad_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    model = CharCNNSpacingModel(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        pad_idx=pad_idx,
        dropout=DROPOUT,
    ).to(device)

    def init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    config: Dict[str, Any] = {
        "vocab_size": vocab_size,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_classes": NUM_CLASSES,
        "pad_idx": pad_idx,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_seq_len": MAX_SEQ_LEN,
        "train_ratio": TRAIN_RATIO,
        "valid_ratio": VALID_RATIO,
        "test_ratio": TEST_RATIO,
        "random_seed": RANDOM_SEED,
        "model_type": "CharCNNSpacingModel",
    }

    best_model_path = os.path.join(MODEL_DIR, "best_model.pt")
    latest_model_path = os.path.join(MODEL_DIR, "latest_model.pt")
    best_model_quant_path = os.path.join(MODEL_DIR, "best_model_quantized.pt")
    latest_model_quant_path = os.path.join(MODEL_DIR, "latest_model_quantized.pt")

    best_val_f1 = 0.0
    epochs_without_improvement = 0

    print("[Train] Start training")
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{NUM_EPOCHS} =====")

        train_loss, train_acc = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(
            model=model,
            data_loader=valid_loader,
            criterion=criterion,
            device=device,
            desc="Valid",
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
            f"Val Precision: {val_prec:.4f}, Val Recall: {val_rec:.4f}, Val F1: {val_f1:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        scheduler.step(val_loss)

        save_model(model, latest_model_path, char2idx, config)
        save_quantized_model(model, latest_model_quant_path, char2idx, config)

        if val_f1 > best_val_f1 + MIN_DELTA:
            best_val_f1 = val_f1
            epochs_without_improvement = 0

            print(
                f"[Epoch {epoch}] Validation F1 improved to {best_val_f1:.4f} -> saving new best model"
            )
            save_model(model, best_model_path, char2idx, config)
            save_quantized_model(model, best_model_quant_path, char2idx, config)

        else:
            epochs_without_improvement += 1
            print(
                f"[Epoch {epoch}] No validation F1 improvement "
                f"({epochs_without_improvement}/{PATIENCE})"
            )

        if epochs_without_improvement >= PATIENCE:
            print(
                f"[EarlyStopping] No improvement for {PATIENCE} epochs. Stopping training."
            )
            break

    print("\n[Train] Training finished")

    if os.path.exists(best_model_path):
        print("[Test] Loading best model for test evaluation")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("[Test] No best model found, using latest model for test evaluation")

    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        desc="Test",
    )

    print(
        f"[Test] Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, "
        f"Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}"
    )

    print("\n[Examples] Spacing restoration on a few test samples:")
    model.eval()
    num_examples = min(5, len(test_samples))
    for i in range(num_examples):
        no_space_text, labels = test_samples[i]
        restored = restore_spacing(no_space_text, model, char2idx, device)
        print(f"Input (no space): {no_space_text}")
        print(f"Model output   : {restored}")
        print("-" * 50)


if __name__ == "__main__":
    main()
