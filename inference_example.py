import os
import time
import gc
from typing import List, Dict, Any, Tuple

import torch
from torch import nn
import torch.nn.functional as F


# ============================
# Special tokens (must match training)
# ============================

PAD_TOKEN: str = "<PAD>"
UNK_TOKEN: str = "<UNK>"
MASK_TOKEN: str = "<MASK>"


# ============================
# Model components
# ============================

class DropPath(nn.Module):
    """
    Stochastic Depth (drop path) for regularization.
    Randomly drops entire residual paths during training.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, ...].

        Returns:
            Tensor with some paths randomly dropped during training.
        """
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        # Broadcast random tensor over all non-batch dimensions.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        random_tensor = random_tensor.div(keep_prob)
        return x * random_tensor


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise attention.

    It re-weights each channel using global average pooled features.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, C, L].

        Returns:
            Tensor of shape [B, C, L] with channel-wise re-weighting.
        """
        # Global average pooling over sequence length.
        s = x.mean(dim=-1)  # [B, C]
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))  # [B, C]
        # Rescale each channel.
        return x * s.unsqueeze(-1)


class MultiScaleConvBlock(nn.Module):
    """
    Multi-scale convolution block.

    It uses several convolution branches with different kernel sizes
    in parallel and concatenates the outputs along the channel dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        assert len(kernel_sizes) > 0, "kernel_sizes must not be empty."

        self.convs = nn.ModuleList()
        num_branches = len(kernel_sizes)

        # Distribute out_channels across branches.
        base_channels = out_channels // num_branches
        extra = out_channels - base_channels * num_branches

        branch_channels: List[int] = []
        for i in range(num_branches):
            ch = base_channels + (1 if i < extra else 0)
            branch_channels.append(ch)

        # Create one Conv1d for each kernel size.
        for k, ch in zip(kernel_sizes, branch_channels):
            padding = k // 2
            self.convs.append(
                nn.Conv1d(in_channels, ch, kernel_size=k, padding=padding)
            )

        # LayerNorm is applied over the channel dimension after transpose to [B, L, C].
        self.layer_norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, C_in, L].

        Returns:
            Tensor of shape [B, C_out, L].
        """
        # Parallel convolutions.
        outputs = [conv(x) for conv in self.convs]  # each [B, C_branch, L]
        out = torch.cat(outputs, dim=1)  # [B, C_out, L]
        out = F.gelu(out)

        # Apply LayerNorm over channels.
        out = out.transpose(1, 2)  # [B, L, C_out]
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = out.transpose(1, 2)  # [B, C_out, L]

        return out


class BidirectionalDilatedBlock(nn.Module):
    """
    Bidirectional dilated convolution block.

    It processes the sequence in both forward and backward directions
    using dilated convolutions and combines them with a learnable gate.
    A Squeeze-and-Excitation block and LayerNorm are applied as well.
    """

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()

        padding = dilation * (kernel_size - 1) // 2

        # Forward convolution.
        self.conv_forward = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        # Backward convolution.
        self.conv_backward = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        # Gate for combining forward and backward features.
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.se = SqueezeExcitation(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, H, L].

        Returns:
            Tensor of shape [B, H, L] after bidirectional dilated convolutions
            with gating, SE, residual connection, and LayerNorm.
        """
        residual = x

        # Forward branch.
        forward_out = F.gelu(self.conv_forward(x))  # [B, H, L]

        # Backward branch: reverse, convolve, reverse back.
        backward_in = torch.flip(x, dims=[-1])
        backward_out = F.gelu(self.conv_backward(backward_in))
        backward_out = torch.flip(backward_out, dims=[-1])  # [B, H, L]

        # Concatenate forward and backward features.
        combined = torch.cat([forward_out, backward_out], dim=1)  # [B, 2H, L]
        combined = combined.transpose(1, 2)  # [B, L, 2H]

        # Compute gate.
        gate = torch.sigmoid(self.gate(combined))  # [B, L, H]
        gate = gate.transpose(1, 2)  # [B, H, L]

        # Weighted combination.
        out = gate * forward_out + (1.0 - gate) * backward_out

        # Channel-wise attention.
        out = self.se(out)

        # Dropout + stochastic depth + residual.
        out = self.dropout(out)
        out = residual + self.drop_path(out)

        # LayerNorm over channel dimension.
        out = out.transpose(1, 2)  # [B, L, H]
        out = self.layer_norm(out)
        out = out.transpose(1, 2)  # [B, H, L]

        return out


class CharCNNSpacingModelImproved(nn.Module):
    """
    Improved character-level CNN model for Korean spacing correction.

    Architecture:
        1. Character embedding + learnable positional encoding.
        2. Multi-scale convolution for initial feature extraction.
        3. Stack of bidirectional dilated blocks with increasing dilation.
        4. Position-wise classifier with residual shortcut.

    The design focuses on:
        - Bidirectional context (forward/backward convolutions).
        - Multi-scale features (different convolution kernel sizes).
        - Channel attention (Squeeze-and-Excitation).
        - Stochastic depth for better generalization.
        - Efficient convolutions for fast inference.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_classes: int,
        pad_idx: int,
        dropout: float = 0.2,
        drop_path_rate: float = 0.1,
        num_blocks: int = 6,
        max_pos_len: int = 512,
    ) -> None:
        super().__init__()

        # Character embedding table.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        # Learnable positional embeddings.
        # During training max_pos_len is 512, and it must be consistent
        # with the checkpoint.
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_pos_len, embed_dim) * 0.02
        )

        # Multi-scale convolution to project embeddings to hidden_dim.
        self.input_proj = MultiScaleConvBlock(
            in_channels=embed_dim,
            out_channels=hidden_dim,
            kernel_sizes=(3, 5, 7),
            dropout=dropout,
        )

        # Bidirectional dilated blocks with increasing dilation rate.
        dilations = [1, 2, 4, 8, 16, 32][:num_blocks]
        drop_path_rates = [
            drop_path_rate * i / max(num_blocks - 1, 1) for i in range(num_blocks)
        ]

        self.blocks = nn.ModuleList(
            [
                BidirectionalDilatedBlock(
                    hidden_dim=hidden_dim,
                    kernel_size=3,
                    dilation=d,
                    dropout=dropout,
                    drop_path=dp,
                )
                for d, dp in zip(dilations, drop_path_rates)
            ]
        )

        # Position-wise classifier with a residual shortcut.
        self.pre_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.shortcut = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape [B, L] with character indices.

        Returns:
            logits: Tensor of shape [B, L, num_classes] with class logits
                    for each character position.
        """
        B, L = input_ids.shape

        # Embedding lookup.
        x = self.embedding(input_ids)  # [B, L, E]

        # Positional encoding (interpolate if sequence is longer).
        if L <= self.pos_embedding.size(1):
            pos_emb = self.pos_embedding[:, :L, :]
        else:
            # Interpolate along the sequence length dimension.
            pos_emb = F.interpolate(
                self.pos_embedding.transpose(1, 2),
                size=L,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)  # [1, L, E]

        x = x + pos_emb  # [B, L, E]
        x = x.transpose(1, 2)  # [B, E, L]

        # Multi-scale projection.
        x = self.input_proj(x)  # [B, H, L]

        # Stack of bidirectional dilated blocks.
        for block in self.blocks:
            x = block(x)  # [B, H, L]

        x = x.transpose(1, 2)  # [B, L, H]

        # Residual classifier.
        shortcut = self.shortcut(x)          # [B, L, C]
        x = self.pre_classifier(x)           # [B, L, H]
        logits = self.classifier(x) + shortcut  # [B, L, C]

        return logits


# ============================
# Checkpoint loading (FP32 / Quantized)
# ============================

def load_fp32_model_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, int], Dict[str, Any]]:
    """
    Load a FP32 model checkpoint and reconstruct the CharCNNSpacingModelImproved instance.

    Args:
        checkpoint_path: Path to the FP32 checkpoint (.pt).
        device:         Target device for the model ("cpu" or "cuda").

    Returns:
        model:    Loaded FP32 model on the specified device, in eval mode.
        char2idx: Character-to-index dictionary.
        config:   Configuration dictionary stored in the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    char2idx: Dict[str, int] = checkpoint["vocab"]
    config: Dict[str, Any] = checkpoint["config"]

    model = CharCNNSpacingModelImproved(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=config["num_classes"],
        pad_idx=config["pad_idx"],
        dropout=config["dropout"],
        drop_path_rate=config.get("drop_path_rate", 0.0),
        num_blocks=config.get("num_blocks", 6),
        # Training used max_pos_len=512; config does not store it explicitly.
        max_pos_len=512,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, char2idx, config


def load_quantized_model_from_fp32(
    fp32_checkpoint_path: str,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, int], Dict[str, Any]]:
    """
    Load the FP32 checkpoint on CPU and apply dynamic int8 quantization using torchao.

    NOTE:
        - The model is always quantized and run on CPU, because torchao int8
          quantization is CPU-oriented.
        - If torchao is not available or quantization fails for some reason,
          this function falls back to a plain FP32 model on CPU.

    Args:
        fp32_checkpoint_path: Path to the FP32 checkpoint (.pt).
        device:              Target device for the quantized model.
                             This should be torch.device("cpu").

    Returns:
        model:    Quantized (or FP32 fallback) model on CPU, in eval mode.
        char2idx: Character-to-index dictionary.
        config:   Configuration dictionary from the checkpoint.
    """
    cpu_device = torch.device("cpu")
    checkpoint = torch.load(fp32_checkpoint_path, map_location=cpu_device)

    char2idx: Dict[str, int] = checkpoint["vocab"]
    config: Dict[str, Any] = checkpoint["config"]

    # Build FP32 model on CPU.
    model = CharCNNSpacingModelImproved(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        num_classes=config["num_classes"],
        pad_idx=config["pad_idx"],
        dropout=config["dropout"],
        drop_path_rate=config.get("drop_path_rate", 0.0),
        num_blocks=config.get("num_blocks", 6),
        max_pos_len=512,
    ).to(cpu_device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Try to apply torchao dynamic int8 quantization.
    try:
        from torchao.quantization import (
            quantize_,
            int8_dynamic_activation_int8_weight,
        )

        quantize_(model, int8_dynamic_activation_int8_weight())
        print("[Quant] Applied torchao int8_dynamic_activation_int8_weight quantization.")
    except Exception as e:
        print(
            f"[Quant] Warning: Failed to apply torchao quantization ({e}). "
            f"Using FP32 model on CPU instead."
        )

    # Ensure final device is the requested device (expected to be CPU).
    model.to(device)
    model.eval()

    return model, char2idx, config


# ============================
# Spacing restoration (inference)
# ============================

def restore_spacing(
    text: str,
    model: nn.Module,
    char2idx: Dict[str, int],
    device: torch.device,
    space_threshold: float = 0.5,
) -> str:
    """
    Restore spacing for an input string using the trained model.

    Steps:
        1. Remove existing spaces from the input.
        2. Convert each character to an integer index.
        3. Run the model to get probabilities.
        4. Insert a space if P(space) >= space_threshold.

    Args:
        text:            Input string (possibly with missing or wrong spaces).
        model:           Trained spacing model.
        char2idx:        Character-to-index mapping.
        device:          Device where the model resides.
        space_threshold: Probability threshold for inserting a space.

    Returns:
        A string with predicted spaces.
    """
    model.eval()

    # Remove all spaces to let the model decide spacing.
    no_space = text.replace(" ", "")

    if not no_space:
        return text

    unk_idx = char2idx.get(UNK_TOKEN)
    input_ids: List[int] = [
        char2idx.get(ch, unk_idx) for ch in no_space
    ]

    # Shape: [1, seq_len].
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)                   # [1, L, C]
        probs = torch.softmax(logits, dim=-1)          # [1, L, C]
        space_probs = probs[..., 1].squeeze(0)         # [L]

    output_chars: List[str] = []
    for ch, p_space in zip(no_space, space_probs.tolist()):
        output_chars.append(ch)
        if p_space >= space_threshold:
            output_chars.append(" ")

    return "".join(output_chars).strip()


def run_inference_examples(
    examples: List[str],
    model: nn.Module,
    char2idx: Dict[str, int],
    device: torch.device,
    space_threshold: float = 0.5,
) -> List[Tuple[str, str]]:
    """
    Run spacing restoration on a list of example sentences.

    Args:
        examples:        List of input strings.
        model:           Spacing model.
        char2idx:        Character-to-index mapping.
        device:          Device for inference.
        space_threshold: Threshold for inserting a space.

    Returns:
        List of (input_text, restored_text) pairs.
    """
    results: List[Tuple[str, str]] = []

    for text in examples:
        restored = restore_spacing(
            text=text,
            model=model,
            char2idx=char2idx,
            device=device,
            space_threshold=space_threshold,
        )
        results.append((text, restored))

    return results


# ============================
# Benchmark main
# ============================

def main() -> None:
    """
    Benchmark script:

      - Load FP32 model and measure:
            * load time
            * inference time (for a set of example sentences)
            * unload time

      - Load + quantize model (CPU int8 via torchao) and measure:
            * load + quantization time
            * inference time
            * unload time

      - Print example outputs and a summary of timings.
    """
    # Decide devices.
    fp32_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quant_device = torch.device("cpu")  # torchao int8 is CPU-oriented.

    print(f"[Device] FP32 model device: {fp32_device}")
    print(f"[Device] Quantized model device (forced CPU): {quant_device}")

    # Checkpoint path (must match training script).
    model_dir = "models"
    fp32_checkpoint_path = os.path.join(model_dir, "best_model.pt")

    if not os.path.exists(fp32_checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {fp32_checkpoint_path}. "
            f"Please train the model first (training script)."
        )

    # Example sentences for testing.
    examples: List[str] = [
        "오늘날씨가좋다",
        "한국어띄어쓰기모델테스트중입니다",
        "이문장은공백이없습니다",
        "어젯밤업데이트된신규인공지능기반문서요약시스템이예상하지못한형태의입력데이터를처리하는과정에서문맥을일부잘못해석했다는보고가개발팀내부슬랙채널에공유되었는데이를검증하기위해테스트벤치에서추가재현실험을진행하려고했지만실험환경이완전히동일하게구성되지않아서정확한원인파악에어려움이있다고한다",
    ]

    # ------------------------------------------------------------------
    # 1. FP32 model: load -> inference -> unload
    # ------------------------------------------------------------------
    print("\n[FP32] Loading model...")
    if fp32_device.type == "cuda":
        # Synchronize GPU before measuring time.
        torch.cuda.synchronize()

    fp32_load_start = time.perf_counter()
    model_fp32, char2idx_fp32, config_fp32 = load_fp32_model_checkpoint(
        fp32_checkpoint_path,
        fp32_device,
    )
    if fp32_device.type == "cuda":
        torch.cuda.synchronize()
    fp32_load_end = time.perf_counter()
    fp32_load_time = fp32_load_end - fp32_load_start
    print(f"[FP32] Load time: {fp32_load_time:.6f} seconds")

    # Inference timing for FP32 model.
    print("\n[FP32] Running inference on examples...")
    if fp32_device.type == "cuda":
        torch.cuda.synchronize()
    fp32_infer_start = time.perf_counter()
    fp32_results = run_inference_examples(
        examples=examples,
        model=model_fp32,
        char2idx=char2idx_fp32,
        device=fp32_device,
        space_threshold=0.5,
    )
    if fp32_device.type == "cuda":
        torch.cuda.synchronize()
    fp32_infer_end = time.perf_counter()
    fp32_infer_time = fp32_infer_end - fp32_infer_start
    print(f"[FP32] Inference time (for {len(examples)} sentences): {fp32_infer_time:.6f} seconds")

    # Print FP32 example outputs.
    print("\n[FP32] Example outputs")
    print("-" * 60)
    for inp, out in fp32_results:
        print(f"Input : {inp}")
        print(f"Output: {out}")
        print("-" * 60)

    # Unload FP32 model and measure time.
    print("\n[FP32] Unloading model...")
    if fp32_device.type == "cuda":
        torch.cuda.synchronize()
    fp32_unload_start = time.perf_counter()
    del model_fp32
    # Free GPU memory if needed.
    if fp32_device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    fp32_unload_end = time.perf_counter()
    fp32_unload_time = fp32_unload_end - fp32_unload_start
    print(f"[FP32] Unload time: {fp32_unload_time:.6f} seconds")

    # ------------------------------------------------------------------
    # 2. Quantized model: load+quant -> inference -> unload
    # ------------------------------------------------------------------
    print("\n[Quant] Loading and quantizing model from FP32 checkpoint...")
    quant_load_start = time.perf_counter()
    model_quant, char2idx_quant, config_quant = load_quantized_model_from_fp32(
        fp32_checkpoint_path,
        quant_device,
    )
    quant_load_end = time.perf_counter()
    quant_load_time = quant_load_end - quant_load_start
    print(f"[Quant] Load + quantization time: {quant_load_time:.6f} seconds")

    # Inference timing for quantized model.
    print("\n[Quant] Running inference on examples...")
    quant_infer_start = time.perf_counter()
    quant_results = run_inference_examples(
        examples=examples,
        model=model_quant,
        char2idx=char2idx_quant,
        device=quant_device,
        space_threshold=0.5,
    )
    quant_infer_end = time.perf_counter()
    quant_infer_time = quant_infer_end - quant_infer_start
    print(f"[Quant] Inference time (for {len(examples)} sentences): {quant_infer_time:.6f} seconds")

    # Print quantized example outputs.
    print("\n[Quant] Example outputs")
    print("-" * 60)
    for inp, out in quant_results:
        print(f"Input : {inp}")
        print(f"Output: {out}")
        print("-" * 60)

    # Unload quantized model and measure time.
    print("\n[Quant] Unloading model...")
    quant_unload_start = time.perf_counter()
    del model_quant
    gc.collect()
    quant_unload_end = time.perf_counter()
    quant_unload_time = quant_unload_end - quant_unload_start
    print(f"[Quant] Unload time: {quant_unload_time:.6f} seconds")

    # ------------------------------------------------------------------
    # 3. Summary
    # ------------------------------------------------------------------
    print("\n[Summary] Timing results (seconds)")
    print("-" * 60)
    print(
        f"FP32  - load: {fp32_load_time:.6f}, "
        f"infer: {fp32_infer_time:.6f}, "
        f"unload: {fp32_unload_time:.6f}"
    )
    print(
        f"Quant - load: {quant_load_time:.6f}, "
        f"infer: {quant_infer_time:.6f}, "
        f"unload: {quant_unload_time:.6f}"
    )
    print("-" * 60)


if __name__ == "__main__":
    main()
