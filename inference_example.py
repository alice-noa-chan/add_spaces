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