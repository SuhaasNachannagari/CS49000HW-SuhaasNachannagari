# validate_compare.py
import json
import random
from typing import List, Dict, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from transformers import BertTokenizer

# Your duplicates (must be in same folder)
import hw1_duplicate as hw1
import hw2_duplicate as hw2


# -------------------------
# Utils
# -------------------------
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str) -> List[Dict[str, Union[str, int]]]:
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -------------------------
# HW1 mini-batch trainer (independent of hw1_duplicate's trainer)
# -------------------------
def train_mlp_step_losses(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle_seed: int,
    lr: float,
    epochs: int,
    device: torch.device,
) -> List[float]:
    model.to(device)
    model.train()

    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    ds = TensorDataset(x_tensor, y_tensor)

    g = torch.Generator()
    g.manual_seed(shuffle_seed)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g, drop_last=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    step_losses: List[float] = []

    for _ in range(epochs):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            step_losses.append(loss.item())

    return step_losses


@torch.no_grad()
def eval_mlp_accuracy(model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    model.eval()
    model.to(device)
    x_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)
    logits = model(x_tensor)
    preds = torch.argmax(logits, dim=1)
    return float((preds == y_tensor).float().mean().item())


@torch.no_grad()
def eval_bert_accuracy(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device).view(-1)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(logits, dim=1)

        correct += int((preds == labels).sum().item())
        total += int(labels.numel())
    return correct / max(total, 1)


# -------------------------
# Plotting
# -------------------------
def plot_side_by_side_same_axes(curves: Dict[str, List[float]], out_png: str) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    def moving_average(x: np.ndarray, window: int) -> np.ndarray:
        """Simple centered-ish moving average (via convolution)."""
        if len(x) < window:
            return x
        kernel = np.ones(window, dtype=np.float32) / window
        return np.convolve(x, kernel, mode="valid")

    names = list(curves.keys())
    losses_list = [np.array(curves[n], dtype=np.float32) for n in names]

    # --- smoothing hyperparam (same for all models for fairness) ---
    window = 300  # try 200, 300, 500 depending on how smooth you want

    # Compute smoothed curves for consistent axis limits too
    smoothed_list = [moving_average(v, window) for v in losses_list]

    # Use BOTH raw + smoothed to compute y-limits (so red line never clips)
    all_vals = np.concatenate(losses_list + smoothed_list) if losses_list else np.array([0.0], dtype=np.float32)
    y_min, y_max = float(all_vals.min()), float(all_vals.max())

    max_steps_raw = max((len(v) for v in losses_list), default=1)
    max_steps_smooth = max((len(v) for v in smoothed_list), default=1)
    # smoothed curve starts at step (window-1), so max x could extend a bit
    max_x = max(max_steps_raw, (window - 1) + max_steps_smooth)

    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4), sharex=True, sharey=True)
    if len(names) == 1:
        axes = [axes]

    for ax, name, raw in zip(axes, names, losses_list):
        # Raw curve (your original)
        ax.plot(range(len(raw)), raw)

        # Smoothed red trend line
        smooth = moving_average(raw, window)
        if len(smooth) > 0:
            x_smooth = np.arange(window - 1, window - 1 + len(smooth))
            ax.plot(x_smooth, smooth, color="red", linewidth=2)

        ax.set_title(name)
        ax.set_xlim(0, max_x)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Gradient update step")

    axes[0].set_ylabel("Loss")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.show()
    print(f"Saved plot -> {out_png}")


# -------------------------
# Main
# -------------------------
def main():
    # Fixed constraints (same across all 4 models)
    SHUFFLE_SEED = 12345
    BATCH_SIZE = 16

    # You ARE allowed to vary these per-model
    HP = {
        "mlp_onehot": {"lr": 1e-3, "epochs": 6},
        "mlp_bow": {"lr": 1e-3, "epochs": 6},
        "mlp_word2vec_avg": {"lr": 5e-4, "epochs": 10},
        "bert_cls_linear": {"lr": 2e-5, "epochs": 3},
    }

    set_global_seed(SHUFFLE_SEED)
    device = get_device()
    print(f"Device: {device}")
    print(f"Batch size (fixed): {BATCH_SIZE}")
    print(f"Shuffle seed (fixed): {SHUFFLE_SEED}")

    # Load data
    train_data = read_jsonl("train.jsonl")
    valid_data = read_jsonl("valid.jsonl")

    train_texts = [str(d.get("headline", "") or "") for d in train_data]
    train_y = np.array([int(d["is_sarcastic"]) for d in train_data], dtype=np.int64)

    valid_texts = [str(d.get("headline", "") or "") for d in valid_data]
    valid_y = np.array([int(d["is_sarcastic"]) for d in valid_data], dtype=np.int64)

    # -------------------------
    # HW1 features
    # -------------------------
    print("\n[HW1] Building featurizer + features (this will load Word2Vec .bin)...")
    featurizer = hw1.TextFeaturizer(train_texts, w2v_path="GoogleNews-vectors-negative300.bin")

    X_onehot = np.array([featurizer.to_one_hot(t) for t in train_texts], dtype=np.float32)
    X_bow = np.array([featurizer.to_bow(t) for t in train_texts], dtype=np.float32)
    X_w2v = np.array([featurizer.to_word2vec(t) for t in train_texts], dtype=np.float32)

    Xv_onehot = np.array([featurizer.to_one_hot(t) for t in valid_texts], dtype=np.float32)
    Xv_bow = np.array([featurizer.to_bow(t) for t in valid_texts], dtype=np.float32)
    Xv_w2v = np.array([featurizer.to_word2vec(t) for t in valid_texts], dtype=np.float32)

    # -------------------------
    # Train 3x MLP
    # -------------------------
    curves: Dict[str, List[float]] = {}

    print("\nTraining MLP (one-hot)...")
    mlp_onehot = hw1.SarcasmMLP(input_size=X_onehot.shape[1], hidden_sizes=[64, 32], output_size=2)
    curves["mlp_onehot"] = train_mlp_step_losses(
        mlp_onehot, X_onehot, train_y,
        batch_size=BATCH_SIZE, shuffle_seed=SHUFFLE_SEED,
        lr=HP["mlp_onehot"]["lr"], epochs=HP["mlp_onehot"]["epochs"],
        device=device
    )
    acc_onehot = eval_mlp_accuracy(mlp_onehot, Xv_onehot, valid_y, device)

    print("Training MLP (BoW)...")
    mlp_bow = hw1.SarcasmMLP(input_size=X_bow.shape[1], hidden_sizes=[64, 32], output_size=2)
    curves["mlp_bow"] = train_mlp_step_losses(
        mlp_bow, X_bow, train_y,
        batch_size=BATCH_SIZE, shuffle_seed=SHUFFLE_SEED,
        lr=HP["mlp_bow"]["lr"], epochs=HP["mlp_bow"]["epochs"],
        device=device
    )
    acc_bow = eval_mlp_accuracy(mlp_bow, Xv_bow, valid_y, device)

    print("Training MLP (avg word2vec)...")
    mlp_w2v = hw1.SarcasmMLP(input_size=X_w2v.shape[1], hidden_sizes=[64, 32], output_size=2)
    curves["mlp_word2vec_avg"] = train_mlp_step_losses(
        mlp_w2v, X_w2v, train_y,
        batch_size=BATCH_SIZE, shuffle_seed=SHUFFLE_SEED,
        lr=HP["mlp_word2vec_avg"]["lr"], epochs=HP["mlp_word2vec_avg"]["epochs"],
        device=device
    )
    acc_w2v = eval_mlp_accuracy(mlp_w2v, Xv_w2v, valid_y, device)

    # -------------------------
    # Train BERT (HW2)
    # -------------------------
    print("\n[HW2] Training BERT...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    bert_train_ds = hw2.SarcasmDataset(train_data, tokenizer)
    bert_valid_ds = hw2.SarcasmDataset(valid_data, tokenizer)

    g = torch.Generator()
    g.manual_seed(SHUFFLE_SEED)
    bert_train_dl = DataLoader(bert_train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    bert_valid_dl = DataLoader(bert_valid_ds, batch_size=BATCH_SIZE, shuffle=False)

    bert_model = hw2.SarcasmBERT()
    curves["bert_cls_linear"] = hw2.train_loop_step_losses(
        bert_model, bert_train_dl, device,
        lr=HP["bert_cls_linear"]["lr"], epochs=HP["bert_cls_linear"]["epochs"]
    )
    acc_bert = eval_bert_accuracy(bert_model, bert_valid_dl, device)

    # -------------------------
    # Plot (side-by-side, identical axes)
    # -------------------------
    print("\nPlotting loss curves...")


    print("Saved raw step-loss curves -> loss_step_curves_raw.npz")
    plot_side_by_side_same_axes(curves, out_png="loss_curves.png")

    # -------------------------
    # Print quick results summary (nice for your writeup)
    # -------------------------
    print("\nValidation accuracies:")
    print(f"  mlp_onehot       : {acc_onehot:.4f}")
    print(f"  mlp_bow          : {acc_bow:.4f}")
    print(f"  mlp_word2vec_avg : {acc_w2v:.4f}")
    print(f"  bert_cls_linear  : {acc_bert:.4f}")

    print("\nStep counts (number of gradient updates):")
    for name, losses in curves.items():
        print(f"  {name:16s}: {len(losses)}")

    print("\nDone.")


if __name__ == "__main__":
    main()