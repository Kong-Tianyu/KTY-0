import json
import math
import random
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_json_embed(path: str) -> Tuple[torch.Tensor, Iterable[str]]:
    with open(path, "r") as f:
        data = json.load(f)
    keys = sorted(data.keys())
    embed = torch.tensor([data[k] for k in keys], dtype=torch.float32)
    if embed.dim() == 3 and embed.shape[1] == 1:
        embed = embed.squeeze(1)
    return embed, keys


def load_numpy(path: str) -> torch.Tensor:
    return torch.tensor(np.load(path), dtype=torch.float32)


class MultiModalDataset(Dataset):
    def __init__(self, seq_embed: torch.Tensor, desc_embed: torch.Tensor, labels: torch.Tensor):
        min_len = min(seq_embed.shape[0], desc_embed.shape[0], labels.shape[0])
        self.seq = seq_embed[:min_len]
        self.desc = desc_embed[:min_len]
        self.label = labels[:min_len]

    def __len__(self) -> int:
        return self.seq.shape[0]

    def __getitem__(self, idx: int):
        return {"seq": self.seq[idx], "desc": self.desc[idx], "label": self.label[idx]}


def torch_standardize(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True) + 1e-6
    return (x - mean) / std


def prepare_dataset(seq_path: str, desc_path: str, y_path: str) -> Tuple[MultiModalDataset, int, int, int]:
    seq_embed, _ = load_json_embed(seq_path)
    desc_embed, _ = load_json_embed(desc_path)
    labels = load_numpy(y_path)

    seq_embed = torch_standardize(seq_embed)
    desc_embed = torch_standardize(desc_embed)
    labels = torch_standardize(labels)

    dataset = MultiModalDataset(seq_embed, desc_embed, labels)
    return dataset, seq_embed.shape[1], desc_embed.shape[1], labels.shape[1]


def _unpack_output(output) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(output, (tuple, list)) and len(output) > 0:
        pred = output[0]
        aux = output[1] if len(output) > 1 else None
    else:
        pred, aux = output, None
    return pred, aux


def train_and_evaluate(
    model_builder: Callable[[int, int, int], nn.Module],
    seq_path: str,
    desc_path: str,
    y_path: str,
    *,
    batch_size: int = 16,
    epochs: int = 50,
    lr: float = 1e-4,
    aux_loss_weight: float = 0.0,
    optimizer_cls: Callable = torch.optim.Adam,
    optimizer_kwargs: Optional[dict] = None,
    device: Optional[str] = None,
    seed: int = 42,
) -> Tuple[nn.Module, float, torch.Tensor]:
    seed_everything(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    optimizer_kwargs = optimizer_kwargs or {}

    dataset, seq_dim, desc_dim, out_dim = prepare_dataset(seq_path, desc_path, y_path)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)

    model = model_builder(seq_dim, desc_dim, out_dim).to(device)
    optimizer = optimizer_cls(model.parameters(), lr=lr, **optimizer_kwargs)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in progress:
            seq = batch["seq"].to(device)
            desc = batch["desc"].to(device)
            label = batch["label"].to(device)

            optimizer.zero_grad()
            pred, aux_loss = _unpack_output(model(seq, desc))
            loss = criterion(pred, label)
            if aux_loss is not None and aux_loss_weight:
                loss = loss + aux_loss_weight * aux_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.6f}")

        train_rmse = math.sqrt(epoch_loss / len(train_loader))
        print(f"[Epoch {epoch + 1}] Train RMSE = {train_rmse:.6f}")

    model.eval()
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            seq = batch["seq"].to(device)
            desc = batch["desc"].to(device)
            label = batch["label"].to(device)
            pred, _ = _unpack_output(model(seq, desc))
            all_preds.append(pred.cpu())
            all_labels.append(label.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    rmse = math.sqrt(nn.MSELoss()(all_preds, all_labels).item())
    print("\nFinal Test RMSE:", rmse)
    return model, rmse, all_preds
