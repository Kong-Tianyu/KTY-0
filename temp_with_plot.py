import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rmse_shared import train_and_evaluate


# ======================================================
# Losses (论文一致)
# ======================================================
def load_balance_loss(gate_scores: torch.Tensor) -> torch.Tensor:
    """
    gate_scores: [N_tokens, N_experts]
    """
    importance = gate_scores.sum(dim=0)
    importance = importance / (importance.sum() + 1e-9)
    return (importance * importance).sum() * gate_scores.shape[1]


def entropy_loss(gate_scores: torch.Tensor) -> torch.Tensor:
    """
    Encourages confident routing (论文公式 7)
    """
    return -(gate_scores * torch.log(gate_scores + 1e-9)).sum(dim=1).mean()


# ======================================================
# Router
# ======================================================
class Router(nn.Module):
    def __init__(self, in_channels: int, num_experts: int, temperature: float = 1.5):
        super().__init__()
        self.gate = nn.Linear(in_channels, num_experts)
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x) / self.temperature
        return F.softmax(logits, dim=-1)


# ======================================================
# Expert
# ======================================================
class FeedForwardExpert(nn.Module):
    def __init__(self, channels: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ======================================================
# Dynamic Top-P routing
# ======================================================
def dynamic_top_p_routing(
    gate_scores: torch.Tensor,
    p_threshold: float,
):
    """
    gate_scores: [N_tokens, N_experts]
    return: [N_tokens, N_experts] bool mask
    """
    sorted_score, sorted_idx = torch.sort(
        gate_scores, dim=-1, descending=True
    )
    cumsum = torch.cumsum(sorted_score, dim=-1)

    select_sorted = cumsum <= p_threshold
    select_sorted[:, 0] = True  # 至少选择 1 个 expert

    selected_mask = torch.zeros_like(gate_scores, dtype=torch.bool)
    selected_mask.scatter_(1, sorted_idx, select_sorted)
    return selected_mask


# ======================================================
# MoE FeedForward (论文核心)
# ======================================================
class MoEFeedForward(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_dim: int,
        num_experts: int = 4,
        dropout: float = 0.1,
        p_threshold: float = 0.4,  # 论文超参 p
    ):
        super().__init__()
        self.num_experts = num_experts
        self.p_threshold = p_threshold

        self.router = Router(channels, num_experts)
        self.experts = nn.ModuleList(
            [FeedForwardExpert(channels, hidden_dim, dropout) for _ in range(num_experts)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: [B, S, C]
        B, S, C = x.shape
        flat = x.view(B * S, C)

        gate_scores = self.router(flat)  # [N_tokens, E]

        selected_mask = dynamic_top_p_routing(
            gate_scores, self.p_threshold
        )

        out = torch.zeros_like(flat)

        for e in range(self.num_experts):
            token_idx = torch.nonzero(selected_mask[:, e], as_tuple=False).squeeze(1)
            if token_idx.numel() == 0:
                continue
            expert_out = self.experts[e](flat[token_idx])
            out[token_idx] += gate_scores[token_idx, e].unsqueeze(1) * expert_out

        moe_out = self.dropout(out.view(B, S, C))

        lb_loss = load_balance_loss(gate_scores)
        ent_loss = entropy_loss(gate_scores)

        return moe_out, lb_loss, ent_loss


# ======================================================
# Transformer Encoder Layer
# ======================================================
class MoETransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        num_experts: int = 4,
        p_threshold: float = 0.4,
        layer_scale_init: float = 1e-2,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.gamma1 = nn.Parameter(layer_scale_init * torch.ones(d_model))
        self.gamma2 = nn.Parameter(layer_scale_init * torch.ones(d_model))

        self.moe_ffn = MoEFeedForward(
            channels=d_model,
            hidden_dim=dim_feedforward,
            num_experts=num_experts,
            dropout=dropout,
            p_threshold=p_threshold,
        )

    def forward(self, src: torch.Tensor):
        attn_out, _ = self.self_attn(src, src, src, need_weights=False)
        src = src + self.gamma1.view(1, 1, -1) * self.dropout1(attn_out)
        src = self.norm1(src)

        ffn_out, lb_loss, ent_loss = self.moe_ffn(src)
        src = src + self.gamma2.view(1, 1, -1) * self.dropout2(ffn_out)
        src = self.norm2(src)

        return src, lb_loss, ent_loss


# ======================================================
# MultiModal Predictor
# ======================================================
class MultiModalPredictor(nn.Module):
    def __init__(
        self,
        seq_dim: int,
        desc_dim: int,
        output_dim: int,
        embed_dim: int = 512,
        ff_dim: int = 1024,
        num_layers: int = 4,
        dropout: float = 0.2,
        moe_experts: int = 4,
        p_threshold: float = 0.4,
        use_fusion_token: bool = True,
    ):
        super().__init__()

        self.seq_proj = nn.Sequential(nn.Linear(seq_dim, embed_dim), nn.ReLU())
        self.desc_proj = nn.Sequential(nn.Linear(desc_dim, embed_dim), nn.ReLU())

        self.use_fusion_token = use_fusion_token
        if use_fusion_token:
            self.fusion_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.attn_pool_q = nn.Parameter(torch.randn(embed_dim))

        for h in range(16, 0, -1):
            if embed_dim % h == 0:
                num_heads = h
                break

        self.encoder_layers = nn.ModuleList(
            [
                MoETransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_dim,
                    dropout=dropout,
                    num_experts=moe_experts,
                    p_threshold=p_threshold,
                )
                for _ in range(num_layers)
            ]
        )

        self.encoder_norm = nn.LayerNorm(embed_dim)

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, output_dim),
        )

    def forward(self, seq: torch.Tensor, desc: torch.Tensor):
        seq_emb = self.seq_proj(seq)
        desc_emb = self.desc_proj(desc)

        tokens = [seq_emb.unsqueeze(1), desc_emb.unsqueeze(1)]
        if self.use_fusion_token:
            fusion = self.fusion_token.expand(seq_emb.size(0), -1, -1)
            tokens = [fusion] + tokens

        x = torch.cat(tokens, dim=1)

        total_lb, total_ent = 0.0, 0.0
        for layer in self.encoder_layers:
            x, lb, ent = layer(x)
            total_lb += lb
            total_ent += ent

        x = self.encoder_norm(x)

        attn_scores = torch.einsum("bsd,d->bs", x, self.attn_pool_q) / math.sqrt(x.size(-1))
        attn_weights = torch.softmax(attn_scores, dim=1)
        rep = (attn_weights.unsqueeze(-1) * x).sum(dim=1)

        out = self.decoder(rep)

        return out, total_lb / len(self.encoder_layers), total_ent / len(self.encoder_layers)


# ======================================================
# Builder
# ======================================================
def build_model(seq_dim: int, desc_dim: int, output_dim: int) -> nn.Module:
    return MultiModalPredictor(seq_dim, desc_dim, output_dim)


# ... [前面的代码保持不变] ...

# ======================================================
# Train (修改版，添加RMSE记录和绘图功能)
# ======================================================
if __name__ == "__main__":
    dataset_name = "adamson"
    base_path = f"/data1/kty/孔天予/KTY的毕设/{dataset_name}_5000"
    
    # 导入训练函数
    from rmse_shared import train_and_evaluate
    
    # 修改train_and_evaluate函数以记录RMSE并绘图
    import matplotlib.pyplot as plt
    
    def train_and_evaluate_with_plot(
        model_builder,
        seq_path: str,
        desc_path: str,
        y_path: str,
        batch_size: int = 16,
        epochs: int = 50,
        lr: float = 1e-4,
        aux_loss_weight: float = 0.0,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        import random
        import numpy as np
        import torch
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        
        def seed_everything(seed: int):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        seed_everything(seed)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 准备数据集
        dataset, seq_dim, desc_dim, out_dim = train_and_evaluate.__globals__['prepare_dataset'](
            seq_path, desc_path, y_path
        )
        
        g = torch.Generator()
        g.manual_seed(seed)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)
        
        model = model_builder(seq_dim, desc_dim, out_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # 记录RMSE历史
        train_rmse_list = []
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch in progress:
                seq = batch["seq"].to(device)
                desc = batch["desc"].to(device)
                label = batch["label"].to(device)

                optimizer.zero_grad()
                output = model(seq, desc)
                
                # 处理模型输出（可能包含aux loss）
                if isinstance(output, tuple):
                    pred = output[0]
                    if len(output) > 2:  # 如果有aux loss
                        aux_loss = output[1] + output[2]  # 假设有两个aux loss
                    else:
                        aux_loss = output[1] if len(output) > 1 else None
                else:
                    pred = output
                    aux_loss = None
                
                loss = criterion(pred, label)
                if aux_loss is not None and aux_loss_weight:
                    loss = loss + aux_loss_weight * aux_loss
                
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                progress.set_postfix(loss=f"{loss.item():.6f}")

            train_rmse = math.sqrt(epoch_loss / len(train_loader))
            train_rmse_list.append(train_rmse)
            print(f"[Epoch {epoch + 1}] Train RMSE = {train_rmse:.6f}")
        
        # 绘制RMSE曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), train_rmse_list, 'b-', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('RMSE', fontsize=12)
        plt.title('Training RMSE vs. Epoch', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 智能设置x轴刻度
        if epochs <= 20:
            plt.xticks(range(1, epochs + 1))
        else:
            step = max(1, epochs // 10)
            plt.xticks(range(0, epochs + 1, step))
            
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(f'rmse_vs_epoch_{dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 评估模型
        model.eval()
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                seq = batch["seq"].to(device)
                desc = batch["desc"].to(device)
                label = batch["label"].to(device)
                output = model(seq, desc)
                if isinstance(output, tuple):
                    pred = output[0]
                else:
                    pred = output
                all_preds.append(pred.cpu())
                all_labels.append(label.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        final_rmse = math.sqrt(nn.MSELoss()(all_preds, all_labels).item())
        print(f"\nFinal Test RMSE: {final_rmse:.6f}")
        
        # 打印RMSE历史
        print("\nTraining RMSE History:")
        for epoch, rmse in enumerate(train_rmse_list, 1):
            print(f"Epoch {epoch:3d}: {rmse:.6f}")
        
        return model, final_rmse, all_preds, train_rmse_list
    
    # 使用修改后的训练函数
    model, final_rmse, predictions, rmse_history = train_and_evaluate_with_plot(
        model_builder=build_model,
        seq_path=os.path.join(base_path, "seq_embed.json"),
        desc_path=os.path.join(base_path, "desc_embed.json"),
        y_path=os.path.join(base_path, "Y.npy"),
        batch_size=8,
        epochs=300,
        aux_loss_weight=None,
        seed=42,
    )

  
