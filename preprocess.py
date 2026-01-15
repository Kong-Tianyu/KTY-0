"""
Adamson 数据集预处理脚本（专用版本）
生成 GRAPE Stage B 所需的所有文件：

    ctrl.npy                # 控制组平均表达
    Y.npy                   # 每个条件的平均表达
    cond2gene_idx.npy       # 条件对应被敲除基因 index（支持多基因 KO）
    genes.json              # 基因名列表
    conditions.json         # 条件名称列表

仅适用于 Adamson 数据集。
"""

import scanpy as sc
import numpy as np
import json
import os

# ----------------------------------------------
# 将条件字符串解析为基因名字列表
# ----------------------------------------------
def parse_condition(cond_str):
    """
    Adamson condition 示例：
        "STAT1" → ["STAT1"]
        "STAT1+CEBPB" → ["CEBPB", "STAT1"]
        "ctrl" → []

    返回排序后的基因列表。
    """
    if cond_str.lower() == "ctrl":
        return []
    genes = [g for g in cond_str.split("+") if g.lower() != "ctrl"]
    return sorted(genes)


# ----------------------------------------------
# 预处理主函数
# ----------------------------------------------
def preprocess_adamson(
    adata_path,
    save_dir,
    target_gene_num=5000
):

    print("📥 Loading Adamson AnnData...")
    adata = sc.read_h5ad(adata_path)

    # 检查条件列，一般 Adamson 数据集也使用 obs['condition'] 或 obs['perturbation']
    if "condition" in adata.obs.columns:
        cond_col = "condition"
    elif "perturbation" in adata.obs.columns:
        cond_col = "perturbation"
    else:
        raise KeyError("❌ Adamson 数据集必须包含 obs['condition'] 或 obs['perturbation'] 列")

    # 使用 gene_name (Adamson 原生列)
    if "gene_name" not in adata.var.columns:
        print("⚠ 未找到 gene_name，使用 var_names 替代。")
        adata.var["gene_name"] = adata.var_names

    # ----------------------------------------------
    # Step 1: 筛选高度变异基因（5000 基因）
    # ----------------------------------------------
    print(f"🔍 Selecting top {target_gene_num} Highly Variable Genes (HVG)...")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=target_gene_num,
        subset=True
    )
    print(f"➡ HVG selected. Shape after filtering: {adata.shape}")

    # 基因名列表
    gene_names = adata.var["gene_name"].tolist()
    num_genes = len(gene_names)
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/genes.json", "w") as f:
        json.dump(gene_names, f, indent=2)
    print(f"🧬 Saved gene list: {num_genes} genes")

    # ----------------------------------------------
    # Step 2: 提取条件名称
    # ----------------------------------------------
    conditions = adata.obs[cond_col].astype(str).tolist()
    unique_conditions = sorted(set(conditions))
    with open(f"{save_dir}/conditions.json", "w") as f:
        json.dump(unique_conditions, f, indent=2)
    print(f"📌 Found {len(unique_conditions)} unique perturbation conditions")

    # ----------------------------------------------
    # Step 3: 获取表达矩阵
    # ----------------------------------------------
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()

    # ----------------------------------------------
    # Step 4: 生成 ctrl.npy
    # ----------------------------------------------
    print("🧪 Computing control expression (ctrl.npy)")
    ctrl_mask = np.array([c.lower() == "ctrl" for c in conditions])
    if ctrl_mask.sum() == 0:
        raise ValueError("❌ Adamson 数据中未找到 ctrl 细胞")
    ctrl = X[ctrl_mask].mean(axis=0)
    np.save(f"{save_dir}/ctrl.npy", ctrl)

    # ----------------------------------------------
    # Step 5: 生成 Y.npy 和 cond2gene_idx.npy
    # ----------------------------------------------
    print("📊 Computing perturbation expression profiles...")
    Y_list = []
    cond_gene_idx_list = []

    for cond in unique_conditions:
        if cond.lower() == "ctrl":
            continue

        mask = np.array([c == cond for c in conditions])
        expr = X[mask].mean(axis=0)
        Y_list.append(expr)

        # 解析 KO 基因名
        ko_genes = parse_condition(cond)
        ko_idx = [gene_names.index(g) for g in ko_genes if g in gene_names]
        missing_genes = [g for g in ko_genes if g not in gene_names]
        for g in missing_genes:
            print(f"⚠ Warning: KO gene {g} not in selected gene list")
        cond_gene_idx_list.append(ko_idx)

    Y = np.vstack(Y_list)
    np.save(f"{save_dir}/Y.npy", Y)
    np.save(f"{save_dir}/cond2gene_idx.npy", np.array(cond_gene_idx_list, dtype=object))

    print("✅ All Adamson preprocessing completed!")
    print(f"Saved to: {save_dir}")


# ----------------------------------------------
# 主入口
# ----------------------------------------------
if __name__ == "__main__":
    # ❗ 修改为你的 Adamson 数据路径
    adata_path = "/data1/kty/孔天予/GRAPE-main/data/adamson/perturb_processed.h5ad"

    # ❗ 输出目录
    save_dir = "./result/adamson_5000"

    preprocess_adamson(
        adata_path=adata_path,
        save_dir=save_dir,
        target_gene_num=5000
    )
