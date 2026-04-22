"""阶段 3：深度学习基线 —— LightGCN 和 SASRec（通过 RecBole 框架训练，用自有 evaluator 评估）。"""
import os
import sys
import json
import glob
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("baseline_deep", log_file=str(ROOT / "results" / "baseline_deep.log"))


# ─────────────────────────── 1. 数据格式转换 ────────────────────────────

def convert_to_recbole_format():
    inter_dir = ROOT / "data" / "processed" / "recbole"
    inter_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
    val   = pd.read_csv(ROOT / "data" / "processed" / "val.csv")
    test  = pd.read_csv(ROOT / "data" / "processed" / "test.csv")

    def rename(df):
        return df.rename(columns={
            "user_id":   "user_id:token",
            "movie_id":  "item_id:token",
            "rating":    "rating:float",
            "timestamp": "timestamp:float",
        })

    all_data = pd.concat([train, val, test], ignore_index=True)
    out_path = inter_dir / "ml-1m.inter"
    rename(all_data).to_csv(out_path, sep="\t", index=False)
    logger.info(f"RecBole .inter 文件已保存：{out_path}  ({len(all_data)} 条)")
    return str(inter_dir), train, val, test


# ─────────────────────────── 2. RecBole 训练辅助 ────────────────────────

def build_recbole_config(model_name: str, inter_dir: str, n_epochs: int,
                         extra_params: dict = None) -> dict:
    """生成 RecBole 运行配置字典（不包含 neg_sampling，由各模型自行传入）。"""
    cfg = {
        "data_path":        inter_dir,
        "dataset":          "ml-1m",
        "USER_ID_FIELD":    "user_id",
        "ITEM_ID_FIELD":    "item_id",
        "RATING_FIELD":     "rating",
        "TIME_FIELD":       "timestamp",
        "load_col": {
            "inter": ["user_id", "item_id", "rating", "timestamp"]
        },
        "epochs":           n_epochs,
        "train_batch_size": 4096,
        "eval_batch_size":  4096,
        "learning_rate":    0.001,
        "eval_args": {
            "split":    {"RS": [0.8, 0.1, 0.1]},
            "order":    "RO",
            "group_by": "user",
            "mode":     {"valid": "full", "test": "full"},
        },
        "metrics":      ["Recall", "NDCG"],
        "topk":         [10],
        "valid_metric": "Recall@10",
        "use_gpu":      torch.cuda.is_available(),
        "show_progress": True,
        "checkpoint_dir": str(ROOT / "results" / "models"),
    }
    if extra_params:
        cfg.update(extra_params)
    return cfg


def train_with_recbole(model_name: str, inter_dir: str, n_epochs: int,
                       extra_params: dict = None):
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import init_seed, init_logger, get_model, get_trainer

    cfg_dict = build_recbole_config(model_name, inter_dir, n_epochs, extra_params)
    config = Config(model=model_name, config_dict=cfg_dict)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    dataset = create_dataset(config)
    train_data, valid_data, _ = data_preparation(config, dataset)

    model_cls = get_model(config["model"])(config, train_data.dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model_cls)

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=True
    )
    logger.info(f"[{model_name}] best valid score={best_valid_score:.4f}, result={best_valid_result}")
    return trainer, model_cls, dataset, config


def load_saved_recbole_model(model_name: str, inter_dir: str, extra_params: dict = None):
    """加载已保存的最新 RecBole 模型（跳过训练）。"""
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import init_seed, init_logger, get_model, get_trainer

    # 找到最新保存的模型文件
    pattern = str(ROOT / "results" / "models" / f"{model_name}-*.pth")
    saved_files = sorted(glob.glob(pattern))
    if not saved_files:
        raise FileNotFoundError(f"未找到已保存的 {model_name} 模型文件：{pattern}")
    saved_path = saved_files[-1]
    logger.info(f"[{model_name}] 加载已保存模型：{saved_path}")

    cfg_dict = build_recbole_config(model_name, inter_dir, n_epochs=1, extra_params=extra_params)
    config = Config(model=model_name, config_dict=cfg_dict)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    dataset = create_dataset(config)
    train_data, valid_data, _ = data_preparation(config, dataset)

    model_cls = get_model(config["model"])(config, train_data.dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model_cls)

    # 加载权重（weights_only=False 兼容 PyTorch 2.6+）
    checkpoint = torch.load(saved_path, map_location=config["device"], weights_only=False)
    model_cls.load_state_dict(checkpoint["state_dict"])
    logger.info(f"[{model_name}] 模型权重加载成功")

    return trainer, model_cls, dataset, config


# ─────────────────────────── 3. RecBoleWrapper ──────────────────────────

class RecBoleWrapper:
    """将 RecBole 模型包装成 BaseRecommender 接口（支持 LightGCN 和 SASRec）。"""

    def __init__(self, name: str, recbole_model, dataset, config,
                 train_df: pd.DataFrame):
        self.name = name
        self._model = recbole_model
        self._dataset = dataset
        self._config = config
        self._model.eval()

        uid_field = dataset.uid_field
        iid_field = dataset.iid_field

        # user_id → recbole token
        self._uid2token = {}
        for uid in train_df["user_id"].unique():
            tok = dataset.field2token_id[uid_field].get(str(uid))
            if tok is not None:
                self._uid2token[int(uid)] = tok

        # item_id → recbole token（过滤 [PAD]/[UNK] 等非数字键）
        self._iid2token = {}
        for key, tok in dataset.field2token_id[iid_field].items():
            try:
                self._iid2token[int(key)] = tok
            except (ValueError, TypeError):
                pass
        self._token2iid = {v: k for k, v in self._iid2token.items()}

        # 用户训练集已交互物品
        self._user_train_items = defaultdict(set)
        for _, row in train_df.iterrows():
            self._user_train_items[int(row["user_id"])].add(int(row["movie_id"]))

        # 候选物品 token（排除 padding=0）
        self._all_item_tokens = torch.tensor(
            [t for t in self._token2iid.keys() if t != 0],
            dtype=torch.long
        )
        logger.info(f"[{name}] 候选物品数：{len(self._all_item_tokens)}")

        # 判断是否是序列模型（SASRec 等）
        self._is_sequential = hasattr(recbole_model, "ITEM_SEQ")
        if self._is_sequential:
            # 构建用户历史序列（按时间戳排序）
            try:
                self._max_seq_len = config["MAX_ITEM_LIST_LENGTH"]
            except KeyError:
                self._max_seq_len = 50
            self._user_sequences = {}
            user_sorted = train_df.sort_values("timestamp")
            for uid, group in user_sorted.groupby("user_id"):
                seq = [self._iid2token[iid] for iid in group["movie_id"].tolist()
                       if iid in self._iid2token]
                seq = seq[-self._max_seq_len:]  # 保留最近的
                self._user_sequences[int(uid)] = seq
            logger.info(f"[{name}] 序列模型，已构建 {len(self._user_sequences)} 个用户历史序列")

    def predict(self, user_id: int, top_k: int = 20):
        uid_token = self._uid2token.get(user_id)
        if uid_token is None:
            return []

        device = self._config["device"]

        from recbole.data.interaction import Interaction

        if self._is_sequential:
            # 使用 full_sort_predict 一次性获得所有物品分数
            seq = self._user_sequences.get(user_id, [])
            seq_len = len(seq)
            if seq_len == 0:
                return []
            padded = seq + [0] * (self._max_seq_len - seq_len)
            item_seq = torch.tensor([padded], dtype=torch.long, device=device)
            item_length = torch.tensor([seq_len], dtype=torch.long, device=device)
            inter = Interaction({
                self._model.ITEM_SEQ:     item_seq,
                self._model.ITEM_SEQ_LEN: item_length,
            })
            with torch.no_grad():
                all_scores = self._model.full_sort_predict(inter).squeeze(0).cpu().numpy()
            # all_scores 索引对应 dataset 内的 token id（包含 padding=0）
            # 取出候选物品的分数
            cand_tokens = self._all_item_tokens.numpy()
            scores = all_scores[cand_tokens]
        else:
            # LightGCN 等：逐物品打分
            n_items = len(self._all_item_tokens)
            user_tensor = torch.full((n_items,), uid_token, dtype=torch.long, device=device)
            item_tensor = self._all_item_tokens.to(device)
            inter = Interaction({
                self._dataset.uid_field: user_tensor,
                self._dataset.iid_field: item_tensor,
            })
            with torch.no_grad():
                scores = self._model.predict(inter).cpu().numpy()

        # 排除训练集已交互物品
        train_tokens = {self._iid2token[iid]
                        for iid in self._user_train_items[user_id]
                        if iid in self._iid2token}
        mask = np.array([t not in train_tokens for t in self._all_item_tokens.numpy()])
        scores[~mask] = -np.inf

        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [self._token2iid[self._all_item_tokens[i].item()] for i in top_indices]


# ─────────────────────────── 4. 评估 ────────────────────────────────────

def evaluate_model(wrapped_model, test_df: pd.DataFrame, ks=(10, 20)) -> dict:
    from src.evaluation.metrics import recall_at_k, ndcg_at_k

    user_test = defaultdict(list)
    for _, row in test_df.iterrows():
        user_test[int(row["user_id"])].append(int(row["movie_id"]))

    results = defaultdict(list)
    max_k = max(ks)

    for user_id, ground_truth in tqdm(user_test.items(),
                                      desc=f"Evaluating {wrapped_model.name}"):
        recommended = wrapped_model.predict(user_id, top_k=max_k)
        for k in ks:
            results[f"Recall@{k}"].append(recall_at_k(recommended, ground_truth, k))
            results[f"NDCG@{k}"].append(ndcg_at_k(recommended, ground_truth, k))

    avg = {m: float(np.mean(v)) for m, v in results.items()}
    logger.info(f"[{wrapped_model.name}] {avg}")
    return avg


# ─────────────────────────── 主流程 ──────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("阶段 3：深度学习基线 LightGCN + SASRec")
    logger.info("=" * 60)

    # ── 任务 1：数据转换 ──────────────────────────────────────────────────
    logger.info("\n[任务1] 转换数据格式为 RecBole .inter")
    inter_dir, train_df, val_df, test_df = convert_to_recbole_format()

    results = {}

    # ── 任务 2：LightGCN ─────────────────────────────────────────────────
    logger.info("\n[任务2] LightGCN（加载已保存模型）")
    lightgcn_params = {
        "embedding_size": 64,
        "n_layers":       3,
        "reg_weight":     1e-4,
        "neg_sampling":   {"uniform": 1},   # LightGCN 使用 BPR loss，需要负采样
    }
    try:
        # 优先加载已保存模型（节省训练时间）
        saved = glob.glob(str(ROOT / "results" / "models" / "LightGCN-*.pth"))
        if saved:
            trainer, lgcn_model, lgcn_dataset, lgcn_config = load_saved_recbole_model(
                "LightGCN", inter_dir, extra_params=lightgcn_params
            )
        else:
            trainer, lgcn_model, lgcn_dataset, lgcn_config = train_with_recbole(
                "LightGCN", inter_dir, n_epochs=100, extra_params=lightgcn_params
            )
        lgcn_wrapper = RecBoleWrapper(
            "LightGCN", lgcn_model, lgcn_dataset, lgcn_config, train_df
        )
        results["LightGCN"] = evaluate_model(lgcn_wrapper, test_df)
    except Exception as e:
        logger.error(f"LightGCN 失败：{e}", exc_info=True)
        results["LightGCN"] = {"error": str(e)}

    # ── 任务 3：SASRec ───────────────────────────────────────────────────
    logger.info("\n[任务3] 训练 SASRec（RecBole）")
    # SASRec：序列推荐，CE loss，时间序排列
    sasrec_params = {
        "hidden_size":              64,
        "num_attention_heads":      2,
        "num_layers":               2,
        "MAX_ITEM_LIST_LENGTH":     50,
        "loss_type":                "CE",
        "train_neg_sample_args":    None,   # 覆盖 overall.yaml 默认的 uniform 负采样
        "train_batch_size":         2048,
        "eval_batch_size":          2048,
        "eval_args": {              # 序列推荐必须用 TO（时间序）
            "split":    {"RS": [0.8, 0.1, 0.1]},
            "order":    "TO",
            "group_by": "user",
            "mode":     {"valid": "full", "test": "full"},
        },
    }
    try:
        saved_sas = glob.glob(str(ROOT / "results" / "models" / "SASRec-*.pth"))
        if saved_sas:
            trainer_s, sas_model, sas_dataset, sas_config = load_saved_recbole_model(
                "SASRec", inter_dir, extra_params=sasrec_params
            )
        else:
            trainer_s, sas_model, sas_dataset, sas_config = train_with_recbole(
                "SASRec", inter_dir, n_epochs=100, extra_params=sasrec_params
            )
        sas_wrapper = RecBoleWrapper(
            "SASRec", sas_model, sas_dataset, sas_config, train_df
        )
        results["SASRec"] = evaluate_model(sas_wrapper, test_df)
    except Exception as e:
        logger.error(f"SASRec 失败：{e}", exc_info=True)
        results["SASRec"] = {"error": str(e)}

    # ── 任务 4：保存结果 ─────────────────────────────────────────────────
    logger.info("\n[任务4] 整合并保存结果")
    cf_path = ROOT / "results" / "metrics" / "baseline_cf.json"
    cf_results = json.loads(cf_path.read_text()) if cf_path.exists() else {}

    deep_results = {**cf_results, **results}
    out_path = ROOT / "results" / "metrics" / "baseline_deep.json"
    out_path.write_text(json.dumps(deep_results, indent=2, ensure_ascii=False))
    logger.info(f"结果已保存至 {out_path}")

    logger.info("\n=== 全部基线结果汇总 ===")
    for model_name, metrics in deep_results.items():
        if isinstance(metrics, dict) and "error" not in metrics:
            logger.info(f"{model_name:12s}  "
                        f"R@10={metrics.get('Recall@10', 0):.4f}  "
                        f"N@10={metrics.get('NDCG@10', 0):.4f}  "
                        f"R@20={metrics.get('Recall@20', 0):.4f}  "
                        f"N@20={metrics.get('NDCG@20', 0):.4f}")
        else:
            logger.warning(f"{model_name}: {metrics}")


if __name__ == "__main__":
    main()
