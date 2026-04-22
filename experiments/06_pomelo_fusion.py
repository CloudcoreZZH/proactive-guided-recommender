"""阶段 6：三通道融合架构 Pomelo 评估脚本。

评估内容：
  1. focused / balanced / discovery 三种模式的完整指标（500用户，seed=42）
  2. SASRec 在同 500 用户上的对照评估
  3. 权重敏感性分析（准确性-多样性权衡曲线）
  4. 通道来源统计
  5. 5个用户案例对比（SASRec vs Pomelo-balanced）
"""
import os
import sys
import json
import glob
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import setup_logger
from src.evaluation.metrics import recall_at_k, ndcg_at_k, diversity, novelty, serendipity
from src.retrieval.faiss_index import FaissIndex

logger = setup_logger("pomelo_fusion", log_file=str(ROOT / "results" / "pomelo_fusion.log"))

FAISS_INDEX_PATH = "D:/bigdata_pomelo/data/embeddings/faiss_index.bin"


# ─────────────────────────── 数据加载 ────────────────────────────────────

def load_data():
    train = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
    test  = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
    movies = pd.read_csv(
        ROOT / "data" / "raw" / "ml-1m" / "movies.dat",
        sep="::", engine="python", header=None,
        names=["movie_id", "title", "genres"], encoding="latin-1"
    )
    return train, test, movies


def build_user_structures(train: pd.DataFrame):
    user_train_items: dict = defaultdict(list)
    user_train_timestamps: dict = defaultdict(list)
    for _, row in train.iterrows():
        uid, mid, ts = int(row["user_id"]), int(row["movie_id"]), int(row["timestamp"])
        user_train_items[uid].append(mid)
        user_train_timestamps[uid].append((ts, mid))
    for uid in user_train_timestamps:
        user_train_timestamps[uid].sort()
    return dict(user_train_items), dict(user_train_timestamps)


def build_movie_popularity(train: pd.DataFrame) -> dict:
    counts = train.groupby("movie_id").size()
    max_count = counts.max()
    return {int(mid): count / max_count for mid, count in counts.items()}


# ─────────────────────────── SASRec 加载 ─────────────────────────────────

def load_sasrec(train: pd.DataFrame):
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import init_seed, init_logger, get_model, get_trainer
    import torch

    inter_dir = str(ROOT / "data" / "processed" / "recbole")
    sasrec_params = {
        "hidden_size":           64,
        "num_attention_heads":   2,
        "num_layers":            2,
        "MAX_ITEM_LIST_LENGTH":  50,
        "loss_type":             "CE",
        "train_neg_sample_args": None,
        "train_batch_size":      2048,
        "eval_batch_size":       2048,
        "eval_args": {
            "split":    {"RS": [0.8, 0.1, 0.1]},
            "order":    "TO",
            "group_by": "user",
            "mode":     {"valid": "full", "test": "full"},
        },
        "data_path":        inter_dir,
        "dataset":          "ml-1m",
        "USER_ID_FIELD":    "user_id",
        "ITEM_ID_FIELD":    "item_id",
        "RATING_FIELD":     "rating",
        "TIME_FIELD":       "timestamp",
        "load_col": {"inter": ["user_id", "item_id", "rating", "timestamp"]},
        "epochs":           1,
        "use_gpu":          torch.cuda.is_available(),
        "show_progress":    False,
        "checkpoint_dir":   str(ROOT / "results" / "models"),
    }

    saved = sorted(glob.glob(str(ROOT / "results" / "models" / "SASRec-*.pth")))
    if not saved:
        raise FileNotFoundError("未找到 SASRec 模型文件")
    saved_path = saved[-1]
    logger.info(f"加载 SASRec 模型：{saved_path}")

    config = Config(model="SASRec", config_dict=sasrec_params)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    dataset = create_dataset(config)
    train_data, valid_data, _ = data_preparation(config, dataset)
    model_cls = get_model(config["model"])(config, train_data.dataset).to(config["device"])
    get_trainer(config["MODEL_TYPE"], config["model"])(config, model_cls)

    checkpoint = torch.load(saved_path, map_location=config["device"], weights_only=False)
    model_cls.load_state_dict(checkpoint["state_dict"])
    logger.info("SASRec 权重加载成功")

    # 使用 03_baseline_deep.py 中的 RecBoleWrapper（文件名以数字开头，用 importlib 加载）
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "baseline_deep", ROOT / "experiments" / "03_baseline_deep.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    wrapper = mod.RecBoleWrapper("SASRec", model_cls, dataset, config, train)
    return wrapper


# ─────────────────────────── 评估函数 ────────────────────────────────────

def evaluate_users(
    model,
    eval_users: list,
    user_test_items: dict,
    user_train_items: dict,
    movie_vectors_arr: np.ndarray,
    item_id_to_idx: dict,
    item_popularity: dict,
    ks=(10, 20),
    with_channels: bool = False,
) -> dict:
    """对指定用户列表评估全部指标，返回均值字典。"""
    max_k = max(ks)
    results = defaultdict(list)
    channel_counts = defaultdict(int)
    channel_hits = defaultdict(int)

    for uid in tqdm(eval_users, desc=f"Evaluating {model.name}"):
        gt = user_test_items.get(uid, [])
        if not gt:
            continue

        if with_channels and hasattr(model, "predict_with_channels"):
            channel_recs = model.predict_with_channels(uid, top_k=max_k)
            recommended = [r["movie_id"] for r in channel_recs]
            for r in channel_recs:
                channel_counts[r["channel"]] += 1
                if r["movie_id"] in set(gt):
                    channel_hits[r["channel"]] += 1
        else:
            recommended = model.predict(uid, top_k=max_k)

        history = user_train_items.get(uid, [])

        for k in ks:
            results[f"Recall@{k}"].append(recall_at_k(recommended, gt, k))
            results[f"NDCG@{k}"].append(ndcg_at_k(recommended, gt, k))

        results["Diversity"].append(
            diversity(recommended[:max_k], movie_vectors_arr, item_id_to_idx)
        )
        results["Novelty"].append(
            novelty(recommended[:max_k], item_popularity)
        )
        results["Serendipity"].append(
            serendipity(recommended[:max_k], gt, history, movie_vectors_arr, item_id_to_idx)
        )

    avg = {m: float(np.mean(v)) for m, v in results.items()}
    if with_channels:
        avg["_channel_counts"] = dict(channel_counts)
        avg["_channel_hits"] = dict(channel_hits)
    return avg


# ─────────────────────────── 主流程 ──────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-run", action="store_true", help="只跑10个用户快速验证")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("阶段 6：Pomelo 三通道融合架构评估")
    logger.info("=" * 60)

    # ── 1. 加载数据 ──────────────────────────────────────────────────────
    logger.info("[1] 加载数据...")
    train, test, movies = load_data()
    user_train_items, user_train_timestamps = build_user_structures(train)
    item_popularity = build_movie_popularity(train)

    user_test_items = defaultdict(list)
    for _, row in test.iterrows():
        user_test_items[int(row["user_id"])].append(int(row["movie_id"]))

    # 电影类型字典
    movie_genres = dict(zip(movies["movie_id"].astype(int), movies["genres"]))

    # ── 2. 加载向量 ──────────────────────────────────────────────────────
    logger.info("[2] 加载电影向量...")
    movie_vectors_arr = np.load(ROOT / "data" / "embeddings" / "movie_vectors.npy").astype(np.float32)
    movie_id_map = json.load(open(ROOT / "data" / "embeddings" / "movie_id_map.json"))
    item_id_to_idx = {int(mid): i for i, mid in enumerate(movie_id_map)}
    logger.info(f"  向量矩阵：{movie_vectors_arr.shape}")

    # ── 3. 加载 FAISS ────────────────────────────────────────────────────
    logger.info("[3] 加载 FAISS 索引...")
    faiss_idx = FaissIndex(dimension=1024)
    faiss_idx.load(FAISS_INDEX_PATH)
    logger.info(f"  FAISS 向量数：{faiss_idx.index.ntotal}")

    # ── 4. 加载 SASRec ───────────────────────────────────────────────────
    logger.info("[4] 加载 SASRec 模型...")
    sasrec_wrapper = load_sasrec(train)

    # ── 5. 构建 Pomelo ───────────────────────────────────────────────────
    logger.info("[5] 构建 Pomelo 推荐器...")
    from src.models.pomelo import PomeloRecommender

    pomelo = PomeloRecommender(
        sasrec_wrapper=sasrec_wrapper,
        faiss_index=faiss_idx,
        movie_vectors_arr=movie_vectors_arr,
        movie_id_map=movie_id_map,
        movie_genres=movie_genres,
        movie_popularity=item_popularity,
        user_train_items=user_train_items,
    )

    # ── 6. 选取评估用户（seed=42，500人，与 RAG 相同）────────────────────
    all_test_users = list(user_test_items.keys())
    rng = np.random.default_rng(42)
    eval_users = rng.choice(all_test_users, size=min(500, len(all_test_users)), replace=False).tolist()
    if args.test_run:
        eval_users = eval_users[:10]
    logger.info(f"[6] 评估用户数：{len(eval_users)}")

    results_all = {}

    # ── 7. SASRec 对照评估（同 500 用户）────────────────────────────────
    logger.info("[7] SASRec 对照评估（500用户）...")
    sasrec_wrapper.name = "SASRec"
    sas_metrics = evaluate_users(
        sasrec_wrapper, eval_users, user_test_items,
        user_train_items, movie_vectors_arr, item_id_to_idx, item_popularity
    )
    results_all["SASRec_500"] = sas_metrics
    logger.info(f"  SASRec: {sas_metrics}")

    # ── 8. Pomelo 三种模式评估 ───────────────────────────────────────────
    for mode in ["focused", "balanced", "discovery"]:
        logger.info(f"[8] Pomelo-{mode} 评估...")
        pomelo.set_mode(mode)
        pomelo.name = f"Pomelo-{mode}"
        with_ch = (mode == "balanced")
        metrics = evaluate_users(
            pomelo, eval_users, user_test_items,
            user_train_items, movie_vectors_arr, item_id_to_idx, item_popularity,
            with_channels=with_ch,
        )
        results_all[f"Pomelo-{mode}"] = metrics
        logger.info(f"  Pomelo-{mode}: { {k:v for k,v in metrics.items() if not k.startswith('_')} }")

    # ── 9. 权重敏感性分析 ────────────────────────────────────────────────
    logger.info("[9] 权重敏感性分析...")
    sensitivity = []
    for exploit_w in np.arange(0.1, 0.85, 0.1):
        explore_w = round(0.9 - exploit_w, 2)
        seren_w = 0.1
        pomelo.set_weights(exploit_w, explore_w, seren_w)
        pomelo.name = f"Pomelo-e{exploit_w:.1f}"
        m = evaluate_users(
            pomelo, eval_users, user_test_items,
            user_train_items, movie_vectors_arr, item_id_to_idx, item_popularity,
            ks=[10],
        )
        entry = {
            "exploit_w": round(float(exploit_w), 2),
            "explore_w": round(float(explore_w), 2),
            "serendipity_w": seren_w,
            "Recall@10": m["Recall@10"],
            "Diversity": m["Diversity"],
        }
        sensitivity.append(entry)
        logger.info(f"  exploit={exploit_w:.1f}: Recall@10={m['Recall@10']:.4f}, Diversity={m['Diversity']:.4f}")

    results_all["sensitivity"] = sensitivity

    # ── 10. 通道来源统计（balanced 模式）────────────────────────────────
    balanced_metrics = results_all.get("Pomelo-balanced", {})
    channel_counts = balanced_metrics.pop("_channel_counts", {})
    channel_hits   = balanced_metrics.pop("_channel_hits", {})
    total_recs = sum(channel_counts.values()) or 1
    total_hits = sum(channel_hits.values()) or 1
    channel_stats = {
        ch: {
            "count": channel_counts.get(ch, 0),
            "ratio": channel_counts.get(ch, 0) / total_recs,
            "hits":  channel_hits.get(ch, 0),
            "hit_rate": channel_hits.get(ch, 0) / max(channel_counts.get(ch, 1), 1),
        }
        for ch in ["exploit", "explore", "serendipity"]
    }
    results_all["channel_stats"] = channel_stats
    logger.info(f"[10] 通道统计：{channel_stats}")

    # ── 11. 保存结果 ─────────────────────────────────────────────────────
    out_path = ROOT / "results" / "metrics" / "pomelo.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_all, f, ensure_ascii=False, indent=2)
    logger.info(f"[11] 结果已保存：{out_path}")

    # ── 12. 案例分析（5个用户）──────────────────────────────────────────
    logger.info("[12] 生成案例分析...")
    pomelo.set_mode("balanced")
    pomelo.name = "Pomelo-balanced"

    # 按训练集交互数选5个不同活跃度的用户
    user_activity = {uid: len(user_train_items.get(uid, [])) for uid in eval_users}
    sorted_users = sorted(user_activity.items(), key=lambda x: x[1])
    n = len(sorted_users)
    case_users = [sorted_users[i][0] for i in [0, n//4, n//2, 3*n//4, n-1]]

    movie_title = dict(zip(movies["movie_id"].astype(int), movies["title"]))
    cases = []
    for uid in case_users:
        history_ids = [mid for _, mid in (user_train_timestamps.get(uid, []))[-10:]]
        history_titles = [movie_title.get(m, str(m)) for m in history_ids]

        sasrec_wrapper.name = "SASRec"
        sas_recs = sasrec_wrapper.predict(uid, top_k=10)
        sas_titles = [movie_title.get(m, str(m)) for m in sas_recs]

        pomelo_recs = pomelo.predict_with_channels(uid, top_k=10)
        pomelo_list = [
            {"movie_id": r["movie_id"], "title": movie_title.get(r["movie_id"], str(r["movie_id"])),
             "channel": r["channel"]}
            for r in pomelo_recs
        ]

        cases.append({
            "user_id": uid,
            "activity": user_activity[uid],
            "history_last10": history_titles,
            "sasrec_top10": sas_titles,
            "pomelo_balanced_top10": pomelo_list,
        })

    cases_path = ROOT / "results" / "cases" / "pomelo_vs_sasrec.json"
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cases_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    logger.info(f"案例已保存：{cases_path}")

    # ── 最终汇总 ─────────────────────────────────────────────────────────
    logger.info("\n=== 最终结果汇总 ===")
    for name, m in results_all.items():
        if name in ("sensitivity", "channel_stats"):
            continue
        r10 = m.get("Recall@10", 0)
        n10 = m.get("NDCG@10", 0)
        div = m.get("Diversity", 0)
        nov = m.get("Novelty", 0)
        ser = m.get("Serendipity", 0)
        logger.info(f"{name:20s}  R@10={r10:.4f}  N@10={n10:.4f}  "
                    f"Div={div:.4f}  Nov={nov:.4f}  Ser={ser:.6f}")


if __name__ == "__main__":
    main()
