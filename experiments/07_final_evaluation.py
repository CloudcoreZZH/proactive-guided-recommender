"""阶段 7：最终评估 —— 汇总所有方法指标，生成对比图表，预计算 Demo 推荐数据。

输出：
  results/metrics/final_comparison.csv
  results/figures/fig1_accuracy.png
  results/figures/fig2_diversity_novelty.png
  results/figures/fig3_radar.png
  results/figures/fig4_tradeoff.png
  results/figures/fig5_channel_contribution.png
  data/processed/demo_recommendations.json
"""
import os
import sys
import json
import glob
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("final_eval", log_file=str(ROOT / "results" / "final_evaluation.log"))

FAISS_INDEX_PATH = "D:/bigdata_pomelo/data/embeddings/faiss_index.bin"

# ── 字体设置（支持中文）────────────────────────────────────────────────────
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150


# ─────────────────────────── 指标加载 ────────────────────────────────────

def load_all_metrics() -> dict:
    """从各阶段 JSON 文件汇总所有方法的指标。"""
    metrics_dir = ROOT / "results" / "metrics"

    cf = json.loads((metrics_dir / "baseline_cf.json").read_text())
    deep = json.loads((metrics_dir / "baseline_deep.json").read_text())
    rag_raw = json.loads((metrics_dir / "rag.json").read_text())
    pomelo_raw = json.loads((metrics_dir / "pomelo.json").read_text())

    rag = {
        "Recall@10": rag_raw["all_users"]["recall@10"],
        "NDCG@10":   rag_raw["all_users"]["ndcg@10"],
        "Recall@20": rag_raw["all_users"]["recall@20"],
        "NDCG@20":   rag_raw["all_users"]["ndcg@20"],
    }

    all_metrics = {
        "ItemCF":           cf["ItemCF"],
        "MF-BPR":           cf["MF-BPR"],
        "LightGCN":         deep["LightGCN"],
        "SASRec":           deep["SASRec"],
        "RAG":              rag,
        "SASRec(500)":      pomelo_raw["SASRec_500"],
        "Pomelo-focused":   pomelo_raw["Pomelo-focused"],
        "Pomelo-balanced":  pomelo_raw["Pomelo-balanced"],
        "Pomelo-discovery": pomelo_raw["Pomelo-discovery"],
    }
    return all_metrics, pomelo_raw


def build_comparison_df(all_metrics: dict) -> pd.DataFrame:
    rows = []
    for name, m in all_metrics.items():
        rows.append({
            "Method":     name,
            "Recall@10":  round(m.get("Recall@10", float("nan")), 4),
            "NDCG@10":    round(m.get("NDCG@10",   float("nan")), 4),
            "Recall@20":  round(m.get("Recall@20", float("nan")), 4),
            "NDCG@20":    round(m.get("NDCG@20",   float("nan")), 4),
            "Diversity":  round(m.get("Diversity",  float("nan")), 4),
            "Novelty":    round(m.get("Novelty",    float("nan")), 4),
            "Serendipity":round(m.get("Serendipity",float("nan")), 6),
        })
    return pd.DataFrame(rows)


# ─────────────────────────── 图表生成 ────────────────────────────────────

COLORS = {
    "ItemCF":           "#4C72B0",
    "MF-BPR":           "#DD8452",
    "LightGCN":         "#55A868",
    "SASRec":           "#C44E52",
    "RAG":              "#8172B2",
    "SASRec(500)":      "#937860",
    "Pomelo-focused":   "#DA8BC3",
    "Pomelo-balanced":  "#8C8C8C",
    "Pomelo-discovery": "#CCB974",
}


def fig1_accuracy(all_metrics: dict, out_dir: Path):
    """图1：Recall@10 和 NDCG@10 柱状图（所有方法）。"""
    methods = list(all_metrics.keys())
    r10 = [all_metrics[m].get("Recall@10", 0) for m in methods]
    n10 = [all_metrics[m].get("NDCG@10",   0) for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width/2, r10, width, label="Recall@10",
                   color=[COLORS.get(m, "#999") for m in methods], alpha=0.85)
    bars2 = ax.bar(x + width/2, n10, width, label="NDCG@10",
                   color=[COLORS.get(m, "#999") for m in methods], alpha=0.55)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("图1：各方法准确性对比（Recall@10 & NDCG@10）")
    ax.legend()
    ax.set_ylim(0, max(r10) * 1.25)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.001, f"{h:.3f}",
                ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    path = out_dir / "fig1_accuracy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"图1 已保存：{path}")


def fig2_diversity_novelty(all_metrics: dict, out_dir: Path):
    """图2：Diversity 和 Novelty 柱状图（有这两项指标的方法）。"""
    methods = [m for m in all_metrics if not np.isnan(all_metrics[m].get("Diversity", float("nan")))]
    div = [all_metrics[m]["Diversity"] for m in methods]
    nov = [all_metrics[m]["Novelty"]   for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width/2, div, width, label="Diversity",
                    color=[COLORS.get(m, "#999") for m in methods], alpha=0.85)
    bars2 = ax2.bar(x + width/2, nov, width, label="Novelty",
                    color=[COLORS.get(m, "#999") for m in methods], alpha=0.45)

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("Diversity", color="#333")
    ax2.set_ylabel("Novelty",   color="#666")
    ax1.set_title("图2：多样性与新颖性对比")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    path = out_dir / "fig2_diversity_novelty.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"图2 已保存：{path}")


def fig3_radar(all_metrics: dict, out_dir: Path):
    """图3：Pomelo 三模式雷达图（Recall@10, NDCG@10, Diversity, Novelty 归一化）。"""
    modes = ["Pomelo-focused", "Pomelo-balanced", "Pomelo-discovery"]
    labels = ["Recall@10", "NDCG@10", "Diversity", "Novelty"]

    # 归一化到 [0,1]（以三模式最大值为基准）
    raw = {m: [all_metrics[m].get(l, 0) for l in labels] for m in modes}
    maxvals = [max(raw[m][i] for m in modes) for i in range(len(labels))]
    norm = {m: [raw[m][i] / maxvals[i] if maxvals[i] > 0 else 0
                for i in range(len(labels))] for m in modes}

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    mode_colors = ["#4C72B0", "#55A868", "#C44E52"]

    for mode, color in zip(modes, mode_colors):
        vals = norm[mode] + norm[mode][:1]
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=mode)
        ax.fill(angles, vals, alpha=0.1, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("图3：Pomelo 三模式雷达图（归一化）", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    path = out_dir / "fig3_radar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"图3 已保存：{path}")


def fig4_tradeoff(pomelo_raw: dict, out_dir: Path):
    """图4：准确性-多样性权衡曲线（来自敏感性分析）。"""
    sens = pomelo_raw.get("sensitivity", [])
    if not sens:
        logger.warning("无敏感性分析数据，跳过图4")
        return

    recall = [s["Recall@10"] for s in sens]
    div    = [s["Diversity"]  for s in sens]
    exploit_w = [s["exploit_w"] for s in sens]

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(div, recall, c=exploit_w, cmap="RdYlGn_r", s=80, zorder=3)
    ax.plot(div, recall, "--", color="#aaa", linewidth=1, zorder=2)

    for i, ew in enumerate(exploit_w):
        ax.annotate(f"e={ew:.1f}", (div[i], recall[i]),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Exploit 权重")
    ax.set_xlabel("Diversity")
    ax.set_ylabel("Recall@10")
    ax.set_title("图4：准确性-多样性权衡曲线（Serendipity=0.1 固定）")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "fig4_tradeoff.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"图4 已保存：{path}")


def fig5_channel_contribution(pomelo_raw: dict, out_dir: Path):
    """图5：Pomelo-balanced 通道贡献分析（推荐占比 vs 命中率）。"""
    ch_stats = pomelo_raw.get("channel_stats", {})
    if not ch_stats:
        logger.warning("无通道统计数据，跳过图5")
        return

    channels = ["exploit", "explore", "serendipity"]
    ch_labels = ["Exploit\n(精准)", "Explore\n(多样)", "Serendipity\n(意外)"]
    ratios    = [ch_stats[c]["ratio"]    for c in channels]
    hit_rates = [ch_stats[c]["hit_rate"] for c in channels]

    x = np.arange(len(channels))
    width = 0.35
    ch_colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width/2, ratios,    width, color=ch_colors, alpha=0.85, label="推荐占比")
    bars2 = ax2.bar(x + width/2, hit_rates, width, color=ch_colors, alpha=0.45, label="命中率")

    ax1.set_xticks(x)
    ax1.set_xticklabels(ch_labels, fontsize=10)
    ax1.set_ylabel("推荐占比")
    ax2.set_ylabel("命中率（Hit Rate）")
    ax1.set_title("图5：Pomelo-balanced 三通道贡献分析")

    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.0%}",
                 ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.0002, f"{h:.2%}",
                 ha="center", va="bottom", fontsize=9)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()
    path = out_dir / "fig5_channel_contribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"图5 已保存：{path}")


# ─────────────────────────── Demo 推荐预计算 ─────────────────────────────

def precompute_demo_recommendations(eval_users: list, n_users: int = 200):
    """预计算 Demo 所需的推荐结果，保存到 data/processed/demo_recommendations.json。

    包含方法：ItemCF, SASRec, Pomelo-focused/balanced/discovery
    RAG 从缓存读取。
    """
    import importlib.util
    import torch
    import pandas as pd

    logger.info("[Demo预计算] 开始...")

    train = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
    test  = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
    movies = pd.read_csv(
        ROOT / "data" / "raw" / "ml-1m" / "movies.dat",
        sep="::", engine="python", header=None,
        names=["movie_id", "title", "genres"], encoding="latin-1"
    )
    movie_title  = dict(zip(movies["movie_id"].astype(int), movies["title"]))
    movie_genres = dict(zip(movies["movie_id"].astype(int), movies["genres"]))

    # 用户训练集结构
    user_train_items: dict = defaultdict(list)
    user_train_ts: dict = defaultdict(list)
    for _, row in train.iterrows():
        uid, mid, ts = int(row["user_id"]), int(row["movie_id"]), int(row["timestamp"])
        user_train_items[uid].append(mid)
        user_train_ts[uid].append((ts, mid))
    for uid in user_train_ts:
        user_train_ts[uid].sort()

    user_test_items: dict = defaultdict(list)
    for _, row in test.iterrows():
        user_test_items[int(row["user_id"])].append(int(row["movie_id"]))

    # 取前 n_users 个用户（已是 seed=42 随机采样的 500 人子集）
    demo_users = eval_users[:n_users]

    # ── ItemCF ──────────────────────────────────────────────────────────
    logger.info("[Demo预计算] 训练 ItemCF...")
    from src.models.itemcf import ItemCF
    itemcf = ItemCF(top_k_similar=50)
    itemcf.fit(train)

    # ── SASRec / Pomelo ──────────────────────────────────────────────────
    logger.info("[Demo预计算] 加载 SASRec...")
    spec = importlib.util.spec_from_file_location(
        "baseline_deep", ROOT / "experiments" / "03_baseline_deep.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    import glob as _glob
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import init_seed, init_logger, get_model, get_trainer

    inter_dir = str(ROOT / "data" / "processed" / "recbole")
    sasrec_params = {
        "hidden_size": 64, "num_attention_heads": 2, "num_layers": 2,
        "MAX_ITEM_LIST_LENGTH": 50, "loss_type": "CE",
        "train_neg_sample_args": None, "train_batch_size": 2048, "eval_batch_size": 2048,
        "eval_args": {"split": {"RS": [0.8, 0.1, 0.1]}, "order": "TO",
                      "group_by": "user", "mode": {"valid": "full", "test": "full"}},
        "data_path": inter_dir, "dataset": "ml-1m",
        "USER_ID_FIELD": "user_id", "ITEM_ID_FIELD": "item_id",
        "RATING_FIELD": "rating", "TIME_FIELD": "timestamp",
        "load_col": {"inter": ["user_id", "item_id", "rating", "timestamp"]},
        "epochs": 1, "use_gpu": torch.cuda.is_available(), "show_progress": False,
        "checkpoint_dir": str(ROOT / "results" / "models"),
    }
    saved = sorted(_glob.glob(str(ROOT / "results" / "models" / "SASRec-*.pth")))
    saved_path = saved[-1]
    config = Config(model="SASRec", config_dict=sasrec_params)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    dataset = create_dataset(config)
    train_data, valid_data, _ = data_preparation(config, dataset)
    model_cls = get_model(config["model"])(config, train_data.dataset).to(config["device"])
    get_trainer(config["MODEL_TYPE"], config["model"])(config, model_cls)
    checkpoint = torch.load(saved_path, map_location=config["device"], weights_only=False)
    model_cls.load_state_dict(checkpoint["state_dict"])
    sasrec_wrapper = mod.RecBoleWrapper("SASRec", model_cls, dataset, config, train)
    logger.info("[Demo预计算] SASRec 加载完成")

    # ── FAISS + 向量 ─────────────────────────────────────────────────────
    movie_vectors_arr = np.load(ROOT / "data" / "embeddings" / "movie_vectors.npy").astype(np.float32)
    movie_id_map = json.load(open(ROOT / "data" / "embeddings" / "movie_id_map.json"))
    from src.retrieval.faiss_index import FaissIndex
    faiss_idx = FaissIndex(dimension=1024)
    faiss_idx.load(FAISS_INDEX_PATH)

    item_popularity = {
        int(mid): cnt / train.groupby("movie_id").size().max()
        for mid, cnt in train.groupby("movie_id").size().items()
    }

    from src.models.pomelo import PomeloRecommender
    pomelo = PomeloRecommender(
        sasrec_wrapper=sasrec_wrapper,
        faiss_index=faiss_idx,
        movie_vectors_arr=movie_vectors_arr,
        movie_id_map=movie_id_map,
        movie_genres=movie_genres,
        movie_popularity=item_popularity,
        user_train_items=dict(user_train_items),
    )

    # ── RAG 缓存 ─────────────────────────────────────────────────────────
    rag_cache_path = ROOT / "data" / "processed" / "user_rankings_cache.json"
    rag_cache = {}
    if rag_cache_path.exists():
        raw_rag = json.loads(rag_cache_path.read_text(encoding="utf-8"))
        for k, v in raw_rag.items():
            rag_cache[int(k)] = v
        logger.info(f"[Demo预计算] RAG 缓存加载：{len(rag_cache)} 用户")

    # ── 逐用户预计算 ─────────────────────────────────────────────────────
    demo_data = {}
    for uid in tqdm(demo_users, desc="预计算推荐"):
        history_sorted = [mid for _, mid in sorted(user_train_ts.get(uid, []))]
        history_last10 = history_sorted[-10:]

        # ItemCF
        itemcf_recs = itemcf.predict(uid, top_k=10)

        # SASRec
        sasrec_recs = sasrec_wrapper.predict(uid, top_k=10)

        # Pomelo 三模式
        pomelo_recs = {}
        for mode in ["focused", "balanced", "discovery"]:
            pomelo.set_mode(mode)
            pomelo_recs[mode] = pomelo.predict_with_channels(uid, top_k=10)

        # RAG（从缓存取，格式为 movie_id 列表）
        rag_recs = []
        if uid in rag_cache:
            cached = rag_cache[uid]
            if isinstance(cached, list):
                rag_recs = [int(x) if isinstance(x, (int, float)) else int(x.get("movie_id", 0))
                            for x in cached[:10]]

        demo_data[uid] = {
            "history_last10": [
                {"movie_id": m, "title": movie_title.get(m, str(m)),
                 "genres": movie_genres.get(m, "")}
                for m in history_last10
            ],
            "test_items": user_test_items.get(uid, []),
            "ItemCF": [{"movie_id": m, "title": movie_title.get(m, str(m)),
                        "genres": movie_genres.get(m, "")} for m in itemcf_recs],
            "SASRec": [{"movie_id": m, "title": movie_title.get(m, str(m)),
                        "genres": movie_genres.get(m, "")} for m in sasrec_recs],
            "Pomelo-focused":   [{"movie_id": r["movie_id"],
                                   "title": movie_title.get(r["movie_id"], str(r["movie_id"])),
                                   "genres": movie_genres.get(r["movie_id"], ""),
                                   "channel": r["channel"]} for r in pomelo_recs["focused"]],
            "Pomelo-balanced":  [{"movie_id": r["movie_id"],
                                   "title": movie_title.get(r["movie_id"], str(r["movie_id"])),
                                   "genres": movie_genres.get(r["movie_id"], ""),
                                   "channel": r["channel"]} for r in pomelo_recs["balanced"]],
            "Pomelo-discovery": [{"movie_id": r["movie_id"],
                                   "title": movie_title.get(r["movie_id"], str(r["movie_id"])),
                                   "genres": movie_genres.get(r["movie_id"], ""),
                                   "channel": r["channel"]} for r in pomelo_recs["discovery"]],
            "RAG": [{"movie_id": m, "title": movie_title.get(m, str(m)),
                     "genres": movie_genres.get(m, "")} for m in rag_recs],
        }

    out_path = ROOT / "data" / "processed" / "demo_recommendations.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(demo_data, f, ensure_ascii=False, indent=2)
    logger.info(f"[Demo预计算] 已保存：{out_path}（{len(demo_data)} 用户）")
    return demo_data


# ─────────────────────────── 主流程 ──────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-precompute", action="store_true",
                        help="跳过 Demo 推荐预计算（仅生成图表）")
    parser.add_argument("--precompute-users", type=int, default=200,
                        help="预计算的用户数（默认200）")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("阶段 7：最终评估与图表生成")
    logger.info("=" * 60)

    # ── 1. 加载指标 ──────────────────────────────────────────────────────
    logger.info("[1] 加载所有方法指标...")
    all_metrics, pomelo_raw = load_all_metrics()

    # ── 2. 对比表 ────────────────────────────────────────────────────────
    logger.info("[2] 生成对比表...")
    df = build_comparison_df(all_metrics)
    csv_path = ROOT / "results" / "metrics" / "final_comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"  对比表已保存：{csv_path}")
    logger.info("\n" + df.to_string(index=False))

    # ── 3. 生成图表 ──────────────────────────────────────────────────────
    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[3] 生成图1：准确性对比...")
    fig1_accuracy(all_metrics, fig_dir)

    logger.info("[4] 生成图2：多样性与新颖性...")
    fig2_diversity_novelty(all_metrics, fig_dir)

    logger.info("[5] 生成图3：Pomelo 雷达图...")
    fig3_radar(all_metrics, fig_dir)

    logger.info("[6] 生成图4：权衡曲线...")
    fig4_tradeoff(pomelo_raw, fig_dir)

    logger.info("[7] 生成图5：通道贡献...")
    fig5_channel_contribution(pomelo_raw, fig_dir)

    # ── 4. Demo 预计算 ───────────────────────────────────────────────────
    if not args.skip_precompute:
        logger.info(f"[8] 预计算 Demo 推荐（{args.precompute_users} 用户）...")
        # 复用 seed=42 的 500 用户子集
        import pandas as pd
        test = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
        all_test_users = test["user_id"].unique().tolist()
        rng = np.random.default_rng(42)
        eval_users = rng.choice(all_test_users, size=min(500, len(all_test_users)),
                                replace=False).tolist()
        precompute_demo_recommendations(eval_users, n_users=args.precompute_users)
    else:
        logger.info("[8] 跳过 Demo 预计算")

    logger.info("\n=== 阶段7 完成 ===")
    logger.info(f"图表目录：{fig_dir}")
    logger.info(f"对比表：{csv_path}")


if __name__ == "__main__":
    main()
