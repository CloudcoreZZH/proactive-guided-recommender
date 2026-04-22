"""Pomelo 推荐系统交互式 Demo。

运行命令：
    streamlit run app/streamlit_demo.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── 页面配置 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pomelo 推荐系统 Demo",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 样式 ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.channel-exploit     { background:#4C72B0; color:white; padding:2px 8px;
                        border-radius:4px; font-size:12px; }
.channel-explore     { background:#55A868; color:white; padding:2px 8px;
                        border-radius:4px; font-size:12px; }
.channel-serendipity { background:#C44E52; color:white; padding:2px 8px;
                        border-radius:4px; font-size:12px; }
.hit-badge           { background:#FFD700; color:#333; padding:2px 6px;
                        border-radius:4px; font-size:11px; font-weight:bold; }
.metric-card         { background:#f8f9fa; border-radius:8px; padding:12px;
                        text-align:center; border:1px solid #dee2e6; }
</style>
""", unsafe_allow_html=True)


# ── 数据加载（缓存）────────────────────────────────────────────────────────

@st.cache_data
def load_demo_data():
    path = ROOT / "data" / "processed" / "demo_recommendations.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    # 键转为 int
    return {int(k): v for k, v in raw.items()}


@st.cache_data
def load_movie_vectors():
    arr = np.load(ROOT / "data" / "embeddings" / "movie_vectors.npy").astype(np.float32)
    id_map = json.load(open(ROOT / "data" / "embeddings" / "movie_id_map.json"))
    return arr, {int(mid): i for i, mid in enumerate(id_map)}


# ── 指标计算 ──────────────────────────────────────────────────────────────

def _cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def compute_metrics(recs, gt, history, movie_vectors_arr, item_id_to_idx):
    """计算单用户的 Recall@10, Diversity, Novelty, Serendipity。"""
    rec_ids = [r["movie_id"] for r in recs]
    k = len(rec_ids)
    if k == 0:
        return {}

    # Recall@10
    gt_set = set(gt)
    recall = len(set(rec_ids) & gt_set) / max(len(gt_set), 1)

    # Diversity
    vecs = [movie_vectors_arr[item_id_to_idx[m]] for m in rec_ids if m in item_id_to_idx]
    if len(vecs) >= 2:
        sims = [_cosine_sim(vecs[i], vecs[j])
                for i in range(len(vecs)) for j in range(i+1, len(vecs))]
        diversity = 1 - np.mean(sims)
    else:
        diversity = 0.0

    # Novelty（用 1/log2(1+rank) 近似，rank 用列表位置）
    novelty = float(np.mean([-np.log2((i+1) / (k+1)) for i in range(k)]))

    # Serendipity
    hist_vecs = [movie_vectors_arr[item_id_to_idx[m]] for m in history if m in item_id_to_idx]
    ser_scores = []
    for mid in rec_ids:
        if mid not in item_id_to_idx:
            continue
        vec = movie_vectors_arr[item_id_to_idx[mid]]
        unexpectedness = 1 - max((_cosine_sim(vec, hv) for hv in hist_vecs), default=0.0)
        relevance = 1.0 if mid in gt_set else 0.0
        ser_scores.append(unexpectedness * relevance)
    serendipity = float(np.mean(ser_scores)) if ser_scores else 0.0

    return {
        "Recall@10":   round(recall, 4),
        "Diversity":   round(diversity, 4),
        "Novelty":     round(novelty, 4),
        "Serendipity": round(serendipity, 6),
    }


# ── 通道标签 HTML ─────────────────────────────────────────────────────────

def channel_badge(channel: str) -> str:
    label = {"exploit": "精准", "explore": "多样", "serendipity": "意外"}.get(channel, channel)
    return f'<span class="channel-{channel}">{label}</span>'


def hit_badge() -> str:
    return '<span class="hit-badge">✓ 命中</span>'


# ── 主界面 ────────────────────────────────────────────────────────────────

def main():
    st.title("🍊 Pomelo 个性化推荐系统 Demo")
    st.caption("三通道融合架构：Exploit（精准）× Explore（多样）× Serendipity（意外发现）")

    demo_data = load_demo_data()
    if demo_data is None:
        st.error("未找到预计算推荐数据。请先运行：\n\n"
                 "`python experiments/07_final_evaluation.py`")
        st.stop()

    movie_vectors_arr, item_id_to_idx = load_movie_vectors()

    # ── 侧边栏控制面板 ────────────────────────────────────────────────────
    with st.sidebar:
        st.header("控制面板")

        user_ids = sorted(demo_data.keys())
        selected_uid = st.selectbox(
            "选择用户 ID",
            user_ids,
            format_func=lambda x: f"用户 {x}（{len(demo_data[x]['history_last10'])} 部历史）",
        )

        method = st.radio(
            "推荐方法",
            ["ItemCF", "SASRec", "RAG", "Pomelo"],
            index=3,
        )

        pomelo_mode = "balanced"
        if method == "Pomelo":
            st.markdown("**Pomelo 模式**")
            pomelo_mode = st.select_slider(
                "通道权重模式",
                options=["focused", "balanced", "discovery"],
                value="balanced",
                format_func=lambda x: {
                    "focused":   "专注 (0.9/0.08/0.02)",
                    "balanced":  "平衡 (0.6/0.3/0.1)",
                    "discovery": "探索 (0.3/0.4/0.3)",
                }[x],
            )
            mode_weights = {
                "focused":   (0.9, 0.08, 0.02),
                "balanced":  (0.6, 0.3,  0.1),
                "discovery": (0.3, 0.4,  0.3),
            }[pomelo_mode]
            st.markdown(f"""
| 通道 | 权重 |
|------|------|
| 🎯 Exploit | {mode_weights[0]} |
| 🔍 Explore | {mode_weights[1]} |
| ✨ Serendipity | {mode_weights[2]} |
""")

        st.divider()
        st.markdown("**关于 Pomelo**")
        st.markdown("""
- **Exploit**：SASRec 精准推荐
- **Explore**：MMR 多样性重排
- **Serendipity**：向量中等距离区域
        """)

    # ── 主内容区 ──────────────────────────────────────────────────────────
    user_data = demo_data[selected_uid]

    col_hist, col_recs = st.columns([1, 2])

    # 左列：用户历史
    with col_hist:
        st.subheader(f"用户 {selected_uid} 的观看历史")
        history_items = user_data["history_last10"]
        if history_items:
            for item in history_items:
                st.markdown(f"🎬 **{item['title']}**  \n`{item['genres']}`")
        else:
            st.info("无历史记录")

    # 右列：推荐结果
    with col_recs:
        st.subheader(f"推荐结果 — {method}" +
                     (f" ({pomelo_mode})" if method == "Pomelo" else ""))

        # 取推荐列表
        method_key_map = {
            "ItemCF":  "ItemCF",
            "SASRec":  "SASRec",
            "RAG":     "RAG",
            "Pomelo":  f"Pomelo-{pomelo_mode}",
        }
        recs = user_data.get(method_key_map[method], [])
        gt_set = set(user_data.get("test_items", []))
        history_ids = [item["movie_id"] for item in history_items]

        if not recs:
            st.warning("该用户暂无推荐结果（RAG 可能未缓存此用户）")
        else:
            for i, rec in enumerate(recs, 1):
                mid = rec["movie_id"]
                title = rec.get("title", str(mid))
                genres = rec.get("genres", "")
                channel = rec.get("channel", "")
                is_hit = mid in gt_set

                badges = ""
                if channel:
                    badges += channel_badge(channel) + " "
                if is_hit:
                    badges += hit_badge()

                st.markdown(
                    f"**{i}.** {title}  \n"
                    f"`{genres}`  {badges}",
                    unsafe_allow_html=True,
                )

        # 指标卡片
        st.divider()
        st.markdown("**本次推荐指标**")
        if recs:
            m = compute_metrics(recs, list(gt_set), history_ids,
                                movie_vectors_arr, item_id_to_idx)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Recall@10",   f"{m.get('Recall@10', 0):.3f}")
            c2.metric("Diversity",   f"{m.get('Diversity', 0):.3f}")
            c3.metric("Novelty",     f"{m.get('Novelty', 0):.3f}")
            c4.metric("Serendipity", f"{m.get('Serendipity', 0):.5f}")

    # ── 方法对比（展开区）────────────────────────────────────────────────
    with st.expander("📊 所有方法对比（本用户）"):
        compare_methods = ["ItemCF", "SASRec", "Pomelo-focused", "Pomelo-balanced", "Pomelo-discovery"]
        rows = []
        for mname in compare_methods:
            recs_m = user_data.get(mname, [])
            if not recs_m:
                continue
            m = compute_metrics(recs_m, list(gt_set), history_ids,
                                movie_vectors_arr, item_id_to_idx)
            rows.append({"方法": mname, **m})
        if rows:
            import pandas as pd
            st.dataframe(pd.DataFrame(rows).set_index("方法"), use_container_width=True)

    # ── 全局指标参考 ─────────────────────────────────────────────────────
    with st.expander("📈 全局实验结果（500用户均值）"):
        global_data = {
            "方法":        ["ItemCF", "MF-BPR", "LightGCN", "SASRec", "RAG",
                            "Pomelo-focused", "Pomelo-balanced", "Pomelo-discovery"],
            "Recall@10":  [0.0720, 0.0679, 0.0843, 0.1571, 0.0220,
                           0.1700, 0.1700, 0.1760],
            "NDCG@10":    [0.0368, 0.0332, 0.0421, 0.0780, 0.0151,
                           0.0867, 0.0867, 0.0886],
            "Diversity":  [None, None, None, 0.3439, None,
                           0.3493, 0.3591, 0.3588],
            "Novelty":    [None, None, None, 2.885, None,
                           2.892, 2.692, 2.303],
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(global_data).set_index("方法"), use_container_width=True)
        st.caption("注：Diversity/Novelty 仅在 500 用户子集上计算（SASRec 及 Pomelo 变体）")


if __name__ == "__main__":
    main()
