"""主动引导式推荐 Demo（Streamlit）

运行：
    streamlit run app/guided_demo.py

输入：
    1) 用户兴趣 —— 可用 MovieLens 用户 ID、手动电影标题、或自由文字描述
    2) 目标主题关键词 —— 想把用户引导向哪个领域（如"教育"、"治愈系"、"非血腥的刺激"）
    3) 融合比例 α、返回数量、避免元素（可选）

输出：
    融合向量检索后的候选 + LLM 挑选的 Top-K 推荐（带推荐理由 + 两个分数）
"""
from __future__ import annotations

import sys
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models.guided_recommender import build_default_recommender, GuidedRecommender


# ── 页面 ─────────────────────────────────────────────────────────────────

st.set_page_config(page_title="主动引导式推荐 Demo", page_icon="🧭", layout="wide")

st.markdown("""
<style>
.stProgress > div > div > div { background-color: #4C72B0; }
.reco-card   { background:#fafbfc; border:1px solid #e3e6eb; border-radius:10px;
                padding:14px 16px; margin-bottom:10px; }
.reco-title  { font-size:16px; font-weight:600; color:#222; }
.reco-genre  { color:#666; font-size:13px; }
.reco-reason { color:#2e7d32; font-size:13px; margin-top:6px; }
.score-pill  { display:inline-block; padding:2px 8px; border-radius:12px;
                font-size:11px; margin-right:6px; color:white; }
.pill-user   { background:#4C72B0; }
.pill-topic  { background:#C44E52; }
.pill-fused  { background:#8172B2; }
</style>
""", unsafe_allow_html=True)


# ── 资源加载（进程级缓存） ────────────────────────────────────────────────

@st.cache_resource(show_spinner="加载 BGE / FAISS / LLM（仅首次）...")
def _get_recommender() -> GuidedRecommender:
    return build_default_recommender(project_root=ROOT, with_llm=True)


@st.cache_data
def _load_user_histories():
    """从 train.csv 聚合每个用户最近的观影列表，用于下拉选择。"""
    train = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
    train = train.sort_values(["user_id", "timestamp"])
    hist = defaultdict(list)
    for uid, mid in zip(train["user_id"], train["movie_id"]):
        hist[int(uid)].append(int(mid))
    return {u: items[-20:] for u, items in hist.items()}


# ── UI ───────────────────────────────────────────────────────────────────

def main():
    st.title("🧭 主动引导式推荐 Demo")
    st.caption("从用户现有兴趣出发，把他温和地带向一个新领域。"
                "例：爱看血腥动作的青少年 → 引导到同样刺激但非血腥的悬疑 / 生存冒险。")

    rec = _get_recommender()
    histories = _load_user_histories()

    # ── 侧边栏：输入 ──────────────────────────────────────────────────
    with st.sidebar:
        st.header("① 用户兴趣")
        source = st.radio(
            "输入方式",
            ["MovieLens 用户", "电影标题列表", "自由文字描述"],
            index=0,
        )

        movie_ids_in: list[str] | None = None
        movie_titles_in: list[str] | None = None
        free_text_in: str | None = None

        if source == "MovieLens 用户":
            uid_options = sorted(histories.keys())
            selected_uid = st.selectbox(
                "用户 ID",
                uid_options,
                format_func=lambda u: f"用户 {u}（最近 {len(histories[u])} 部）",
            )
            movie_ids_in = [str(m) for m in histories[selected_uid]]
            with st.expander("查看该用户最近观影", expanded=False):
                for mid in movie_ids_in:
                    t = rec.id_to_title.get(mid, mid)
                    g = rec.id_to_genres.get(mid, "")
                    st.write(f"- {t}（{g}）")

        elif source == "电影标题列表":
            txt = st.text_area(
                "每行一个电影标题（英文，MovieLens 风格）",
                value="Die Hard\nThe Matrix\nTerminator 2: Judgment Day",
                height=120,
            )
            movie_titles_in = [l.strip() for l in txt.splitlines() if l.strip()]

        else:  # 自由文字描述
            free_text_in = st.text_area(
                "用一两句话描述兴趣",
                value="喜欢节奏很快、有打斗和爆炸场面的动作片，也爱看紧张刺激的怪兽电影。",
                height=100,
            )

        st.divider()
        st.header("② 主动引导的目标主题")
        keyword = st.text_input(
            "主题关键词",
            value="不含血腥的惊险刺激",
            help="例：教育启发、科普、治愈系、家庭温情、非血腥的惊险等",
        )

        avoid_raw = st.text_input(
            "要避开的元素（可选，逗号分隔）",
            value="血腥, 露骨暴力",
            help="让 LLM 在挑选时规避的关键词",
        )
        avoid_terms = [t.strip() for t in avoid_raw.split(",") if t.strip()]

        st.divider()
        st.header("③ 调节")
        alpha = st.slider(
            "融合比例 α（贴近用户兴趣 ← → 贴近目标主题）",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        )
        top_k = st.slider("返回数量 K", 3, 15, 8)
        faiss_k = st.slider("FAISS 候选池", 20, 100, 50)
        use_llm = st.toggle("启用 LLM 重排 + 解释", value=True)

        run = st.button("🔍 生成引导推荐", type="primary", use_container_width=True)

    # ── 主区域：结果 ──────────────────────────────────────────────────
    if not run:
        st.info("在左侧设置好 用户兴趣 + 目标主题 + 融合比例 后，点击「生成引导推荐」。")
        _render_intro()
        return

    with st.spinner("正在检索与重排..."):
        try:
            result = rec.recommend(
                target_keyword=keyword,
                movie_ids=movie_ids_in,
                movie_titles=movie_titles_in,
                free_text=free_text_in,
                alpha=alpha,
                top_k=top_k,
                faiss_top_k=faiss_k,
                avoid_terms=avoid_terms,
                use_llm_rerank=use_llm,
                use_llm_profile=True,
            )
        except Exception as e:
            st.error(f"推荐失败：{e}")
            return

    _render_result(result)


def _render_intro():
    st.markdown("""
#### 这个 demo 解决什么问题？
传统推荐只顺着用户的历史偏好输出，无法**主动把用户引导到新领域**。
本 demo 用语义向量融合：

```
q = α · 用户兴趣向量 + (1 − α) · 目标主题向量
候选 = FAISS.search(q, Top-50)
最终 = LLM 从候选中挑选既承接兴趣又属于目标主题的 K 部
```

- `α = 1.0`：忽略主题，纯按用户口味 —— 退化为普通 RAG
- `α = 0.0`：忽略用户，纯按主题 —— 退化为主题搜索
- `α ≈ 0.5`：**主动引导**，两头兼顾，这就是 demo 的主场景

#### 典型场景
- 引导青少年远离血腥暴力 → 保留"刺激感"换掉"血腥"
- 引导只看爽片的人接触**教育 / 科普类**电影
- 给科幻迷推一些带科幻壳子的**艺术 / 哲学**作品
""")


def _render_result(res):
    col_profile, col_topic = st.columns(2)
    with col_profile:
        st.subheader("用户兴趣画像")
        st.write(res.user_profile)
    with col_topic:
        st.subheader(f"目标主题：{res.topic_keyword}")
        st.write(res.topic_description)
        st.caption(f"α = {res.alpha:.2f}｜避免元素：{', '.join(res.avoid_terms) or '—'}｜"
                    f"LLM 重排：{'开' if res.llm_used else '关'}")

    st.divider()
    st.subheader(f"📌 推荐结果（Top-{len(res.recommendations)}）")

    for i, r in enumerate(res.recommendations, 1):
        with st.container():
            st.markdown(
                f"""<div class="reco-card">
                <div class="reco-title">{i}. {r['title']} （{r['year']}）</div>
                <div class="reco-genre">{r['genres']}</div>
                <div>
                    <span class="score-pill pill-fused">融合 {r['score_fused']:.3f}</span>
                    <span class="score-pill pill-user">对用户 {r['score_user']:.3f}</span>
                    <span class="score-pill pill-topic">对主题 {r['score_topic']:.3f}</span>
                </div>
                {f'<div class="reco-reason">💡 {r.get("reason","")}</div>' if r.get("reason") else ""}
                </div>""",
                unsafe_allow_html=True,
            )

    with st.expander("查看全部 FAISS 候选（未经 LLM 重排）", expanded=False):
        df = pd.DataFrame(res.candidates)
        if not df.empty:
            df = df[["movie_id", "title", "year", "genres",
                     "score_fused", "score_user", "score_topic"]]
            st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
