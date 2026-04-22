"""RAG 失效原因诊断

目标：在 RAG 评估的同 500 个用户子集上，测量：
1. FAISS Top-K 召回率（K=10,20,50,100）—— 语义检索阶段的上限
2. LLM 重排后 Top-K 召回率（最终 RAG 输出）
3. SASRec / ItemCF / MF-BPR 在同 500 用户上的召回率 —— 协同过滤对照

关键问题：
- FAISS Top-50 里到底包含多少测试物品？（召回天花板）
- LLM 重排是把正确答案排前，还是挤出去？
- 语义信号和协同信号之间的鸿沟有多大？

所有数据已缓存，无需任何 API 调用；只需加载 BGE 编码画像 + FAISS 查询。
"""
import os
import sys
import json
import random
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['HF_HOME'] = str(PROJECT_ROOT / 'models')
os.environ['TRANSFORMERS_CACHE'] = str(PROJECT_ROOT / 'models')

from src.retrieval.embedder import TextEmbedder
from src.retrieval.faiss_index import FaissIndex

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def main():
    # ---- 路径 ----
    TRAIN = PROJECT_ROOT / 'data' / 'processed' / 'train.csv'
    TEST = PROJECT_ROOT / 'data' / 'processed' / 'test.csv'
    VEC = PROJECT_ROOT / 'data' / 'embeddings' / 'movie_vectors.npy'
    IDMAP = PROJECT_ROOT / 'data' / 'embeddings' / 'movie_id_map.json'
    FAISS_BIN = Path('D:/bigdata_pomelo/data/embeddings/faiss_index.bin')
    PROFILES = PROJECT_ROOT / 'data' / 'processed' / 'user_profiles.json'
    RANKINGS = PROJECT_ROOT / 'data' / 'processed' / 'user_rankings_cache.json'
    OUT = PROJECT_ROOT / 'results' / 'metrics' / 'rag_diagnosis.json'

    # ---- 加载 ----
    logger.info('加载数据')
    train_df = pd.read_csv(TRAIN)
    test_df = pd.read_csv(TEST)
    vectors = np.load(VEC)
    with open(IDMAP, 'r') as f:
        movie_id_map = json.load(f)  # idx -> str mid
    with open(PROFILES, 'r', encoding='utf-8') as f:
        profiles = json.load(f)
    with open(RANKINGS, 'r', encoding='utf-8') as f:
        rankings = json.load(f)

    # 测试集 ground truth
    user_test = {int(r['user_id']): int(r['movie_id']) for _, r in test_df.iterrows()}

    # 用户训练历史（用于 exclude）
    user_train = defaultdict(set)
    for _, r in train_df.iterrows():
        user_train[int(r['user_id'])].add(int(r['movie_id']))

    # 用户交互次数（冷启动判断）
    user_hist_count = train_df.groupby('user_id').size().to_dict()

    # ---- 采样 500 用户（与 RAG 脚本同 seed）----
    random.seed(42)
    np.random.seed(42)
    all_test_users = list(user_test.keys())
    eval_users = random.sample(all_test_users, 500)
    logger.info(f'采样 500 用户；有画像缓存 {sum(str(u) in profiles for u in eval_users)} 个')

    # ---- FAISS 索引 ----
    logger.info('加载 FAISS 索引 + 嵌入器')
    faiss_idx = FaissIndex(dimension=vectors.shape[1])
    faiss_idx.load(str(FAISS_BIN))
    embedder = TextEmbedder(model_name='bge-large', batch_size=32)

    # ---- 计算 FAISS Top-K 命中 ----
    logger.info('计算 FAISS Top-K 召回...')
    hits_faiss = {10: 0, 20: 0, 50: 0, 100: 0, 200: 0, 500: 0}
    # 记录测试物品在 FAISS 候选中的排名（排除训练历史后）
    gt_ranks_faiss = []  # -1 表示未命中

    id_to_idx = {mid: i for i, mid in enumerate(movie_id_map)}

    for uid in tqdm(eval_users, desc='FAISS'):
        profile = profiles.get(str(uid))
        if not profile:
            gt_ranks_faiss.append(-1)
            continue

        query_vec = embedder.encode_single(profile, normalize=True)
        scores, indices = faiss_idx.search(query_vec, top_k=500 + len(user_train[uid]) + 10)

        exclude = user_train[uid]
        filtered_ids = []  # 过滤训练历史后的候选 movie_id 列表
        for idx in indices:
            if idx < 0 or idx >= len(movie_id_map):
                continue
            mid = int(movie_id_map[idx])
            if mid not in exclude:
                filtered_ids.append(mid)
            if len(filtered_ids) >= 500:
                break

        gt = user_test[uid]
        rank = -1
        for i, mid in enumerate(filtered_ids):
            if mid == gt:
                rank = i + 1
                break
        gt_ranks_faiss.append(rank)

        for k in hits_faiss:
            if 0 < rank <= k:
                hits_faiss[k] += 1

    n = len(eval_users)
    recall_faiss = {k: v / n for k, v in hits_faiss.items()}
    logger.info('FAISS 召回率:')
    for k, v in recall_faiss.items():
        logger.info(f'  Top-{k}: {v:.4f} ({hits_faiss[k]}/{n})')

    # ---- LLM 重排 Top-K 命中（来自缓存）----
    logger.info('计算 LLM 重排 Top-K 召回...')
    hits_llm = {10: 0, 20: 0}
    for uid in eval_users:
        key = f'{uid}_20'
        entry = rankings.get(key)
        if not entry:
            continue
        ids = entry.get('ids', [])
        gt = user_test[uid]
        for k in hits_llm:
            if gt in ids[:k]:
                hits_llm[k] += 1
    recall_llm = {k: v / n for k, v in hits_llm.items()}
    logger.info('LLM 重排召回率:')
    for k, v in recall_llm.items():
        logger.info(f'  Top-{k}: {v:.4f} ({hits_llm[k]}/{n})')

    # ---- 在 FAISS 命中的用户中，LLM 重排是否保留命中？----
    logger.info('在 FAISS Top-20 命中的用户中，LLM 重排是否保留命中？')
    faiss_top20_hit_users = [u for u, r in zip(eval_users, gt_ranks_faiss) if 0 < r <= 20]
    llm_top10_kept = 0
    llm_top20_kept = 0
    for uid in faiss_top20_hit_users:
        key = f'{uid}_20'
        entry = rankings.get(key)
        if not entry:
            continue
        ids = entry.get('ids', [])
        gt = user_test[uid]
        if gt in ids[:10]:
            llm_top10_kept += 1
        if gt in ids[:20]:
            llm_top20_kept += 1
    kept = {
        'faiss_top20_hit_users': len(faiss_top20_hit_users),
        'llm_kept_in_top10': llm_top10_kept,
        'llm_kept_in_top20': llm_top20_kept,
        'llm_top10_retention': llm_top10_kept / max(len(faiss_top20_hit_users), 1),
        'llm_top20_retention': llm_top20_kept / max(len(faiss_top20_hit_users), 1),
    }
    logger.info(f'  FAISS Top-20 命中用户: {kept["faiss_top20_hit_users"]}')
    logger.info(f'  LLM 重排 Top-10 保留: {llm_top10_kept} ({kept["llm_top10_retention"]:.2%})')
    logger.info(f'  LLM 重排 Top-20 保留: {llm_top20_kept} ({kept["llm_top20_retention"]:.2%})')

    # ---- 测试物品的 FAISS 排名分布（命中用户中）----
    hit_ranks = [r for r in gt_ranks_faiss if r > 0]
    rank_dist = {
        'count_hit': len(hit_ranks),
        'count_miss': sum(1 for r in gt_ranks_faiss if r < 0),
        'median_rank': float(np.median(hit_ranks)) if hit_ranks else -1,
        'mean_rank': float(np.mean(hit_ranks)) if hit_ranks else -1,
        'p90_rank': float(np.percentile(hit_ranks, 90)) if hit_ranks else -1,
    }
    logger.info(f'FAISS 命中排名分布: 命中{rank_dist["count_hit"]}，中位数 {rank_dist["median_rank"]:.1f}')

    # ---- 用户交互数 vs 命中率 ----
    low_activity = [u for u in eval_users if user_hist_count.get(u, 0) <= 30]
    mid_activity = [u for u in eval_users if 30 < user_hist_count.get(u, 0) <= 100]
    high_activity = [u for u in eval_users if user_hist_count.get(u, 0) > 100]

    def seg_recall(seg_users, at_k):
        hits = 0
        for u in seg_users:
            key = f'{u}_20'
            entry = rankings.get(key)
            if not entry:
                continue
            if user_test[u] in entry['ids'][:at_k]:
                hits += 1
        return hits / max(len(seg_users), 1), len(seg_users)

    seg_stats = {
        'low_activity_le_30': {
            'n': len(low_activity),
            'recall@10': seg_recall(low_activity, 10)[0],
            'recall@20': seg_recall(low_activity, 20)[0],
        },
        'mid_activity_31_100': {
            'n': len(mid_activity),
            'recall@10': seg_recall(mid_activity, 10)[0],
            'recall@20': seg_recall(mid_activity, 20)[0],
        },
        'high_activity_gt_100': {
            'n': len(high_activity),
            'recall@10': seg_recall(high_activity, 10)[0],
            'recall@20': seg_recall(high_activity, 20)[0],
        },
    }
    logger.info(f'按活跃度分层: {seg_stats}')

    # ---- 保存 ----
    out = {
        'eval_users': n,
        'faiss_recall': recall_faiss,
        'faiss_hits': hits_faiss,
        'llm_recall': recall_llm,
        'llm_hits': hits_llm,
        'llm_retention_given_faiss_top20_hit': kept,
        'faiss_gt_rank_distribution': rank_dist,
        'activity_segments': seg_stats,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info(f'结果已保存: {OUT}')

    embedder.clear_cache()


if __name__ == '__main__':
    main()
