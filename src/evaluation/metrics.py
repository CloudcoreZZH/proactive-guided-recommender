"""四个核心评估指标：Recall@K, NDCG@K, Diversity, Novelty, Serendipity。"""
import numpy as np
from typing import Dict, List


def recall_at_k(recommended: List[int], ground_truth: List[int], k: int) -> float:
    """Recall@K：推荐列表中命中的比例。

    Args:
        recommended: 推荐物品列表（已按推荐度排序）
        ground_truth: 真实相关物品列表
        k: 截断位置
    """
    if not ground_truth:
        return 0.0
    rec_set = set(recommended[:k])
    gt_set = set(ground_truth)
    return len(rec_set & gt_set) / len(gt_set)


def ndcg_at_k(recommended: List[int], ground_truth: List[int], k: int) -> float:
    """NDCG@K：归一化折损累积增益。

    对于隐式反馈，相关物品的增益为 1，非相关为 0。
    """
    if not ground_truth:
        return 0.0
    gt_set = set(ground_truth)
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in gt_set:
            dcg += 1.0 / np.log2(i + 2)  # i+2 因为位置从 1 开始

    # 理想 DCG：所有相关物品排在最前面
    ideal_hits = min(len(gt_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def diversity(recommended: List[int], item_vectors: np.ndarray,
              item_id_to_idx: Dict[int, int]) -> float:
    """推荐列表内多样性：平均两两余弦距离。

    Diversity = 1 - (1/K(K-1)) * Σ sim(i,j)
    """
    k = len(recommended)
    if k < 2:
        return 0.0

    vecs = []
    for item_id in recommended:
        if item_id in item_id_to_idx:
            vecs.append(item_vectors[item_id_to_idx[item_id]])
    if len(vecs) < 2:
        return 0.0

    vecs = np.array(vecs)
    # 归一化
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    vecs = vecs / norms

    sim_matrix = vecs @ vecs.T
    n = len(vecs)
    total_sim = (sim_matrix.sum() - np.trace(sim_matrix)) / (n * (n - 1))
    return 1.0 - total_sim


def novelty(recommended: List[int], item_popularity: Dict[int, float]) -> float:
    """推荐新颖性：推荐物品的平均对数流行度倒数。

    Novelty = (1/K) * Σ -log2(popularity(i))
    """
    if not recommended:
        return 0.0
    scores = []
    for item_id in recommended:
        pop = item_popularity.get(item_id, 1e-10)
        pop = max(pop, 1e-10)  # 避免 log(0)
        scores.append(-np.log2(pop))
    return np.mean(scores)


def serendipity(recommended: List[int], ground_truth: List[int],
                user_history: List[int], item_vectors: np.ndarray,
                item_id_to_idx: Dict[int, int]) -> float:
    """Serendipity = Unexpectedness × Relevance (Ge et al., 2010)。

    Unexpectedness(i) = 1 - max_sim(i, user_history)
    Relevance(i) = 1 if i in ground_truth else 0
    """
    if not recommended or not user_history:
        return 0.0

    gt_set = set(ground_truth)

    # 获取用户历史物品的向量
    hist_vecs = []
    for item_id in user_history:
        if item_id in item_id_to_idx:
            hist_vecs.append(item_vectors[item_id_to_idx[item_id]])
    if not hist_vecs:
        return 0.0
    hist_vecs = np.array(hist_vecs)
    hist_norms = np.linalg.norm(hist_vecs, axis=1, keepdims=True)
    hist_norms = np.where(hist_norms == 0, 1, hist_norms)
    hist_vecs = hist_vecs / hist_norms

    scores = []
    for item_id in recommended:
        if item_id not in item_id_to_idx:
            continue
        vec = item_vectors[item_id_to_idx[item_id]]
        norm = np.linalg.norm(vec)
        if norm == 0:
            continue
        vec = vec / norm
        sims = hist_vecs @ vec
        unexpectedness = 1.0 - sims.max()
        relevance = 1.0 if item_id in gt_set else 0.0
        scores.append(unexpectedness * relevance)

    return np.mean(scores) if scores else 0.0


def evaluate_user(recommended: List[int], ground_truth: List[int],
                  k: int = 10) -> Dict[str, float]:
    """对单个用户计算 Recall@K 和 NDCG@K。"""
    return {
        f'Recall@{k}': recall_at_k(recommended, ground_truth, k),
        f'NDCG@{k}': ndcg_at_k(recommended, ground_truth, k),
    }
