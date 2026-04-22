"""阶段5：RAG 增强推荐评估（多线程版）

使用多个 API Key 并发处理，每个线程独享一个 DeepSeek client。
用法：
  python experiments/05_rag_recommendation.py
  python experiments/05_rag_recommendation.py --test-run   # 先用10个用户跑通
  python experiments/05_rag_recommendation.py --num-workers 8
"""
import os
import sys
import json
import time
import logging
import argparse
import random
import re
import threading
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.retrieval.embedder import TextEmbedder
from src.retrieval.faiss_index import FaissIndex
from src.llm.deepseek_client import DeepSeekClient
from src.llm.prompts import USER_PROFILE_PROMPT, RANKING_PROMPT
from src.evaluation.metrics import recall_at_k, ndcg_at_k

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(thread)d] %(message)s'
)
logger = logging.getLogger(__name__)


# ── 全局共享资源（只读或加锁） ──────────────────────────────────
_embedder: TextEmbedder = None
_embedder_lock = threading.Lock()
_faiss_idx: FaissIndex = None
_movie_id_map: list = None
_movie_descriptions: dict = None
_movie_id_to_title: dict = None
_movie_id_to_genres: dict = None

# ── 缓存（写时加锁） ────────────────────────────────────────────
_profile_cache: dict = {}
_profile_lock = threading.Lock()
_ranking_cache: dict = {}
_ranking_lock = threading.Lock()

PROFILE_CACHE_PATH = PROJECT_ROOT / 'data' / 'processed' / 'user_profiles.json'
RANKING_CACHE_PATH = PROJECT_ROOT / 'data' / 'processed' / 'user_rankings_cache.json'

# ── 统计计数器（加锁） ──────────────────────────────────────────
_stats_lock = threading.Lock()
_api_calls = 0


def _save_cache(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_movies(filepath: str) -> pd.DataFrame:
    movies = []
    with open(filepath, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) >= 3:
                movie_id = parts[0]
                title_year = parts[1]
                genres = parts[2]
                if '(' in title_year and title_year.endswith(')'):
                    title = title_year.rsplit('(', 1)[0].strip()
                    year = title_year.rsplit('(', 1)[1].replace(')', '').strip()
                else:
                    title = title_year
                    year = "Unknown"
                movies.append({
                    'movie_id': movie_id,
                    'title': title,
                    'genres': genres.replace('|', ', '),
                    'year': year
                })
    return pd.DataFrame(movies)


def load_api_keys() -> list:
    key_file = Path(__file__).parent.parent.parent / "deepseek apikey.txt"
    keys = []
    if key_file.exists():
        with open(key_file, 'r', encoding='utf-8') as f:
            for line in f:
                key = line.strip()
                if key and not key.startswith('#'):
                    keys.append(key)
    # deduplicate preserving order
    seen = set()
    unique_keys = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique_keys.append(k)
    return unique_keys


def generate_user_profile(user_id: int, client: DeepSeekClient,
                          user_recent_movie_ids: list) -> str:
    """生成并缓存用户画像（线程安全）"""
    global _api_calls
    cache_key = str(user_id)

    with _profile_lock:
        if cache_key in _profile_cache:
            return _profile_cache[cache_key]

    if not user_recent_movie_ids:
        return "用户暂无观影记录"

    movie_lines = []
    for mid in user_recent_movie_ids:
        title = _movie_id_to_title.get(mid, f"电影{mid}")
        genres = _movie_id_to_genres.get(mid, "")
        movie_lines.append(f"- {title}（{genres}）")

    prompt = USER_PROFILE_PROMPT.format(movie_list="\n".join(movie_lines))
    profile = client.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    ).strip()

    with _stats_lock:
        _api_calls += 1

    with _profile_lock:
        _profile_cache[cache_key] = profile
        if len(_profile_cache) % 10 == 0:
            _save_cache(_profile_cache, PROFILE_CACHE_PATH)

    return profile


def faiss_retrieve(profile_text: str, exclude_ids: set, top_k: int = 50) -> list:
    """FAISS 语义检索候选电影（线程安全：embedder 加锁）"""
    with _embedder_lock:
        query_vec = _embedder.encode_single(profile_text, normalize=True)

    scores, indices = _faiss_idx.search(query_vec, top_k=top_k + len(exclude_ids) + 10)

    candidates = []
    for idx in indices:
        if idx < 0 or idx >= len(_movie_id_map):
            continue
        mid_str = _movie_id_map[idx]
        if int(mid_str) not in exclude_ids:
            candidates.append(mid_str)
        if len(candidates) >= top_k:
            break
    return candidates


def parse_ranking_response(response: str, candidates: list) -> tuple:
    """解析 LLM 排序结果，返回 (id_list, reason_list)"""
    ranked_ids = []
    reasons = []

    # 优先匹配 [电影ID] 格式
    pattern = r'\[(\d+)\][^\n—]*(?:——\s*(.+?))?(?=\n|$)'
    for mid_str, reason in re.findall(pattern, response):
        try:
            mid = int(mid_str)
            if mid not in ranked_ids:
                ranked_ids.append(mid)
                reasons.append(reason.strip() if reason else "")
        except ValueError:
            continue

    if ranked_ids:
        return ranked_ids, reasons

    # fallback：按候选标题匹配
    candidate_titles = {_movie_id_to_title.get(mid, ""): mid for mid in candidates}
    for line in response.split('\n'):
        for title, mid in candidate_titles.items():
            if title and title in line:
                mid_int = int(mid)
                if mid_int not in ranked_ids:
                    reason = line.split('——', 1)[1].strip() if '——' in line else ""
                    ranked_ids.append(mid_int)
                    reasons.append(reason)
                break

    return ranked_ids, reasons


def llm_rerank(user_id: int, user_profile: str, candidates: list,
               top_k: int, client: DeepSeekClient) -> tuple:
    """LLM 重排候选，返回 (id_list, reason_list)，带缓存"""
    global _api_calls
    cache_key = f"{user_id}_{top_k}"

    with _ranking_lock:
        if cache_key in _ranking_cache:
            cached = _ranking_cache[cache_key]
            return cached.get('ids', []), cached.get('reasons', [])

    candidate_lines = []
    for i, mid in enumerate(candidates[:20], 1):
        title = _movie_id_to_title.get(mid, f"电影{mid}")
        desc = _movie_descriptions.get(mid, "")[:100]
        candidate_lines.append(f"{i}. [{mid}] {title} —— {desc}")

    prompt = RANKING_PROMPT.format(
        user_profile=user_profile,
        candidate_movies_with_descriptions="\n".join(candidate_lines),
        top_k=top_k,
    )

    try:
        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
        )
        with _stats_lock:
            _api_calls += 1
        ranked_ids, reasons = parse_ranking_response(response, candidates)
    except Exception as e:
        logger.warning(f"LLM rerank failed user {user_id}: {e}, fallback to FAISS order")
        ranked_ids = [int(mid) for mid in candidates[:top_k]]
        reasons = [""] * len(ranked_ids)

    # pad with FAISS order if needed
    if len(ranked_ids) < top_k:
        existing = set(ranked_ids)
        for mid in candidates:
            if int(mid) not in existing:
                ranked_ids.append(int(mid))
                reasons.append("")
            if len(ranked_ids) >= top_k:
                break

    ranked_ids = ranked_ids[:top_k]
    reasons = (reasons + [""] * top_k)[:top_k]

    with _ranking_lock:
        _ranking_cache[cache_key] = {'ids': ranked_ids, 'reasons': reasons}
        if len(_ranking_cache) % 10 == 0:
            _save_cache(_ranking_cache, RANKING_CACHE_PATH)

    return ranked_ids, reasons


def process_user(user_id: int, client: DeepSeekClient,
                 user_train_items: dict, user_train_timestamps: dict,
                 top_k: int = 20) -> tuple:
    """处理单个用户，返回 (user_id, recommended_ids, latency_seconds)"""
    t0 = time.time()
    try:
        exclude_ids = set(user_train_items.get(user_id, []))

        # 最近20部电影（按时间排序）
        sorted_items = user_train_timestamps.get(user_id, [])
        recent_movie_ids = [str(mid) for _, mid in sorted_items[-20:]]

        # 生成画像
        profile = generate_user_profile(user_id, client, recent_movie_ids)

        # FAISS 检索
        candidates = faiss_retrieve(profile, exclude_ids, top_k=50)
        if not candidates:
            return user_id, [], time.time() - t0

        # LLM 重排
        ranked_ids, _ = llm_rerank(user_id, profile, candidates, top_k, client)
        return user_id, ranked_ids, time.time() - t0

    except Exception as e:
        logger.error(f"Error processing user {user_id}: {e}")
        return user_id, [], time.time() - t0


def main():
    global _embedder, _faiss_idx, _movie_id_map, _movie_descriptions
    global _movie_id_to_title, _movie_id_to_genres
    global _profile_cache, _ranking_cache

    parser = argparse.ArgumentParser()
    parser.add_argument('--test-run', action='store_true', help='Only 10 users')
    parser.add_argument('--num-workers', type=int, default=None, help='Override worker count')
    args = parser.parse_args()

    os.environ['HF_HOME'] = str(PROJECT_ROOT / 'models')
    os.environ['TRANSFORMERS_CACHE'] = str(PROJECT_ROOT / 'models')

    TRAIN_PATH = PROJECT_ROOT / 'data' / 'processed' / 'train.csv'
    TEST_PATH = PROJECT_ROOT / 'data' / 'processed' / 'test.csv'
    MOVIES_PATH = PROJECT_ROOT / 'data' / 'raw' / 'ml-1m' / 'movies.dat'
    DESCRIPTIONS_PATH = PROJECT_ROOT / 'data' / 'processed' / 'movie_descriptions.json'
    VECTORS_PATH = PROJECT_ROOT / 'data' / 'embeddings' / 'movie_vectors.npy'
    ID_MAP_PATH = PROJECT_ROOT / 'data' / 'embeddings' / 'movie_id_map.json'
    FAISS_INDEX_PATH = Path('D:/bigdata_pomelo/data/embeddings/faiss_index.bin')
    RESULTS_PATH = PROJECT_ROOT / 'results' / 'metrics' / 'rag.json'
    CASES_PATH = PROJECT_ROOT / 'results' / 'cases' / 'rag_examples.json'
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CASES_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=== Stage 5: RAG Recommendation (Multi-threaded) ===")

    # 1. 加载数据
    logger.info("[1] Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    movies_df = load_movies(str(MOVIES_PATH))

    with open(DESCRIPTIONS_PATH, 'r', encoding='utf-8') as f:
        _movie_descriptions = json.load(f)
    vectors = np.load(VECTORS_PATH)
    with open(ID_MAP_PATH, 'r') as f:
        _movie_id_map = json.load(f)

    _movie_id_to_title = dict(zip(movies_df['movie_id'], movies_df['title']))
    _movie_id_to_genres = dict(zip(movies_df['movie_id'], movies_df['genres']))

    # 构建用户历史索引
    user_train_items = defaultdict(list)
    user_train_timestamps = defaultdict(list)
    for _, row in train_df.iterrows():
        uid, mid, ts = int(row['user_id']), int(row['movie_id']), int(row['timestamp'])
        user_train_items[uid].append(mid)
        user_train_timestamps[uid].append((ts, mid))
    for uid in user_train_timestamps:
        user_train_timestamps[uid].sort()

    user_test_items = {}
    for _, row in test_df.iterrows():
        user_test_items[int(row['user_id'])] = [int(row['movie_id'])]

    # 2. 初始化共享组件
    logger.info("[2] Initializing shared components...")
    _embedder = TextEmbedder(model_name='bge-large', batch_size=32)
    logger.info("  Embedder loaded")

    _faiss_idx = FaissIndex(dimension=vectors.shape[1])
    _faiss_idx.load(str(FAISS_INDEX_PATH))
    logger.info(f"  FAISS index loaded: {_faiss_idx.ntotal} vectors")

    # 3. 加载 API keys & 初始化 clients
    keys = load_api_keys()
    if not keys:
        logger.error("No API keys found!")
        return
    num_workers = args.num_workers or len(keys)
    num_workers = min(num_workers, len(keys))
    logger.info(f"  API keys: {len(keys)}, workers: {num_workers}")

    clients = [DeepSeekClient(api_key=keys[i % len(keys)]) for i in range(num_workers)]

    # 4. 加载缓存
    _profile_cache = {}
    _ranking_cache = {}
    if PROFILE_CACHE_PATH.exists():
        with open(PROFILE_CACHE_PATH, 'r', encoding='utf-8') as f:
            _profile_cache = json.load(f)
        logger.info(f"  Profile cache: {len(_profile_cache)} entries")
    if RANKING_CACHE_PATH.exists():
        with open(RANKING_CACHE_PATH, 'r', encoding='utf-8') as f:
            _ranking_cache = json.load(f)
        logger.info(f"  Ranking cache: {len(_ranking_cache)} entries")

    # 5. 采样用户
    all_test_users = list(user_test_items.keys())
    random.seed(42)
    np.random.seed(42)
    n_eval = 10 if args.test_run else 500
    eval_users = random.sample(all_test_users, min(n_eval, len(all_test_users)))
    user_train_count = train_df.groupby('user_id').size().to_dict()
    cold_users = [u for u in eval_users if user_train_count.get(u, 0) <= 10]
    logger.info(f"[3] Eval users: {len(eval_users)}, cold-start: {len(cold_users)}")

    # 6. 多线程处理
    logger.info(f"[4] Processing with {num_workers} threads...")
    results_map = {}  # user_id -> recommended_ids
    latencies = []
    total_start = time.time()

    # 将用户轮转分配给 worker（确保每个 worker 用对应的 key）
    def make_worker(worker_idx: int):
        client = clients[worker_idx]
        def fn(uid):
            return process_user(uid, client, user_train_items,
                                user_train_timestamps, top_k=20)
        return fn

    with tqdm(total=len(eval_users), desc="Evaluating") as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_user = {}
            for i, uid in enumerate(eval_users):
                worker_idx = i % num_workers
                future = executor.submit(
                    process_user, uid, clients[worker_idx],
                    user_train_items, user_train_timestamps, 20
                )
                future_to_user[future] = uid

            for future in as_completed(future_to_user):
                uid, recs, latency = future.result()
                results_map[uid] = recs
                latencies.append(latency)
                pbar.update(1)
                if len(results_map) % 50 == 0:
                    elapsed = time.time() - total_start
                    remaining = (elapsed / len(results_map)) * (len(eval_users) - len(results_map))
                    logger.info(f"  Progress: {len(results_map)}/{len(eval_users)}, "
                                f"elapsed={elapsed:.0f}s, ETA={remaining:.0f}s")

    total_time = time.time() - total_start

    # 最终保存缓存
    _save_cache(_profile_cache, PROFILE_CACHE_PATH)
    _save_cache(_ranking_cache, RANKING_CACHE_PATH)

    # 7. 计算指标
    logger.info("[5] Computing metrics...")
    def compute_metrics(user_list, label=""):
        recall10_list, ndcg10_list = [], []
        recall20_list, ndcg20_list = [], []
        for uid in user_list:
            gt = user_test_items.get(uid, [])
            recs = results_map.get(uid, [])
            if not gt or not recs:
                continue
            recall10_list.append(recall_at_k(recs, gt, 10))
            ndcg10_list.append(ndcg_at_k(recs, gt, 10))
            recall20_list.append(recall_at_k(recs, gt, 20))
            ndcg20_list.append(ndcg_at_k(recs, gt, 20))
        metrics = {
            'recall@10': float(np.mean(recall10_list)) if recall10_list else 0.0,
            'ndcg@10':   float(np.mean(ndcg10_list))   if ndcg10_list else 0.0,
            'recall@20': float(np.mean(recall20_list)) if recall20_list else 0.0,
            'ndcg@20':   float(np.mean(ndcg20_list))   if ndcg20_list else 0.0,
            'evaluated_users': len(recall10_list),
        }
        logger.info(f"  [{label}] Recall@10={metrics['recall@10']:.4f}, "
                    f"NDCG@10={metrics['ndcg@10']:.4f}, "
                    f"Recall@20={metrics['recall@20']:.4f}")
        return metrics

    all_metrics = compute_metrics(eval_users, "All")
    cold_metrics = compute_metrics(cold_users, "Cold-start") if cold_users else {}

    avg_api_per_user = _api_calls / max(len(eval_users), 1)
    avg_latency = float(np.mean(latencies)) if latencies else 0.0

    logger.info(f"\n[6] Cost Statistics:")
    logger.info(f"  Total API calls: {_api_calls}")
    logger.info(f"  Avg calls/user: {avg_api_per_user:.2f}")
    logger.info(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info(f"  Avg latency/user: {avg_latency:.2f}s")
    logger.info(f"  Estimated calls for all 6040 users: {avg_api_per_user * 6040:.0f}")

    # 8. 保存结果
    final_results = {
        'evaluated_users': len(eval_users),
        'cold_start_users': len(cold_users),
        'test_run': args.test_run,
        'all_users': all_metrics,
        'cold_start': cold_metrics,
        'avg_api_calls_per_user': round(avg_api_per_user, 2),
        'avg_latency_per_user_seconds': round(avg_latency, 3),
        'total_api_calls': _api_calls,
        'total_time_seconds': round(total_time, 1),
        'num_workers': num_workers,
        'note': 'RAG evaluated on 500 sampled users; FAISS Top-50 candidate pool; '
                'scores for non-candidate movies set to 0 (unfavorable to RAG)'
    }
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    logger.info(f"\nResults saved to: {RESULTS_PATH}")

    # 9. 保存案例
    logger.info("[7] Saving example cases...")
    case_users = random.sample(eval_users, min(10, len(eval_users)))
    cases = {}
    for uid in case_users:
        recs = results_map.get(uid, [])
        profile = _profile_cache.get(str(uid), "")
        ranking_key = f"{uid}_20"
        reasons = _ranking_cache.get(ranking_key, {}).get('reasons', [""] * len(recs))
        history_ids = [str(m) for _, m in user_train_timestamps.get(uid, [])[-10:]]
        cases[str(uid)] = {
            'user_profile': profile,
            'history_titles': [_movie_id_to_title.get(mid, mid) for mid in history_ids],
            'recommendations': [
                {'movie_id': mid, 'title': _movie_id_to_title.get(str(mid), str(mid)),
                 'reason': (reasons[i] if i < len(reasons) else "")}
                for i, mid in enumerate(recs[:10])
            ]
        }
    with open(CASES_PATH, 'w', encoding='utf-8') as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    logger.info(f"Cases saved to: {CASES_PATH}")

    # 10. 最终汇总
    logger.info("\n=== Final Summary ===")
    logger.info(f"RAG (all {len(eval_users)} users):")
    logger.info(f"  Recall@10: {all_metrics['recall@10']:.4f}")
    logger.info(f"  NDCG@10:   {all_metrics['ndcg@10']:.4f}")
    logger.info(f"  Recall@20: {all_metrics['recall@20']:.4f}")
    logger.info(f"  NDCG@20:   {all_metrics['ndcg@20']:.4f}")
    if cold_metrics:
        logger.info(f"RAG (cold-start {len(cold_users)} users):")
        logger.info(f"  Recall@10: {cold_metrics['recall@10']:.4f}")
        logger.info(f"  NDCG@10:   {cold_metrics['ndcg@10']:.4f}")

    _embedder.clear_cache()


if __name__ == "__main__":
    main()
