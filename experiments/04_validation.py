"""阶段4：验证与测试脚本

测试 FAISS 检索功能、用户画像生成等
"""
import os
import sys
import json
import time
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent  # pomelo/ 目录
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_movies(filepath):
    """加载电影数据"""
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


def main():
    """主函数"""
    # 设置模型路径
    os.environ['HF_HOME'] = str(PROJECT_ROOT / 'models')
    os.environ['TRANSFORMERS_CACHE'] = str(PROJECT_ROOT / 'models')

    # 路径配置 - 使用ASCII路径保存索引
    VECTORS_PATH = PROJECT_ROOT / 'data' / 'embeddings' / 'movie_vectors.npy'
    INDEX_PATH = Path('D:/bigdata_pomelo/data/embeddings/faiss_index.bin')
    DESCRIPTIONS_PATH = PROJECT_ROOT / 'data' / 'processed' / 'movie_descriptions.json'
    MOVIES_PATH = PROJECT_ROOT / 'data' / 'raw' / 'ml-1m' / 'movies.dat'
    RESULTS_PATH = PROJECT_ROOT / 'results' / 'metrics' / 'retrieval_validation.json'

    # 确保结果目录存在
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=== Stage 4 Validation ===")

    # 1. 加载数据
    logger.info("\n[1] Loading data...")
    with open(DESCRIPTIONS_PATH, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)
    logger.info(f"  Loaded {len(descriptions)} movie descriptions")

    vectors = np.load(VECTORS_PATH)
    logger.info(f"  Loaded vectors: {vectors.shape}")

    # 加载电影信息
    movies_df = load_movies(str(MOVIES_PATH))
    movie_id_to_title = dict(zip(movies_df['movie_id'], movies_df['title']))

    # 2. FAISS 索引统计
    logger.info("\n[2] FAISS Index Statistics...")
    import faiss
    index = faiss.read_index(str(INDEX_PATH))
    logger.info(f"  Vector count: {index.ntotal}")
    logger.info(f"  Vector dimension: {vectors.shape[1]}")

    # 3. 语义检索测试
    logger.info("\n[3] Semantic Search Test...")
    test_queries = [
        "科幻太空探索",
        "浪漫爱情喜剧",
        "恐怖悬疑惊悚"
    ]

    # 加载embedder
    from src.retrieval.embedder import TextEmbedder
    embedder = TextEmbedder(model_name='bge-large', batch_size=32)

    results = {}
    for query_text in test_queries:
        query_vector = embedder.encode_single(query_text)
        start = time.time()
        scores, indices = index.search(query_vector.reshape(1, -1), 10)
        elapsed = (time.time() - start) * 1000

        movie_ids = []
        movie_titles = []
        for idx in indices[0]:
            if idx < len(descriptions):
                mid = list(descriptions.keys())[idx]
                movie_ids.append(mid)
                movie_titles.append(movie_id_to_title.get(mid, f"Movie {mid}"))

        results[query_text] = {
            'query': query_text,
            'movies': movie_titles[:5],
            'scores': scores[0][:5].tolist(),
            'latency_ms': elapsed
        }

        logger.info(f"\n  Query: '{query_text}'")
        logger.info(f"  Latency: {elapsed:.2f}ms")
        logger.info(f"  Top-5 results:")
        for i, (title, score) in enumerate(zip(movie_titles[:5], scores[0][:5])):
            logger.info(f"    {i+1}. {title} (score: {score:.4f})")

    # 4. 随机用户推荐测试
    logger.info("\n[4] Random User Recommendation Test...")
    train_path = PROJECT_ROOT / 'data' / 'processed' / 'train.csv'
    if train_path.exists():
        train_df = pd.read_csv(train_path)
        user_ids = train_df['user_id'].unique()[:3]

        from src.retrieval.user_profile import UserProfileVectorizer
        vectorizer = UserProfileVectorizer(embedder)

        movie_id_map_path = PROJECT_ROOT / 'data' / 'embeddings' / 'movie_id_map.json'
        with open(movie_id_map_path, 'r') as f:
            movie_id_map = json.load(f)

        movie_id_to_idx = {mid: idx for idx, mid in enumerate(movie_id_map)}
        movie_id_to_vector = {mid: vectors[idx] for mid, idx in movie_id_to_idx.items() if idx < len(vectors)}

        for uid in user_ids:
            user_movies = train_df[train_df['user_id'] == uid]['movie_id'].tolist()[:20]
            user_movie_ids = [str(mid) for mid in user_movies]

            # 用向量平均生成用户画像
            profile_vector = vectorizer.generate_profile_with_averaging(
                user_movie_ids, movie_id_to_vector
            )

            # 检索相似电影
            scores, indices = index.search(profile_vector.reshape(1, -1), 10)

            recommended_titles = []
            for idx in indices[0]:
                if idx < len(movie_id_map):
                    mid = movie_id_map[idx]
                    recommended_titles.append(movie_id_to_title.get(mid, f"Movie {mid}"))

            logger.info(f"\n  User {uid}:")
            logger.info(f"    History: {len(user_movies)} movies")
            logger.info(f"    Top-5 recommendations: {recommended_titles[:5]}")

    # 5. 统计信息汇总
    logger.info("\n[5] Summary Statistics...")

    # 计算描述长度统计
    desc_lengths = [len(d) for d in descriptions.values()]
    logger.info(f"  Descriptions: {len(descriptions)} movies")
    logger.info(f"  Avg description length: {np.mean(desc_lengths):.1f} chars")

    # 验证向量数量
    expected_movies = len(descriptions)
    actual_vectors = vectors.shape[0]
    logger.info(f"  Vectors: {actual_vectors} (expected ~{expected_movies})")

    # FAISS查询延迟
    if results:
        avg_latency = np.mean([r['latency_ms'] for r in results.values()])
        logger.info(f"  Avg FAISS query latency: {avg_latency:.2f}ms")

    # 6. 保存验证结果
    validation_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'statistics': {
            'num_descriptions': len(descriptions),
            'num_vectors': actual_vectors,
            'vector_dimension': vectors.shape[1],
            'avg_description_length': float(np.mean(desc_lengths)),
            'avg_query_latency_ms': float(avg_latency) if results else None
        },
        'semantic_search_results': results,
        'validation_status': 'pass' if avg_latency < 10 else 'warning'
    }

    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, ensure_ascii=False, indent=2)

    logger.info(f"\n=== Validation Complete ===")
    logger.info(f"Results saved to: {RESULTS_PATH}")
    logger.info(f"Status: {validation_results['validation_status'].upper()}")

    if validation_results['validation_status'] == 'pass':
        logger.info("All tests passed! FAISS query latency < 10ms")
    else:
        logger.warning("FAISS query latency exceeds 10ms threshold")

    # 清理GPU缓存
    embedder.clear_cache()


if __name__ == "__main__":
    main()
