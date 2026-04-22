"""数据预处理模块：过滤、划分、生成交互矩阵。"""
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from src.data.loader import load_ratings, load_movies, load_users
from src.utils.config import get_data_path
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def filter_interactions(ratings: pd.DataFrame,
                        min_user: int = 5,
                        min_item: int = 5) -> pd.DataFrame:
    """过滤评分少于阈值的用户和电影，迭代直至稳定。"""
    prev_len = 0
    while len(ratings) != prev_len:
        prev_len = len(ratings)
        user_counts = ratings['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user].index
        ratings = ratings[ratings['user_id'].isin(valid_users)]

        item_counts = ratings['movie_id'].value_counts()
        valid_items = item_counts[item_counts >= min_item].index
        ratings = ratings[ratings['movie_id'].isin(valid_items)]

    logger.info(f"After filtering: {len(ratings)} interactions, "
                f"{ratings['user_id'].nunique()} users, "
                f"{ratings['movie_id'].nunique()} items")
    return ratings.reset_index(drop=True)


def leave_one_out_split(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按用户时间戳做 Leave-One-Out 划分。

    每个用户的最后一次交互 -> test，倒数第二次 -> val，其余 -> train。
    """
    ratings = ratings.sort_values(['user_id', 'timestamp'])

    test_idx = ratings.groupby('user_id')['timestamp'].idxmax()
    test = ratings.loc[test_idx]
    remaining = ratings.drop(test_idx)

    val_idx = remaining.groupby('user_id')['timestamp'].idxmax()
    val = remaining.loc[val_idx]
    train = remaining.drop(val_idx)

    logger.info(f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    logger.info(f"Ratio - Train: {len(train)/len(ratings):.4f}, "
                f"Val: {len(val)/len(ratings):.4f}, "
                f"Test: {len(test)/len(ratings):.4f}")

    return train, val, test


def build_interaction_matrix(ratings: pd.DataFrame,
                             user_map: dict = None,
                             item_map: dict = None) -> tuple[csr_matrix, dict, dict]:
    """生成 user-item 交互矩阵（稀疏矩阵）。

    Returns:
        matrix: CSR 稀疏矩阵
        user_map: {原始 user_id -> 矩阵行索引}
        item_map: {原始 movie_id -> 矩阵列索引}
    """
    if user_map is None:
        unique_users = sorted(ratings['user_id'].unique())
        user_map = {uid: idx for idx, uid in enumerate(unique_users)}
    if item_map is None:
        unique_items = sorted(ratings['movie_id'].unique())
        item_map = {iid: idx for idx, iid in enumerate(unique_items)}

    rows = ratings['user_id'].map(user_map).values
    cols = ratings['movie_id'].map(item_map).values
    vals = np.ones(len(ratings), dtype=np.float32)

    matrix = csr_matrix(
        (vals, (rows, cols)),
        shape=(len(user_map), len(item_map))
    )
    logger.info(f"Interaction matrix shape: {matrix.shape}, nnz: {matrix.nnz}")
    return matrix, user_map, item_map


def preprocess_and_save(data_dir: str = None, output_dir: str = None):
    """完整预处理流程：加载、过滤、划分、保存。"""
    ratings = load_ratings(data_dir)

    ratings = filter_interactions(ratings, min_user=5, min_item=5)

    train, val, test = leave_one_out_split(ratings)

    if output_dir is None:
        output_dir = get_data_path('processed')
    os.makedirs(output_dir, exist_ok=True)

    train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    logger.info(f"Saved train/val/test to {output_dir}")

    return train, val, test


if __name__ == '__main__':
    preprocess_and_save()
