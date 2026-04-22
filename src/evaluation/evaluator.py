"""统一评估器：批量评估推荐模型。"""
import numpy as np
import pandas as pd
from typing import Dict, List
from collections import defaultdict
from tqdm import tqdm
from src.evaluation.metrics import recall_at_k, ndcg_at_k
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class Evaluator:
    """统一评估器，支持批量评估多个模型。"""

    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame,
                 ks: List[int] = None):
        """
        Args:
            train_data: 训练集 DataFrame (user_id, movie_id, rating, timestamp)
            test_data: 测试集 DataFrame
            ks: 要评估的 K 值列表
        """
        self.ks = ks or [10, 20]

        # 构建用户的训练集交互（用于排除已交互物品）
        self.user_train_items = defaultdict(set)
        for _, row in train_data.iterrows():
            self.user_train_items[int(row['user_id'])].add(int(row['movie_id']))

        # 构建用户的测试集真实物品
        self.user_test_items = defaultdict(list)
        for _, row in test_data.iterrows():
            self.user_test_items[int(row['user_id'])].append(int(row['movie_id']))

        # 所有物品集合
        self.all_items = set()
        for items in self.user_train_items.values():
            self.all_items.update(items)
        for items in self.user_test_items.values():
            self.all_items.update(items)

        self.test_users = list(self.user_test_items.keys())
        logger.info(f"Evaluator initialized: {len(self.test_users)} test users, "
                    f"{len(self.all_items)} items")

    def evaluate(self, model, max_k: int = None) -> Dict[str, float]:
        """评估单个模型。

        Args:
            model: 推荐模型（需实现 predict 方法）
            max_k: 最大 K 值（默认取 self.ks 中最大值）

        Returns:
            指标字典，如 {'Recall@10': 0.08, 'NDCG@10': 0.20, ...}
        """
        if max_k is None:
            max_k = max(self.ks)

        results = defaultdict(list)

        for user_id in tqdm(self.test_users, desc=f"Evaluating {model.name}"):
            ground_truth = self.user_test_items[user_id]
            if not ground_truth:
                continue

            recommended = model.predict(user_id, top_k=max_k)

            for k in self.ks:
                results[f'Recall@{k}'].append(
                    recall_at_k(recommended, ground_truth, k))
                results[f'NDCG@{k}'].append(
                    ndcg_at_k(recommended, ground_truth, k))

        # 计算平均值
        avg_results = {}
        for metric, values in results.items():
            avg_results[metric] = float(np.mean(values))

        logger.info(f"[{model.name}] Results:")
        for metric, value in sorted(avg_results.items()):
            logger.info(f"  {metric}: {value:.4f}")

        return avg_results

    def evaluate_multiple(self, models: list) -> pd.DataFrame:
        """评估多个模型，返回对比表。"""
        all_results = {}
        for model in models:
            all_results[model.name] = self.evaluate(model)

        df = pd.DataFrame(all_results).T
        df.index.name = 'Model'
        return df
