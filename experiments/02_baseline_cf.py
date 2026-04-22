"""阶段 2 评估脚本：ItemCF 和 MF-BPR 基线模型。"""
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.models.itemcf import ItemCF
from src.models.mf import MatrixFactorization
from src.evaluation.evaluator import Evaluator
from src.utils.config import get_data_path, get_results_path
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    # 加载数据
    logger.info("Loading preprocessed data...")
    train = pd.read_csv(get_data_path('processed', 'train.csv'))
    test = pd.read_csv(get_data_path('processed', 'test.csv'))

    logger.info(f"Train: {len(train)}, Test: {len(test)}")

    # 初始化评估器
    evaluator = Evaluator(train, test, ks=[10, 20])

    results = {}

    # ========== ItemCF ==========
    logger.info("=" * 50)
    logger.info("Training ItemCF...")
    itemcf = ItemCF(top_k_similar=50, use_iuf=True)
    itemcf.fit(train)
    results['ItemCF'] = evaluator.evaluate(itemcf)

    # ========== MF-BPR ==========
    logger.info("=" * 50)
    logger.info("Training MF-BPR...")
    mf = MatrixFactorization(
        embedding_dim=64,
        lr=0.005,
        epochs=100,
        batch_size=4096,
        reg_weight=1e-4
    )
    mf.fit(train)
    results['MF-BPR'] = evaluator.evaluate(mf)

    # 保存结果
    output_path = get_results_path('metrics', 'baseline_cf.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")

    # 打印对比表
    logger.info("=" * 50)
    logger.info("Final Results:")
    df = pd.DataFrame(results).T
    logger.info(f"\n{df.to_string()}")


if __name__ == '__main__':
    main()
