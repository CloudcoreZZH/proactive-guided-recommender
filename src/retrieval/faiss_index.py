"""FAISS 索引管理

使用 FAISS 构建和查询向量索引
"""
import os
import logging
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import faiss

logger = logging.getLogger(__name__)


class FaissIndex:
    """FAISS 索引封装"""

    def __init__(self, dimension: int):
        """
        初始化 FAISS 索引

        Args:
            dimension: 向量维度
        """
        self.dimension = dimension
        self.index: Optional[faiss.Index] = None
        self._built = False

    def build(self, vectors: np.ndarray) -> None:
        """
        构建索引

        Args:
            vectors: 归一化后的向量，形状 (n, dimension)
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}")

        # 使用内积索引（因为向量已归一化，等价于余弦相似度）
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)

        self._built = True
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        搜索最相似的向量

        Args:
            query_vector: 查询向量（单个向量，形状 (dimension,) 或 (1, dimension)）
            top_k: 返回的 Top-K 数量

        Returns:
            (scores, indices) 元组
            - scores: 相似度分数
            - indices: 对应的向量索引
        """
        if not self._built:
            raise RuntimeError("Index not built yet. Call build() first.")

        # 确保维度正确
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        elif query_vector.ndim > 2:
            raise ValueError(f"Invalid query vector shape: {query_vector.shape}")

        # 搜索
        scores, indices = self.index.search(query_vector, top_k)

        return scores[0], indices[0]

    def search_batch(
        self,
        query_vectors: np.ndarray,
        top_k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量搜索

        Args:
            query_vectors: 查询向量，形状 (n, dimension)
            top_k: 返回的 Top-K 数量

        Returns:
            (scores, indices) 元组
        """
        if not self._built:
            raise RuntimeError("Index not built yet. Call build() first.")

        scores, indices = self.index.search(query_vectors, top_k)

        return scores, indices

    def save(self, path: str) -> None:
        """
        保存索引到文件

        Args:
            path: 保存路径
        """
        if not self._built:
            raise RuntimeError("Index not built yet. Cannot save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path))
        logger.info(f"Saved FAISS index to {path}")

    def load(self, path: str) -> None:
        """
        从文件加载索引

        Args:
            path: 索引文件路径
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        self.index = faiss.read_index(str(path))
        self._built = True
        logger.info(f"Loaded FAISS index from {path} ({self.index.ntotal} vectors)")

    @property
    def ntotal(self) -> int:
        """返回索引中的向量数量"""
        if self._built:
            return self.index.ntotal
        return 0

    @property
    def is_built(self) -> bool:
        """返回索引是否已构建"""
        return self._built


def build_and_save_index(
    vectors: np.ndarray,
    output_path: str,
) -> FaissIndex:
    """
    构建并保存 FAISS 索引

    Args:
        vectors: 归一化后的向量
        output_path: 输出路径

    Returns:
        FaissIndex 对象
    """
    dimension = vectors.shape[1]
    faiss_index = FaissIndex(dimension)
    faiss_index.build(vectors)
    faiss_index.save(output_path)

    return faiss_index


def load_or_build_index(
    vectors: np.ndarray,
    index_path: str,
) -> FaissIndex:
    """
    加载已有索引或构建新索引

    Args:
        vectors: 归一化后的向量
        index_path: 索引文件路径

    Returns:
        FaissIndex 对象
    """
    index_path = Path(index_path)

    if index_path.exists():
        faiss_index = FaissIndex(vectors.shape[1])
        faiss_index.load(str(index_path))
    else:
        faiss_index = build_and_save_index(vectors, str(index_path))

    return faiss_index


if __name__ == "__main__":
    import json
    import time

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    PROJECT_ROOT = Path(__file__).parent.parent.parent

    # 确保目录存在
    embeddings_dir = PROJECT_ROOT / 'data' / 'embeddings'
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    vectors_path = embeddings_dir / 'movie_vectors.npy'
    index_path = embeddings_dir / 'faiss_index.bin'

    if vectors_path.exists():
        vectors = np.load(vectors_path)
        logger.info(f"Loaded vectors: {vectors.shape}")

        # 构建索引
        start = time.time()
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        logger.info(f"Built FAISS index with {index.ntotal} vectors")

        # 保存
        faiss.write_index(index, str(index_path))
        logger.info(f"Index built and saved in {time.time() - start:.2f}s")

        # 测试搜索
        query = vectors[0].reshape(1, -1)
        start = time.time()
        scores, indices = index.search(query, top_k=10)
        logger.info(f"Search completed in {time.time() - start*1000:.2f}ms")
        logger.info(f"Top-10 scores: {scores[0][:5]}")
    else:
        logger.warning(f"Vectors not found at {vectors_path}")
