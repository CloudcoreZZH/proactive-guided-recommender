"""文本向量化模块

使用 Sentence-Transformers 加载 BGE 模型，将所有电影描述编码为向量
"""
import os
import logging
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class TextEmbedder:
    """文本向量化器"""

    # 模型选择优先级
    MODEL_CONFIGS = {
        'bge-large': {
            'name': 'BAAI/bge-large-zh-v1.5',
            'dimension': 1024,
            'description': '效果最好，约 1.3GB'
        },
        'bge-base': {
            'name': 'BAAI/bge-base-zh-v1.5',
            'dimension': 768,
            'description': '显存不够时使用'
        },
        'minilm': {
            'name': 'sentence-transformers/all-MiniLM-L6-v2',
            'dimension': 384,
            'description': '最轻量的 fallback'
        }
    }

    def __init__(
        self,
        model_name: str = 'bge-large',
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        """
        初始化向量化器

        Args:
            model_name: 模型名称 ('bge-large', 'bge-base', 'minilm')
            batch_size: 批量大小
            device: 计算设备 ('cuda', 'cpu')，默认自动选择
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device
        self.batch_size = batch_size
        self.model_name = model_name

        config = self.MODEL_CONFIGS[model_name]
        self.model = SentenceTransformer(config['name'], device=device)
        self.dimension = config['dimension']

        logger.info(f"Loaded embedder model: {config['name']} (dim={self.dimension})")
        logger.info(f"Device: {self.device}")

    def encode(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        将文本列表编码为向量

        Args:
            texts: 文本列表
            normalize: 是否归一化（L2 norm）
            show_progress: 是否显示进度条

        Returns:
            numpy 数组，形状为 (n, dimension)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """编码单个文本"""
        embedding = self.model.encode(
            [text],
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return embedding[0]

    def clear_cache(self):
        """清理 GPU 缓存"""
        if self.device == 'cuda':
            del self.model
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")


def build_movie_embeddings(
    descriptions: dict,
    output_dir: str,
    model_name: str = 'bge-large',
    batch_size: int = 32,
) -> Tuple[np.ndarray, list]:
    """
    构建电影向量库

    Args:
        descriptions: {movie_id: description} 字典
        output_dir: 输出目录
        model_name: 模型名称
        batch_size: 批量大小

    Returns:
        (vectors, movie_ids) 元组
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 按 movie_id 排序确保顺序一致
    movie_ids = sorted(descriptions.keys())
    texts = [descriptions[mid] for mid in movie_ids]

    logger.info(f"Building embeddings for {len(texts)} movies...")
    logger.info(f"Model: {model_name}, Batch size: {batch_size}")

    # 初始化向量化器
    embedder = TextEmbedder(model_name=model_name, batch_size=batch_size)

    # 编码
    vectors = embedder.encode(texts, normalize=True)

    # 保存向量
    vectors_path = output_dir / 'movie_vectors.npy'
    np.save(vectors_path, vectors)
    logger.info(f"Saved vectors to {vectors_path}")

    # 保存 movie_id 映射
    id_map_path = output_dir / 'movie_id_map.json'
    import json
    with open(id_map_path, 'w', encoding='utf-8') as f:
        json.dump(movie_ids, f, ensure_ascii=False)
    logger.info(f"Saved movie_id map to {id_map_path}")

    # 清理缓存
    embedder.clear_cache()

    return vectors, movie_ids


if __name__ == "__main__":
    import json

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 设置模型下载路径
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    os.environ['HF_HOME'] = str(PROJECT_ROOT / 'models')
    os.environ['TRANSFORMERS_CACHE'] = str(PROJECT_ROOT / 'models')
    logger.info(f"HF_HOME: {os.environ['HF_HOME']}")

    # 加载描述
    descriptions_path = PROJECT_ROOT / 'data' / 'processed' / 'movie_descriptions.json'
    with open(descriptions_path, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)

    logger.info(f"Loaded {len(descriptions)} movie descriptions")

    # 构建向量
    vectors, movie_ids = build_movie_embeddings(
        descriptions=descriptions,
        output_dir=str(PROJECT_ROOT / 'data' / 'embeddings'),
        model_name='bge-large',
        batch_size=32,
    )

    logger.info(f"\n=== Summary ===")
    logger.info(f"Movies: {len(movie_ids)}")
    logger.info(f"Vector shape: {vectors.shape}")
