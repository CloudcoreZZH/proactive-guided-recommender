"""用户画像向量化模块

根据用户历史观看电影生成"兴趣画像"，并将其向量化用于检索
"""
import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from .embedder import TextEmbedder
from ..llm.deepseek_client import DeepSeekClient

logger = logging.getLogger(__name__)


class UserProfileVectorizer:
    """用户画像向量化器"""

    def __init__(
        self,
        embedder: TextEmbedder,
        deepseek_client: Optional[DeepSeekClient] = None,
    ):
        """
        初始化用户画像向量化器

        Args:
            embedder: 文本向量化器
            deepseek_client: DeepSeek客户端（如果为None则使用fallback方案）
        """
        self.embedder = embedder
        self.deepseek_client = deepseek_client

    def generate_profile_with_llm(
        self,
        user_movie_ids: List[str],
        movie_id_to_description: Dict[str, str],
        movie_id_to_title: Dict[str, str],
    ) -> str:
        """
        使用 LLM 生成用户画像文本

        Args:
            user_movie_ids: 用户观看过的电影ID列表
            movie_id_to_description: 电影ID到描述的映射
            movie_id_to_title: 电影ID到标题的映射

        Returns:
            用户画像描述文本
        """
        if self.deepseek_client is None:
            raise ValueError("DeepSeek client not available")

        # 构建电影列表
        movie_list = []
        for mid in user_movie_ids[:50]:  # 最多50部
            title = movie_id_to_title.get(mid, f"Movie {mid}")
            movie_list.append(title)

        if not movie_list:
            return "用户暂无观影记录"

        # 调用 LLM 生成画像
        profile = self.deepseek_client.generate_user_profile(movie_list)

        return profile

    def generate_profile_with_averaging(
        self,
        user_movie_ids: List[str],
        movie_id_to_vector: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        使用向量平均生成用户画像

        Args:
            user_movie_ids: 用户观看过的电影ID列表
            movie_id_to_vector: 电影ID到向量的映射

        Returns:
            用户画像向量
        """
        vectors = []
        for mid in user_movie_ids:
            if mid in movie_id_to_vector:
                vectors.append(movie_id_to_vector[mid])

        if not vectors:
            # 返回零向量
            return np.zeros(self.embedder.dimension)

        # 计算加权平均（可以加权重，比如按评分）
        profile_vector = np.mean(vectors, axis=0)

        # 归一化
        norm = np.linalg.norm(profile_vector)
        if norm > 0:
            profile_vector = profile_vector / norm

        return profile_vector

    def profile_to_vector(self, profile_text: str) -> np.ndarray:
        """
        将画像文本转为向量

        Args:
            profile_text: 用户画像文本

        Returns:
            画像向量
        """
        return self.embedder.encode_single(profile_text, normalize=True)

    def generate_profile_vector(
        self,
        user_movie_ids: List[str],
        movie_id_to_description: Dict[str, str],
        movie_id_to_title: Dict[str, str],
        movie_id_to_vector: Optional[Dict[str, np.ndarray]] = None,
        use_llm: bool = True,
    ) -> np.ndarray:
        """
        生成用户画像向量

        Args:
            user_movie_ids: 用户观看过的电影ID列表
            movie_id_to_description: 电影ID到描述的映射
            movie_id_to_title: 电影ID到标题的映射
            movie_id_to_vector: 电影ID到向量的映射
            use_llm: 是否使用LLM生成画像（False则用向量平均）

        Returns:
            用户画像向量
        """
        if use_llm and self.deepseek_client:
            # 使用 LLM 生成画像文本，然后转为向量
            profile_text = self.generate_profile_with_llm(
                user_movie_ids, movie_id_to_description, movie_id_to_title
            )
            profile_vector = self.profile_to_vector(profile_text)
            logger.info(f"Generated LLM profile: {profile_text[:50]}...")
        else:
            # 使用向量平均
            if movie_id_to_vector is None:
                raise ValueError("movie_id_to_vector required for averaging method")
            profile_vector = self.generate_profile_with_averaging(
                user_movie_ids, movie_id_to_vector
            )
            logger.info("Generated profile with vector averaging")

        return profile_vector


def load_user_history(train_path: str) -> Dict[int, List[int]]:
    """
    加载用户历史观影记录

    Args:
        train_path: 训练集路径

    Returns:
        {user_id: [movie_id, ...]} 字典
    """
    df = pd.read_csv(train_path)
    user_history = df.groupby('user_id')['item_id'].apply(list).to_dict()
    return user_history


def build_user_profiles(
    user_ids: List[int],
    user_history: Dict[int, List[int]],
    movie_descriptions: Dict[str, str],
    movie_vectors: np.ndarray,
    movie_id_map: List[str],
    embedder: TextEmbedder,
    deepseek_client: Optional[DeepSeekClient] = None,
    use_llm: bool = True,
) -> Dict[int, np.ndarray]:
    """
    为多个用户批量生成画像向量

    Args:
        user_ids: 用户ID列表
        user_history: 用户历史记录
        movie_descriptions: 电影描述
        movie_vectors: 电影向量
        movie_id_map: 电影ID列表（索引对应向量）
        embedder: 向量化器
        deepseek_client: DeepSeek客户端
        use_llm: 是否使用LLM

    Returns:
        {user_id: profile_vector} 字典
    """
    # 构建 movie_id 到索引的映射
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(movie_id_map)}

    # 构建 movie_id 到向量的映射
    movie_id_to_vector = {}
    for mid, idx in movie_id_to_idx.items():
        movie_id_to_vector[mid] = movie_vectors[idx]

    # 构建 movie_id 到标题的映射
    movie_id_to_title = {}
    for mid, desc in list(movie_descriptions.items())[:10]:
        movie_id_to_title[mid] = f"Movie {mid}"

    # 初始化画像向量化器
    vectorizer = UserProfileVectorizer(embedder, deepseek_client)

    # 批量生成
    user_profiles = {}
    for uid in user_ids:
        if uid not in user_history:
            # 新用户，返回零向量
            user_profiles[uid] = np.zeros(embedder.dimension)
            continue

        movie_ids = [str(mid) for mid in user_history[uid]]

        try:
            profile_vector = vectorizer.generate_profile_vector(
                user_movie_ids=movie_ids,
                movie_id_to_description=movie_descriptions,
                movie_id_to_title=movie_id_to_title,
                movie_id_to_vector=movie_id_to_vector,
                use_llm=use_llm,
            )
            user_profiles[uid] = profile_vector
        except Exception as e:
            logger.error(f"Error generating profile for user {uid}: {e}")
            user_profiles[uid] = np.zeros(embedder.dimension)

    return user_profiles


if __name__ == "__main__":
    import json

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    PROJECT_ROOT = Path(__file__).parent.parent.parent

    # 设置模型下载路径
    os.environ['HF_HOME'] = str(PROJECT_ROOT / 'models')
    os.environ['TRANSFORMERS_CACHE'] = str(PROJECT_ROOT / 'models')

    # 加载数据
    descriptions_path = PROJECT_ROOT / 'data' / 'processed' / 'movie_descriptions.json'
    with open(descriptions_path, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)

    logger.info(f"Loaded {len(descriptions)} movie descriptions")

    # 测试向量平均方法
    embedder = TextEmbedder(model_name='bge-large', batch_size=32)
    vectorizer = UserProfileVectorizer(embedder)

    # 模拟用户历史
    test_movie_ids = ['1', '2', '3', '10', '50']
    test_vectors = {mid: np.random.randn(1024) for mid in test_movie_ids}
    test_vectors = {mid: v / np.linalg.norm(v) for mid, v in test_vectors.items()}

    profile_vec = vectorizer.generate_profile_with_averaging(
        test_movie_ids, test_vectors
    )

    logger.info(f"Profile vector shape: {profile_vec.shape}")
    logger.info("User profile generation (fallback) working!")
