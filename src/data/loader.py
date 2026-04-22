"""MovieLens-1M 数据加载模块。"""
import os
import pandas as pd
from src.utils.config import get_data_path
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_ratings(data_dir: str = None) -> pd.DataFrame:
    """加载 ratings.dat，返回 DataFrame。"""
    if data_dir is None:
        data_dir = get_data_path('raw', 'ml-1m')
    path = os.path.join(data_dir, 'ratings.dat')
    logger.info(f"Loading ratings from {path}")
    df = pd.read_csv(
        path, sep='::', engine='python', header=None,
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        encoding='latin-1'
    )
    logger.info(f"Loaded {len(df)} ratings")
    return df


def load_movies(data_dir: str = None) -> pd.DataFrame:
    """加载 movies.dat，返回 DataFrame。"""
    if data_dir is None:
        data_dir = get_data_path('raw', 'ml-1m')
    path = os.path.join(data_dir, 'movies.dat')
    logger.info(f"Loading movies from {path}")
    df = pd.read_csv(
        path, sep='::', engine='python', header=None,
        names=['movie_id', 'title', 'genres'],
        encoding='latin-1'
    )
    logger.info(f"Loaded {len(df)} movies")
    return df


def load_users(data_dir: str = None) -> pd.DataFrame:
    """加载 users.dat，返回 DataFrame。"""
    if data_dir is None:
        data_dir = get_data_path('raw', 'ml-1m')
    path = os.path.join(data_dir, 'users.dat')
    logger.info(f"Loading users from {path}")
    df = pd.read_csv(
        path, sep='::', engine='python', header=None,
        names=['user_id', 'gender', 'age', 'occupation', 'zip'],
        encoding='latin-1'
    )
    logger.info(f"Loaded {len(df)} users")
    return df


def load_all(data_dir: str = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """加载所有三个数据文件。"""
    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)
    users = load_users(data_dir)
    return ratings, movies, users
