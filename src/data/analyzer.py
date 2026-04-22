"""数据分析模块：统计信息与可视化。"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from src.utils.config import get_results_path
from src.utils.logger import setup_logger

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

logger = setup_logger(__name__)


def compute_statistics(ratings: pd.DataFrame,
                       movies: pd.DataFrame,
                       users: pd.DataFrame) -> dict:
    """计算数据集的基本统计信息。"""
    n_users = ratings['user_id'].nunique()
    n_movies = ratings['movie_id'].nunique()
    n_ratings = len(ratings)
    sparsity = 1.0 - n_ratings / (n_users * n_movies)

    stats = {
        'n_users': n_users,
        'n_movies': n_movies,
        'n_ratings': n_ratings,
        'sparsity': sparsity,
        'avg_ratings_per_user': n_ratings / n_users,
        'avg_ratings_per_movie': n_ratings / n_movies,
        'rating_mean': ratings['rating'].mean(),
        'rating_std': ratings['rating'].std(),
    }

    logger.info("Dataset Statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

    return stats


def plot_user_activity(ratings: pd.DataFrame, save_dir: str = None):
    """绘制用户活跃度分布。"""
    user_counts = ratings.groupby('user_id').size()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(user_counts, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Number of Ratings')
    axes[0].set_ylabel('Number of Users')
    axes[0].set_title('User Activity Distribution')

    axes[1].hist(user_counts, bins=50, edgecolor='black', alpha=0.7, color='steelblue', log=True)
    axes[1].set_xlabel('Number of Ratings')
    axes[1].set_ylabel('Number of Users (log scale)')
    axes[1].set_title('User Activity Distribution (Log Scale)')

    plt.tight_layout()
    _save_fig(fig, 'user_activity_distribution.png', save_dir)
    return fig


def plot_item_popularity(ratings: pd.DataFrame, save_dir: str = None):
    """绘制电影热门度分布。"""
    item_counts = ratings.groupby('movie_id').size().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(item_counts, bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[0].set_xlabel('Number of Ratings')
    axes[0].set_ylabel('Number of Movies')
    axes[0].set_title('Movie Popularity Distribution')

    axes[1].plot(range(len(item_counts)), item_counts.values, color='coral', linewidth=0.8)
    axes[1].set_xlabel('Movie Rank')
    axes[1].set_ylabel('Number of Ratings')
    axes[1].set_title('Movie Popularity Long Tail')
    axes[1].set_yscale('log')

    plt.tight_layout()
    _save_fig(fig, 'item_popularity_distribution.png', save_dir)
    return fig


def plot_rating_distribution(ratings: pd.DataFrame, save_dir: str = None):
    """绘制评分分布。"""
    fig, ax = plt.subplots(figsize=(8, 5))

    rating_counts = ratings['rating'].value_counts().sort_index()
    ax.bar(rating_counts.index, rating_counts.values, width=0.6,
           edgecolor='black', alpha=0.7, color='mediumpurple')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Count')
    ax.set_title('Rating Distribution')
    ax.set_xticks([1, 2, 3, 4, 5])

    for i, (r, c) in enumerate(zip(rating_counts.index, rating_counts.values)):
        ax.text(r, c + 1000, f'{c:,}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    _save_fig(fig, 'rating_distribution.png', save_dir)
    return fig


def plot_genre_distribution(movies: pd.DataFrame, save_dir: str = None):
    """绘制电影类型分布。"""
    all_genres = movies['genres'].str.split('|').explode()
    genre_counts = all_genres.value_counts()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(genre_counts)), genre_counts.values, color='teal', alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(genre_counts)))
    ax.set_yticklabels(genre_counts.index)
    ax.set_xlabel('Number of Movies')
    ax.set_title('Genre Distribution')
    ax.invert_yaxis()

    for i, v in enumerate(genre_counts.values):
        ax.text(v + 5, i, str(v), va='center', fontsize=9)

    plt.tight_layout()
    _save_fig(fig, 'genre_distribution.png', save_dir)
    return fig


def _save_fig(fig, filename: str, save_dir: str = None):
    """Save figure to disk."""
    if save_dir is None:
        save_dir = get_results_path('figures', 'data_analysis')
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved figure: {path}")
    plt.close(fig)
