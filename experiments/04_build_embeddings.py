"""阶段4：向量化与语义检索构建 - 批量生成电影描述

使用 DeepSeek API 生成电影文本描述，支持断点续传和多进程并行
用法：
  单进程: python 04_build_embeddings.py
  多进程: python 04_build_embeddings.py --num-workers 4
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path
from multiprocessing import Process, Queue

# 设置项目根目录（pomelo文件夹）
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

from src.llm.deepseek_client import DeepSeekClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(process)d] %(message)s'
)
logger = logging.getLogger(__name__)


def load_movies(filepath: str) -> pd.DataFrame:
    """加载电影数据"""
    movies = []
    with open(filepath, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) >= 3:
                movie_id = parts[0]
                title_year = parts[1]
                genres = parts[2]

                # 解析标题和年份
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


def worker_process(
    worker_id: int,
    total_workers: int,
    movies_df: pd.DataFrame,
    output_path: str,
    api_key_index: int,
    keys: list,
    batch_limit: int = None,
):
    """
    工作进程：处理分配给它的电影
    每个 worker 写独立文件（worker_{id}.json），避免多进程竞态覆盖。
    """
    pid = os.getpid()
    logger.info(f"[Worker {worker_id}] Process {pid} started with API key index {api_key_index}")

    # 初始化客户端
    try:
        api_client = DeepSeekClient(api_key=keys[api_key_index])
    except Exception as e:
        logger.error(f"[Worker {worker_id}] Failed to initialize API client: {e}")
        return

    output_dir = Path(output_path).parent
    # 每个 worker 写自己的独立文件，彻底避免竞态覆盖
    worker_file = output_dir / f"worker_{worker_id}.json"

    # 加载主文件 + 本 worker 已有文件（断点续传）
    done_ids = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                main_desc = json.load(f)
            done_ids.update(main_desc.keys())
        except:
            pass

    my_descriptions = {}
    if worker_file.exists():
        try:
            with open(worker_file, 'r', encoding='utf-8') as f:
                my_descriptions = json.load(f)
            done_ids.update(my_descriptions.keys())
            logger.info(f"[Worker {worker_id}] Resumed {len(my_descriptions)} from own file")
        except:
            pass

    # 先过滤未完成电影，再按 modulo 分配（避免剩余集中在单个 worker）
    remaining = movies_df[~movies_df['movie_id'].isin(done_ids)].reset_index(drop=True)
    assigned_mask = [(i % total_workers) == worker_id for i in range(len(remaining))]
    movies_assigned = remaining[assigned_mask].copy()

    if batch_limit:
        movies_assigned = movies_assigned.head(batch_limit)

    logger.info(f"[Worker {worker_id}] Need to process {len(movies_assigned)} movies")

    if len(movies_assigned) == 0:
        logger.info(f"[Worker {worker_id}] Nothing to do, exiting")
        return

    # 批量生成，只写自己的文件
    errors = []
    for idx, (_, row) in enumerate(movies_assigned.iterrows()):
        movie_id = row['movie_id']
        try:
            description = api_client.generate_movie_description(
                title=row['title'],
                genres=row['genres'],
                year=row['year']
            )
            my_descriptions[movie_id] = description

            if (idx + 1) % 20 == 0:
                with open(worker_file, 'w', encoding='utf-8') as f:
                    json.dump(my_descriptions, f, ensure_ascii=False, indent=2)
                logger.info(f"[Worker {worker_id}] Progress: {idx + 1}/{len(movies_assigned)}")

        except Exception as e:
            logger.error(f"[Worker {worker_id}] Error for movie {movie_id}: {e}")
            errors.append((movie_id, str(e)))

    # 最终保存 worker 文件
    with open(worker_file, 'w', encoding='utf-8') as f:
        json.dump(my_descriptions, f, ensure_ascii=False, indent=2)

    logger.info(f"[Worker {worker_id}] Done! Generated {len(my_descriptions)} descriptions, {len(errors)} errors")


def merge_worker_files(output_path: str, num_workers: int) -> dict:
    """合并所有 worker 文件到主文件"""
    output_dir = Path(output_path).parent
    merged = {}

    # 先加载主文件（如果存在）
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                merged = json.load(f)
            logger.info(f"Loaded {len(merged)} from main file")
        except:
            pass

    # 合并所有 worker 文件
    for i in range(num_workers):
        wfile = output_dir / f"worker_{i}.json"
        if wfile.exists():
            try:
                with open(wfile, 'r', encoding='utf-8') as f:
                    wdata = json.load(f)
                merged.update(wdata)
                logger.info(f"Merged worker_{i}.json: {len(wdata)} entries")
            except Exception as e:
                logger.error(f"Failed to merge worker_{i}.json: {e}")

    # 写回主文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    logger.info(f"Merge complete: {len(merged)} total descriptions")
    return merged


def load_api_keys() -> list:
    """加载所有 API keys"""
    # 使用项目根目录的直接父目录
    key_file = Path(__file__).parent.parent.parent / "deepseek apikey.txt"
    key_file = key_file.resolve()

    keys = []
    if key_file.exists():
        with open(key_file, 'r', encoding='utf-8') as f:
            for line in f:
                key = line.strip()
                if key and not key.startswith('#'):
                    keys.append(key)

    logger.info(f"Loaded {len(keys)} API keys from {key_file}")
    return keys


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Generate movie descriptions using DeepSeek API')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--batch-limit', type=int, default=None, help='Limit movies per worker (for testing)')
    parser.add_argument('--continue', dest='continue_mode', action='store_true', help='Continue from existing descriptions')
    args = parser.parse_args()

    num_workers = args.num_workers
    batch_limit = args.batch_limit

    # 设置环境变量：模型下载到当前目录
    os.environ['HF_HOME'] = str(PROJECT_ROOT / 'models')
    os.environ['TRANSFORMERS_CACHE'] = str(PROJECT_ROOT / 'models')
    logger.info(f"HF_HOME set to: {os.environ['HF_HOME']}")

    # 加载 API keys
    keys = load_api_keys()
    if not keys:
        logger.error("No API keys found!")
        return

    # 确保 workers 数量不超过 key 数量
    if num_workers > len(keys):
        logger.warning(f"Workers ({num_workers}) > API keys ({len(keys)}), using {len(keys)} workers")
        num_workers = len(keys)

    # 路径配置
    movies_path = PROJECT_ROOT / 'data' / 'raw' / 'ml-1m' / 'movies.dat'
    output_path = PROJECT_ROOT / 'data' / 'processed' / 'movie_descriptions.json'

    logger.info(f"Loading movies from {movies_path}")
    movies_df = load_movies(str(movies_path))
    logger.info(f"Loaded {len(movies_df)} movies")

    if num_workers == 1:
        # 单进程模式
        logger.info("Running in single-worker mode")
        api_client = DeepSeekClient(api_key=keys[0])

        existing_descriptions = {}
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_descriptions = json.load(f)
                logger.info(f"Loaded {len(existing_descriptions)} existing descriptions")
            except:
                pass

        descriptions = existing_descriptions.copy()
        movies_to_process = movies_df[~movies_df['movie_id'].isin(descriptions.keys())]

        if batch_limit:
            movies_to_process = movies_to_process.head(batch_limit)

        logger.info(f"Need to generate {len(movies_to_process)} descriptions")

        for idx, row in tqdm(movies_to_process.iterrows(), total=len(movies_to_process),
                             desc="Generating descriptions"):
            movie_id = row['movie_id']
            try:
                description = api_client.generate_movie_description(
                    title=row['title'],
                    genres=row['genres'],
                    year=row['year']
                )
                descriptions[movie_id] = description

                if len(descriptions) % 20 == 0:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(descriptions, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Error for movie {movie_id}: {e}")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(descriptions, f, ensure_ascii=False, indent=2)

    else:
        # 多进程模式
        logger.info(f"Running in multi-worker mode with {num_workers} workers")

        # 加载已有描述
        existing_descriptions = {}
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_descriptions = json.load(f)
                logger.info(f"Loaded {len(existing_descriptions)} existing descriptions")
            except:
                pass

        # 启动工作进程
        processes = []
        for i in range(num_workers):
            p = Process(
                target=worker_process,
                args=(i, num_workers, movies_df, str(output_path), i % len(keys), keys, batch_limit)
            )
            p.start()
            processes.append(p)
            logger.info(f"Started worker {i}")

        # 等待所有进程完成
        for p in processes:
            p.join()
            logger.info(f"Worker process finished")

        # 合并所有 worker 文件到主文件
        logger.info("Merging worker files...")
        merge_worker_files(str(output_path), num_workers)

    # 输出统计
    with open(output_path, 'r', encoding='utf-8') as f:
        final_descriptions = json.load(f)

    logger.info(f"\n=== Summary ===")
    logger.info(f"Total movies: {len(movies_df)}")
    logger.info(f"Descriptions generated: {len(final_descriptions)}")
    logger.info(f"Output: {output_path}")

    # 展示样例
    logger.info("\n=== Sample Descriptions ===")
    sample_movies = ['1', '2', '3', '10', '50']
    for movie_id in sample_movies:
        if movie_id in final_descriptions:
            movie_info = movies_df[movies_df['movie_id'] == movie_id].iloc[0]
            logger.info(f"\n[{movie_id}] {movie_info['title']} ({movie_info['year']})")
            logger.info(f"  {final_descriptions[movie_id][:150]}...")


if __name__ == "__main__":
    main()