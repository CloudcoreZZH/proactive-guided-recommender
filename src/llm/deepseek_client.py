"""DeepSeek API 客户端封装

提供统一的 API 调用接口，支持重试和错误处理
"""
import os
import time
import logging
from typing import Optional, List, Dict, Any

from openai import OpenAI

logger = logging.getLogger(__name__)


class DeepSeekClient:
    """DeepSeek API 客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        初始化 DeepSeek 客户端

        Args:
            api_key: DeepSeek API Key，如果不提供则从环境变量读取
            base_url: API 基础 URL
            max_retries: 最大重试次数
            retry_delay: 重试间隔（秒）
        """
        # 读取所有 API Keys（支持每行一个）
        all_keys = []
        if api_key is None:
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if api_key is None:
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                potential_paths = [
                    os.path.join(base_dir, "deepseek apikey.txt"),
                ]
                for config_path in potential_paths:
                    config_path = os.path.normpath(config_path)
                    if os.path.exists(config_path):
                        with open(config_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                key = line.strip()
                                if key and not key.startswith('#'):
                                    all_keys.append(key)
                        break

        if all_keys:
            # 使用第一个 key
            api_key = all_keys[0]
            logger.info(f"Found {len(all_keys)} API keys, using the first one")
        elif api_key is None:
            raise ValueError("DeepSeek API Key not found. Please provide it or set DEEPSEEK_API_KEY environment variable.")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        logger.info("DeepSeekClient initialized successfully")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "deepseek-chat",
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str:
        """
        发送聊天请求到 DeepSeek API

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大 token 数
            stream: 是否流式输出

        Returns:
            API 返回的文本内容
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                )

                if stream:
                    return response
                else:
                    return response.choices[0].message.content

            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

        return ""

    def generate_movie_description(
        self,
        title: str,
        genres: str,
        year: str,
    ) -> str:
        """
        生成电影描述

        Args:
            title: 电影标题
            genres: 电影类型
            year: 发行年份

        Returns:
            电影描述文本
        """
        from .prompts import MOVIE_DESCRIPTION_PROMPT

        prompt = MOVIE_DESCRIPTION_PROMPT.format(
            title=title,
            genres=genres,
            year=year
        )

        response = self.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return response.strip()

    def generate_user_profile(
        self,
        movie_list: List[str],
    ) -> str:
        """
        生成用户画像

        Args:
            movie_list: 用户观看过的电影列表

        Returns:
            用户画像描述
        """
        from .prompts import USER_PROFILE_PROMPT

        prompt = USER_PROFILE_PROMPT.format(
            movie_list="\n".join(f"- {m}" for m in movie_list)
        )

        response = self.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        return response.strip()


def get_deepseek_client() -> DeepSeekClient:
    """获取 DeepSeek 客户端单例"""
    if not hasattr(get_deepseek_client, "_instance"):
        get_deepseek_client._instance = DeepSeekClient()
    return get_deepseek_client._instance
