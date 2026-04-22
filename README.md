# Pomelo（破茧）—— 个性化内容推荐系统

> 实现并对比五种推荐方法，验证"三通道融合推荐架构"的有效性

## 项目简介

本项目构建了一个完整的个性化电影推荐系统，涵盖从传统协同过滤到 LLM 增强推荐的多种方法，并提出"三通道融合架构"（Pomelo），在保持推荐准确率的同时提升多样性、新颖性和意外发现度。

## 实现方法

| 方法 | 类别 | 说明 |
|------|------|------|
| ItemCF | 协同过滤 | 基于物品的协同过滤，余弦相似度 + IUF 惩罚 |
| MF-BPR | 矩阵分解 | PyTorch 实现，BPR 损失函数 |
| LightGCN | 图神经网络 | 基于 RecBole 框架 |
| SASRec | 序列推荐 | 基于 RecBole 框架 |
| RAG | LLM 增强 | FAISS 检索 + DeepSeek 重排 |
| Pomelo | 三通道融合 | 精准 + 探索 + Serendipity 通道 |

## 数据集

- **MovieLens-1M**：约 100 万评分，6040 用户，3883 电影

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Key

# 数据预处理
python -m src.data.preprocessor

# 运行实验（按阶段）
python experiments/02_baseline_cf.py
python experiments/03_baseline_deep.py
python experiments/05_rag_recommendation.py
python experiments/06_pomelo_fusion.py
python experiments/07_final_evaluation.py

# 启动 Demo
streamlit run app/streamlit_demo.py
```

## 评估指标

- **Recall@K**：召回率
- **NDCG@K**：归一化折损累积增益
- **Diversity**：推荐列表内多样性
- **Novelty**：推荐新颖性
- **Serendipity**：意外发现度

## 项目结构

```
pomelo/
├── config/          # 配置文件
├── data/            # 数据目录
├── src/             # 源代码
├── experiments/     # 实验脚本
├── results/         # 实验结果
├── app/             # Streamlit Demo
└── tests/           # 单元测试
```

## 技术栈

Python 3.10 | PyTorch 2.0+ | RecBole | FAISS | Sentence-Transformers | DeepSeek API | Streamlit
