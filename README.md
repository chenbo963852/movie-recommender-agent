# 基于 RAG + Agent 的电影推荐系统

本项目实现了一个基于 RAG + Agent 的电影推荐系统，结合 FastAPI、Qdrant 向量检索、BM25 关键词召回、MovieLens 用户画像、TMDB / MovieLens 质量分与混合排序策略，支持自然语言约束解析、已看电影过滤、个性化推荐与 strict / fallback 多阶段 Agent 推荐流程。

## 项目简介

本系统不仅支持普通的相似电影推荐，还支持用户输入自然语言需求，例如：

```text
推荐 2010 年之后的高评分科幻片，不要恐怖片，推荐 5 部
```

系统会自动解析出结构化参数，例如：

```text
genre = Science Fiction
exclude_genre = Horror
year_from = 2010
min_vote_average = 7.0
top_k = 5
```

然后通过向量检索、BM25 召回、规则过滤、质量分排序和 Agent fallback 策略返回最终推荐结果。

## 核心功能

- 基于 SentenceTransformer 的电影语义向量检索
- 基于 Qdrant 的本地向量数据库
- 基于 BM25 的关键词召回
- 向量召回 + BM25 召回 + 质量分的混合排序
- 支持 MovieLens 用户画像的个性化推荐
- 支持已看电影过滤，避免推荐用户已经看过的电影
- 支持自然语言推荐需求解析
- 支持 strict / fallback 多阶段 Agent 推荐流程
- 支持 MovieLens 无数据泄漏离线评估
- 支持 Agent 参数抽取与约束满足率评估

## 系统架构

```text
用户自然语言请求
    |
    v
Intent Parser / Local LLM
    |
    v
结构化推荐参数
    |
    v
Qdrant 向量召回 + BM25 关键词召回
    |
    v
按 genre / year / rating / vote count 过滤
    |
    v
融合 TMDB / MovieLens 质量分进行混合排序
    |
    v
Agent strict / fallback 多阶段推荐
    |
    v
生成最终推荐结果与推荐理由
```

## 推荐系统离线评估

| 模型 | HitRate@10 | Recall@10 | NDCG@10 | Coverage@10 | Seen Violation |
|---|---:|---:|---:|---:|---:|
| Popular Baseline | 0.0910 | 0.0515 | 0.0301 | 0.0021 | 0 |
| Vector Seed | 0.0350 | 0.0185 | 0.0164 | 0.0773 | 0 |
| Personalized Rerank | 0.0420 | 0.0230 | 0.0177 | 0.0597 | 0 |
| Main Hybrid Recommender | 0.0370 | 0.0195 | 0.0134 | 0.0390 | 0 |

本项目构建了无数据泄漏的 MovieLens 离线评估流程，对比了热门推荐、向量召回、个性化重排与主混合推荐模型。

Popular Baseline 在 HitRate@10 上更高，说明热门电影在离线留出评估中具有天然优势；而向量召回和混合推荐方法能够提升 Coverage，覆盖更多长尾电影，并且支持个性化推荐、自然语言约束和已看电影过滤。

系统 Seen Violation Rate 为 0，说明推荐结果中不会出现用户已经看过的电影。

## Agent 评估

| 指标 | 结果 |
|---|---:|
| 测试样本数 | 20 |
| 参数完全匹配率 | 0.9000 |
| Genre 抽取准确率 | 1.0000 |
| Exclude Genre 抽取准确率 | 1.0000 |
| Year From 抽取准确率 | 0.9500 |
| Min Vote Count 抽取准确率 | 0.9500 |
| Top-K 抽取准确率 | 1.0000 |
| 非空返回率 | 0.9500 |
| Fallback 成功率 | 0.6667 |
| 约束满足率 | 0.8333 |

Agent 评估用于验证系统是否能正确理解用户自然语言需求，并将其转换为推荐系统可执行的结构化参数。

## 技术栈

- Python
- FastAPI
- Qdrant
- SentenceTransformers
- BM25
- MovieLens
- TMDB metadata
- Pandas
- PyTorch
- Transformers

## 项目结构

```text
qdrant_search_demo/
├── app/
│   ├── main.py
│   ├── schemas.py
│   └── services/
│       ├── agent_recommendation_service.py
│       ├── bm25_service.py
│       ├── embedding_service.py
│       ├── intent_parser_service.py
│       ├── local_llm_service.py
│       ├── qdrant_service.py
│       ├── ratings_service.py
│       ├── recommendation_service.py
│       ├── retrieval_service.py
│       ├── user_profile_service.py
│       └── user_seen_movies_service.py
├── scripts/
├── data/
│   └── eval/
├── requirements.txt
├── README.md
└── .gitignore
```

## 运行方式

安装依赖：

```bash
pip install -r requirements.txt
```

启动 FastAPI 服务：

```bash
uvicorn app.main:app --reload
```

打开接口文档：

```text
http://127.0.0.1:8000/docs
```

## 数据与模型说明

本仓库不上传大型本地文件，包括：

- 本地 embedding / LLM 模型文件
- Qdrant 本地向量数据库文件
- MovieLens 原始大数据集
- TMDB 原始数据
- BM25 缓存文件
- 中间处理生成的 parquet / csv 文件

如需完整复现，需要单独下载 MovieLens 数据集，并准备本地模型文件。

## 项目亮点

本项目不是一个简单的推荐 demo，而是实现了一个较完整的推荐系统工程闭环，包括：

- 检索召回
- 混合排序
- 用户画像
- 已看过滤
- 自然语言约束解析
- Agent fallback
- 推荐理由生成
- 离线评估
- Agent 评估

适合作为推荐系统、RAG、Agent 和机器学习工程方向的简历项目。
