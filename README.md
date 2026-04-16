# Lumina Insight (智光洞察) 🚀

Lumina Insight 是一个企业级的 RAG（检索增强生成）后端系统。基于 FastAPI 构建，采用异步非阻塞架构，专为处理包含复杂表格的 PDF 文档和高并发企业知识问答场景设计。

## ✨ 核心特性

* **多模态文档解析**: 摒弃传统的纯文本提取，集成 `PyMuPDF4LLM`，将复杂的 PDF 表格与排版完美还原为 Markdown 语义结构。
* **企业级权限隔离**: 基于 PostgreSQL + `pgvector` 的多租户架构，在底层 SQL 层面实现严密的（Public/Private/Uploader）Metadata 权限过滤。
* **精准检索漏斗**: 采用 "向量粗排 (L2 Distance) + Rerank 语义精排" 的混合检索架构，大幅降低大模型幻觉。
* **结构化流式输出 (SSE)**: 专为现代前端设计的流式接口，实现"首帧下发溯源引用源 -> 随后流式打字输出回答"的丝滑体验。
* **异步连接池**: 基于 SQLAlchemy 和 asyncpg，轻松应对高并发数据库读写。

## 🛠️ 技术栈

* **框架**: FastAPI, Uvicorn
* **ORM & 数据库**: SQLAlchemy, PostgreSQL, pgvector (运行于 Docker)
* **AI 引擎**: 阿里云百炼 (DashScope) - Qwen-Plus (大模型), gte-rerank (重排序), text-embedding-v1 (向量)
* **文档处理**: PyMuPDF4LLM, LangChain (仅用于切分和流式包装)

## 🚀 快速启动

### 1. 环境准备
确保你的电脑已安装 Git、Python 3.10+ 和 Docker Desktop。

```bash
git clone [https://github.com/你的用户名/lumina-insight.git](https://github.com/你的用户名/lumina-insight.git)
cd lumina-insight
python -m venv venv
source venv/Scripts/activate  # Windows 用户
pip install -r requirements.txt
```

### 2. 启动数据库基础设施
```bash
docker-compose up -d
```

### 3. 配置环境变量
在项目根目录创建 .env 文件，填入你的阿里云 API Key：
```bash
DASHSCOPE_API_KEY="sk-你的真实百炼Key"
DATABASE_URL="postgresql+asyncpg://postgres:mysecretpassword@localhost:5432/lumina_db"
```

### 4. 数据入库与启动服务
```bash
# 将测试 PDF 置于 data/ 目录下并执行入库(默认名：test.pdf)
python ingest_data.py

# 启动 FastAPI 服务
uvicorn main:app --reload
```
服务启动后，可通过 http://127.0.0.1:8000/docs 访问自动生成的交互式 API 文档。