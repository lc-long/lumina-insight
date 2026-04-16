import os
import dashscope
from typing import List
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_community.embeddings import DashScopeEmbeddings
from core.models import KnowledgeChunk

# 配置百炼的 API Key 给 dashscope SDK 使用 (Rerank 需要)
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

class EnterpriseRetriever:
    def __init__(self):
        self.embeddings_model = DashScopeEmbeddings(model="text-embedding-v1")

    async def hybrid_search(self, db: AsyncSession, query: str, user_id: str, top_k: int = 3) -> List[dict]:
        """
        企业级混合检索核心逻辑
        1. 向量转换 -> 2. 数据库向量粗排 + 权限过滤 -> 3. Rerank 模型精排
        """
        print(f"\n[1/3] 正在将用户问题转化为向量: '{query}'")
        # 1. 将用户的查询转化为高维向量
        query_vector = self.embeddings_model.embed_query(query)

        print(f"[2/3] 正在数据库中进行带权限过滤的向量召回 (粗排 Top 15)...")
        # 2. 构建 SQLAlchemy 异步查询
        # 使用 pgvector 的 l2_distance (<->) 运算符计算距离
        stmt = (
            select(KnowledgeChunk)
            # 权限隔离核心：要么是公开文档，要么是自己上传的私有文档
            .filter(
                or_(
                    KnowledgeChunk.access_level == "public",
                    KnowledgeChunk.uploader_id == user_id
                )
            )
            # 按照向量 L2 距离从小到大排序 (越小越相似)
            .order_by(KnowledgeChunk.embedding.l2_distance(query_vector))
            # 粗排多召回一些，给 Rerank 留足空间
            .limit(15) 
        )

        result = await db.execute(stmt)
        # 获取粗排召回的文档对象
        recalled_chunks = result.scalars().all()
        
        if not recalled_chunks:
            print("未召回到任何有权限的知识块。")
            return []

        print(f"[3/3] 召回完成 (共 {len(recalled_chunks)} 条)，正在请求 Rerank 大模型进行语义精排...")
        
        # 3. Rerank 精排
        # 将我们数据库查出来的文本，拼装成 DashScope 需要的格式
        documents_for_rerank = [chunk.content for chunk in recalled_chunks]
        
        # 调用阿里云的文本重排序模型
        resp = dashscope.TextReRank.call(
            model="gte-rerank",
            query=query,
            documents=documents_for_rerank,
            top_n=top_k,
            return_documents=True # 让 API 把排序后的文本一起返回
        )
        
        if resp.status_code != 200:
            raise Exception(f"Rerank 接口报错: {resp.message}")

        # 4. 数据组装与溯源信息打包
        final_results = []
        # Rerank 返回的 results 包含了新的排序 index 和 relevance_score (相关性得分)
        for reranked_item in resp.output.results:
            # 根据原数组的 index 找回对应的数据库对象，以获取溯源元数据
            original_chunk = recalled_chunks[reranked_item.index]
            
            final_results.append({
                "content": original_chunk.content,
                "score": reranked_item.relevance_score,
                # 极其重要的企业级溯源字段
                "source": original_chunk.source_file,
                "page": original_chunk.page_number,
                "uploader": original_chunk.uploader_id
            })
            
        print("检索链路执行完毕！")
        return final_results