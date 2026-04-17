import os
import re
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
        print(f"\n[1/4] 正在将用户问题转化为向量: '{query}'")
        query_vector = self.embeddings_model.embed_query(query)

        # ---------------------------------------------------------
        # 1. 提取潜在的文件名 (纯净正则：只匹配 英文/数字/横杠/下划线 + 后缀)
        # ---------------------------------------------------------
        potential_files = re.findall(r'[a-zA-Z0-9_-]+\.[a-zA-Z0-9]+', query)
        
        # 基础权限过滤条件
        permission_filter = or_(
            KnowledgeChunk.access_level == "public",
            KnowledgeChunk.uploader_id == user_id
        )

        # ---------------------------------------------------------
        # 2. 第一路：纯向量召回 (兜底语义，粗排 15 条)
        # ---------------------------------------------------------
        stmt_vector = (
            select(KnowledgeChunk)
            .filter(permission_filter)
            .order_by(KnowledgeChunk.embedding.l2_distance(query_vector))
            .limit(15)
        )
        result_vector = await db.execute(stmt_vector)
        vector_chunks = result_vector.scalars().all()

        # ---------------------------------------------------------
        # 3. 第二路：文件名精准召回
        # ---------------------------------------------------------
        keyword_chunks = []
        if potential_files:
            print(f"      -> 🎯 嗅探到纯净文件名线索 {potential_files}，触发强制召回！")
            like_conditions = [KnowledgeChunk.source_file.ilike(f"%{f}%") for f in potential_files]
            
            stmt_keyword = (
                select(KnowledgeChunk)
                .filter(permission_filter)
                .filter(or_(*like_conditions))
                .limit(top_k) # 既然是独占模式，直接限制最多取 top_k 条即可
            )
            result_keyword = await db.execute(stmt_keyword)
            keyword_chunks = result_keyword.scalars().all()

        # ---------------------------------------------------------
        # 4. 🌟 核心排他逻辑：独占模式 vs 常规模式
        # ---------------------------------------------------------
        final_results = []
        
        # 【模式 A】：如果命中了文件名，进入“独占模式”
        if keyword_chunks:
            print(f"      -> 🛡️ 触发独占模式：只返回目标文件内容，屏蔽其他语义干扰！")
            for chunk in keyword_chunks:
                final_results.append({
                    "content": chunk.content,
                    "score": 0.9999,  # 赋予极高的假分数，保证排在第一
                    "source": chunk.source_file,
                    "page": chunk.page_number,
                    "uploader": chunk.uploader_id,
                    "images": chunk.meta_info.get("images", []) 
                })
            # 独占模式下，直接截取并返回，彻底跳过向量结果和 Rerank 模型！
            print(f"[4/4] 检索链路执行完毕（独占模式）！最终输出 {len(final_results)} 条。")
            return final_results[:top_k]

        # 【模式 B】：如果没有命中文件名，进入常规的“向量 + Rerank 模式”
        if vector_chunks:
            print(f"[3/4] 正在请求 Rerank 大模型对 {len(vector_chunks)} 条进行精排...")
            documents_for_rerank = [chunk.content for chunk in vector_chunks]
            
            try:
                resp = dashscope.TextReRank.call(
                    model="gte-rerank",
                    query=query,
                    documents=documents_for_rerank,
                    top_n=top_k, 
                    return_documents=True 
                )
                
                if resp.status_code == 200:
                    for reranked_item in resp.output.results:
                        original_chunk = vector_chunks[reranked_item.index]
                        final_results.append({
                            "content": original_chunk.content,
                            "score": reranked_item.relevance_score,
                            "source": original_chunk.source_file,
                            "page": original_chunk.page_number,
                            "uploader": original_chunk.uploader_id,
                            "images": original_chunk.meta_info.get("images", []) 
                        })
                else:
                    print(f"❌ Rerank 接口报错: {resp.message}")
            except Exception as e:
                print(f"❌ 请求 Rerank 时发生异常: {str(e)}")
        else:
            print("未召回到任何有权限的知识块。")

        # 截取前 top_k 个结果
        final_results = final_results[:top_k]
        print(f"[4/4] 检索链路执行完毕！最终输出 {len(final_results)} 条。")
        
        return final_results