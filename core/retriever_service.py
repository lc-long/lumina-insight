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

    # 🌟 修改点 1：函数签名增加了 target_files 参数，默认值为 None
    async def hybrid_search(self, db: AsyncSession, query: str, user_id: str, top_k: int = 3, target_files: List[str] = None) -> List[dict]:
        print(f"\n[1/4] 正在将用户问题转化为向量: '{query}'")
        query_vector = self.embeddings_model.embed_query(query)

        # 🌟 修改点 2：删除了内部的 potential_files 正则表达式，直接使用传进来的 target_files
        
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
        # 🌟 修改点 3：用 target_files 替代原来的 potential_files
        if target_files:
            print(f"      -> 🎯 嗅探到外部传入的文件名线索 {target_files}，触发强制召回！")
            like_conditions = [KnowledgeChunk.source_file.ilike(f"%{f}%") for f in target_files]
            
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
        # 🌟 修复核心：只要前端传了 target_files，就必须进入独占模式，绝不允许向下“滑坡”
        if target_files:
            print(f"      -> 🛡️ 触发独占模式拦截：用户明确指定了文件 {target_files}")
            
            # 补丁：如果在数据库里根本没找到这个文件，直接返回空！截断后续的向量搜索
            if not keyword_chunks:
                print(f"      -> ❌ 数据库中未找到指定文件，已阻断向量搜索，防止幻觉！")
                return [] 
                
            for chunk in keyword_chunks:
                final_results.append({
                    "content": chunk.meta_info.get("parent_content", chunk.content),
                    "score": 0.9999,  # 赋予极高的假分数，保证排在第一
                    "source": chunk.source_file,
                    "page": chunk.page_number,
                    "uploader": chunk.uploader_id,
                    "images": chunk.meta_info.get("images", []) 
                })
            # 独占模式下，直接截取并返回，彻底跳过向量结果和 Rerank 模型！
            print(f"[4/4] 检索链路执行完毕（独占模式）！最终输出 {len(final_results)} 条。")
            return final_results[:top_k]

        # 【模式 B】：如果没有命中文件名 (即 target_files 为空)，才进入常规的“向量 + Rerank 模式”
        if vector_chunks:
            print(f"[3/4] 向量检索搜到了 {len(vector_chunks)} 个相关子块，正在追溯完整的父文档上下文...")
            
            # 🌟 核心逻辑：父文档重组与去重
            parent_docs_dict = {}
            for chunk in vector_chunks:
                # 尝试获取父级 ID 和父级内容
                parent_id = chunk.meta_info.get("parent_id")
                # 构造统一格式的数据字典（安全脱离 SQLAlchemy 模型）
                doc_info = {
                    "id": parent_id or chunk.id, # 如果没有 parent_id (老数据)，就用自己的 id
                    # 如果有父内容就用父内容，没有就用自己原本的子内容兜底
                    "content": chunk.meta_info.get("parent_content", chunk.content),
                    "source": chunk.source_file,
                    "page": chunk.page_number,
                    "uploader": chunk.uploader_id,
                    "images": chunk.meta_info.get("images", [])
                }
                
                # 利用字典的 Key 唯一性，天然过滤掉属于同一个父块的多个子块
                if doc_info["id"] not in parent_docs_dict:
                    parent_docs_dict[doc_info["id"]] = doc_info
            
            # 去重后，真正要送去精排的、包含大段落上下文的候选文档
            reconstructed_parents = list(parent_docs_dict.values())
            print(f"      -> 🧩 去重组装后，提取出 {len(reconstructed_parents)} 个不重复的父级完整段落，准备精排...")

            documents_for_rerank = [item["content"] for item in reconstructed_parents]
            
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
                        # 从重组后的父文档列表中获取对应项
                        original_item = reconstructed_parents[reranked_item.index]
                        final_results.append({
                            "content": original_item["content"], # 这里已经是大段落的父内容了
                            "score": reranked_item.relevance_score,
                            "source": original_item["source"],
                            "page": original_item["page"],
                            "uploader": original_item["uploader"],
                            "images": original_item["images"]
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