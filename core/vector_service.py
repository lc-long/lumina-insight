from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_community.embeddings import DashScopeEmbeddings
from core.models import KnowledgeChunk
from langchain_core.documents import Document

class VectorService:
    def __init__(self):
        # 初始化阿里云 Embedding 模型
        self.embeddings_model = DashScopeEmbeddings(model="text-embedding-v1")

    async def upsert_documents(self, db: AsyncSession, documents: List[Document]):
        """
        将经过加工的 Document 对象存入数据库
        """
        # 1. 批量提取文本内容，准备生成向量
        texts = [doc.page_content for doc in documents]
        
        print(f"正在为 {len(texts)} 个知识块生成向量索引...")
        # 2. 调用阿里云接口获取向量 (这是耗时操作)
        embeddings = self.embeddings_model.embed_documents(texts)
        
        # 3. 构造数据库模型对象
        chunks_to_save = []
        for doc, vector in zip(documents, embeddings):
            chunk = KnowledgeChunk(
                content=doc.page_content,
                embedding=vector,
                uploader_id=doc.metadata.get("uploader_id"),
                access_level=doc.metadata.get("access_level"),
                source_file=doc.metadata.get("source"),
                page_number=doc.metadata.get("page"),
                meta_info=doc.metadata
            )
            chunks_to_save.append(chunk)
        
        # 4. 批量存入数据库
        db.add_all(chunks_to_save)
        await db.commit()
        print("入库成功！所有知识块已持久化并支持向量搜索。")