from sqlalchemy import Column, Integer, String, JSON, Text, ForeignKey
from sqlalchemy.orm import DeclarativeBase
from pgvector.sqlalchemy import Vector # 核心：导入 pgvector 支持

class Base(DeclarativeBase):
    pass

class KnowledgeChunk(Base):
    """
    知识块表：存储切分后的文本、对应的向量以及权限元数据
    """
    __tablename__ = "knowledge_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False)           # 原始文本内容
    
    # 向量字段：维度通常为 1536 (对应阿里云 text-embedding-v1 或 OpenAI 标配)
    # pgvector 允许我们像定义普通列一样定义向量列
    embedding = Column(Vector(1536))                 
    
    # 业务元数据与权限
    uploader_id = Column(String(50), index=True)      # 上传者ID，加索引方便快速过滤
    access_level = Column(String(20), default="private") # 权限等级：private/public
    
    # 溯源信息
    source_file = Column(String(255))                 # 文件名
    page_number = Column(Integer)                     # 页码
    
    # 预留字段：存储其他原始元数据（如原始 PDF 的作者、创建日期等）
    meta_info = Column(JSON, default={})

    def __repr__(self):
        return f"<KnowledgeChunk(source={self.source_file}, page={self.page_number}, access={self.access_level})>"