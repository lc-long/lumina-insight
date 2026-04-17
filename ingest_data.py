import asyncio
import os
import sys

# 导入我们之前写好的核心组件
from core.database import AsyncSessionLocal
from core.document_manager import DocumentManager
from core.vector_service import VectorService

async def ingest_pipeline(file_path: str, uploader_id: str, access_level: str):
    """
    完整的离线数据入库流水线
    """
    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到文件 {file_path}")
        print("请确保你已经在 data/ 目录下放置了对应的 PDF 文件。")
        sys.exit(1)

    print("==================================================")
    print(f"🚀 开始企业级文档入库流程: {file_path}")
    print(f"👤 上传者: {uploader_id} | 🔒 权限级别: {access_level}")
    print("==================================================\n")

    # 1. 实例化核心服务
    doc_manager = DocumentManager()
    vector_service = VectorService()

    # 2. 提取与切分 (Extract & Split)
    # 调用我们写的解析器，将文档切成小块并打上权限 Metadata
    print("[1/3] 正在解析文档并切分文档块...")
    chunks = doc_manager.process_document_with_permissions(
        file_path=file_path,
        uploader_id=uploader_id,
        access_level=access_level
    )
    
    if not chunks:
        print("❌ 警告：未从文档中提取到任何有效文本。")
        return

    # 3. 向量化与持久化入库 (Embed & Load)
    print(f"\n[2/3] 准备将 {len(chunks)} 个数据块向量化并写入 PostgreSQL...")
    
    # 获取数据库连接并执行入库
    async with AsyncSessionLocal() as db:
        try:
            await vector_service.upsert_documents(db=db, documents=chunks)
            print("\n✅ [3/3] 数据入库大功告成！")
        except Exception as e:
            print(f"\n❌ 入库失败，数据库报错: {str(e)}")

if __name__ == "__main__":
    # 模拟 User_A 上传了一份公开的公司制度文档
    # 你可以修改这里的路径，指向你实际放入 data/ 目录下的 pdf
    target_file = "data/records.xlsx"  # 也可以测试 docx、pptx、csv 等 Office 文档
    
    # 启动异步流水线
    asyncio.run(ingest_pipeline(
        file_path=target_file,
        uploader_id="User_A",
        access_level="public"
    ))