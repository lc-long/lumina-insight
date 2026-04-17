import asyncio
from sqlalchemy import delete
from core.database import AsyncSessionLocal
from core.models import KnowledgeChunk

async def clear_old_pdf(file_name: str):
    print(f"正在连接数据库，准备清理 {file_name} 的旧数据...")
    
    async with AsyncSessionLocal() as db:
        try:
            # 构造删除语句：DELETE FROM knowledge_chunks WHERE source_file = 'test2.pdf'
            stmt = delete(KnowledgeChunk).where(KnowledgeChunk.source_file == file_name)
            
            # 执行删除并提交事务
            result = await db.execute(stmt)
            await db.commit()
            
            # result.rowcount 可以获取删除了多少条数据
            print(f"✅ 清理完成！共删除了 {result.rowcount} 条相关的旧知识块。")
            
        except Exception as e:
            print(f"❌ 清理失败: {e}")
            await db.rollback()

if __name__ == "__main__":
    # 指定你要删除的旧文件名
    target_file = "test2.pdf" 
    asyncio.run(clear_old_pdf(target_file))