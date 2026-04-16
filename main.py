import asyncio
from core.database import engine
from core.models import Base

async def init_db():
    async with engine.begin() as conn:
        # 这一行会自动识别 models.py 里的类，并在数据库中创建对应的表
        # 注意：这需要 pgvector 插件已在数据库中激活 (我们之前做过了)
        await conn.run_sync(Base.metadata.create_all)
    print("数据库表结构初始化完成。")

if __name__ == "__main__":
    asyncio.run(init_db())