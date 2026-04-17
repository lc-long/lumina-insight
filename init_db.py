import asyncio
from core.database import engine
from core.models import Base

async def init_db():
    print("🚀 正在连接数据库并初始化表结构...")
    async with engine.begin() as conn:
        # 这个命令会扫描 models.py 里的所有类
        # 如果数据库里没有对应的表，它就会自动创建
        await conn.run_sync(Base.metadata.create_all)
    print("✅ 数据库表初始化完成！现在你可以使用 ChatMessage 表了。")

if __name__ == "__main__":
    asyncio.run(init_db())