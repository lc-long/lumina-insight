import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from core.database import engine
from core.models import Base
from api.chat import router as chat_router # 导入我们刚才写的路由

# 初始化 FastAPI
app = FastAPI(title="Lumina Insight API", description="企业级 RAG 知识库系统")

# 配置跨域，允许前端项目访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册子路由
app.include_router(chat_router)

async def init_db():
    async with engine.begin() as conn:
        # 这一行会自动识别 models.py 里的类，并在数据库中创建对应的表
        # 注意：这需要 pgvector 插件已在数据库中激活 (我们之前做过了)
        await conn.run_sync(Base.metadata.create_all)
    print("数据库表结构初始化完成。")

# if __name__ == "__main__":
#     asyncio.run(init_db())
if __name__ == "__main__":
    # 在生产环境中通常不在这里 run，但为了本地开发方便，我们保留
    # 先异步建表，再启动 uvicorn
    # 注意：在 asyncio.run 中启动 uvicorn 会有事件循环冲突，
    # 推荐的现代写法是直接在外部用命令行启动，这里只做兼容。
    pass