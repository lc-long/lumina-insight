import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

# ==========================================
# 1. 配置数据库连接 URL
# ==========================================
# 实际开发中，这个 URL 必须写在 .env 文件里！
# 格式: postgresql+asyncpg://用户名:密码@主机地址:端口/数据库名
# 这里我们先写死作为演示，稍后你会把它移到 .env 中
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+asyncpg://postgres:yourpassword@localhost:5432/lumina_db"
)

# ==========================================
# 2. 创建异步引擎 (Engine) —— 这就是我们的“餐厅”
# ==========================================
engine = create_async_engine(
    DATABASE_URL,
    echo=False,                # 生产环境设为 False，否则会打印出所有 SQL 语句，导致日志爆炸
    pool_size=10,              # 【核心配置】常驻的连接数 (长包了 10 张桌子)
    max_overflow=20,           # 【核心配置】当 10 张桌子满了，最多允许再临时加 20 张桌子。超过 30 个请求就在外排队
    pool_timeout=30,           # 【核心配置】排队最多等 30 秒，等不到就报错 (TimeoutError)，防止无限死锁
    pool_recycle=1800,         # 【核心配置】每 1800 秒 (半小时) 强制回收一次连接，防止数据库端因为长时间不活跃而悄悄掐断连接
)

# ==========================================
# 3. 创建会话工厂 (Session Local) —— 这相当于“领位员”
# ==========================================
# 每次调用它，它就会从 engine 的池子里拿出一个连接交给你
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autocommit=False,          # 必须手动 commit 才能生效，保证事务安全
    autoflush=False,
    expire_on_commit=False,    # 提交后不让对象过期，方便在异步流中继续读取对象属性
)

# ==========================================
# 4. 依赖注入函数 (用于 FastAPI 路由)
# ==========================================
async def get_db():
    """
    提供给 FastAPI 的依赖项。
    功能：从连接池借一个连接 -> 让接口处理业务 -> 无论成功还是报错，最后自动归还连接
    """
    session = AsyncSessionLocal()
    try:
        # yield 会把 session 交给 FastAPI 的路由去用
        yield session
    finally:
        # 业务处理完，必须 close。
        # 注意：这里的 close 不是关闭 TCP 连接，而是把“桌子”还给连接池
        await session.close()