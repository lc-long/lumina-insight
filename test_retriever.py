import asyncio
from core.database import AsyncSessionLocal
from core.retriever_service import EnterpriseRetriever

async def test_search():
    # 模拟用户查询
    query = "开题报告中，英文的论文题目是什么？"
    user_id = "User_A" # 模拟当前登录的用户
    
    retriever = EnterpriseRetriever()
    
    # 从我们之前写的连接池借一个连接
    async with AsyncSessionLocal() as db:
        results = await retriever.hybrid_search(
            db=db, 
            query=query, 
            user_id=user_id, 
            top_k=2 # 只取最核心的 2 条
        )
        
        print("\n================ 最终精排结果 ================")
        for i, res in enumerate(results):
            print(f"[{i+1}] 得分: {res['score']:.4f}")
            print(f"    来源: {res['source']} (第 {res['page']} 页)")
            print(f"    权限拥有者: {res['uploader']}")
            print(f"    内容: {res['content']}\n")

if __name__ == "__main__":
    asyncio.run(test_search())