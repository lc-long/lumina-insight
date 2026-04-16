import os
import json
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

# 导入我们之前写好的模块
from core.database import get_db
from core.retriever_service import EnterpriseRetriever

# 导入 LangChain 依赖
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

router = APIRouter()
retriever = EnterpriseRetriever()

# 1. 初始化流式大模型
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0.1,  # RAG 场景要求严谨，温度调低
    streaming=True    # 必须开启流式开关
)

# 2. 定义前端请求的数据格式
class ChatRequest(BaseModel):
    question: str
    user_id: str = "User_A"  # 模拟前端传过来的当前登录用户

# 3. 构造企业级严谨 Prompt
template = """你是一个专业的企业级知识库助手。请严格基于以下【已知信息】来回答用户的问题。
在回答时，请逻辑清晰、分点阐述。
【极其重要】：如果【已知信息】中找不到答案，你必须直接回复“根据当前权限的内部资料，我无法回答该问题”，严禁胡编乱造！

【已知信息】:
{context}

用户问题: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 4. 核心：异步流式生成器
async def rag_stream_generator(request: ChatRequest, db: AsyncSession):
    try:
        # A. 触发混合检索与精排
        results = await retriever.hybrid_search(
            db=db, 
            query=request.question, 
            user_id=request.user_id, 
            top_k=3
        )
        
        # 如果什么都没搜到，直接返回无权限/无数据提示，不用消耗大模型 Token
        if not results:
            yield f"data: {json.dumps({'type': 'content', 'data': '根据当前权限的内部资料，未检索到相关内容。'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # B. 提取溯源信息和上下文文本
        sources = []
        context_texts = []
        for res in results:
            context_texts.append(res['content'])
            sources.append({
                "source": res['source'],
                "page": res['page'],
                "score": round(res['score'], 4)
            })
            
        # C. 【魔法时刻】首先把溯源信息作为第一帧推给前端
        # 前端收到这个 type: sources 后，就可以立马在页面侧边栏渲染出引用文档卡片
        yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
        
        # D. 拼接上下文，交给大模型生成回答
        context_str = "\n\n---\n\n".join(context_texts)
        chain = prompt | llm
        
        # E. 监听大模型的流式吐字，包装成 JSON 发给前端
        async for chunk in chain.astream({"context": context_str, "question": request.question}):
            if chunk.content: # 过滤空字符
                yield f"data: {json.dumps({'type': 'content', 'data': chunk.content})}\n\n"
                
        # F. 明确告诉前端传输结束
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
    except Exception as e:
        # 捕获异常并通过流推送给前端，避免前端一直转圈等待
        yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

# 5. 暴露接口
@router.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    企业级 RAG 流式问答接口
    """
    # 将生成器包装进 StreamingResponse，指定媒体类型为 SSE
    return StreamingResponse(
        rag_stream_generator(request, db), 
        media_type="text/event-stream"
    )