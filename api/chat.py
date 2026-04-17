import os
import json
import base64 # 新增 base64 库
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db
from core.retriever_service import EnterpriseRetriever

# 导入 LangChain 依赖
from langchain_openai import ChatOpenAI
# 新增引入 HumanMessage 用于构建多模态消息体
from langchain_core.messages import HumanMessage 

router = APIRouter()
retriever = EnterpriseRetriever()

# 1. 升级为多模态视觉大模型
llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-vl-max",  # 从 qwen-plus 修改为视觉加强版
    temperature=0.1,
    streaming=True
)

class ChatRequest(BaseModel):
    question: str
    user_id: str = "User_A"

# 修改 Prompt 为纯字符串，因为后续要手动组装多模态 Message
template_str = """你是一个专业的企业级知识库助手。请严格基于以下【已知信息】及提供的【图片】来回答用户的问题。
在回答时，请逻辑清晰、分点阐述。
如果【已知信息】和【图片】中找不到答案，你必须直接回复“根据当前权限的内部资料，我无法回答该问题”。

【已知信息】:
{context}

用户问题: {question}
"""

async def rag_stream_generator(request: ChatRequest, db: AsyncSession):
    try:
        results = await retriever.hybrid_search(
            db=db, 
            query=request.question, 
            user_id=request.user_id, 
            top_k=3
        )
        
        if not results:
            yield f"data: {json.dumps({'type': 'content', 'data': '根据当前权限的内部资料，未检索到相关内容。'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        sources = []
        context_texts = []
        all_image_paths = [] # 新增：用于收集本次检索召回的所有相关图片

        for res in results:
            context_texts.append(res['content'])
            
            # 收集去重后的真实有效图片路径
            for img_path in res.get('images', []):
                if img_path not in all_image_paths and os.path.exists(img_path):
                    all_image_paths.append(img_path)

            sources.append({
                "source": res['source'],
                "page": res['page'],
                "score": round(res['score'], 4)
            })
            
        yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
        
        # 拼接文本上下文
        context_str = "\n\n---\n\n".join(context_texts)
        final_prompt_text = template_str.replace("{context}", context_str).replace("{question}", request.question)
        
        # 构造 LangChain 要求的 OpenAI 兼容多模态消息体格式
        message_content = [{"type": "text", "text": final_prompt_text}]
        
        # 遍历所有被召回的本地图片
        print(f"🖼️ [Debug] 准备喂给大模型的真实图片路径: {all_image_paths}") # <--- 加上这一行
        # 遍历所有被召回的本地图片，转为 Base64 拼接到 Prompt 中
        for img_path in all_image_paths:
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode('utf-8')
                # 简单判断 MIME 类型
                mime_type = "image/png" if img_path.lower().endswith('.png') else "image/jpeg"
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{img_b64}"}
                })
        
        messages = [HumanMessage(content=message_content)]
        
        # 直接调用 llm.astream 处理包含图文的多模态对象
        async for chunk in llm.astream(messages):
            if chunk.content:
                yield f"data: {json.dumps({'type': 'content', 'data': chunk.content})}\n\n"
                
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"

# 5. 暴露接口
@router.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """
    多模态 RAG 流式问答接口
    """
    # 将生成器包装进 StreamingResponse，指定媒体类型为 SSE
    return StreamingResponse(
        rag_stream_generator(request, db), 
        media_type="text/event-stream"
    )