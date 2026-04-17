import os
import re
import json
import base64 # 新增 base64 库
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db
from core.retriever_service import EnterpriseRetriever
from sqlalchemy import select # 增加导入
from core.models import ChatMessage

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
    user_id: str = "User_A",
    session_id: str = "default_session" # 🌟 新增：由前端生成的唯一会话ID

# 专门用于重写问题的重写 Prompt
REWRITE_TEMPLATE = """根据以下对话历史和用户提出的最新问题，
将其重新改写为一个完整的、不依赖上下文的独立问题。
如果最新问题已经很完整，则保持原样输出。严禁回答问题，只需重写问题。

对话历史:
{history}

最新问题: {question}
独立问题:"""

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
        # ---------------------------------------------------------
        # 1. 记忆模块：获取最近 5 轮的历史记录 (10条消息)
        # ---------------------------------------------------------
        hist_stmt = (
            select(ChatMessage)
            .filter(ChatMessage.session_id == request.session_id)
            .order_by(ChatMessage.timestamp.desc())
            .limit(10)
        )
        hist_res = await db.execute(hist_stmt)
        history_objs = list(reversed(hist_res.scalars().all()))
        history_str = "\n".join([f"{m.role}: {m.content}" for m in history_objs])

        # ---------------------------------------------------------
        # 2. 意图模块：问题重写 (Query Rewriting)
        # ---------------------------------------------------------
        if history_str:
            rewrite_prompt = REWRITE_TEMPLATE.format(history=history_str, question=request.question)
            rewrite_res = await llm.ainvoke(rewrite_prompt)
            standalone_question = rewrite_res.content.strip()
            print(f"🔄 [问题重写] '{request.question}' -> '{standalone_question}'")
        else:
            standalone_question = request.question

        # 🌟 核心修复：只在“原始问题”中探测文件名，防止历史记录里的文件名干扰检索
        # 这样如果你这句没提 kt.png，我们就不会进入独占模式
        current_files = re.findall(r'[a-zA-Z0-9_-]+\.[a-zA-Z0-9]+', request.question)

        # ---------------------------------------------------------
        # 3. 检索模块：使用重写后的问题去搜数据库
        # ---------------------------------------------------------
        results = await retriever.hybrid_search(
            db=db, 
            query=standalone_question, 
            user_id=request.user_id, 
            top_k=3,
            target_files=current_files # 🌟 新增参数：显式传递当前指定的文件
        )
        
        if not results:
            yield f"data: {json.dumps({'type': 'content', 'data': '根据当前权限的内部资料，未检索到相关内容。'})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # ---------------------------------------------------------
        # 4. 组装模块：提取溯源信息、文本上下文和图片路径
        # ---------------------------------------------------------
        sources = []
        context_texts = []
        all_image_paths = []

        for res in results:
            context_texts.append(res['content'])
            
            # 收集有效图片路径
            for img_path in res.get('images', []):
                if img_path not in all_image_paths and os.path.exists(img_path):
                    all_image_paths.append(img_path)

            sources.append({
                "source": res['source'],
                "page": res['page'],
                "score": round(res['score'], 4)
            })
            
        yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
        
        # ---------------------------------------------------------
        # 5. 多模态生成模块：图文并茂地丢给 Qwen-VL
        # ---------------------------------------------------------
        context_str = "\n\n---\n\n".join(context_texts)
        # 使用多模态专属的 template_str，结合重写后的独立问题
        final_prompt_text = template_str.replace("{context}", context_str).replace("{question}", standalone_question)
        
        # 构造 LangChain 视觉大模型消息体
        message_content = [{"type": "text", "text": final_prompt_text}]
        
        for img_path in all_image_paths:
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode('utf-8')
                mime_type = "image/png" if img_path.lower().endswith('.png') else "image/jpeg"
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{img_b64}"}
                })
        
        messages = [HumanMessage(content=message_content)]
        
        # 流式输出并记录完整回答
        full_answer = ""
        async for chunk in llm.astream(messages):
            if chunk.content:
                full_answer += chunk.content
                yield f"data: {json.dumps({'type': 'content', 'data': chunk.content})}\n\n"
                
        # ---------------------------------------------------------
        # 6. 持久化模块：把这一次的问答存入数据库记忆中
        # ---------------------------------------------------------
        new_user_msg = ChatMessage(
            session_id=request.session_id, 
            user_id=request.user_id, 
            role="user", 
            content=request.question # 存用户的原始问题
        )
        new_ai_msg = ChatMessage(
            session_id=request.session_id, 
            user_id=request.user_id, 
            role="assistant", 
            content=full_answer
        )
        db.add_all([new_user_msg, new_ai_msg])
        await db.commit()

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