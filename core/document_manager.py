import re
import os
import dashscope # 新增：引入 dashscope 用于调用视觉模型
from dashscope import MultiModalConversation # 新增：多模态对话接口
import pymupdf4llm
from markitdown import MarkItDown
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter

# 确保读取到 .env 中的百炼 API Key
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

class DocumentManager:
    def __init__(self):
        # 统一使用 Markdown 切分器
        self.text_splitter = MarkdownTextSplitter(
            chunk_size=600,
            chunk_overlap=50
        )
        # 初始化微软 Office 文档解析引擎
        self.office_parser = MarkItDown()

    def _ocr_with_qwen_vl(self, file_path: str) -> str:
        """
        调用 Qwen-VL 视觉大模型进行智能 OCR 与排版还原
        """
        # 视觉模型需要绝对路径，并且要求是以 file:// 开头的本地 URI 格式
        abs_path = os.path.abspath(file_path)
        # 兼容 Windows 路径中的反斜杠
        abs_path = abs_path.replace('\\', '/')
        local_file_uri = f"file://{abs_path}"
        
        # 精心设计的企业级 Prompt，让视觉大模型不仅提取文字，还要还原表格
        messages = [{
            'role': 'user',
            'content': [
                {'image': local_file_uri},
                {'text': '你是一个专业的文档解析助手。请仔细阅读图中的所有文字、表格和排版结构，并将其精准地转化为 Markdown 格式输出。如果有表格，请严格使用 Markdown 表格语法。不要输出任何解释性的废话，只输出纯 Markdown 内容。'}
            ]
        }]
        
        file_name = os.path.basename(file_path)
        print(f"👁️  [视觉大模型] 正在呼叫 Qwen-VL 分析图像像素: {file_name}...")
        
        # 调用多模态大模型
        response = MultiModalConversation.call(
            model='qwen-vl-plus',
            messages=messages
        )
        
        if response.status_code == 200:
            # 提取模型返回的 Markdown 文本
            try:
                # 兼容 DashScope 返回的多模态 JSON 结构
                return response.output.choices[0].message.content[0]['text']
            except Exception:
                return str(response.output.choices[0].message.content)
        else:
            raise Exception(f"OCR 视觉模型调用失败: {response.code} - {response.message}")

    def _extract_to_markdown(self, file_path: str) -> str:
        """
        核心路由逻辑：嗅探文件格式，并调用对应的底层引擎提取 Markdown
        """
        # 获取文件后缀名
        ext = file_path.lower().split('.')[-1]
        
        try:
            if ext == 'pdf':
                print(f"📄 [嗅探器] 检测到 PDF，正在路由给 PyMuPDF4LLM 引擎...")
                # 新增：创建专门的目录存放提取出的图片
                os.makedirs("data/images", exist_ok=True)
                # 新增：开启 write_images，并将图片统一保存到指定目录
                return pymupdf4llm.to_markdown(file_path, write_images=True, image_path="data/images")
            
            elif ext in ['docx', 'xlsx', 'pptx', 'csv']:
                print(f"📊 [嗅探器] 检测到 Office 文档 (.{ext})，正在路由给 MarkItDown 引擎...")
                result = self.office_parser.convert(file_path)
                return result.text_content
                
            elif ext in ['png', 'jpg', 'jpeg']:
                print(f"🖼️  [嗅探器] 检测到图片扫描件 (.{ext})，触发多模态视觉链路...")
                return self._ocr_with_qwen_vl(file_path) # 核心新增路由
                
            elif ext in ['txt', 'md']:
                print(f"📝 [嗅探器] 检测到纯文本/Markdown，直接读取...")
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                raise ValueError(f"系统暂不支持解析 .{ext} 格式的文件。")
                
        except Exception as e:
            print(f"❌ 解析文件 {file_path} 时发生严重错误: {e}")
            return ""

    def process_document_with_permissions(self, file_path: str, uploader_id: str, access_level: str) -> List[Document]:
        md_text = self._extract_to_markdown(file_path)
        
        if not md_text or not md_text.strip():
            return []

        # 🌟 核心修复：在【切分之前】进行全局扫描，提取整篇文档的所有图片
        all_doc_images = re.findall(r'!\[.*?\]\((.*?)\)', md_text)
        
        ext = file_path.lower().split('.')[-1]
        if ext in ['png', 'jpg', 'jpeg']:
            all_doc_images.append(file_path)

        raw_doc = Document(
            page_content=md_text,
            metadata={"source": file_path, "page": 1} 
        )
        
        chunks = self.text_splitter.split_documents([raw_doc])
        
        processed_chunks = []
        for chunk in chunks:
            file_name = os.path.basename(file_path)
            
            chunk.metadata.update({
                "uploader_id": uploader_id,
                "access_level": access_level,
                "doc_type": ext,
                "source": file_name,
                # 🌟 核心修复：把全局图片赋予每一个知识块。
                # 这样不管大模型召回了哪一段文字，都能顺带把这篇论文的图拿过去看
                "images": all_doc_images  
            })
            processed_chunks.append(chunk)
            
        print(f"✅ 解析完成！共切分 {len(processed_chunks)} 个块，发现全局图片 {len(all_doc_images)} 张。")
        return processed_chunks