import os
import pymupdf4llm
from markitdown import MarkItDown
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter

class DocumentManager:
    def __init__(self):
        # 统一使用 Markdown 切分器
        self.text_splitter = MarkdownTextSplitter(
            chunk_size=600,
            chunk_overlap=50
        )
        # 初始化微软 Office 文档解析引擎
        self.office_parser = MarkItDown()

    def _extract_to_markdown(self, file_path: str) -> str:
        """
        核心路由逻辑：嗅探文件格式，并调用对应的底层引擎提取 Markdown
        """
        # 获取文件后缀名
        ext = file_path.lower().split('.')[-1]
        
        try:
            if ext == 'pdf':
                print(f"📄 [嗅探器] 检测到 PDF，正在路由给 PyMuPDF4LLM 引擎...")
                return pymupdf4llm.to_markdown(file_path)
            
            elif ext in ['docx', 'xlsx', 'pptx', 'csv']:
                print(f"📊 [嗅探器] 检测到 Office 文档 (.{ext})，正在路由给 MarkItDown 引擎...")
                # MarkItDown 能极其完美地把 Excel 的行列转化为 Markdown 表格
                result = self.office_parser.convert(file_path)
                return result.text_content
            
            elif ext in ['txt', 'md']:
                print(f"📝 [嗅探器] 检测到纯文本/Markdown，直接读取...")
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                raise ValueError(f"系统暂不支持解析 .{ext} 格式的文件。")
                
        except Exception as e:
            print(f"❌ 解析文件 {file_path} 时发生严重错误: {e}")
            return ""

    # 注意：方法名从 process_pdf_with_permissions 改为了 process_document_with_permissions
    def process_document_with_permissions(self, file_path: str, uploader_id: str, access_level: str) -> List[Document]:
        """
        通用的全格式文档处理与打标入口
        """
        # 1. 动态获取全局 Markdown 文本
        md_text = self._extract_to_markdown(file_path)
        
        if not md_text.strip():
            return []

        # 2. 包装为 LangChain Document 对象
        raw_doc = Document(
            page_content=md_text,
            metadata={"source": file_path, "page": 1} # Office 暂时统称 1 页
        )
        
        # 3. 使用 Markdown 切分器智能切块
        chunks = self.text_splitter.split_documents([raw_doc])
        
        # 4. 注入企业级元数据 (Metadata)
        processed_chunks = []
        for chunk in chunks:
            # 获取实际的文件名而不是完整路径
            file_name = os.path.basename(file_path)
            
            chunk.metadata.update({
                "uploader_id": uploader_id,
                "access_level": access_level,
                "doc_type": file_path.split('.')[-1], # 记录原始文档类型
                "source": file_name # 覆盖掉绝对路径，防止暴露服务器目录结构
            })
            processed_chunks.append(chunk)
            
        print(f"✅ 解析完成！共提取并切分为 {len(processed_chunks)} 个带标签的知识块。")
        return processed_chunks