import os
import pymupdf4llm
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter # 改用专门切分 Markdown 的利器

class DocumentManager:
    def __init__(self):
        # 使用 Markdown 切分器，它会尽量保证不把一个完整的 Markdown 表格切成两半
        self.text_splitter = MarkdownTextSplitter(
            chunk_size=600,
            chunk_overlap=50
        )

    def process_pdf_with_permissions(self, file_path: str, uploader_id: str, access_level: str) -> List[Document]:
        """
        使用 PyMuPDF4LLM 进行高级版面解析，提取 Markdown 并打标
        """
        print(f"正在使用 PyMuPDF4LLM 进行多模态版面解析 (提取 Markdown 表格): {file_path}")
        
        # 1. 核心大招：直接把整个 PDF 连同表格转成排版精美的 Markdown 字符串
        try:
            md_text = pymupdf4llm.to_markdown(file_path)
        except Exception as e:
            print(f"PDF 转换 Markdown 失败: {e}")
            return []

        # 2. 将全局 Markdown 文本包装成 LangChain 认识的 Document 对象
        # 我们目前先整体作为一个 Document，然后再交给切分器去切
        raw_doc = Document(
            page_content=md_text,
            metadata={"source": file_path, "page": 1} # PyMuPDF4LLM 默认合并为一个长文本，我们统称第一页，企业里可以按页单独提取
        )
        
        # 3. 智能 Markdown 切分
        chunks = self.text_splitter.split_documents([raw_doc])
        
        # 4. 注入企业级元数据 (Metadata)
        processed_chunks = []
        for chunk in chunks:
            chunk.metadata.update({
                "uploader_id": uploader_id,
                "access_level": access_level,
                "doc_type": "pdf"
            })
            processed_chunks.append(chunk)
            
        print(f"解析完成，共提取并切分为 {len(processed_chunks)} 个包含 Markdown 表格的知识块。")
        return processed_chunks

# 测试代码
if __name__ == "__main__":
    test_pdf_path = "../data/test.pdf" # 换成你刚才报错的那个带表格的 pdf 名字
    if os.path.exists(test_pdf_path):
        manager = DocumentManager()
        chunks = manager.process_pdf_with_permissions(test_pdf_path, "User_A", "public")
        
        # 打印出第一个 Chunk，看看是不是完美的 Markdown 表格！
        if chunks:
            print("\n====== 提取出的核心 Markdown 内容预览 ======\n")
            print(chunks[5].page_content[:500]) # 打印前 500 个字符看看有没有 |---|
    else:
        print(f"找不到测试文件: {test_pdf_path}")