import os
from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentManager:
    def __init__(self):
        # 初始化高级文本切分器
        # 企业级文档通常段落分明，按照段落、句子来切分比纯按字符数切分更不容易切断上下文
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )

    def process_pdf_with_permissions(self, file_path: str, uploader_id: str, access_level: str) -> List[Document]:
        """
        解析 PDF 并为切分后的每一个文本块打上权限标签
        :param file_path: 文件物理路径
        :param uploader_id: 上传者的用户ID
        :param access_level: 权限级别 ("private" 或 "public")
        """
        print(f"正在使用 PDFPlumber 深度解析: {file_path}")
        # PDFPlumber 能够更好地识别 PDF 里的文本流和底层表格结构
        loader = PDFPlumberLoader(file_path)
        raw_docs = loader.load()
        
        # 1. 文本切分
        chunks = self.text_splitter.split_documents(raw_docs)
        
        # 2. 【核心】注入企业级元数据 (Metadata)
        processed_chunks = []
        for chunk in chunks:
            # chunk.metadata 默认带有 source (文件路径) 和 page (页码)
            # 我们强行注入业务权限字段
            chunk.metadata.update({
                "uploader_id": uploader_id,
                "access_level": access_level,
                "doc_type": "pdf"
            })
            processed_chunks.append(chunk)
            
        print(f"解析完成，共切分为 {len(processed_chunks)} 个知识块，已打标。")
        return processed_chunks

# 测试代码 (仅在直接运行该文件时执行)
if __name__ == "__main__":
    # 模拟测试：请确保在项目根目录的 data/ 文件夹下随便放一个 test.pdf
    test_pdf_path = "../data/test.pdf"
    
    if os.path.exists(test_pdf_path):
        manager = DocumentManager()
        # 模拟 User_001 上传了一份私人合同
        chunks = manager.process_pdf_with_permissions(
            file_path=test_pdf_path, 
            uploader_id="User_001", 
            access_level="private"
        )
        
        # 打印第一个知识块的元数据看看效果
        print("\n第一块数据的元数据(Metadata)状态:")
        print(chunks[0].metadata)
    else:
        print("请在 data/ 目录下放置一个 test.pdf 用于测试解析器。")