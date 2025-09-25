# 修正/增强后的 KnowledgeBase 示例（带详细中文注释）

import time
import uuid
import logging
from io import StringIO
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text_to_fp
from langchain_text_splitters import RecursiveCharacterTextSplitter
import threading
import os
import pickle


class KnowledgeBase:
    def __init__(self, embed_model_name: str = 'all-MiniLM-L6-v2', model_cache_dir: str = './model'):
        """
        初始化知识库：
        - 加载 SentenceTransformer 模型（用于生成文本向量）
        - 初始化 FAISS 索引和几个映射字典
        """
        self.embed_model = SentenceTransformer(embed_model_name, cache_folder=model_cache_dir)

        # FAISS 索引对象（向量会存储在这里，常驻内存）
        self.faiss_index: Optional[faiss.Index] = None

        # 三个映射表：
        # 1. original_id -> 文本块内容
        self.faiss_contents_map: Dict[str, str] = {}
        # 2. original_id -> metadata（来源文件路径、doc_id 等）
        self.faiss_metadatas_map: Dict[str, Dict[str, Any]] = {}
        # 3. 向量顺序列表（index 的第 i 个向量对应的 original_id）
        self.faiss_id_order_for_index: List[str] = []

        # 线程锁（保证多线程/并发时索引安全）
        self.lock = threading.Lock()

    def clear(self):
        """
        清空当前索引与映射（相当于重置知识库）
        """
        with self.lock:
            self.faiss_index = None
            self.faiss_contents_map.clear()
            self.faiss_metadatas_map.clear()
            self.faiss_id_order_for_index.clear()

    @staticmethod
    def extract_text(file_path: str) -> str:
        """
        从 PDF 文件中提取纯文本
        - 使用 pdfminer.high_level.extract_text_to_fp
        - 返回字符串形式的全文
        """
        output = StringIO()
        with open(file_path, 'rb') as fh:
            extract_text_to_fp(fh, output)
        return output.getvalue()

    def _process_single_pdf(self, file_path: str, chunk_size: int = 400, chunk_overlap: int = 40) -> Tuple[List[str], List, List[str]]:
        """
        处理单个 PDF 文件：
        1. 提取全文
        2. 按 chunk_size 进行分块（避免向量化时文本太长）
        3. 生成每个 chunk 的 metadata 和 original_id
        """
        text = self.extract_text(file_path)

        # 使用递归分割器，先按段落再按句子再按空格拆分
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "，", "；", "：", " "]
        )
        chunks: List[str] = splitter.split_text(text)

        if not chunks:
            raise ValueError("文档内容为空或无法提取文本")

        # 生成唯一 doc_id（用 uuid 避免冲突）
        doc_id = f"doc_{uuid.uuid4().hex}"

        # 每个 chunk 对应的 metadata
        metadatas = [{"source": file_path, "doc_id": doc_id} for _ in chunks]

        # 每个 chunk 的唯一 ID（doc_id_chunk_index）
        original_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

        return chunks, metadatas, original_ids

    def process_pdfs(self, file_paths: List[str], batch_size: int = 32) -> Dict[str, Any]:
        """
        批量处理多个 PDF 文件，并将所有 chunk 添加到索引中。
        - 会生成向量并添加到 FAISS 索引
        - 会更新内容/元数据映射
        - 返回每个文件的处理结果和索引统计
        """
        processed_results = []   # 保存每个文件的处理情况
        all_chunks: List[str] = []  # 所有 chunk 的文本
        all_metadatas: List[str] = []  # 所有 chunk 的 metadata
        all_original_ids = []    # 所有 chunk 的 original_id

        # Step 1: 提取并分块
        for idx, path in enumerate(file_paths, start=1):
            try:
                chunks, metadatas, original_ids = self._process_single_pdf(path)
                all_chunks.extend(chunks)
                all_metadatas.extend(metadatas)
                all_original_ids.extend(original_ids)
                processed_results.append({"file": path, "status": "ok", "chunks": len(chunks)})
            except Exception as e:
                # 出错时记录日志
                logging.exception(f"Failed processing {path}")
                processed_results.append({"file": path, "status": "error", "error": str(e)})

        if not all_chunks:
            # 如果没有成功提取任何 chunk
            return {"summary": processed_results, "indexed": 0}

        # Step 2: 分批计算向量
        embeddings = []
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            # SentenceTransformer 支持直接返回 numpy
            emb = self.embed_model.encode(batch, show_progress_bar=True, convert_to_numpy=True)
            embeddings.append(np.asarray(emb, dtype=np.float32))
        embeddings_np = np.vstack(embeddings).astype('float32')

        # Step 3: 构建/更新 FAISS 索引
        with self.lock:
            # 构建 faiss 数据库
            self.faiss_index = None
            self.faiss_contents_map.clear()
            self.faiss_metadatas_map.clear()
            self.faiss_id_order_for_index.clear()
            dim = embeddings_np.shape[1]
            self.faiss_index = faiss.IndexFlatL2(dim)
            self.faiss_index.add(embeddings_np)

            # 记录顺序和对应的内容/metadata
            for i, original_id in enumerate(all_original_ids):
                self.faiss_contents_map[original_id] = all_chunks[i]
                self.faiss_metadatas_map[original_id] = all_metadatas[i]
                self.faiss_id_order_for_index.append(original_id)

            total = self.faiss_index.ntotal

        logging.info(f"FAISS 索引构建/扩展完成，共索引 {total} 个向量")

        return {"summary": processed_results, "indexed": len(all_chunks), "total_indexed": total}

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        在索引中搜索：
        1. 把 query 编码成向量
        2. 在 FAISS 索引中查找前 k 个最近邻
        3. 映射回原始文本和 metadata
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []

        # 编码 query（注意转换为 float32）
        q_emb = self.embed_model.encode([query], convert_to_numpy=True).astype('float32')

        # 执行检索，返回距离 D 和索引位置 I
        D, I = self.faiss_index.search(q_emb, k)

        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            original_id = self.faiss_id_order_for_index[idx]
            results.append({
                "original_id": original_id,
                "text": self.faiss_contents_map.get(original_id),
                "metadata": self.faiss_metadatas_map.get(original_id),
                "score": float(dist)   # 距离值，数值越小越相似（因为用 L2）
            })
        return results

    def save(self, dir_path: str):
        """
        保存索引和映射到磁盘：
        - FAISS 索引保存为 index.faiss
        - 其他映射用 pickle 保存
        """
        os.makedirs(dir_path, exist_ok=True)
        with self.lock:
            if self.faiss_index is not None:
                faiss.write_index(self.faiss_index, os.path.join(dir_path, "index.faiss"))
            with open(os.path.join(dir_path, "id_order.pkl"), "wb") as f:
                pickle.dump(self.faiss_id_order_for_index, f)
            with open(os.path.join(dir_path, "contents_map.pkl"), "wb") as f:
                pickle.dump(self.faiss_contents_map, f)
            with open(os.path.join(dir_path, "metadatas_map.pkl"), "wb") as f:
                pickle.dump(self.faiss_metadatas_map, f)

    def load(self, dir_path: str):
        """
        从磁盘加载索引和映射：
        - 读取 FAISS 索引文件
        - 读取 pickle 存储的映射
        """
        with self.lock:
            idx_path = os.path.join(dir_path, "index.faiss")
            if os.path.exists(idx_path):
                self.faiss_index = faiss.read_index(idx_path)
            with open(os.path.join(dir_path, "id_order.pkl"), "rb") as f:
                self.faiss_id_order_for_index = pickle.load(f)
            with open(os.path.join(dir_path, "contents_map.pkl"), "rb") as f:
                self.faiss_contents_map = pickle.load(f)
            with open(os.path.join(dir_path, "metadatas_map.pkl"), "rb") as f:
                self.faiss_metadatas_map = pickle.load(f)



if __name__ == '__main__':
    base = KnowledgeBase()
    base.process_pdfs(['./document/lhc.pdf'])
    for name, chunk in base.faiss_contents_map.items():
        print(chunk, '\n\n')