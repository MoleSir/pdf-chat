from app.index import FlatL2Index, BM25Index
import uuid
import logging
from io import StringIO
from typing import List, Tuple, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text_to_fp
from langchain_text_splitters import RecursiveCharacterTextSplitter
import threading


class KnowledgeBase:
    def __init__(self, embed_model: str = 'all-MiniLM-L6-v2', model_cache_dir: str = './model'):
        # 两种索引
        self.l2_index = FlatL2Index(embed_model, model_cache_dir)
        self.bm25_index =BM25Index()

        # 映射表
        self.contents_map: Dict[str, str] = {}
        self.metadatas_map: Dict[str, Dict[str, Any]] = {}
        self.id_order: List[str] = []

        self.lock = threading.Lock()

    def clear(self):
        with self.lock:
            self.l2_index.clear()
            self.bm25_index.clear()
            self.contents_map.clear()
            self.metadatas_map.clear()
            self.id_order.clear()

    @staticmethod
    def _extract_text(file_path: str) -> str:
        output = StringIO()
        with open(file_path, 'rb') as fh:
            extract_text_to_fp(fh, output)
        return output.getvalue()

    def _extract_pdf(self, file_path: str, chunk_size: int = 400, chunk_overlap: int = 40) -> Tuple[List[str], List, List[str]]:
        text = KnowledgeBase._extract_text(file_path)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "，", "；", "：", " "]
        )
        chunks: List[str] = splitter.split_text(text)
        if not chunks:
            raise ValueError("文档内容为空或无法提取文本")

        doc_id = f"doc_{uuid.uuid4().hex}"
        metadatas = [{"source": file_path, "doc_id": doc_id} for _ in chunks]
        original_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        return chunks, metadatas, original_ids

    def process_pdfs(self, file_paths: List[str], chunk_size: int = 400, chunk_overlap: int = 40) -> Dict[str, Any]:
        processed_results = []
        all_chunks: List[str] = []
        all_metadatas: List = []
        all_original_ids: List[str] = []

        # Step 1: 提取并分块
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "，", "；", "：", " "]
        )
        for idx, file_path in enumerate(file_paths, start=1):
            try:
                # chunks, metadatas, original_ids = self._process_single_pdf(path)
                text = KnowledgeBase._extract_text(file_path)
                chunks: List[str] = splitter.split_text(text)
                if not chunks:
                    raise ValueError("文档内容为空或无法提取文本")
                
                doc_id = f"doc_{uuid.uuid4().hex}"
                metadatas = [{"source": file_path, "doc_id": doc_id} for _ in chunks]
                original_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

                all_chunks.extend(chunks)
                all_metadatas.extend(metadatas)
                all_original_ids.extend(original_ids)
                processed_results.append({"file": file_path, "status": "ok", "chunks": len(chunks)})
            except Exception as e:
                logging.exception(f"Failed processing {file_path}")
                processed_results.append({"file": file_path, "status": "error", "error": str(e)})

        if not all_chunks:
            return {"summary": processed_results, "indexed": 0}

        # Step 2: 建立索引
        with self.lock:
            self.l2_index.clear()
            self.bm25_index.clear()
            self.l2_index.build(all_chunks)
            self.bm25_index.build(all_chunks)

            self.contents_map.clear()
            self.metadatas_map.clear()
            self.id_order.clear()
            for i, original_id in enumerate(all_original_ids):
                self.contents_map[original_id] = all_chunks[i]
                self.metadatas_map[original_id] = all_metadatas[i]
                self.id_order.append(original_id)

        return {"summary": processed_results, "indexed": len(all_chunks)}

    def search(self, query: str, k: int = 5, method: str = "faiss") -> List[Dict[str, Any]]:
        if method == "faiss":
            if not self.l2_index:
                return []
            results = self.l2_index.search(query, k=k)
        elif method == "bm25":
            if not self.bm25_index:
                return []
            results = self.bm25_index.search(query, top_k=k)
        else:
            raise ValueError(f"Unknown search method {method}")

        output = []
        for score, idx in results:
            if idx < 0 or idx >= len(self.id_order):
                continue
            original_id = self.id_order[idx]
            output.append({
                "original_id": original_id,
                "text": self.contents_map.get(original_id),
                "metadata": self.metadatas_map.get(original_id),
                "score": float(score)
            })
        return output


if __name__ == '__main__':
    base = KnowledgeBase()
    base.process_pdfs(['./document/lhc.pdf'])
    for name, chunk in base.contents_map.items():
        print(chunk, '\n\n')