from app.index import FlatL2Index, BM25Index
import uuid
import logging
from io import StringIO
from typing import List, Tuple, Dict, Any
from sentence_transformers import CrossEncoder
from pdfminer.high_level import extract_text_to_fp
from langchain_text_splitters import RecursiveCharacterTextSplitter
import threading
from dataclasses import dataclass


@dataclass
class SearchResult:
    original_id: str
    text: str
    metadata: str 
    score: float


class KnowledgeBase:
    MODEL_CACHE_DIR = './model'

    def __init__(self, embed_model: str = 'all-MiniLM-L6-v2'):
        # 两种索引
        self.l2_index = FlatL2Index(embed_model, KnowledgeBase.MODEL_CACHE_DIR)
        self.bm25_index =BM25Index()
        self.cross_encoder = None

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
            self.cross_encoder = None

    def enable_cross_encoder(self):
        self.cross_encoder = CrossEncoder(
            'sentence-transformers/distiluse-base-multilingual-cased-v2',
            cache_folder=KnowledgeBase.MODEL_CACHE_DIR,
        )

    def disable_cross_encoder(self):
        # 释放保证内存
        self.cross_encoder = None

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

    def search(self, query: str, top_k: int = 5, alpha: float = 0.7) -> List[SearchResult]:
        """
        hybrid_merge
        """
        results_vec = self.l2_index.search(query, top_k=top_k)
        results_bm25 = self.bm25_index.search(query, top_k=top_k)
        results = KnowledgeBase._hybrid_merge_scores(results_vec, results_bm25, alpha)
        return self._pack_search_result(results)

    def search_vector(self, query: str, top_k: int = 5) -> List[SearchResult]:
        results = self.l2_index.search(query, top_k=top_k)
        return self._pack_search_result(results)
    
    def search_bm2(self, query: str, top_k: int = 5) -> List[SearchResult]:
        results = self.bm25_index.search(query, top_k=top_k)
        return self._pack_search_result(results)
    
    def rerank(self, query: str, results: List[SearchResult], top_k: int = 3) -> List[SearchResult]:
        if self.cross_encoder is None:
            return results
        # 构建 cross 模型输入 [(查询,选中的文档)]
        cross_inputs = [[query, result.text] for result in results]

        try:
            # 计算相关性得分
            scores = self.cross_encoder.predict(cross_inputs)
            # 重新赋值得分
            for score, result in zip(scores, results):
                result.score = float(score)
            # 按得分排序
            results = sorted(results, key=lambda x: x.score, reverse=True)
            # 返回前K个结果
            if top_k >= len(results):
                return results
            else:
                return results[:top_k]
        except Exception as e:
            logging.error(f"交叉编码器重排序失败: {str(e)}")
            return results
    
    def _pack_search_result(self, results: List[Tuple[float, int]]) -> List[SearchResult]:
        output = []
        for score, idx in results:
            if idx < 0 or idx >= len(self.id_order):
                continue
            original_id = self.id_order[idx]
            result = SearchResult(
                original_id, 
                self.contents_map.get(original_id),
                self.metadatas_map.get(original_id),
                score
            )
            output.append(result)
        return output

    def _hybrid_merge_scores(
        list1: List[Tuple[float, int]], 
        list2: List[Tuple[float, int]], 
        alpha: float = 0.7
    ) -> List[Tuple[float, int]]:
        """
        Hybrid merge of two score lists, returning List[(score, idx)].
        """
        merged_dict = {}

        # --- list1 ---
        if list1:
            scores1 = [s for s, _ in list1]
            max1, min1 = max(scores1), min(scores1)
            for s, idx in list1:
                norm_s = (s - min1) / (max1 - min1 + 1e-8)
                merged_dict[idx] = alpha * norm_s

        # --- list2 ---
        if list2:
            scores2 = [s for s, _ in list2]
            max2, min2 = max(scores2), min(scores2)
            for s, idx in list2:
                norm_s = (s - min2) / (max2 - min2 + 1e-8)
                if idx in merged_dict:
                    merged_dict[idx] += (1 - alpha) * norm_s
                else:
                    merged_dict[idx] = (1 - alpha) * norm_s

        # --- sort and return (score, idx) ---
        merged_list = sorted(
            [(score, int(idx)) for idx, score in merged_dict.items()],
            key=lambda x: x[0],
            reverse=True
        )
        return merged_list


if __name__ == '__main__':
    base = KnowledgeBase()
    base.process_pdfs(['./document/lhc.pdf'])
    for name, chunk in base.contents_map.items():
        print(chunk, '\n\n')