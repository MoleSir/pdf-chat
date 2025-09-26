from typing import List, Tuple
import faiss
import jieba
import rank_bm25
import numpy as np
from sentence_transformers import SentenceTransformer


class BM25Index:
    def __init__(self):
        self.bm25_index = None

    def build(self, documents: List[str]):
        tokenized_corpus: List[List[str]] = []
        for doc in documents:  
            # str -> list[str]
            tokens = list(jieba.cut(doc))
            tokenized_corpus.append(tokens)
        self.bm25_index = rank_bm25.BM25Okapi(tokenized_corpus)

    def clear(self):
        self.bm25_index = None

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, int]]:
        if not self.bm25_index:
            return []
        
        # 对查询进行分词
        tokenized_query: list[str] = list(jieba.cut(query))
        # 获取 BM25 得分
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # 获取得分最高的 top_k 索引
        top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
        
        # 返回 [(score, index), ...]
        return [(float(bm25_scores[idx]), int(idx)) for idx in top_indices]


class FlatL2Index:
    def __init__(self, embed_model: str = 'all-MiniLM-L6-v2', model_cache_dir: str = './model'):
        self.embed_model = SentenceTransformer(embed_model, cache_folder=model_cache_dir, local_files_only=True)
        self.faiss_index = None

    def build(self, documents: List[str], batch_size: int = 400):
        embeddings = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            # SentenceTransformer 支持直接返回 numpy
            emb = self.embed_model.encode(batch, show_progress_bar=True, convert_to_numpy=True)
            embeddings.append(np.asarray(emb, dtype=np.float32))
        embeddings_np = np.vstack(embeddings).astype('float32')

        dim = embeddings_np.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dim)
        self.faiss_index.add(embeddings_np)

    def clear(self):
        self.faiss_index = None

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, int]]:
        if not self.faiss_index:
            return []
        
        # 编码 query（注意转换为 float32）
        q_emb = self.embed_model.encode([query], convert_to_numpy=True).astype('float32')
        # 执行检索，返回距离 D 和索引位置 I
        D, I = self.faiss_index.search(q_emb, top_k)

        return list(zip(map(float, D[0]), I[0]))


if __name__ == "__main__":
    documents = [
        "苹果公司发布了新的iPhone手机。",
        "谷歌开发了新的搜索算法。",
        "微软推出了Windows系统的新版本。",
        "特斯拉发布了新款电动车。",
        "人工智能正在改变世界。"
    ]

    bm25 = BM25Index()
    bm25.build(documents)
    queries = [
        "iPhone 手机",
        "搜索引擎",
        "电动车",
        "AI 技术"
    ]
    for query in queries:
        print(f"\nQuery: {query}")
        results = bm25.search(query, top_k=3)
        for s, idx in results:
            print(f"  - Doc[{idx}] -> {documents[idx]}")

    floatl2 = FlatL2Index()
    floatl2.build(documents)
    queries = [
        "手机",
        "搜索引擎",
        "电动车",
        "人工智能"
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        results = floatl2.search(q, k=3)
        for score, idx in results:
            print(f"  - dist={score:.4f}, Doc[{idx}] -> {documents[idx]}")
